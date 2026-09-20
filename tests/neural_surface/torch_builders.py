"""Builders for the torch side of this context, kept apart from ``builders.py`` on purpose.

``builders.py`` is imported by every test module here, including the ones that must run with no
optional extra installed, so it may not touch ``torch``. This module may, and only the modules that
already require the extra import it -- plus the contract harness in ``tests/parametric_pricing``,
which imports it behind a ``find_spec`` guard for the same reason it imports ``jax_builders`` that
way.

**A small network and a short budget for the whole session.** A cold start on the shipped defaults
costs about a second, and most tests here are about the port's shape rather than the fit's
accuracy -- purity, determinism, the layout of the answer, what a warm start does to the previous
surface. Those run on :data:`TEST_SETTINGS` and :data:`TEST_SPEC`, a sixteen-by-sixteen network
trained for a few hundred steps, which answers in under a tenth of a second. The tests that are
about accuracy ask for the shipped defaults explicitly, and pay for them.
"""

from __future__ import annotations

from tests.neural_surface.builders import make_grid_spec, make_mesh, make_weighting
from tests.parametric_pricing.builders import make_market_snapshot
from volengine.contracts.calibrated_surface import CalibratedSurface, SurfaceStatus
from volengine.contracts.market_snapshot import MarketSnapshot
from volengine.neural_surface.adapters.torch_learner import (
    NetworkSpec,
    TorchFitSettings,
    TorchLearner,
)
from volengine.neural_surface.application.acl import (
    fit_metrics,
    to_calibrated_surface,
    to_training_batch,
    to_training_samples,
)
from volengine.neural_surface.domain.invariants import ArbitrageMesh
from volengine.neural_surface.domain.training_batch import TrainingBatch

TEST_SPEC = NetworkSpec(hidden=(16, 16))
"""Two hidden layers of sixteen: the same kind of network as the shipped one, a quarter the size."""

TEST_SETTINGS = TorchFitSettings(cold_steps=300, warm_steps=5)
"""A short cold budget and the smallest interesting warm one. Rates, penalties and seed are the
shipped defaults, so a test that bends one of those is bending the only thing it claims to."""

DENSE_MONEYNESS = tuple(-0.40 + 0.05 * step for step in range(17))
"""Seventeen strikes across the quoted band, for the tests that measure how well the network fits:
the five-strike default snapshot is enough to exercise the plumbing and too thin to say anything
about a fit."""


def make_torch_learner(
    settings: TorchFitSettings | None = None,
    spec: NetworkSpec | None = None,
    penalty_mesh: ArbitrageMesh | None = None,
) -> TorchLearner:
    """A learner on the session's small network and short budget, one knob if a test needs it."""
    return TorchLearner(
        penalty_mesh=make_mesh() if penalty_mesh is None else penalty_mesh,
        settings=TEST_SETTINGS if settings is None else settings,
        spec=TEST_SPEC if spec is None else spec,
    )


def make_snapshot_batch(snapshot: MarketSnapshot | None = None) -> TrainingBatch:
    """One published snapshot, translated into the batch the learner is handed, fresh points only.

    Built through the real ACL rather than assembled by hand: what the learner receives *in the
    engine* is what the tests should hand it, and a batch written directly in a test could be one
    no inversion would ever produce.
    """
    snapshot = make_market_snapshot(log_moneyness=DENSE_MONEYNESS) if snapshot is None else snapshot
    fresh = to_training_samples(snapshot, make_weighting())
    batch = to_training_batch(snapshot, fresh=fresh, replayed=())
    assert batch is not None, "the builder's snapshot must produce a trainable batch"
    return batch


def publish_through_learner(snapshot: MarketSnapshot) -> CalibratedSurface:
    """Snapshot to batch to cold start to ``CalibratedSurface``, exactly as the use case would.

    The neural producer's entry in the contract harness. The gate is deliberately not applied
    here: publication is the use case's judgement and is tested there, and what the harness needs
    is the translation, which must work for whatever the learner came back with -- the same terms
    as the harness's own ``published`` for the calibrators.
    """
    batch = make_snapshot_batch(snapshot)
    learner = make_torch_learner()
    surface = learner.update(None, batch)
    published = to_calibrated_surface(
        surface=surface,
        snapshot=snapshot,
        grid=make_grid_spec(),
        producer_id=learner.producer_id,
        surface_id=f"{snapshot.snapshot_id}/{learner.producer_id}",
        ts_calibrated=snapshot.ts_exchange,
        status=SurfaceStatus.OK,
        fit=fit_metrics(surface=surface, fresh=batch.fresh, n_iterations=1, duration_ms=0.0),
    )
    assert published is not None, "a trained surface must be expressible on the published grid"
    return published
