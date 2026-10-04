"""A real network with NaN weights, through the use case: the F3-D debt row's regression test.

``test_train_on_snapshot.py`` covers the divergence path with a surface written as a formula. This
module covers it with the thing that actually diverges in production: a ``TorchSurface`` whose
weights one bad gradient step has turned to NaN. The learner is the real one for the good cycle and
a stub handing back the poisoned surface for the bad one, because no configuration of a healthy
learner diverges on demand -- and a test that waited for one would be testing the optimiser's luck.

Torch-only, so the file name matches ``conftest.py``'s collection rule and is skipped without the
``neural`` extra.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
import torch

from tests.neural_surface.builders import (
    StubLearner,
    make_buffer,
    make_grid_spec,
    make_mesh,
    make_schedule,
    make_thresholds,
    make_weighting,
)
from tests.neural_surface.torch_builders import make_snapshot_batch, make_torch_learner
from tests.parametric_pricing.builders import NOW, make_market_snapshot
from tests.support import RecordingMetrics
from volengine.contracts.calibrated_surface import SurfaceStatus
from volengine.contracts.events import CalibrationFailed, SurfaceCalibrated
from volengine.neural_surface.adapters.torch_learner import TorchSurface
from volengine.neural_surface.application.train_on_snapshot import TrainOnSnapshot
from volengine.neural_surface.application.training_state import TrainingState
from volengine.neural_surface.domain.errors import SurfaceEvaluationError
from volengine.neural_surface.domain.invariants import check_surface
from volengine.platform.clock import ManualClock


def poisoned(surface: TorchSurface) -> TorchSurface:
    """The same network with every weight NaN -- what a single exploding step leaves behind."""
    return replace(
        surface, weights=tuple(torch.full_like(w, float("nan")) for w in surface.weights)
    )


def trained_surface() -> TorchSurface:
    return make_torch_learner().update(None, make_snapshot_batch())


def make_use_case(learner: StubLearner) -> TrainOnSnapshot:
    return TrainOnSnapshot(
        learner=learner,
        state=TrainingState(),
        buffer=make_buffer(),
        clock=ManualClock(NOW),
        metrics=RecordingMetrics(),
        weighting=make_weighting(),
        grid=make_grid_spec(),
        mesh=make_mesh(),
        # Generous: the small test network's calendar crossings are of order 1e-4 (docs/SEAMS.md),
        # and the good cycle has to be published for there to be anything to republish.
        thresholds=make_thresholds(butterfly=1e-3, calendar=1e-3),
        schedule=make_schedule(),
        rng=np.random.default_rng(20261004),
    )


def test_a_nan_weight_network_really_is_unevaluable() -> None:
    """The guard on the test below: the poisoned surface is refused by the domain, not by luck."""
    with pytest.raises(SurfaceEvaluationError):
        check_surface(poisoned(trained_surface()), make_mesh())


def test_a_nan_weight_network_yields_the_refuse_and_republish_pair() -> None:
    good = trained_surface()
    learner = StubLearner(surface=good, producer_id="mlp-torch")
    use_case = make_use_case(learner)
    published = use_case.handle(make_market_snapshot())
    assert isinstance(published[0], SurfaceCalibrated), "the good cycle must be published"

    learner.answer_with(poisoned(good))
    events = use_case.handle(make_market_snapshot(snapshot_id="BTC-DERIBIT:00000001"))

    assert len(events) == 2
    failed, republished = events
    assert isinstance(failed, CalibrationFailed)
    assert "diverged" in failed.reason
    assert isinstance(republished, SurfaceCalibrated)
    assert republished.surface.status is SurfaceStatus.STALE_REPUBLISH
    assert republished.surface.surface_id == published[0].surface.surface_id
