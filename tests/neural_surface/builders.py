"""Valid Neural Surface objects, one per type, with a knob for every field a test bends.

The convention this repo tests by: a builder returns **one valid object**, and a test changes the
minimum needed to make its point -- either through a keyword here or through
``dataclasses.replace``, which re-runs ``__post_init__``. What a test says is then exactly what it
is probing, instead of five parameters of noise around one poisoned value.

Shared from a module rather than from ``conftest.py``: conftest is where pytest looks for fixtures
and hooks it *injects*, and importing from it is discouraged because it is loaded by collection
magic rather than by an import anyone can follow. See ``tests/support.py``.

This context needs one thing the others do not: **stand-in surfaces**. ``LearnedSurface`` is a
Protocol whose only production implementation is a PyTorch model that does not exist yet and that
the domain is forbidden to import anyway, so the doubles below are how the gate of ADR-010 gets
tested at all. That is the argument for the Protocol made concrete -- an invariant checkable
against a surface written in four lines of numpy is an invariant that can be trusted before the
model it will judge has been trained even once.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta

import numpy as np
from numpy.typing import NDArray

from volengine.neural_surface.application.acl import Weighting
from volengine.neural_surface.application.grid_spec import GridSpec
from volengine.neural_surface.application.train_on_snapshot import GateThresholds, TrainingSchedule
from volengine.neural_surface.domain.errors import NeuralSurfaceError
from volengine.neural_surface.domain.invariants import ArbitrageMesh, ArbitrageReport
from volengine.neural_surface.domain.learned_surface import LearnedSurface
from volengine.neural_surface.domain.replay_buffer import ReplayBuffer, StratificationSpec
from volengine.neural_surface.domain.training_batch import TrainingBatch, TrainingSample

NAIVE = datetime(2026, 7, 27, 12, 0)
"""An instant with no zone. Every entry point in this context must refuse it."""

NOW = datetime(2026, 7, 27, 12, 0, tzinfo=UTC)
"""The snapshot instant, comfortably before every expiry these builders imply."""

NEAR_TENOR = 1.0 / 12.0
FAR_TENOR = 3.0 / 12.0
"""One and three months in years, rounded clean so a test can do the algebra by hand."""


def make_sample(
    log_moneyness: float = -0.05,
    tenor_years: float = NEAR_TENOR,
    implied_vol: float = 0.65,
    weight: float = 0.30,
    ts_observed: datetime = NOW,
) -> TrainingSample:
    """One quote as the network sees it: just off the forward, at one month, at 65% vol.

    Crypto vols run high, so 0.65 is an ordinary at-the-money number rather than a stressed one.
    Its total variance is ``0.65**2 / 12``, which no test should have to look up: the point of a
    single default is that the arithmetic is reproducible in a comment.
    """
    return TrainingSample(
        log_moneyness=log_moneyness,
        tenor_years=tenor_years,
        implied_vol=implied_vol,
        weight=weight,
        ts_observed=ts_observed,
    )


FRESH_SAMPLES = (
    make_sample(log_moneyness=-0.05, tenor_years=NEAR_TENOR, implied_vol=0.65, weight=0.30),
    make_sample(log_moneyness=0.10, tenor_years=NEAR_TENOR, implied_vol=0.62, weight=0.30),
)
"""Two quotes from this snapshot, at the near tenor, either side of the forward."""

REPLAYED_SAMPLES = (
    make_sample(
        log_moneyness=-0.25,
        tenor_years=FAR_TENOR,
        implied_vol=0.70,
        weight=0.20,
        ts_observed=NOW - timedelta(minutes=5),
    ),
    make_sample(
        log_moneyness=0.30,
        tenor_years=FAR_TENOR,
        implied_vol=0.66,
        weight=0.20,
        ts_observed=NOW - timedelta(minutes=12),
    ),
)
"""Two older wing quotes at the far tenor, which is exactly what the replay buffer is for: the
fresh pair above concentrates near the money and at one expiry, and a batch of only those would
teach the network to forget everywhere else."""


def make_batch(
    market_id: str = "BTC-DERIBIT",
    snapshot_id: str = "snap-000001",
    ts_snapshot: datetime = NOW,
    samples: tuple[TrainingSample, ...] | None = None,
    n_fresh: int = 2,
) -> TrainingBatch:
    """One fine-tuning step: two fresh quotes followed by two drawn from the buffer.

    The default respects the ordering convention -- ``samples[:n_fresh]`` are this snapshot's --
    so a test that changes ``n_fresh`` alone is testing the split and nothing else. Weights sum to
    one, which is not required but makes any weighted mean in an assertion readable.
    """
    if samples is None:
        samples = FRESH_SAMPLES + REPLAYED_SAMPLES
    return TrainingBatch(
        market_id=market_id,
        snapshot_id=snapshot_id,
        ts_snapshot=ts_snapshot,
        samples=samples,
        n_fresh=n_fresh,
    )


def make_spec(
    moneyness_edges: tuple[float, ...] = (-0.40, -0.10, 0.10, 0.40),
    tenor_edges: tuple[float, ...] = (0.02, 0.10, 0.50),
    capacity_per_cell: int = 3,
    max_age_seconds: float = 600.0,
) -> StratificationSpec:
    """Six cells: three moneyness bands by two tenor bands, three samples each, ten-minute memory.

    Small on purpose. A capacity of three is the smallest that lets eviction happen twice in a row
    without emptying the cell, and six cells is enough for a stratified draw to visibly differ
    from a uniform one while still being countable by hand in an assertion.
    """
    return StratificationSpec(
        moneyness_edges=moneyness_edges,
        tenor_edges=tenor_edges,
        capacity_per_cell=capacity_per_cell,
        max_age_seconds=max_age_seconds,
    )


def make_buffer(
    spec: StratificationSpec | None = None,
    samples: tuple[TrainingSample, ...] = (),
) -> ReplayBuffer:
    """A buffer over :func:`make_spec`'s cells, optionally pre-filled.

    ``spec`` defaults to ``None`` rather than to ``make_spec()`` because a default argument is
    evaluated once at import time, and a buffer is the one mutable object in this context: two
    tests sharing one would share its contents.
    """
    buffer = ReplayBuffer(spec=make_spec() if spec is None else spec)
    buffer.extend(samples)
    return buffer


MESH_K = tuple(-0.60 + 0.05 * step for step in range(25))
"""A uniform moneyness mesh from -60% to +60% in steps of 5%.

Built by repeated multiply-add rather than by accumulation so that consecutive differences agree
to the last bit or two: the mesh has to satisfy the uniformity the gate enforces, and a mesh
assembled by ``x += h`` drifts far enough to fail it. Twenty-five points leaves twenty-three after
the two edges are dropped by the central differences."""


def make_mesh(
    log_moneyness: tuple[float, ...] = MESH_K,
    tenors: tuple[float, ...] = (NEAR_TENOR, FAR_TENOR),
) -> ArbitrageMesh:
    """The dense mesh the hard gate is evaluated on: 25 moneyness points at two tenors.

    Two tenors, not one, because it is the smallest mesh on which the calendar condition says
    anything at all.
    """
    return ArbitrageMesh(log_moneyness=log_moneyness, tenors=tenors)


def make_report(
    butterfly_violation: float = 0.0,
    calendar_violation: float = 0.0,
    n_points_judged: int = 46,
    n_butterfly_violations: int = 0,
    n_calendar_violations: int = 0,
) -> ArbitrageReport:
    """A clean verdict over :func:`make_mesh`: 23 judged points at each of two tenors, no breach.

    Every count is consistent with the mesh above, so a test that bends one number is bending the
    only thing it claims to be about.
    """
    return ArbitrageReport(
        butterfly_violation=butterfly_violation,
        calendar_violation=calendar_violation,
        n_points_judged=n_points_judged,
        n_butterfly_violations=n_butterfly_violations,
        n_calendar_violations=n_calendar_violations,
    )


@dataclass(frozen=True, slots=True)
class FlatVolSurface:
    """A surface at one volatility everywhere: ``w(k, T) = vol**2 * T``. Arbitrage-free by hand.

    The reference against which the gate's own correctness is checked. Flat in ``k`` means
    ``w' = w'' = 0``, so Durrleman's function collapses to exactly ``1`` at every point -- a
    verdict a test can assert on the nose rather than within a tolerance -- and total variance
    grows linearly in the tenor, which is the calendar condition satisfied with room to spare.

    A frozen dataclass with a plain ``version`` attribute, which satisfies the Protocol's read-only
    property: structural conformance never asks how a member is stored.
    """

    vol: float = 0.65
    version: int = 1

    def total_variance(
        self, k: NDArray[np.float64], tenors: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """``(len(tenors), len(k))`` of ``vol**2 * T``, constant along the moneyness axis."""
        column = np.asarray(tenors, dtype=np.float64) * self.vol * self.vol
        return np.repeat(column[:, None], len(k), axis=1)


@dataclass(frozen=True, slots=True)
class CallableSurface:
    """Any surface a test can write as a formula in ``(k, T)``, healthy or poisoned.

    ``fn`` receives two arrays already broadcast to the full ``(len(tenors), len(k))`` mesh and
    returns the total variance on it, so a test states the shape it wants -- a smile, a wing that
    dives negative, a calendar that runs backwards, a grid of NaN -- in one expression, with no
    optimiser and no model in the way.
    """

    fn: Callable[[NDArray[np.float64], NDArray[np.float64]], NDArray[np.float64]]
    version: int = 1

    def total_variance(
        self, k: NDArray[np.float64], tenors: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """Evaluate ``fn`` over the mesh, tenor-major."""
        k_mesh, tenor_mesh = np.meshgrid(
            np.asarray(k, dtype=np.float64), np.asarray(tenors, dtype=np.float64)
        )
        return np.asarray(self.fn(k_mesh, tenor_mesh), dtype=np.float64)


@dataclass(frozen=True, slots=True)
class TransposedSurface:
    """A surface that answers with the axes the wrong way round: ``(len(k), len(tenors))``.

    The mistake a real adapter makes on its first day, and the one a shape check exists for. It is
    also the mistake that would otherwise pass unnoticed on a square mesh, which is why no builder
    here offers equal counts on both axes.
    """

    vol: float = 0.65
    version: int = 1

    def total_variance(
        self, k: NDArray[np.float64], tenors: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """The right numbers with the axes swapped."""
        column = np.asarray(k, dtype=np.float64) * 0.0 + self.vol * self.vol
        return np.repeat(column[:, None], len(tenors), axis=1)


@dataclass(frozen=True, slots=True)
class CountingSurface:
    """A flat surface that records every mesh it was asked to evaluate.

    For the tests that assert *who* evaluates and how often -- that the gate evaluates the surface
    once rather than once per condition, and that it passes the mesh it was given rather than one
    of its own.
    """

    vol: float = 0.65
    version: int = 1
    calls: list[tuple[int, int]] = field(default_factory=list)

    def total_variance(
        self, k: NDArray[np.float64], tenors: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """Record the mesh dimensions, then answer like :class:`FlatVolSurface`."""
        self.calls.append((len(tenors), len(k)))
        column = np.asarray(tenors, dtype=np.float64) * self.vol * self.vol
        return np.repeat(column[:, None], len(k), axis=1)


def make_weighting(
    spread_scale: float = 0.05,
    flagged_factor: float = 0.25,
    unpaired_itm_factor: float = 0.10,
) -> Weighting:
    """The same three numbers the parametric builder uses, and they have to stay the same.

    Design 6.5 compares two producers fitted to one market; the weights are what that comparison
    holds constant. A test suite that drifted them apart would be silently testing two different
    experiments.
    """
    return Weighting(
        spread_scale=spread_scale,
        flagged_factor=flagged_factor,
        unpaired_itm_factor=unpaired_itm_factor,
    )


def make_grid_spec(k_min: float = -0.4, k_max: float = 0.4, n_nodes: int = 9) -> GridSpec:
    """A mesh wider than the quoted band, matching the parametric producer's node for node."""
    return GridSpec(k_min=k_min, k_max=k_max, n_nodes=n_nodes)


def make_schedule(
    replay_size: int = 4,
    restart_seconds: float | None = None,
) -> TrainingSchedule:
    """Replay on, restarts off. A restart is what a test asking about one switches on."""
    return TrainingSchedule(replay_size=replay_size, restart_seconds=restart_seconds)


def make_thresholds(butterfly: float = 1e-6, calendar: float = 1e-6) -> GateThresholds:
    """Tight enough that a genuinely broken surface is refused, loose enough for rounding."""
    return GateThresholds(butterfly=butterfly, calendar=calendar)


class StubLearner:
    """A ``SurfaceLearner`` that hands back whatever surface the test put in it.

    Records the ``previous`` it was given on each call, which is the only way to assert that
    fine-tuning is actually chained -- a learner that accepted a history and ignored it would turn
    continuous training into a cold restart every snapshot with nothing raising.
    """

    def __init__(
        self,
        surface: LearnedSurface | None = None,
        producer_id: str = "mlp-stub",
        failure: NeuralSurfaceError | None = None,
    ) -> None:
        self._surface = surface if surface is not None else FlatVolSurface()
        self._producer_id = producer_id
        self._failure = failure
        self.calls: list[tuple[LearnedSurface | None, TrainingBatch]] = []

    @property
    def producer_id(self) -> str:
        return self._producer_id

    def answer_with(self, surface: LearnedSurface) -> None:
        """Change what the next call returns, so one test can drive two consecutive cycles."""
        self._surface = surface

    def update(self, previous: LearnedSurface | None, batch: TrainingBatch) -> LearnedSurface:
        self.calls.append((previous, batch))
        if self._failure is not None:
            raise self._failure
        return self._surface
