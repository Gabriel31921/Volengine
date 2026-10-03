"""Every degraded path of the engine, walked on the real Deribit chain rather than a generator.

F3-F's review of the paths ``test_degradation.py`` (F2-08) reaches on synthetic data, repeated on
the golden fixture: a snapshot marked degraded by Market Data, a slice refused by the acceptance
rule, a whole cycle refused with a good surface behind it (ADR-006) and without one, and Risk
ageing what it was handed (Design 7.2). Each path is driven through the real use cases with the
example's real configuration, one snapshot at a time -- "replay is a ``for`` statement", as
``CalibrateOnSnapshot`` puts it -- so the sequence is the test's and no scheduler decides it.

**What the review found, and where it is asserted.** Every path behaves as its ADR says, with one
gap: a surface published ``DEGRADED`` -- fitted to a degraded snapshot, or missing an expiry whose
slice was refused -- reaches Risk as a report marked ``NORMAL``. ``SurfaceView`` carries no status
by design, on the argument that ``ts_snapshot`` already says everything a status would; that is
true of ``STALE_REPUBLISH`` and not of ``DEGRADED``, which is not a matter of age. The degradation
is counted (``risk.surface.received{status=DEGRADED}``) and is not printed on the report. Asserted
below as it stands, and recorded in ``docs/SEAMS.md``: closing it changes Risk, which F3-F is the
stage that proves does not need to change for a second market.
"""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timedelta

import pytest

from tests.entrypoints.builders import RefusingAfter
from tests.entrypoints.test_deribit_fixture import EXAMPLE, published_snapshots
from tests.support import RecordingMetrics
from volengine.contracts.calibrated_surface import CalibratedSurface, SurfaceStatus
from volengine.contracts.events import CalibrationFailed, Event, SurfaceCalibrated
from volengine.contracts.market_snapshot import MarketSnapshot
from volengine.entrypoints.config import AppConfig, load_config
from volengine.parametric_pricing.adapters.scipy_calibrator import ScipyCalibrator
from volengine.parametric_pricing.application.calibrate_on_snapshot import (
    Acceptance,
    CalibrateOnSnapshot,
)
from volengine.parametric_pricing.application.calibration_state import CalibrationState
from volengine.parametric_pricing.domain.ports import Calibrator
from volengine.platform.clock import ManualClock
from volengine.risk.application.compute_report import ComputeReportUseCase
from volengine.risk.application.surface_cache import LastValueSurfaceProvider, ProducerSurfaces
from volengine.risk.domain.freshness_policy import FreshnessDecision
from volengine.risk.domain.risk_report import RiskReport

pytestmark = pytest.mark.e2e

RMSE_BETWEEN_THE_SLICES_BP = 50.0
"""An acceptance threshold between the fixture's two slices: the three-month one fits to 18.7 bp
and the seven-day one to 80.6 bp (``examples/deribit-live.toml``), so this refuses one of them."""

HIGHER_THAN_MEASURED_COVERAGE = 0.9
"""A coverage floor above the 0.66 admissible ratio the fixture holds at every snapshot."""


class Session:
    """The fit and the report of one producer on the fixture's market, on one manual clock.

    The clock is moved to each snapshot's own receipt instant before it is fitted, and wherever a
    test sends it before a report -- which is how Risk's ageing is reached without waiting.
    """

    def __init__(
        self,
        config: AppConfig,
        calibrator: Calibrator | None = None,
        acceptance: Acceptance | None = None,
    ) -> None:
        calibration = config.calibration
        inner = calibrator if calibrator is not None else ScipyCalibrator(calibration.fit)
        self.clock = ManualClock(published_snapshots()[0].ts_local)
        self.metrics = RecordingMetrics()
        self.fit = CalibrateOnSnapshot(
            calibrator=inner,
            state=CalibrationState(),
            clock=self.clock,
            metrics=self.metrics,
            weighting=calibration.weighting,
            grid=calibration.grid,
            acceptance=acceptance if acceptance is not None else calibration.acceptance,
        )
        self.cache = LastValueSurfaceProvider(clock=self.clock, metrics=self.metrics)
        self.report = ComputeReportUseCase(
            provider=ProducerSurfaces(self.cache, inner.producer_id),
            portfolio=config.risk.portfolio,
            policy=config.risk.freshness,
            clock=self.clock,
            metrics=self.metrics,
            settings=config.risk.settings,
            expected_producer_id=inner.producer_id,
        )
        self.market_id = config.markets[0].market_id

    def handle(self, snapshot: MarketSnapshot) -> tuple[Event, ...]:
        """Fit at the snapshot's own instant, and hand every published surface to Risk."""
        self.move_to(snapshot.ts_local)
        events = self.fit.handle(snapshot)
        for event in events:
            if isinstance(event, SurfaceCalibrated):
                self.cache.accept(event.surface)
        return events

    def move_to(self, instant: datetime) -> None:
        self.clock.advance((instant - self.clock.now()).total_seconds())

    def value(self) -> RiskReport:
        return self.report.compute(self.market_id)


@pytest.fixture(scope="module")
def config() -> AppConfig:
    return load_config(EXAMPLE)


@pytest.fixture(scope="module")
def snapshots() -> list[MarketSnapshot]:
    return published_snapshots()


def surface_in(events: tuple[Event, ...]) -> CalibratedSurface:
    published = [event.surface for event in events if isinstance(event, SurfaceCalibrated)]
    assert len(published) == 1
    return published[0]


# --- a degraded snapshot: Market Data flags, the calibrator still fits


def test_a_low_coverage_venue_chain_is_published_degraded(config: AppConfig) -> None:
    market = config.markets[0]
    strict = replace(
        market,
        snapshot=replace(
            market.snapshot,
            material_move_threshold=0.0,
            max_quiet_seconds=None,
            min_coverage_ratio=HIGHER_THAN_MEASURED_COVERAGE,
        ),
    )

    degraded = published_snapshots(strict)

    assert len(degraded) >= 20
    assert all(one.quality.degraded for one in degraded)


def test_a_degraded_venue_chain_is_still_fitted_and_its_surface_says_so(
    config: AppConfig, snapshots: list[MarketSnapshot]
) -> None:
    """Ingestion flags, the calibrator decides: the fit runs and labels what it publishes."""
    session = Session(config)
    last = snapshots[-1]
    marked = replace(last, quality=replace(last.quality, degraded=True))

    surface = surface_in(session.handle(marked))

    assert surface.status is SurfaceStatus.DEGRADED
    assert len(surface.grid.tenors) == 2


# --- a slice refused, the rest published


def test_a_slice_over_the_rmse_is_dropped_and_the_surface_goes_out_degraded(
    config: AppConfig, snapshots: list[MarketSnapshot]
) -> None:
    session = Session(config, acceptance=Acceptance(max_rmse_vol_bp=RMSE_BETWEEN_THE_SLICES_BP))

    surface = surface_in(session.handle(snapshots[-1]))

    assert surface.status is SurfaceStatus.DEGRADED
    assert [expiry.date().isoformat() for expiry in surface.grid.expiries] == ["2026-12-25"]
    assert "pricing.slice.rmse_exceeded" in session.metrics.counter_names()


def test_the_example_s_own_acceptance_keeps_both_slices(
    config: AppConfig, snapshots: list[MarketSnapshot]
) -> None:
    """The guard on the test above: the drop is the threshold's, not the fixture's."""
    surface = surface_in(Session(config).handle(snapshots[-1]))

    assert surface.status is SurfaceStatus.OK
    assert len(surface.grid.expiries) == 2


def test_a_degraded_surface_values_the_book_as_normal_and_is_only_counted(
    config: AppConfig, snapshots: list[MarketSnapshot]
) -> None:
    """The gap the review found, as it stands: the label stops at Risk's boundary.

    The book is written on 25DEC26, the slice that survived, so it is valued -- and the report says
    ``NORMAL`` about a surface its producer published as ``DEGRADED``. Only the metric knows.
    """
    session = Session(config, acceptance=Acceptance(max_rmse_vol_bp=RMSE_BETWEEN_THE_SLICES_BP))
    session.handle(snapshots[-1])

    report = session.value()

    assert report.positions
    assert report.freshness is FreshnessDecision.NORMAL
    assert any(
        name == "risk.surface.received" and tags.get("status") == SurfaceStatus.DEGRADED.value
        for name, _, tags in session.metrics.counters
    )


# --- a whole cycle refused (ADR-006)


def test_a_refusal_with_nothing_behind_it_publishes_only_the_failure(
    config: AppConfig, snapshots: list[MarketSnapshot]
) -> None:
    session = Session(config, calibrator=RefusingAfter(ScipyCalibrator(), good_cycles=0))

    events = session.handle(snapshots[-1])

    assert [type(event) for event in events] == [CalibrationFailed]


def test_risk_with_no_surface_at_all_says_so(
    config: AppConfig, snapshots: list[MarketSnapshot]
) -> None:
    session = Session(config, calibrator=RefusingAfter(ScipyCalibrator(), good_cycles=0))
    session.handle(snapshots[-1])

    report = session.value()

    assert report.freshness is FreshnessDecision.REJECT
    assert report.ts_snapshot is None
    assert report.message is not None and "no surface" in report.message


@pytest.fixture(scope="module")
def republished(
    config: AppConfig, snapshots: list[MarketSnapshot]
) -> tuple[Session, tuple[Event, ...], MarketSnapshot]:
    """A good cycle on one real snapshot, then a refused cycle on a later one."""
    session = Session(
        config, calibrator=RefusingAfter(ScipyCalibrator(config.calibration.fit), good_cycles=1)
    )
    good = snapshots[-2]
    first = surface_in(session.handle(good))
    assert first.status is SurfaceStatus.OK
    return session, session.handle(snapshots[-1]), good


def test_a_refusal_after_a_good_cycle_republishes_the_last_good_surface(
    republished: tuple[Session, tuple[Event, ...], MarketSnapshot],
) -> None:
    _, events, good = republished

    assert [type(event) for event in events] == [CalibrationFailed, SurfaceCalibrated]
    stale = surface_in(events)
    assert stale.status is SurfaceStatus.STALE_REPUBLISH
    assert stale.source_snapshot_id == good.snapshot_id
    assert stale.ts_snapshot == good.ts_exchange


def test_risk_ages_a_republished_surface_from_its_original_snapshot(
    config: AppConfig, republished: tuple[Session, tuple[Event, ...], MarketSnapshot]
) -> None:
    """Design 7.2 on real data: NORMAL, then DEGRADED, then REJECT, as the snapshot ages."""
    session, events, _ = republished
    stale = surface_in(events)
    policy = config.risk.freshness
    reports: list[RiskReport] = []
    for age in (1.0, policy.warn_seconds + 1.0, policy.reject_seconds + 1.0):
        session.move_to(stale.ts_snapshot + timedelta(seconds=age))
        reports.append(session.value())

    assert [report.freshness for report in reports] == [
        FreshnessDecision.NORMAL,
        FreshnessDecision.DEGRADED,
        FreshnessDecision.REJECT,
    ]
    assert reports[1].positions
    assert reports[2].positions == ()
    assert reports[2].message is not None and "no valid surface" in reports[2].message
