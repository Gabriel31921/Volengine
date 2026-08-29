"""One calibration cycle, judged by what it publishes.

The acceptance rule of ADR-006 and the stale republish behind it are the two behaviours this
context exists to demonstrate, and both are asserted here by calling one method and looking at
the events that came back. No bus, no event loop, no waiting -- which is the whole argument for
the handler shape this use case has.
"""

from __future__ import annotations

from dataclasses import replace

import pytest

from tests.parametric_pricing.builders import (
    FAR,
    FAR_TENOR,
    NEAR,
    NEAR_TENOR,
    NOW,
    StubCalibrator,
    make_calibration_result,
    make_grid_spec,
    make_market_snapshot,
    make_params,
    make_slice_result,
    make_weighting,
)
from tests.support import RecordingMetrics
from volengine.contracts.calibrated_surface import SurfaceStatus
from volengine.contracts.events import CalibrationFailed, SurfaceCalibrated
from volengine.parametric_pricing.application.calibrate_on_snapshot import (
    Acceptance,
    CalibrateOnSnapshot,
)
from volengine.parametric_pricing.application.calibration_state import CalibrationState
from volengine.parametric_pricing.domain.calibration import CalibrationResult
from volengine.parametric_pricing.domain.errors import CalibrationError
from volengine.parametric_pricing.domain.svi_slice import SVIParams
from volengine.platform.clock import ManualClock


def make_use_case(
    calibrator: StubCalibrator | None = None,
    max_rmse_vol_bp: float = 50.0,
    state: CalibrationState | None = None,
    clock: ManualClock | None = None,
) -> tuple[CalibrateOnSnapshot, StubCalibrator, CalibrationState, RecordingMetrics]:
    calibrator = calibrator if calibrator is not None else StubCalibrator()
    state = state if state is not None else CalibrationState()
    metrics = RecordingMetrics()
    use_case = CalibrateOnSnapshot(
        calibrator=calibrator,
        state=state,
        clock=clock if clock is not None else ManualClock(NOW),
        metrics=metrics,
        weighting=make_weighting(),
        grid=make_grid_spec(),
        acceptance=Acceptance(max_rmse_vol_bp=max_rmse_vol_bp),
    )
    return use_case, calibrator, state, metrics


def surfaces(events: tuple[object, ...]) -> list[SurfaceCalibrated]:
    return [event for event in events if isinstance(event, SurfaceCalibrated)]


def failures(events: tuple[object, ...]) -> list[CalibrationFailed]:
    return [event for event in events if isinstance(event, CalibrationFailed)]


def fitted_result(
    rmse_vol_bp: float = 12.0,
    max_err_vol_bp: float = 31.0,
    converged: bool = True,
    at_bound: bool = False,
    params: SVIParams | None = None,
) -> CalibrationResult:
    """A result covering both expiries of the default snapshot, with one knob per slice."""
    return CalibrationResult(
        slices=tuple(
            make_slice_result(
                expiry=expiry,
                tenor_years=tenor,
                rmse_vol_bp=rmse_vol_bp,
                max_err_vol_bp=max_err_vol_bp,
                converged=converged,
                at_bound=at_bound,
                params=params,
            )
            for expiry, tenor in ((NEAR, NEAR_TENOR), (FAR, FAR_TENOR))
        ),
        n_iterations=5,
        duration_ms=2.0,
    )


# --- the accepted cycle


def test_an_accepted_fit_publishes_one_surface() -> None:
    use_case, _, _, _ = make_use_case()

    events = use_case.handle(make_market_snapshot())

    assert len(events) == 1
    assert len(surfaces(events)) == 1


def test_the_published_surface_names_the_snapshot_it_was_fitted_to() -> None:
    """The link that makes the pipeline attributable: arrival order proves nothing (ADR-003)."""
    use_case, _, _, _ = make_use_case()
    snapshot = make_market_snapshot(snapshot_id="BTC-DERIBIT:00000007")

    surface = surfaces(use_case.handle(snapshot))[0].surface

    assert surface.source_snapshot_id == "BTC-DERIBIT:00000007"


def test_the_published_surface_carries_the_snapshots_own_instant() -> None:
    """Not the fit's: staleness downstream is measured against the market (ADR-006)."""
    use_case, _, _, _ = make_use_case()

    surface = surfaces(use_case.handle(make_market_snapshot()))[0].surface

    assert surface.ts_snapshot == NOW


def test_two_producers_fitting_one_snapshot_publish_distinguishable_surfaces() -> None:
    """An id built from the snapshot alone would give both producers' surfaces one identity."""
    snapshot = make_market_snapshot()
    svi, _, _, _ = make_use_case(calibrator=StubCalibrator(producer_id="svi-scipy"))
    neural, _, _, _ = make_use_case(calibrator=StubCalibrator(producer_id="mlp-torch"))

    first = surfaces(svi.handle(snapshot))[0].surface
    second = surfaces(neural.handle(snapshot))[0].surface

    assert first.surface_id != second.surface_id
    assert first.source_snapshot_id == second.source_snapshot_id


def test_the_same_snapshot_always_mints_the_same_surface_id() -> None:
    use_case, _, _, _ = make_use_case()
    snapshot = make_market_snapshot()

    first = surfaces(use_case.handle(snapshot))[0].surface
    second = surfaces(use_case.handle(snapshot))[0].surface

    assert first.surface_id == second.surface_id


# --- the warm start


def test_the_first_cycle_is_a_cold_start() -> None:
    use_case, calibrator, _, _ = make_use_case()

    use_case.handle(make_market_snapshot())

    assert calibrator.calls == [None]


def test_the_second_cycle_starts_from_the_first_ones_parameters() -> None:
    """A calibrator that accepted a history and ignored it would pass every other test here."""
    use_case, calibrator, _, _ = make_use_case()
    use_case.handle(make_market_snapshot())

    use_case.handle(make_market_snapshot(snapshot_id="BTC-DERIBIT:00000001"))

    warm = calibrator.calls[1]
    assert warm is not None
    assert set(warm) == {NEAR, FAR}


def test_a_refused_slice_does_not_become_a_warm_start() -> None:
    """Starting the next search from a pinned parameter walks it straight back to the same edge."""
    calibrator = StubCalibrator(result=fitted_result(at_bound=True))
    use_case, _, state, _ = make_use_case(calibrator=calibrator)

    use_case.handle(make_market_snapshot())

    assert state.warm_start is None


# --- the acceptance rule


def test_a_slice_over_the_rmse_threshold_is_not_published() -> None:
    calibrator = StubCalibrator(result=fitted_result(rmse_vol_bp=90.0, max_err_vol_bp=200.0))
    use_case, _, _, _ = make_use_case(calibrator=calibrator, max_rmse_vol_bp=50.0)

    events = use_case.handle(make_market_snapshot())

    assert failures(events)
    assert not surfaces(events)


def test_a_slice_pinned_against_a_bound_is_not_published_however_good_its_rmse() -> None:
    """The residual of a projection onto the boundary, not the residual of a fit (ADR-006)."""
    calibrator = StubCalibrator(result=fitted_result(rmse_vol_bp=1.0, at_bound=True))
    use_case, _, _, _ = make_use_case(calibrator=calibrator)

    assert not surfaces(use_case.handle(make_market_snapshot()))


def test_a_slice_that_ran_out_of_budget_is_not_published() -> None:
    calibrator = StubCalibrator(result=fitted_result(rmse_vol_bp=1.0, converged=False))
    use_case, _, _, _ = make_use_case(calibrator=calibrator)

    assert not surfaces(use_case.handle(make_market_snapshot()))


def test_a_healthy_slice_is_published() -> None:
    """The vacuous-pass guard: the rule above must not simply reject everything."""
    calibrator = StubCalibrator(result=fitted_result(rmse_vol_bp=1.0))
    use_case, _, _, _ = make_use_case(calibrator=calibrator)

    assert surfaces(use_case.handle(make_market_snapshot()))


def test_a_surface_missing_one_expiry_is_published_as_degraded() -> None:
    """A hole in the term structure the grid alone cannot explain."""
    calibrator = StubCalibrator(
        result=CalibrationResult(
            slices=(
                make_slice_result(expiry=NEAR, tenor_years=NEAR_TENOR, rmse_vol_bp=1.0),
                make_slice_result(expiry=FAR, tenor_years=FAR_TENOR, at_bound=True),
            ),
            n_iterations=5,
            duration_ms=2.0,
        )
    )
    use_case, _, _, _ = make_use_case(calibrator=calibrator)

    surface = surfaces(use_case.handle(make_market_snapshot()))[0].surface

    assert surface.status is SurfaceStatus.DEGRADED
    assert surface.grid.expiries == (NEAR,)


def test_a_clean_fit_on_a_clean_snapshot_is_published_as_ok() -> None:
    use_case, _, _, _ = make_use_case()

    surface = surfaces(use_case.handle(make_market_snapshot()))[0].surface

    assert surface.status is SurfaceStatus.OK


def test_a_degraded_snapshot_can_only_produce_a_degraded_surface() -> None:
    """The input's quality block does not travel with the surface, so it has to be said here."""
    use_case, _, _, _ = make_use_case()

    surface = surfaces(use_case.handle(make_market_snapshot(degraded=True)))[0].surface

    assert surface.status is SurfaceStatus.DEGRADED


def test_a_degraded_snapshot_is_still_fitted() -> None:
    """Publishing it was Market Data's decision; second-guessing it here would be two layers."""
    use_case, _, _, _ = make_use_case()

    assert surfaces(use_case.handle(make_market_snapshot(degraded=True)))


@pytest.mark.parametrize("bad", [0.0, -1.0, float("nan"), float("inf")])
def test_an_unusable_rmse_threshold_is_refused(bad: float) -> None:
    with pytest.raises(ValueError, match="RMSE threshold"):
        Acceptance(max_rmse_vol_bp=bad)


# --- the refusal, and what goes out behind it


def test_a_refusal_with_no_history_publishes_the_failure_alone() -> None:
    """The ordinary state at start-up: there is no old surface to be stale about."""
    calibrator = StubCalibrator(result=fitted_result(at_bound=True))
    use_case, _, _, _ = make_use_case(calibrator=calibrator)

    events = use_case.handle(make_market_snapshot())

    assert len(events) == 1
    assert isinstance(events[0], CalibrationFailed)


def test_a_refusal_republishes_the_previous_surface_as_stale() -> None:
    """An honest old surface beats silence: a consumer cannot tell silence from a dead process."""
    calibrator = StubCalibrator()
    use_case, _, _, _ = make_use_case(calibrator=calibrator)
    good = surfaces(use_case.handle(make_market_snapshot()))[0].surface

    calibrator.answer_with(fitted_result(at_bound=True))
    events = use_case.handle(make_market_snapshot(snapshot_id="BTC-DERIBIT:00000001"))

    republished = surfaces(events)[0].surface
    assert republished.status is SurfaceStatus.STALE_REPUBLISH
    assert republished.ts_snapshot == good.ts_snapshot
    assert republished.surface_id == good.surface_id


def test_the_failure_is_published_before_the_republished_surface() -> None:
    """A recording should read cause before consequence."""
    calibrator = StubCalibrator()
    use_case, _, _, _ = make_use_case(calibrator=calibrator)
    use_case.handle(make_market_snapshot())

    calibrator.answer_with(fitted_result(at_bound=True))
    events = use_case.handle(make_market_snapshot(snapshot_id="BTC-DERIBIT:00000001"))

    assert isinstance(events[0], CalibrationFailed)
    assert isinstance(events[1], SurfaceCalibrated)


def test_a_snapshot_with_nothing_invertible_is_a_refusal_rather_than_a_crash() -> None:
    use_case, _, _, _ = make_use_case()
    snapshot = make_market_snapshot()
    dead = replace(
        snapshot,
        slices=tuple(
            replace(
                slice_data,
                quotes=tuple(replace(quote, mid=1e9) for quote in slice_data.quotes),
            )
            for slice_data in snapshot.slices
        ),
    )

    events = use_case.handle(dead)

    assert failures(events)
    assert "implied volatility" in failures(events)[0].reason


def test_a_calibrator_that_fails_outright_is_reported_rather_than_propagated() -> None:
    """A failure is a data point in the comparison, not an exception for a loop to swallow."""
    calibrator = StubCalibrator(failure=CalibrationError("the optimiser could not be built"))
    use_case, _, _, _ = make_use_case(calibrator=calibrator)

    events = use_case.handle(make_market_snapshot())

    assert "optimiser" in failures(events)[0].reason


def test_a_refusal_names_the_producer_that_failed() -> None:
    """The one place ``producer_id`` is genuinely meant to be read."""
    calibrator = StubCalibrator(result=fitted_result(at_bound=True), producer_id="svi-scipy")
    use_case, _, _, _ = make_use_case(calibrator=calibrator)

    assert failures(use_case.handle(make_market_snapshot()))[0].producer_id == "svi-scipy"


def test_a_cycle_never_returns_nothing() -> None:
    """Silence downstream is indistinguishable from a process that died."""
    calibrator = StubCalibrator(result=fitted_result(at_bound=True))
    use_case, _, _, _ = make_use_case(calibrator=calibrator)

    assert use_case.handle(make_market_snapshot())


# --- what the cycle reports


def test_the_fit_quality_is_gauged() -> None:
    use_case, _, _, metrics = make_use_case()

    use_case.handle(make_market_snapshot())

    assert metrics.gauge_value("pricing.rmse_vol_bp") > 0


def test_a_refusal_is_counted() -> None:
    calibrator = StubCalibrator(result=fitted_result(at_bound=True))
    use_case, _, _, metrics = make_use_case(calibrator=calibrator)

    use_case.handle(make_market_snapshot())

    assert "pricing.calibration.refused" in metrics.counter_names()


def test_a_pinned_slice_is_counted_apart_from_one_that_merely_missed() -> None:
    """A market where parameters keep pinning is a market the model cannot represent."""
    calibrator = StubCalibrator(result=fitted_result(at_bound=True))
    use_case, _, _, metrics = make_use_case(calibrator=calibrator)

    use_case.handle(make_market_snapshot())

    assert "pricing.slice.at_bound" in metrics.counter_names()
    assert "pricing.slice.rmse_exceeded" not in metrics.counter_names()


def test_a_slice_that_only_missed_the_threshold_is_counted_as_such() -> None:
    calibrator = StubCalibrator(result=fitted_result(rmse_vol_bp=90.0, max_err_vol_bp=200.0))
    use_case, _, _, metrics = make_use_case(calibrator=calibrator, max_rmse_vol_bp=50.0)

    use_case.handle(make_market_snapshot())

    assert "pricing.slice.rmse_exceeded" in metrics.counter_names()
    assert "pricing.slice.at_bound" not in metrics.counter_names()


def test_the_cycle_duration_is_measured_on_the_injected_clock() -> None:
    """A wall clock here would make a replay's published duration depend on the machine."""
    clock = ManualClock(NOW)
    use_case, _, _, _ = make_use_case(clock=clock)

    surface = surfaces(use_case.handle(make_market_snapshot()))[0].surface

    assert surface.fit.duration_ms == 0.0


def test_a_warm_started_cycle_may_report_zero_iterations() -> None:
    """Zero is the normal case in a calm market, and must never be read as a failure."""
    calibrator = StubCalibrator(result=replace(make_calibration_result(), n_iterations=0))
    use_case, _, _, _ = make_use_case(calibrator=calibrator)

    surface = surfaces(use_case.handle(make_market_snapshot()))[0].surface

    assert surface.fit.n_iterations == 0


def test_a_surface_whose_every_slice_collapsed_is_refused_rather_than_published() -> None:
    """``SVIParams`` admits a zero-variance slice; ``VolGrid`` cannot carry one."""
    collapsed = make_params(a=0.0, b=0.0, rho=0.0, m=0.0, sigma=0.2)
    calibrator = StubCalibrator(result=fitted_result(params=collapsed))
    use_case, _, _, _ = make_use_case(calibrator=calibrator)

    events = use_case.handle(make_market_snapshot())

    assert failures(events)
    assert "evaluated on the grid" in failures(events)[0].reason
