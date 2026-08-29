"""The memory a pure ``Calibrator`` refuses to keep.

Two properties carry the weight here: that a rejected slice never becomes a warm start, and that
an expiry which stopped trading stops being remembered. Both are invisible in a single cycle and
both compound over a session -- the first makes a bad cycle self-sustaining, the second makes the
mapping grow for as long as the process runs.
"""

from __future__ import annotations

from datetime import UTC, datetime

from tests.parametric_pricing.builders import (
    FAR,
    FAR_TENOR,
    NEAR,
    NEAR_TENOR,
    NOW,
    make_calibration_task,
    make_grid_spec,
    make_params,
    make_slice_result,
)
from volengine.contracts.calibrated_surface import CalibratedSurface, SurfaceStatus
from volengine.parametric_pricing.application.acl import to_calibrated_surface
from volengine.parametric_pricing.application.calibration_state import CalibrationState


def a_surface(surface_id: str = "surface-1") -> CalibratedSurface:
    surface = to_calibrated_surface(
        task=make_calibration_task(),
        accepted=[make_slice_result(expiry=NEAR, tenor_years=NEAR_TENOR)],
        n_iterations=1,
        duration_ms=1.0,
        grid=make_grid_spec(),
        producer_id="svi-stub",
        surface_id=surface_id,
        ts_calibrated=NOW,
        status=SurfaceStatus.OK,
    )
    assert surface is not None
    return surface


# --- the warm start


def test_a_fresh_state_has_no_warm_start() -> None:
    """``None`` and not an empty mapping: the port's ``previous`` documents one cold-start state."""
    assert CalibrationState().warm_start is None


def test_accepted_parameters_become_the_next_cycles_starting_point() -> None:
    state = CalibrationState()
    fitted = make_slice_result(expiry=NEAR, tenor_years=NEAR_TENOR)

    state.accept([fitted], [NEAR])

    warm = state.warm_start
    assert warm is not None
    assert warm[NEAR] == fitted.params


def test_the_warm_start_is_keyed_by_expiry_rather_than_by_position() -> None:
    """Strikes and whole expiries are born and die between snapshots (ADR-013)."""
    state = CalibrationState()
    near = make_slice_result(expiry=NEAR, tenor_years=NEAR_TENOR, params=make_params(a=0.04))
    far = make_slice_result(expiry=FAR, tenor_years=FAR_TENOR, params=make_params(a=0.09))

    state.accept([near, far], [NEAR, FAR])

    warm = state.warm_start
    assert warm is not None
    assert warm[NEAR].a == 0.04
    assert warm[FAR].a == 0.09


def test_a_later_cycle_overwrites_the_parameters_of_an_expiry() -> None:
    state = CalibrationState()
    state.accept([make_slice_result(params=make_params(a=0.04))], [NEAR])

    state.accept([make_slice_result(params=make_params(a=0.07))], [NEAR])

    warm = state.warm_start
    assert warm is not None
    assert warm[NEAR].a == 0.07


def test_an_expiry_that_stopped_trading_is_forgotten() -> None:
    """Nothing overwrites a settled contract's entry, so without pruning it lives forever."""
    state = CalibrationState()
    state.accept(
        [
            make_slice_result(expiry=NEAR, tenor_years=NEAR_TENOR),
            make_slice_result(expiry=FAR, tenor_years=FAR_TENOR),
        ],
        [NEAR, FAR],
    )

    state.accept([make_slice_result(expiry=FAR, tenor_years=FAR_TENOR)], [FAR])

    warm = state.warm_start
    assert warm is not None
    assert set(warm) == {FAR}


def test_an_expiry_still_listed_but_not_accepted_keeps_its_previous_parameters() -> None:
    """A refused cycle must not throw away a warm start that an earlier cycle earned."""
    state = CalibrationState()
    state.accept([make_slice_result(expiry=NEAR, params=make_params(a=0.04))], [NEAR])

    state.accept([], [NEAR])

    warm = state.warm_start
    assert warm is not None
    assert warm[NEAR].a == 0.04


def test_the_warm_start_cannot_be_mutated_through_the_property() -> None:
    """The state owns it; a caller holding a live dict could rewrite a fit's starting point."""
    state = CalibrationState()
    state.accept([make_slice_result(expiry=NEAR)], [NEAR])

    warm = state.warm_start
    assert warm is not None
    dict(warm).clear()

    assert state.warm_start is not None
    assert NEAR in state.warm_start


def test_a_state_pruned_to_nothing_reports_a_cold_start_again() -> None:
    state = CalibrationState()
    state.accept([make_slice_result(expiry=NEAR)], [NEAR])

    state.accept([], [datetime(2027, 1, 1, tzinfo=UTC)])

    assert state.warm_start is None


# --- the last good surface


def test_a_fresh_state_has_nothing_to_republish() -> None:
    """The one case where a refusal has no fallback, and is honest about it."""
    assert CalibrationState().last_good is None


def test_a_remembered_surface_is_what_gets_republished() -> None:
    state = CalibrationState()
    surface = a_surface()

    state.remember(surface)

    assert state.last_good is surface


def test_remembering_a_second_surface_replaces_the_first() -> None:
    state = CalibrationState()
    state.remember(a_surface(surface_id="surface-1"))
    second = a_surface(surface_id="surface-2")

    state.remember(second)

    assert state.last_good is second
