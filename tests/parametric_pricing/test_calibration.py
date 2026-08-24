from __future__ import annotations

from dataclasses import FrozenInstanceError, replace

import pytest

from tests.parametric_pricing.builders import (
    FAR,
    FAR_TENOR,
    NAIVE,
    NEAR,
    NEAR_TENOR,
    make_calibration_result,
    make_calibration_task,
    make_slice_result,
    make_slice_task,
)
from tests.support import replace_field

# --- SliceTask: the scalars


def test_slice_task_rejects_a_naive_expiry() -> None:
    with pytest.raises(ValueError, match="timezone-aware"):
        replace(make_slice_task(), expiry=NAIVE)


@pytest.mark.parametrize("field", ["tenor_years", "forward"])
@pytest.mark.parametrize("bad", [0.0, -1.0, float("nan"), float("inf")])
def test_slice_task_rejects_a_non_positive_scalar(field: str, bad: float) -> None:
    with pytest.raises(ValueError, match="positive and finite"):
        replace_field(make_slice_task(), field, bad)


# --- SliceTask: the three parallel tuples


@pytest.mark.parametrize("field", ["log_moneyness", "implied_vol", "weights"])
def test_slice_task_rejects_tuples_of_different_lengths(field: str) -> None:
    """Element ``i`` of all three describes one quote, so a short tuple silently re-pairs them."""
    task = make_slice_task()
    shortened = getattr(task, field)[:-1]

    with pytest.raises(ValueError, match="same length"):
        replace_field(task, field, shortened)


def test_slice_task_rejects_an_empty_slice() -> None:
    with pytest.raises(ValueError, match="at least one quote"):
        replace(make_slice_task(), log_moneyness=(), implied_vol=(), weights=())


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_slice_task_rejects_a_non_finite_log_moneyness(bad: float) -> None:
    with pytest.raises(ValueError, match="log-moneyness at index 1"):
        replace(make_slice_task(), log_moneyness=(-0.30, bad, 0.05, 0.25))


@pytest.mark.parametrize("bad", [0.0, -0.65, float("nan"), float("inf")])
def test_slice_task_rejects_a_non_positive_implied_vol(bad: float) -> None:
    with pytest.raises(ValueError, match="implied volatility at index 2"):
        replace(make_slice_task(), implied_vol=(0.72, 0.65, bad, 0.66))


@pytest.mark.parametrize("bad", [-0.30, float("nan"), float("inf")])
def test_slice_task_rejects_an_unusable_weight(bad: float) -> None:
    with pytest.raises(ValueError, match="weight at index 3"):
        replace(make_slice_task(), weights=(0.15, 0.35, 0.30, bad))


def test_slice_task_rejects_a_negative_weight_whose_neighbours_cover_it() -> None:
    """The per-element guard, not the aggregate one: a bare ``sum(weights) > 0`` passes here."""
    poisoned = (0.15, -0.35, 0.30, 0.20)
    assert sum(poisoned) > 0

    with pytest.raises(ValueError, match="weight at index 1"):
        replace(make_slice_task(), weights=poisoned)


def test_slice_task_rejects_weights_that_sum_to_zero() -> None:
    """Every weight legal on its own, and no quote left with any influence on the fit."""
    with pytest.raises(ValueError, match="sum to a positive number"):
        replace(make_slice_task(), weights=(0.0, 0.0, 0.0, 0.0))


def test_slice_task_accepts_a_single_zero_weight() -> None:
    """How a flagged quote stays visible in the residuals without steering the parameters."""
    task = replace(make_slice_task(), weights=(0.15, 0.0, 0.30, 0.20))

    assert task.weights[1] == 0.0


@pytest.mark.parametrize("bad", [(-0.30, -0.10, -0.10, 0.25), (-0.30, 0.25, 0.05, 0.30)])
def test_slice_task_rejects_a_non_increasing_log_moneyness(bad: tuple[float, ...]) -> None:
    """One volatility per strike, from the out-of-the-money twin: a repeated ``k`` is a repeated
    strike, and an unsorted axis breaks the index-by-index pairing with the other two tuples."""
    with pytest.raises(ValueError, match="strictly increasing"):
        replace(make_slice_task(), log_moneyness=bad)


def test_slice_task_accepts_a_single_quote() -> None:
    """The value object admits, the calibrator judges (ADR-008).

    Five free parameters through one point is not a fit, and it is still a market state the
    engine must be able to represent and report on. Conditioning is answered by regularising
    towards the neighbouring slice, not by refusing to build the object.
    """
    task = make_slice_task(log_moneyness=(0.05,), implied_vol=(0.62,), weights=(1.0,))

    assert len(task.log_moneyness) == 1


# --- CalibrationTask


@pytest.mark.parametrize("field", ["market_id", "snapshot_id"])
def test_calibration_task_rejects_an_empty_identifier(field: str) -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        replace_field(make_calibration_task(), field, "")


def test_calibration_task_rejects_a_naive_snapshot_instant() -> None:
    with pytest.raises(ValueError, match="timezone-aware"):
        replace(make_calibration_task(), ts_snapshot=NAIVE)


def test_calibration_task_rejects_no_slices() -> None:
    with pytest.raises(ValueError, match="at least one slice"):
        replace(make_calibration_task(), slices=())


@pytest.mark.parametrize("second_tenor", [NEAR_TENOR, NEAR_TENOR / 2.0])
def test_calibration_task_rejects_slices_out_of_tenor_order(second_tenor: float) -> None:
    with pytest.raises(ValueError, match="strictly increasing in tenor"):
        replace(
            make_calibration_task(),
            slices=(
                make_slice_task(expiry=NEAR, tenor_years=NEAR_TENOR),
                make_slice_task(expiry=FAR, tenor_years=second_tenor),
            ),
        )


def test_calibration_task_rejects_a_repeated_expiry() -> None:
    """The tenors are strictly increasing, so only the uniqueness check can fire here."""
    with pytest.raises(ValueError, match="unique expiries"):
        replace(
            make_calibration_task(),
            slices=(
                make_slice_task(expiry=NEAR, tenor_years=NEAR_TENOR),
                make_slice_task(expiry=NEAR, tenor_years=FAR_TENOR),
            ),
        )


def test_calibration_task_accepts_a_single_slice() -> None:
    """A one-expiry market is real, and no ordering rule can fail on it."""
    task = make_calibration_task(slices=(make_slice_task(),))

    assert len(task.slices) == 1


# --- SliceResult


def test_slice_result_rejects_a_naive_expiry() -> None:
    with pytest.raises(ValueError, match="timezone-aware"):
        replace(make_slice_result(), expiry=NAIVE)


@pytest.mark.parametrize("bad", [0.0, -1.0, float("nan"), float("inf")])
def test_slice_result_rejects_a_non_positive_tenor(bad: float) -> None:
    with pytest.raises(ValueError, match="positive and finite"):
        replace(make_slice_result(), tenor_years=bad)


@pytest.mark.parametrize("bad", [-1.0, float("nan"), float("inf")])
def test_slice_result_rejects_an_unusable_rmse(bad: float) -> None:
    with pytest.raises(ValueError, match="RMSE in vol basis points"):
        replace(make_slice_result(), rmse_vol_bp=bad, max_err_vol_bp=1e6)


@pytest.mark.parametrize("bad", [-1.0, float("nan"), float("inf")])
def test_slice_result_rejects_an_unusable_maximum_error(bad: float) -> None:
    with pytest.raises(ValueError, match="maximum error in vol basis points"):
        replace(make_slice_result(), max_err_vol_bp=bad)


def test_slice_result_accepts_a_perfect_fit() -> None:
    """Zero error is a legitimate result -- five parameters through four points can be exact --
    and the non-negativity guards must not be written as truthiness tests."""
    result = make_slice_result(rmse_vol_bp=0.0, max_err_vol_bp=0.0)

    assert result.rmse_vol_bp == 0.0


def test_slice_result_rejects_a_maximum_error_below_its_rmse() -> None:
    """Arithmetically impossible, so it catches the two fields filled in the wrong order."""
    with pytest.raises(ValueError, match="cannot be below the RMSE"):
        replace(make_slice_result(), rmse_vol_bp=31.0, max_err_vol_bp=12.0)


def test_slice_result_accepts_a_maximum_error_equal_to_its_rmse() -> None:
    """What a single-quote fit reports: one error is its own root-mean-square."""
    result = make_slice_result(rmse_vol_bp=12.0, max_err_vol_bp=12.0, n_quotes_used=1)

    assert result.max_err_vol_bp == result.rmse_vol_bp


@pytest.mark.parametrize("bad", [0, -1])
def test_slice_result_rejects_a_fit_on_no_quotes(bad: int) -> None:
    with pytest.raises(ValueError, match="at least one quote"):
        replace(make_slice_result(), n_quotes_used=bad)


def test_slice_result_admits_an_unhealthy_fit() -> None:
    """The type holds a rejected calibration on purpose (ADR-006).

    A fit that did not converge and finished pinned at a bound is not an exception, it is the
    ordinary outcome the use case turns into ``CalibrationFailed`` plus a stale republish. It
    has to be representable for that decision to be made anywhere.
    """
    result = make_slice_result(
        rmse_vol_bp=940.0, max_err_vol_bp=2100.0, converged=False, at_bound=True
    )

    assert not result.converged
    assert result.at_bound


# --- CalibrationResult


def test_calibration_result_rejects_no_slices() -> None:
    with pytest.raises(ValueError, match="at least one slice"):
        replace(make_calibration_result(), slices=())


@pytest.mark.parametrize("second_tenor", [NEAR_TENOR, NEAR_TENOR / 2.0])
def test_calibration_result_rejects_slices_out_of_tenor_order(second_tenor: float) -> None:
    with pytest.raises(ValueError, match="strictly increasing in tenor"):
        replace(
            make_calibration_result(),
            slices=(
                make_slice_result(expiry=NEAR, tenor_years=NEAR_TENOR),
                make_slice_result(expiry=FAR, tenor_years=second_tenor),
            ),
        )


def test_calibration_result_rejects_a_repeated_expiry() -> None:
    """The tenors are strictly increasing, so only the uniqueness check can fire here."""
    with pytest.raises(ValueError, match="unique expiries"):
        replace(
            make_calibration_result(),
            slices=(
                make_slice_result(expiry=NEAR, tenor_years=NEAR_TENOR),
                make_slice_result(expiry=NEAR, tenor_years=FAR_TENOR),
            ),
        )


def test_calibration_result_rejects_a_negative_iteration_count() -> None:
    with pytest.raises(ValueError, match="iteration count"):
        replace(make_calibration_result(), n_iterations=-1)


def test_calibration_result_accepts_zero_iterations() -> None:
    """A warm start that lands on the previous optimum: the cheapest cycle, not a failure."""
    result = make_calibration_result(n_iterations=0)

    assert result.n_iterations == 0


@pytest.mark.parametrize("bad", [-1.0, float("nan"), float("inf")])
def test_calibration_result_rejects_an_unusable_duration(bad: float) -> None:
    with pytest.raises(ValueError, match="duration in milliseconds"):
        replace(make_calibration_result(), duration_ms=bad)


# --- Architecture: all four are frozen, slotted value objects


@pytest.mark.parametrize(
    ("value", "field"),
    [
        (make_slice_task(), "forward"),
        (make_calibration_task(), "market_id"),
        (make_slice_result(), "rmse_vol_bp"),
        (make_calibration_result(), "duration_ms"),
    ],
)
def test_calibration_values_are_frozen_and_slotted(value: object, field: str) -> None:
    assert not hasattr(value, "__dict__")

    with pytest.raises(FrozenInstanceError):
        setattr(value, field, None)
