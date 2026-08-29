"""The walking skeleton's calibrator: what it fits, and what it honestly admits it cannot.

Two things are worth testing about a calibrator that does not optimise. The first is that the
parameters it returns really do describe the volatility it claims -- flat SVI is a degenerate case
and a sign slip or a missing tenor would still produce a perfectly valid ``SVIParams``. The second
is that the metrics beside them are the *real* error: this fit is wrong on any market with a
smile, and a calibrator reporting zero there would take ADR-006's acceptance decision away from
the use case that owns it. So the tests below check the fit where it is exact and check that it
declares itself unfit where it is not.
"""

from __future__ import annotations

import math
from datetime import datetime

import pytest

from tests.parametric_pricing.builders import (
    FAR,
    FAR_TENOR,
    NEAR,
    NEAR_TENOR,
    make_calibration_task,
    make_slice_task,
)
from volengine.parametric_pricing.adapters.flat_vol import PRODUCER_ID, FlatVolCalibrator
from volengine.parametric_pricing.domain.calibration import SliceTask
from volengine.parametric_pricing.domain.ports import Calibrator

FLAT_VOL = 0.55
"""One volatility for every strike of a slice: the case this calibrator fits exactly."""

K_AXIS = (-0.2, -0.1, 0.0, 0.1, 0.2)


def flat_slice(
    vol: float = FLAT_VOL,
    tenor_years: float = NEAR_TENOR,
    expiry: datetime = NEAR,
) -> SliceTask:
    """A slice with no smile at all, which is what the constant provider actually produces."""
    return make_slice_task(
        expiry=expiry,
        tenor_years=tenor_years,
        log_moneyness=K_AXIS,
        implied_vol=tuple(vol for _ in K_AXIS),
        weights=(1.0,) * len(K_AXIS),
    )


def smiled_slice(tenor_years: float = NEAR_TENOR) -> SliceTask:
    """A slice a flat model cannot fit: a wing five vol points above the middle."""
    return make_slice_task(
        tenor_years=tenor_years,
        log_moneyness=K_AXIS,
        implied_vol=(0.60, 0.57, 0.55, 0.57, 0.60),
        weights=(1.0,) * len(K_AXIS),
    )


def test_it_satisfies_the_calibrator_port() -> None:
    """The one place the structural conformance is actually checked -- by mypy, here and in the
    composition root, since nothing inherits from a ``Protocol``."""
    calibrator: Calibrator = FlatVolCalibrator()

    assert calibrator.producer_id == PRODUCER_ID


def test_two_instances_can_run_under_different_names() -> None:
    """Identity comes from the object, which is what lets two producers share one market."""
    assert FlatVolCalibrator(producer_id="flat-b").producer_id == "flat-b"


def test_an_anonymous_producer_is_refused() -> None:
    with pytest.raises(ValueError, match="producer id must not be empty"):
        FlatVolCalibrator(producer_id="   ")


def test_the_parameters_reproduce_the_flat_volatility_at_every_moneyness() -> None:
    """The fit itself: ``w = vol^2 * T`` re-annualises to ``vol`` at every k, and only there.

    Checked away from the nodes as well as on them, because a slice with ``b = 0`` is flat by
    construction and the assertion would pass on the axis even if ``a`` were wrong.
    """
    fitted = FlatVolCalibrator().calibrate(None, make_calibration_task(slices=(flat_slice(),)))
    params = fitted.slices[0].params

    for k in (-1.7, -0.05, 0.0, 0.33, 2.5):
        assert params.implied_vol(k, NEAR_TENOR) == pytest.approx(FLAT_VOL, rel=1e-12)


def test_a_slice_with_no_smile_is_fitted_exactly() -> None:
    """Floating-point noise, not a threshold: an absolute bound in basis points, because a
    relative one around zero is met by anything."""
    fitted = FlatVolCalibrator().calibrate(None, make_calibration_task(slices=(flat_slice(),)))

    assert fitted.slices[0].rmse_vol_bp == pytest.approx(0.0, abs=1e-9)
    assert fitted.slices[0].max_err_vol_bp == pytest.approx(0.0, abs=1e-9)


def test_a_smile_is_reported_as_the_error_it_is() -> None:
    """The guard on the test above: the metrics are computed, not stubbed at zero.

    Five vol points of wing against a flat model is hundreds of basis points of error, which is
    what the acceptance rule of ADR-006 exists to refuse. A calibrator that claimed a perfect fit
    here would have every one of its surfaces published.
    """
    fitted = FlatVolCalibrator().calibrate(None, make_calibration_task(slices=(smiled_slice(),)))
    result = fitted.slices[0]

    assert result.rmse_vol_bp > 100.0
    assert result.max_err_vol_bp >= result.rmse_vol_bp


def test_the_flat_volatility_is_the_weighted_mean_of_the_quotes() -> None:
    """Weighted, because the weights are what the loss of a real calibrator would be written in.

    All the weight on one wing must pull the answer onto that wing; an unweighted mean would sit
    in the middle and the assertion below would fail by four vol points.
    """
    task = make_calibration_task(
        slices=(
            make_slice_task(
                log_moneyness=K_AXIS,
                implied_vol=(0.60, 0.57, 0.55, 0.57, 0.60),
                weights=(1.0, 0.0, 0.0, 0.0, 0.0),
            ),
        )
    )

    fitted = FlatVolCalibrator().calibrate(None, task)

    assert fitted.slices[0].params.implied_vol(0.0, NEAR_TENOR) == pytest.approx(0.60, rel=1e-12)


def test_every_slice_of_the_task_is_answered_in_the_task_order() -> None:
    """One result per slice, ordered by tenor -- which ``CalibrationResult`` requires and which
    a producer that reordered or dropped a slice would break."""
    task = make_calibration_task(
        slices=(
            flat_slice(tenor_years=NEAR_TENOR),
            flat_slice(vol=0.42, tenor_years=FAR_TENOR, expiry=FAR),
        )
    )

    fitted = FlatVolCalibrator().calibrate(None, task)

    assert [one.expiry for one in fitted.slices] == [NEAR, FAR]
    assert fitted.slices[1].params.implied_vol(0.0, FAR_TENOR) == pytest.approx(0.42, rel=1e-12)


def test_it_reports_no_search_and_no_bound() -> None:
    """Zero iterations is legal and meaningful, and ``converged`` says the model did its best.

    ``duration_ms`` is zero because measuring it would mean reading a clock, which is the one
    thing the port forbids -- the use case times the call itself, off the injected ``Clock``.
    """
    fitted = FlatVolCalibrator().calibrate(None, make_calibration_task(slices=(flat_slice(),)))

    assert (fitted.n_iterations, fitted.duration_ms) == (0, 0.0)
    assert (fitted.slices[0].converged, fitted.slices[0].at_bound) == (True, False)


def test_it_counts_the_quotes_it_used() -> None:
    fitted = FlatVolCalibrator().calibrate(None, make_calibration_task(slices=(flat_slice(),)))

    assert fitted.slices[0].n_quotes_used == len(K_AXIS)


def test_the_warm_start_changes_nothing() -> None:
    """A pure function wearing an object's clothes: same task, same answer, warm or cold.

    The property the whole port rests on -- replay reproduces, and two calibrators are comparable
    because neither carries its own history into the comparison.
    """
    calibrator = FlatVolCalibrator()
    task = make_calibration_task(slices=(flat_slice(),))
    cold = calibrator.calibrate(None, task)

    warm = calibrator.calibrate({NEAR: cold.slices[0].params}, task)
    again = calibrator.calibrate(None, task)

    assert warm == cold == again


def test_a_single_quote_slice_is_fitted_rather_than_refused() -> None:
    """``SliceTask`` admits one quote, so this must too: what is not enforced is measured, and a
    slice resting on one point reports it in ``n_quotes_used`` rather than by failing."""
    task = make_calibration_task(
        slices=(make_slice_task(log_moneyness=(0.0,), implied_vol=(0.5,), weights=(1.0,)),)
    )

    fitted = FlatVolCalibrator().calibrate(None, task)

    assert fitted.slices[0].n_quotes_used == 1
    assert math.isfinite(fitted.slices[0].rmse_vol_bp)
