"""What the scipy calibrator recovers, what it refuses to be dragged by, and what it admits.

The generator and the calibrator are both ours, which is the trap ``Plan.md`` names for F2-03 and
F2-05 together: a shared mistake -- the same wrong moneyness convention on both sides, the same
forward -- cancels, and a known-truth test passes with the system broken. So the assertions below
are on the **recovered parameters**, not only on the RMSE, and the quotes are generated from SVI
slices the calibrator's own defaults are nowhere near.

Several tests come in pairs: one asserts that a mechanism protects the fit, and its neighbour
asserts that the thing it protects against really would have wrecked it. Without the second, the
first passes on a calibrator that simply ignores its input.
"""

from __future__ import annotations

import math
from datetime import datetime

import numpy as np
import pytest

from tests.parametric_pricing.builders import (
    FAR,
    FAR_TENOR,
    FORWARD,
    NEAR,
    NEAR_TENOR,
    make_calibration_task,
    make_slice_task,
)
from volengine.parametric_pricing.adapters.scipy_calibrator import (
    A_LIMIT,
    BARRIER_BP,
    PRODUCER_ID,
    FitSettings,
    ScipyCalibrator,
    _residuals,
    _to_vector,
)
from volengine.parametric_pricing.domain.calibration import CalibrationTask, SliceTask
from volengine.parametric_pricing.domain.durrleman import butterfly_violation
from volengine.parametric_pricing.domain.ports import Calibrator
from volengine.parametric_pricing.domain.svi_slice import SVIParams

TRUTH = SVIParams(a=0.0035, b=0.060, rho=-0.35, m=-0.02, sigma=0.10)
"""The generating slice: a one-month crypto smile at roughly 20% at-the-money vol, with the
downside wing lifted. Deliberately not the cold start's guess -- ``sigma`` and the level are
nowhere near what the calibrator assumes when it knows nothing."""

ARBITRAGEABLE = SVIParams(a=0.001, b=0.35, rho=-0.90, m=0.0, sigma=0.02)
"""A slice that really does price a negative density: a steep, sharply skewed, nearly kinked
smile. ``butterfly_violation`` is over two units deep on it, which the penalty has room to close
part of."""

K_AXIS: tuple[float, ...] = tuple(np.linspace(-0.6, 0.6, 21))
"""Twenty-one strikes spanning +-60% in log-forward-moneyness: a well-populated crypto slice,
and enough points that five parameters are comfortably over-determined."""

VOL_NOISE = 0.002
"""Twenty basis points of volatility noise, which is the order of a real bid-ask on a liquid
crypto wing."""


def vols_of(params: SVIParams, k: tuple[float, ...], tenor_years: float) -> tuple[float, ...]:
    """The volatilities a slice implies at each point of an axis: the answer a fit must recover."""
    return tuple(params.implied_vol(one, tenor_years) for one in k)


def noisy_vols(
    params: SVIParams, k: tuple[float, ...], tenor_years: float, seed: int
) -> tuple[float, ...]:
    """The same, perturbed by a seeded normal draw. Seeded so a failure is reproducible rather
    than a story about one unlucky afternoon."""
    rng = np.random.default_rng(seed)
    return tuple(vol + float(rng.normal(0.0, VOL_NOISE)) for vol in vols_of(params, k, tenor_years))


def svi_task(
    params: SVIParams = TRUTH,
    k: tuple[float, ...] = K_AXIS,
    tenor_years: float = NEAR_TENOR,
    expiry: datetime = NEAR,
    weights: tuple[float, ...] | None = None,
    implied_vol: tuple[float, ...] | None = None,
) -> SliceTask:
    """One slice quoted exactly on a known SVI curve, evenly weighted unless a test says so."""
    vols = vols_of(params, k, tenor_years) if implied_vol is None else implied_vol
    return make_slice_task(
        expiry=expiry,
        tenor_years=tenor_years,
        forward=FORWARD,
        log_moneyness=k,
        implied_vol=vols,
        weights=weights if weights is not None else tuple(1.0 / len(k) for _ in k),
    )


def one_slice(task: SliceTask) -> CalibrationTask:
    """The smallest whole task: one expiry, so a test reads one result."""
    return make_calibration_task(slices=(task,))


def fit(task: SliceTask, settings: FitSettings | None = None) -> SVIParams:
    """Cold-fit one slice and hand back the parameters, which is what most tests assert on."""
    return ScipyCalibrator(settings).calibrate(None, one_slice(task)).slices[0].params


# --- known truth


def test_recovers_the_generating_parameters_from_clean_quotes() -> None:
    fitted = fit(svi_task())

    assert fitted.a == pytest.approx(TRUTH.a, abs=1e-6)
    assert fitted.b == pytest.approx(TRUTH.b, abs=1e-6)
    assert fitted.rho == pytest.approx(TRUTH.rho, abs=1e-5)
    assert fitted.m == pytest.approx(TRUTH.m, abs=1e-5)
    assert fitted.sigma == pytest.approx(TRUTH.sigma, abs=1e-5)


def test_fits_clean_quotes_to_within_a_hundredth_of_a_basis_point() -> None:
    result = ScipyCalibrator().calibrate(None, one_slice(svi_task())).slices[0]

    # An explicit `abs` because `pytest.approx` passes on `rel` *or* `abs` and its default `abs`
    # is 1e-12: below that scale a relative bound never binds and the assertion accepts anything.
    assert result.rmse_vol_bp == pytest.approx(0.0, abs=1e-2)
    assert result.max_err_vol_bp == pytest.approx(0.0, abs=1e-2)


def test_recovers_the_generating_parameters_under_realistic_noise() -> None:
    fitted = fit(svi_task(implied_vol=noisy_vols(TRUTH, K_AXIS, NEAR_TENOR, seed=7)))

    # Loose bounds on purpose: twenty basis points of noise on twenty-one quotes cannot pin five
    # parameters to more than a percent or so, and a test that demanded more would be asserting
    # the seed rather than the mathematics.
    assert fitted.a == pytest.approx(TRUTH.a, rel=0.10)
    assert fitted.b == pytest.approx(TRUTH.b, rel=0.05)
    assert fitted.rho == pytest.approx(TRUTH.rho, rel=0.05)
    assert fitted.m == pytest.approx(TRUTH.m, abs=0.01)
    assert fitted.sigma == pytest.approx(TRUTH.sigma, rel=0.05)


def test_reports_an_rmse_of_the_order_of_the_noise_it_was_given() -> None:
    task = svi_task(implied_vol=noisy_vols(TRUTH, K_AXIS, NEAR_TENOR, seed=7))

    result = ScipyCalibrator().calibrate(None, one_slice(task)).slices[0]

    # Twenty basis points in, so an RMSE well under that would mean the fit had bent itself
    # through the noise and one well over it that it had missed the smile entirely.
    assert 5.0 < result.rmse_vol_bp < 40.0


def test_carries_the_expiry_and_tenor_of_the_slice_it_fitted() -> None:
    result = ScipyCalibrator().calibrate(None, one_slice(svi_task())).slices[0]

    assert result.expiry == NEAR
    assert result.tenor_years == NEAR_TENOR


def test_answers_with_one_result_per_slice_in_the_task_order() -> None:
    task = make_calibration_task(
        slices=(
            svi_task(expiry=NEAR, tenor_years=NEAR_TENOR),
            svi_task(expiry=FAR, tenor_years=FAR_TENOR),
        )
    )

    result = ScipyCalibrator().calibrate(None, task)

    assert tuple(one.expiry for one in result.slices) == (NEAR, FAR)


# --- robustness to junk


def junk_task() -> SliceTask:
    """Clean quotes with one wing priced at twice its volatility: the quote the flags missed."""
    vols = list(vols_of(TRUTH, K_AXIS, NEAR_TENOR))
    vols[0] *= 2.0
    return svi_task(implied_vol=tuple(vols))


def test_a_junk_quote_does_not_drag_the_smile() -> None:
    fitted = fit(junk_task())

    assert fitted.b == pytest.approx(TRUTH.b, rel=0.05)
    assert fitted.rho == pytest.approx(TRUTH.rho, rel=0.05)


def test_squared_error_really_would_have_dragged_the_smile() -> None:
    # The guard that stops the test above passing vacuously. A Huber scale far past any residual
    # this slice can produce is plain least squares, which is what the loss would be without
    # Design 5.3's robustness clause.
    fitted = fit(junk_task(), FitSettings(huber_scale_bp=1e9))

    assert fitted.b > 5.0 * TRUTH.b
    assert fitted.rho < -0.85


def test_reports_the_error_against_the_junk_quote_it_refused_to_follow() -> None:
    result = ScipyCalibrator().calibrate(None, one_slice(junk_task())).slices[0]

    # Robust does not mean silent: the quote stays in the residuals, so the acceptance rule of
    # ADR-006 sees a slice that does not describe its own data and refuses it.
    assert result.max_err_vol_bp > 1_000.0


# --- the butterfly penalty


def test_the_butterfly_penalty_pulls_the_fit_out_of_arbitrage() -> None:
    mesh = np.linspace(-1.1, 1.1, 51)
    task = svi_task(params=ARBITRAGEABLE)

    penalised = butterfly_violation(fit(task), mesh)
    unpenalised = butterfly_violation(fit(task, FitSettings(durrleman_penalty_bp=0.0)), mesh)

    assert penalised < 0.5 * unpenalised


def test_without_the_penalty_the_arbitrage_is_reproduced_exactly() -> None:
    # The guard: the slice really is arbitrageable, and a fit with the penalty switched off walks
    # straight back onto it. Without this, the test above would pass on a calibrator that could
    # not reach the violation in the first place.
    mesh = np.linspace(-1.1, 1.1, 51)

    fitted = fit(svi_task(params=ARBITRAGEABLE), FitSettings(durrleman_penalty_bp=0.0))

    assert butterfly_violation(fitted, mesh) == pytest.approx(
        butterfly_violation(ARBITRAGEABLE, mesh), rel=1e-3
    )


def test_the_penalty_costs_accuracy_and_says_so() -> None:
    task = svi_task(params=ARBITRAGEABLE)

    penalised = ScipyCalibrator().calibrate(None, one_slice(task)).slices[0]

    # Leaving the arbitrageable region means no longer passing through the quotes, and the RMSE
    # reports that rather than hiding it: the trade-off is visible to the acceptance rule.
    assert penalised.rmse_vol_bp > 100.0


# --- weights


def zero_weighted_junk() -> SliceTask:
    """The junk slice again, with the offending quote's weight set to zero.

    That is how the ACL keeps a flagged quote in the slice without letting it steer the fit; the
    two tests below check both halves of the sentence.
    """
    weights = [1.0 / len(K_AXIS)] * len(K_AXIS)
    weights[0] = 0.0
    total = sum(weights)
    return svi_task(
        implied_vol=junk_task().implied_vol, weights=tuple(one / total for one in weights)
    )


def test_a_zero_weight_quote_is_not_counted_among_the_quotes_used() -> None:
    result = ScipyCalibrator().calibrate(None, one_slice(zero_weighted_junk())).slices[0]

    assert result.n_quotes_used == len(K_AXIS) - 1


def test_a_zero_weight_quote_does_not_move_the_fit() -> None:
    fitted = fit(zero_weighted_junk())

    assert fitted.b == pytest.approx(TRUTH.b, rel=1e-3)
    assert fitted.rho == pytest.approx(TRUTH.rho, rel=1e-3)


def test_the_same_junk_quote_at_full_weight_really_would_have_moved_it() -> None:
    # The guard for the pair above: the quote is genuinely poisonous, so a zero weight is doing
    # the work rather than the quote being harmless.
    fitted = fit(junk_task())

    assert abs(fitted.b - TRUTH.b) > 100.0 * abs(fit(zero_weighted_junk()).b - TRUTH.b)


def test_the_reported_errors_ignore_the_quote_that_was_ignored() -> None:
    result = ScipyCalibrator().calibrate(None, one_slice(zero_weighted_junk())).slices[0]

    # A quote excluded from the loss is excluded from the report of how well the loss did. The
    # alternative -- a maximum error of several thousand basis points against a quote nobody
    # fitted -- would make the diagnostics unreadable exactly when the wings are flagged.
    assert result.max_err_vol_bp < 10.0


# --- thin slices


THIN_AXIS: tuple[float, ...] = (-0.20, 0.0, 0.25)
"""Three quotes: fewer than the five parameters raw SVI has, which is the case ADR-008 answers by
pinning the shape rather than pretending the smile is determined."""

WARM = SVIParams(a=0.0040, b=0.055, rho=-0.30, m=-0.05, sigma=0.15)
"""A plausible previous cycle, whose ``m`` and ``sigma`` are far enough from the truth to be
recognisable if they survive untouched."""


def test_a_thin_slice_keeps_the_shape_parameters_it_started_from() -> None:
    task = one_slice(svi_task(k=THIN_AXIS))

    fitted = ScipyCalibrator().calibrate({NEAR: WARM}, task).slices[0].params

    assert fitted.m == WARM.m
    assert fitted.sigma == WARM.sigma


def test_a_slice_with_enough_quotes_moves_its_shape_parameters() -> None:
    # The guard: the equality above is pinning, not a fit that happens to leave those two alone.
    task = one_slice(svi_task())

    fitted = ScipyCalibrator().calibrate({NEAR: WARM}, task).slices[0].params

    assert fitted.m != WARM.m
    assert fitted.sigma != WARM.sigma


def test_a_thin_slice_still_fits_the_level_the_wings_and_the_skew() -> None:
    task = one_slice(svi_task(k=THIN_AXIS))

    result = ScipyCalibrator().calibrate({NEAR: WARM}, task).slices[0]

    # Three quotes and three free parameters: the fit should pass through them, which is the
    # honest most a slice this thin supports.
    assert result.rmse_vol_bp == pytest.approx(0.0, abs=1.0)
    assert result.n_quotes_used == len(THIN_AXIS)


def test_the_pinning_threshold_is_configurable() -> None:
    task = one_slice(svi_task(k=THIN_AXIS))

    fitted = (
        ScipyCalibrator(FitSettings(min_quotes_for_free_shape=0))
        .calibrate({NEAR: WARM}, task)
        .slices[0]
        .params
    )

    assert fitted.m != WARM.m


# --- the warm start


def test_a_warm_start_costs_fewer_evaluations_than_a_cold_one() -> None:
    task = one_slice(svi_task())
    calibrator = ScipyCalibrator()
    cold = calibrator.calibrate(None, task)

    warm = calibrator.calibrate({NEAR: cold.slices[0].params}, task)

    assert warm.n_iterations < cold.n_iterations


def test_an_expiry_with_no_warm_start_is_cold_started() -> None:
    # A tenor born since the last cycle (ADR-013) is an ordinary event: the mapping simply has no
    # entry for it, and a calibrator that indexed it blindly would fail on the day a strike list
    # changed.
    task = make_calibration_task(
        slices=(
            svi_task(expiry=NEAR, tenor_years=NEAR_TENOR),
            svi_task(expiry=FAR, tenor_years=FAR_TENOR),
        )
    )

    result = ScipyCalibrator().calibrate({NEAR: TRUTH}, task)

    assert result.slices[1].params.b == pytest.approx(TRUTH.b, abs=1e-5)


def test_a_warm_start_that_ran_out_of_budget_is_retried_from_the_cold_guess() -> None:
    task = one_slice(svi_task())
    settings = FitSettings(max_nfev=1)

    cold = ScipyCalibrator(settings).calibrate(None, task)
    warm = ScipyCalibrator(settings).calibrate({NEAR: WARM}, task)

    # One evaluation each: the warm attempt could not converge inside its budget, so a second
    # attempt ran from the data's own geometry and both costs are reported.
    assert cold.n_iterations == 1
    assert warm.n_iterations == 2


def test_a_healthy_warm_start_is_not_retried() -> None:
    # The guard for the retry above: it fires on failure, not on every cycle. A converged,
    # unpinned warm start costs one attempt and nothing more.
    task = one_slice(svi_task())
    calibrator = ScipyCalibrator()
    cold = calibrator.calibrate(None, task)

    warm = calibrator.calibrate({NEAR: cold.slices[0].params}, task)

    assert warm.n_iterations < 10


# --- what the fit admits about itself


def test_reports_no_convergence_when_the_evaluation_budget_runs_out() -> None:
    result = ScipyCalibrator(FitSettings(max_nfev=1)).calibrate(None, one_slice(svi_task()))

    assert result.slices[0].converged is False


def test_reports_a_bound_when_the_level_runs_past_its_limit() -> None:
    # A total variance far above `A_LIMIT`: 900% vol at a month is not a market, and the fit that
    # comes back is a projection onto the edge of the box rather than a smile. ADR-006 refuses it
    # on exactly this flag, whatever its RMSE says.
    beyond = math.sqrt(2.0 * A_LIMIT / NEAR_TENOR)
    task = svi_task(implied_vol=tuple(beyond for _ in K_AXIS))

    result = ScipyCalibrator().calibrate(None, one_slice(task)).slices[0]

    assert result.at_bound is True


def test_an_ordinary_fit_is_not_reported_at_a_bound() -> None:
    # The guard: the flag means something, rather than firing on every slice.
    result = ScipyCalibrator().calibrate(None, one_slice(svi_task())).slices[0]

    assert result.at_bound is False


FLAT_VOL = 0.55
FLAT_AXIS: tuple[float, ...] = tuple(np.linspace(-0.5, 0.5, 9))
FLAT_SEED = 0
"""A market with no smile at all, plus noise. ``b`` goes to zero, which leaves ``rho``, ``m`` and
``sigma`` describing nothing and free to wander."""


def flat_task() -> SliceTask:
    rng = np.random.default_rng(FLAT_SEED)
    return svi_task(
        k=FLAT_AXIS,
        implied_vol=tuple(FLAT_VOL + float(rng.normal(0.0, VOL_NOISE)) for _ in FLAT_AXIS),
    )


def test_a_flat_market_does_not_pin_its_unidentified_parameters() -> None:
    result = ScipyCalibrator().calibrate(None, one_slice(flat_task())).slices[0]

    # A market with no smile is a legal market, and a perfectly good fit of one must be
    # publishable: without the ridge the optimiser drifts along a plateau into the side of the
    # box, ADR-006 refuses the surface, and risk is fed a stale republish forever.
    assert result.at_bound is False
    assert result.rmse_vol_bp < 50.0


def test_without_the_ridge_a_flat_market_pins_a_parameter() -> None:
    # The guard: the drift is real and the ridge is what stops it, on this exact data.
    result = (
        ScipyCalibrator(FitSettings(ridge_bp=0.0)).calibrate(None, one_slice(flat_task())).slices[0]
    )

    assert result.at_bound is True


def test_an_inadmissible_iterate_is_answered_with_a_finite_barrier() -> None:
    # Reaching into a private function, with the same justification ``test_durrleman.py`` gives
    # for `_curve_and_derivatives`: this is the one path that cannot be produced from outside on
    # demand, and if it regressed the symptom would be a `ValueError` out of `SVIParams.from_free`
    # killing a whole surface for one bad step of one slice.
    task = svi_task()
    mesh = np.linspace(-1.0, 1.0, 11)
    inadmissible = np.array([-3.9, -3.0, 0.0, 0.0, -2.0])

    residuals = _residuals(inadmissible, task, mesh, FitSettings())

    assert residuals.size == len(K_AXIS) + mesh.size
    assert np.all(np.isfinite(residuals))
    assert float(residuals.min()) > 1e5


COLLAPSED = SVIParams(a=-0.125, b=0.5, rho=0.0, m=0.0, sigma=0.25)
"""A legal slice whose total variance is exactly zero at its own minimum.

``a + b * sigma * sqrt(1 - rho^2)`` is ``-0.125 + 0.5 * 0.25``, so ``SVIParams`` admits it -- its
invariant is ``min_total_variance >= 0`` -- while ``durrleman_g`` refuses it, because ``g``
divides by ``w`` twice and there is no density to be non-negative about where the distribution has
collapsed onto a point. That asymmetry is the seam in ``docs/SEAMS.md``, and this is the shape an
optimiser walks through on its way somewhere else.
"""


def test_a_slice_whose_variance_collapses_on_the_mesh_is_answered_with_a_barrier() -> None:
    # The second half of the pair above, and the branch that keeps the seam survivable: the raise
    # is caught and priced, not propagated. Were it propagated, one iterate of one slice would
    # take down the whole surface -- and it would do so only on the chains unlucky enough to put a
    # mesh node exactly on the minimum, which is the worst kind of intermittent failure.
    task = svi_task()
    mesh = np.linspace(-1.0, 1.0, 11)
    assert 0.0 in mesh, "the mesh must sample the point where the variance collapses"

    residuals = _residuals(_to_vector(COLLAPSED.to_free()), task, mesh, FitSettings())

    assert np.all(np.isfinite(residuals))
    # Only the penalty block is the barrier. The quotes are still priced by a perfectly evaluable
    # curve, and asserting that distinguishes this branch from the inadmissible-iterate one above,
    # which answers with a barrier everywhere.
    assert np.all(residuals[len(K_AXIS) :] == BARRIER_BP)
    assert float(np.abs(residuals[: len(K_AXIS)]).max()) < BARRIER_BP


# --- the port, and purity


def test_satisfies_the_calibrator_port() -> None:
    calibrator: Calibrator = ScipyCalibrator()

    result = calibrator.calibrate(None, one_slice(svi_task()))

    assert calibrator.producer_id == PRODUCER_ID
    assert len(result.slices) == 1


def test_the_same_task_twice_gives_the_same_parameters() -> None:
    task = one_slice(svi_task(implied_vol=noisy_vols(TRUTH, K_AXIS, NEAR_TENOR, seed=3)))
    calibrator = ScipyCalibrator()

    first = calibrator.calibrate(None, task).slices[0].params
    second = calibrator.calibrate(None, task).slices[0].params

    # Bit for bit, which is what ADR-004 asks of a replay and what a pure fit gives for free.
    assert first == second


def test_the_duration_is_not_measured() -> None:
    result = ScipyCalibrator().calibrate(None, one_slice(svi_task()))

    # Reading a clock is the one thing the port forbids outright, and it would make two runs over
    # the same recording differ. The use case times the call from its own injected `Clock`.
    assert result.duration_ms == 0.0


def test_the_producer_id_travels_from_the_constructor() -> None:
    assert ScipyCalibrator(producer_id="svi-scipy-b").producer_id == "svi-scipy-b"


def test_an_empty_producer_id_is_refused() -> None:
    with pytest.raises(ValueError, match="producer id"):
        ScipyCalibrator(producer_id="   ")


# --- settings


def test_a_non_positive_huber_scale_is_refused() -> None:
    with pytest.raises(ValueError, match="Huber scale"):
        FitSettings(huber_scale_bp=0.0)


def test_a_negative_penalty_is_refused() -> None:
    with pytest.raises(ValueError, match="Durrleman penalty"):
        FitSettings(durrleman_penalty_bp=-1.0)


def test_a_mesh_of_one_node_is_refused() -> None:
    with pytest.raises(ValueError, match="at least two nodes"):
        FitSettings(durrleman_mesh_nodes=1)


def test_a_negative_mesh_margin_is_refused() -> None:
    with pytest.raises(ValueError, match="mesh margin"):
        FitSettings(durrleman_mesh_margin=-0.1)


def test_a_negative_ridge_is_refused() -> None:
    with pytest.raises(ValueError, match="ridge"):
        FitSettings(ridge_bp=-1.0)


def test_a_nan_ridge_is_refused() -> None:
    # NaN passes every ordering guard, so the check has to test finiteness first.
    with pytest.raises(ValueError, match="ridge"):
        FitSettings(ridge_bp=float("nan"))


def test_an_empty_evaluation_budget_is_refused() -> None:
    with pytest.raises(ValueError, match="evaluation budget"):
        FitSettings(max_nfev=0)
