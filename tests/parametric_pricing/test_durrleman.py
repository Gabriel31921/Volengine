from __future__ import annotations

import numpy as np
import pytest
from numpy.typing import NDArray

from tests.parametric_pricing.builders import (
    FAR,
    FAR_TENOR,
    NEAR,
    NEAR_TENOR,
    make_params,
    make_slice,
)
from volengine.parametric_pricing.domain.durrleman import (
    _curve_and_derivatives,
    butterfly_violation,
    calendar_violation,
    durrleman_g,
)
from volengine.parametric_pricing.domain.svi_slice import SVIParams

GRID = np.linspace(-1.5, 1.5, 3001)
"""A dense band of +-150% in log-forward-moneyness, well past any quoted strike.

Dense because every measure here is a *grid* measure and a sparse grid can step over a narrow
dip in ``g``; wide because the violations these functions exist to catch live in the wings.
"""

VIOLATING = SVIParams(a=0.04, b=0.8, rho=-0.9, m=0.0, sigma=0.05)
"""Steep wings around a very tight bottom: a genuine butterfly arbitrage.

``min(g)`` is about -1.6 over :data:`GRID`, against about +0.32 for the healthy default from the
builders. Both anchors were computed independently of this module, so a rewrite of the formula
that happens to agree with itself still has to agree with them.
"""

DERIVATIVE_GRID = np.linspace(-1.0, 1.0, 41)
"""Coarse on purpose: the finite-difference checks below are about the formula, not coverage."""

FIRST_STEP = 1e-5
SECOND_STEP = 1e-4
"""Central-difference steps, chosen per order rather than shared.

A central difference truncates at ``O(h^2)`` and its rounding error grows as ``eps / h`` for the
first derivative and ``eps / h^2`` for the second, so the two orders have different optima and a
single step size would be sloppy at one end. At these values the first derivative agrees to
about 2e-8 relative and the second to about 5e-7.
"""


def _second_difference(params: SVIParams, k: NDArray[np.float64], h: float) -> NDArray[np.float64]:
    """``(w(k + h) - 2 w(k) + w(k - h)) / h^2``, the textbook central second difference."""
    return (
        params.total_variance(k + h) - 2.0 * params.total_variance(k) + params.total_variance(k - h)
    ) / (h * h)


# --- The analytic derivatives
# Reaching into the private helper deliberately. `g` consumes w' quadratically in one term and
# linearly in another, so a sign slip there still produces a smooth, plausible-looking curve
# that every anchor below would accept at the money. The derivative has to be pinned against a
# computation that shares none of its algebra, and finite differences of `total_variance` are
# the only such computation available inside the domain.


def test_analytic_first_derivative_matches_central_finite_differences() -> None:
    """w' = b * (rho + y / r), checked against the curve it is supposed to differentiate.

    The tolerance carries an absolute floor because w' passes through exactly zero at the
    smile's minimum, where a purely relative comparison has nothing to divide by.
    """
    _, w_prime, _ = _curve_and_derivatives(make_params(), DERIVATIVE_GRID)
    differenced = (
        make_params().total_variance(DERIVATIVE_GRID + FIRST_STEP)
        - make_params().total_variance(DERIVATIVE_GRID - FIRST_STEP)
    ) / (2.0 * FIRST_STEP)
    assert w_prime == pytest.approx(differenced, rel=1e-5, abs=1e-8)


def test_analytic_second_derivative_matches_central_finite_differences() -> None:
    """w'' = b * sigma^2 / r^3, the term that carries the curvature into g."""
    _, _, w_second = _curve_and_derivatives(make_params(), DERIVATIVE_GRID)
    differenced = _second_difference(make_params(), DERIVATIVE_GRID, SECOND_STEP)
    assert w_second == pytest.approx(differenced, rel=1e-4, abs=1e-6)


def test_analytic_derivatives_track_a_skewed_slice_too() -> None:
    """The default is mildly skewed; a violent one is where a wrong rho branch would show.

    Same check on the arbitrageable parameters, whose wings are eight times steeper and whose
    bottom is four times tighter -- the regime where r is small and w'' is large.
    """
    _, w_prime, w_second = _curve_and_derivatives(VIOLATING, DERIVATIVE_GRID)
    first = (
        VIOLATING.total_variance(DERIVATIVE_GRID + FIRST_STEP)
        - VIOLATING.total_variance(DERIVATIVE_GRID - FIRST_STEP)
    ) / (2.0 * FIRST_STEP)
    assert w_prime == pytest.approx(first, rel=1e-5, abs=1e-8)
    assert w_second == pytest.approx(
        _second_difference(VIOLATING, DERIVATIVE_GRID, SECOND_STEP), rel=1e-3, abs=1e-5
    )


# --- durrleman_g


def test_g_is_positive_everywhere_for_a_healthy_slice() -> None:
    """A plausible one-month crypto smile implies a strictly positive density, as it must."""
    assert durrleman_g(make_params(), GRID).min() > 0


def test_g_is_negative_somewhere_for_a_butterfly_violating_slice() -> None:
    """The whole reason SVIParams accepts this set: the violation has to be holdable to be seen.

    Pinned to the independently computed depth rather than only to the sign, so a formula that
    finds *a* negative value for the wrong reason does not pass.
    """
    assert durrleman_g(VIOLATING, GRID).min() == pytest.approx(-1.602, abs=1e-3)


def test_g_is_exactly_one_everywhere_for_a_flat_slice() -> None:
    """b = 0 gives w' = w'' = 0, so every term but the leading square vanishes and g == 1.

    The one case with a closed form nobody can get subtly wrong, and equality rather than
    approximation because each dropped term is an exact floating-point zero, not a small number.
    """
    assert np.all(durrleman_g(make_params(b=0.0), GRID) == 1.0)


def test_g_returns_one_value_per_grid_point() -> None:
    """A reduction slipped in here would turn every measure below into a single-point sample."""
    assert durrleman_g(make_params(), GRID).shape == GRID.shape


def test_g_rejects_a_slice_whose_total_variance_vanishes() -> None:
    """a = b = 0 is a legal SVIParams -- min_total_variance is 0, not negative -- and has no
    density.

    g divides by w twice, so this slice would return inf or a NaN, and both of those survive
    max(0, -min(g)) as a clean 0.0. A collapsed distribution reported as arbitrage-free is the
    worst failure a violation metric has, so the function refuses to answer instead.
    """
    degenerate = make_params(a=0.0, b=0.0)
    assert degenerate.min_total_variance == 0.0

    with pytest.raises(ValueError, match="total variance"):
        durrleman_g(degenerate, GRID)


def test_g_rejects_a_grid_point_where_the_total_variance_overflows() -> None:
    """An infinite w passes `w <= 0` and then divides its way to a g of about 1.

    That is the ordering-guard trap in its numerical form: the guard has to test isfinite first
    and join the bad cases with `or`, or a point with no total variance at all is reported as
    the healthiest on the grid.
    """
    with np.errstate(over="ignore"), pytest.raises(ValueError, match="total variance"):
        durrleman_g(make_params(), np.array([1e200]))


def test_g_rejects_an_empty_grid() -> None:
    """Nothing was looked at is not nothing was found, and only one of them scores 0.0."""
    with pytest.raises(ValueError, match="at least one point"):
        durrleman_g(make_params(), np.array([]))


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_g_rejects_a_non_finite_grid_point(bad: float) -> None:
    """A NaN in the grid propagates into g and then past every ordering test downstream."""
    with pytest.raises(ValueError, match="finite"):
        durrleman_g(make_params(), np.array([-0.1, bad, 0.1]))


# --- butterfly_violation


def test_butterfly_violation_is_exactly_zero_for_a_healthy_slice() -> None:
    """Clamped at zero, so a comfortably admissible slice scores 0 rather than a large negative.

    Without the clamp the penalty term would keep rewarding the optimiser for lifting g long
    after the constraint stopped binding, and the reported metric would have no natural scale.
    """
    assert butterfly_violation(make_params(), GRID) == 0.0

    # Guards the point: the measure is not identically zero on everything it is handed.
    assert butterfly_violation(VIOLATING, GRID) > 0


def test_butterfly_violation_is_the_depth_of_the_worst_negative_g() -> None:
    """Same number, opposite sign: the metric is min(g) reflected, not a count or a mean."""
    assert butterfly_violation(VIOLATING, GRID) == pytest.approx(
        -float(durrleman_g(VIOLATING, GRID).min())
    )


def test_butterfly_violation_grows_when_the_grid_reaches_the_violation() -> None:
    """It sees only what the grid samples, which is why the caller chooses the density.

    The violating slice is clean on a tight band around the money and deeply arbitrageable once
    the grid opens out, so the same parameters score 0.0 and 1.6 depending on where you look.
    """
    assert butterfly_violation(VIOLATING, np.linspace(-0.01, 0.01, 101)) == 0.0
    assert butterfly_violation(VIOLATING, GRID) > 1.0


def test_butterfly_violation_rejects_an_empty_grid() -> None:
    with pytest.raises(ValueError, match="at least one point"):
        butterfly_violation(make_params(), np.array([]))


# --- calendar_violation


def test_calendar_violation_is_zero_for_two_slices_sharing_their_parameters() -> None:
    """Equal total variance is non-decreasing, weakly: the condition is <=, not <.

    A flat term structure is unusual but perfectly arbitrage-free, and a metric that reported it
    as a violation would flag every quiet market.
    """
    shared = make_params()
    near = make_slice(expiry=NEAR, tenor_years=NEAR_TENOR, params=shared)
    far = make_slice(expiry=FAR, tenor_years=FAR_TENOR, params=shared)

    assert calendar_violation(near, far, GRID) == 0.0

    # Guards the point: the two curves are one hundredth of a variance point away from scoring
    # positive, so the 0.0 above is a real verdict and not a function that always returns 0.
    barely_cheaper_far = make_slice(
        expiry=FAR, tenor_years=FAR_TENOR, params=make_params(a=shared.a - 0.01)
    )
    assert calendar_violation(near, barely_cheaper_far, GRID) == pytest.approx(0.01)


def test_calendar_violation_is_positive_when_the_two_curves_cross() -> None:
    """The ADR-008 debt in its usual shape: a steep near wing running over a flatter far one.

    The near slice is richer on the downside and cheaper on the upside, so neither curve
    dominates -- which is exactly the case a single at-the-money comparison would miss.
    """
    near = make_slice(
        expiry=NEAR, tenor_years=NEAR_TENOR, params=make_params(a=0.05, b=0.30, rho=-0.90)
    )
    far = make_slice(
        expiry=FAR, tenor_years=FAR_TENOR, params=make_params(a=0.09, b=0.05, rho=-0.10)
    )
    band = np.linspace(-0.8, 0.8, 1601)

    # Guards the point: the curves really do cross, rather than one sitting above the other.
    excess = near.params.total_variance(band) - far.params.total_variance(band)
    assert excess.max() > 0
    assert excess.min() < 0

    assert calendar_violation(near, far, band) == pytest.approx(float(excess.max()))


def test_calendar_violation_rejects_the_two_slices_in_the_wrong_order() -> None:
    """Sorting them silently would answer with the mirror measure, which looks just as valid."""
    near = make_slice(expiry=NEAR, tenor_years=NEAR_TENOR)
    far = make_slice(expiry=FAR, tenor_years=FAR_TENOR)
    with pytest.raises(ValueError, match="shorter-dated"):
        calendar_violation(far, near, GRID)


def test_calendar_violation_rejects_two_slices_at_the_same_tenor() -> None:
    """Neither is the near one, so there is no direction for the inequality to point in."""
    near = make_slice(expiry=NEAR, tenor_years=NEAR_TENOR)
    twin = make_slice(expiry=FAR, tenor_years=NEAR_TENOR)
    with pytest.raises(ValueError, match="shorter-dated"):
        calendar_violation(near, twin, GRID)


def test_calendar_violation_rejects_an_empty_grid() -> None:
    with pytest.raises(ValueError, match="at least one point"):
        calendar_violation(make_slice(), make_slice(tenor_years=FAR_TENOR), np.array([]))


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_calendar_violation_rejects_a_non_finite_grid_point(bad: float) -> None:
    """A NaN difference is not a crossing, but max(0, nan) is 0 and would read as clean."""
    with pytest.raises(ValueError, match="finite"):
        calendar_violation(
            make_slice(),
            make_slice(tenor_years=FAR_TENOR),
            np.array([-0.1, bad, 0.1]),
        )


def test_calendar_violation_does_not_clip_the_grid_to_the_validity_bands() -> None:
    """The bands are published so the caller can judge; applying them here would hide wings.

    Crossings appear where an aggressive extrapolation of one slice runs under the other, which
    is outside at least one quoted band by construction. This pair is clean inside both bands
    and violating outside them, so a silent intersection would report 0.0.
    """
    near = make_slice(
        expiry=NEAR,
        tenor_years=NEAR_TENOR,
        params=make_params(a=0.05, b=0.20, rho=-0.70),
        k_min=-0.30,
        k_max=0.30,
    )
    far = make_slice(
        expiry=FAR,
        tenor_years=FAR_TENOR,
        params=make_params(a=0.15, b=0.05, rho=-0.10),
        k_min=-0.30,
        k_max=0.30,
    )
    inside = np.linspace(near.k_min, near.k_max, 601)

    assert calendar_violation(near, far, inside) == 0.0
    assert calendar_violation(near, far, np.linspace(-1.5, 1.5, 3001)) > 0
