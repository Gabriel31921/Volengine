from __future__ import annotations

import math
from dataclasses import FrozenInstanceError, replace

import numpy as np
import pytest

from tests.parametric_pricing.builders import (
    FAR,
    FAR_TENOR,
    NAIVE,
    NEAR,
    NEAR_TENOR,
    make_free_params,
    make_params,
    make_slice,
    make_surface,
)
from tests.support import replace_field
from volengine.parametric_pricing.domain.durrleman import durrleman_g
from volengine.parametric_pricing.domain.svi_slice import (
    MIN_POSITIVE,
    FreeParams,
    SVIParams,
    SVISurface,
)

NOT_FINITE = [float("nan"), float("inf"), float("-inf")]

FAR_LEFT = -1000.0
FAR_RIGHT = 1000.0
"""Log-moneyness values where the square root is indistinguishable from ``|k - m|``.

Far enough that the asymptotic slope is reached to eight digits, close enough that the total
variance itself stays in a range where a difference of two of them keeps its precision.
"""


# --- SVIParams: the formula


def test_total_variance_is_flat_across_strikes_when_b_is_zero() -> None:
    """b = 0 kills the whole hyperbola and leaves w(k) = a, the analytic degenerate case."""
    flat = make_params(a=0.04, b=0.0)
    assert flat.total_variance(-0.5) == pytest.approx(0.04)
    assert flat.total_variance(0.0) == pytest.approx(0.04)
    assert flat.total_variance(0.8) == pytest.approx(0.04)


def test_implied_vol_is_the_square_root_of_the_total_variance_over_the_tenor() -> None:
    """The only place the tenor enters: w = sigma^2 * T, so sigma = sqrt(w / T).

    Asserted on the flat slice because it is the one case with a closed form nobody can get
    subtly wrong -- a 0.04 total variance over a quarter of a year is a 40% vol.
    """
    assert make_params(a=0.04, b=0.0).implied_vol(0.3, 0.25) == pytest.approx(0.4)


def test_right_wing_slope_is_b_times_one_plus_rho() -> None:
    """Far to the right the square root becomes (k - m), so w' -> b * (1 + rho).

    One of the two numbers that pin the formula down. A sign slip on rho leaves the level, the
    minimum and the curvature all correct and swaps the two wings, which every other test here
    would happily accept.
    """
    params = make_params(b=0.10, rho=-0.35)
    slope = params.total_variance(FAR_RIGHT + 1.0) - params.total_variance(FAR_RIGHT)
    assert slope == pytest.approx(0.10 * (1.0 - 0.35), rel=1e-6)


def test_left_wing_slope_is_b_times_rho_minus_one() -> None:
    """Far to the left the square root becomes -(k - m), so w' -> b * (rho - 1), negative.

    The mirror of the right wing, and the reason a negative rho is the usual market shape: the
    left wing falls off more slowly than the right one rises, lifting downside vol.
    """
    params = make_params(b=0.10, rho=-0.35)
    slope = params.total_variance(FAR_LEFT + 1.0) - params.total_variance(FAR_LEFT)
    assert slope == pytest.approx(0.10 * (-0.35 - 1.0), rel=1e-6)


def test_minimum_total_variance_is_the_lowest_value_the_curve_attains() -> None:
    """The closed form is what the constructor validates, so it must match a real scan.

    If it did not, the invariant would be guarding a quantity unrelated to the curve and a
    slice dipping into negative variance could be constructed anyway.
    """
    params = make_params()
    scanned = params.total_variance(np.linspace(-5.0, 5.0, 200_001))
    assert params.min_total_variance == pytest.approx(float(scanned.min()), rel=1e-6)


# --- SVIParams: scalar and array evaluation


def test_total_variance_returns_a_plain_float_for_a_scalar() -> None:
    """A numpy scalar leaking out here would only be noticed by whatever serialises it."""
    result = make_params().total_variance(0.1)
    assert type(result) is float


def test_total_variance_returns_an_array_of_the_same_shape_for_an_array() -> None:
    grid = np.linspace(-0.5, 0.5, 7)
    result = make_params().total_variance(grid)
    assert isinstance(result, np.ndarray)
    assert result.shape == grid.shape


def test_total_variance_agrees_between_scalar_and_array_input() -> None:
    """The two overloads are no longer one implementation, and this is what keeps them equal.

    Since ADR-026 the scalar branch is a call into ``shared_kernel.domain.svi.total_variance``
    and the array branch is a numpy expression that cannot be, because rule 1 keeps numpy out of
    the kernel. So the scalar side of this comparison *is* the kernel's answer, and the test is
    what pins the vectorised path to it: a wing evaluated one way in the loss and another way in
    a diagnostic is a difference no downstream assertion would attribute to the right place.
    """
    params = make_params()
    grid = np.array([-0.4, 0.0, 0.25])
    one_by_one = [params.total_variance(float(k)) for k in grid]
    assert params.total_variance(grid) == pytest.approx(one_by_one)


def test_implied_vol_returns_a_plain_float_for_a_scalar() -> None:
    assert type(make_params().implied_vol(0.1, NEAR_TENOR)) is float


def test_implied_vol_returns_an_array_for_an_array() -> None:
    result = make_params().implied_vol(np.linspace(-0.3, 0.3, 5), NEAR_TENOR)
    assert isinstance(result, np.ndarray)
    assert result.shape == (5,)


@pytest.mark.parametrize("bad", [0.0, -1.0, *NOT_FINITE])
def test_implied_vol_rejects_an_invalid_tenor(bad: float) -> None:
    """Zero would return an infinite vol and a negative one a NaN, both silently."""
    with pytest.raises(ValueError, match="tenor"):
        make_params().implied_vol(0.1, bad)


# --- SVIParams: what the constructor refuses


@pytest.mark.parametrize("field", ["a", "b", "rho", "m", "sigma"])
@pytest.mark.parametrize("bad", NOT_FINITE)
def test_params_reject_a_non_finite_field(field: str, bad: float) -> None:
    with pytest.raises(ValueError, match="finite"):
        replace_field(make_params(), field, bad)


@pytest.mark.parametrize("bad", [-1e-12, -0.5, -10.0])
def test_params_reject_a_negative_b(bad: float) -> None:
    """Negative wings would slope inwards: the curve would go to -infinity on both sides."""
    with pytest.raises(ValueError, match="b must be non-negative"):
        replace(make_params(), b=bad)


@pytest.mark.parametrize("bad", [1.0, -1.0, 1.4, -2.5])
def test_params_reject_a_rho_outside_the_open_unit_interval(bad: float) -> None:
    """At |rho| = 1 one wing is flat and the curve is no longer a hyperbola."""
    with pytest.raises(ValueError, match="rho"):
        replace(make_params(), rho=bad)


@pytest.mark.parametrize("bad", [0.0, -0.2])
def test_params_reject_a_non_positive_sigma(bad: float) -> None:
    """sigma = 0 leaves a kink at the minimum, where the butterfly condition needs w''."""
    with pytest.raises(ValueError, match="sigma"):
        replace(make_params(), sigma=bad)


def test_params_reject_a_negative_minimum_total_variance() -> None:
    """A curve dipping below zero claims a negative variance, which is not a surface."""
    with pytest.raises(ValueError, match="minimum total variance"):
        replace(make_params(), a=-1.0)


# --- SVIParams: what the constructor must NOT refuse
# The invariants say "this is a surface", not "this surface is arbitrage-free". Every
# rejection added here would take the calibrator's search path or the violation metric away.


def test_params_accept_a_butterfly_violating_slice() -> None:
    """The decision this module is built around: admissibility is measured, not enforced.

    Steep wings with a tight bottom put a negative risk-neutral density in the middle of the
    smile. That is a real arbitrage and the fit will be rejected on the metric -- but the
    constructor accepts it, because least-squares walks through iterates like this one and
    ``durrleman.butterfly_violation`` cannot report a distance it is forbidden to hold.
    """
    arbitrageable = make_params(a=0.04, b=0.8, rho=-0.9, m=0.0, sigma=0.05)
    assert arbitrageable.b == 0.8

    # Guards the point: on a healthy slice this test would pass while proving nothing.
    grid = np.linspace(-1.0, 1.0, 2001)
    assert durrleman_g(arbitrageable, grid).min() < 0
    assert durrleman_g(make_params(), grid).min() > 0


def test_params_accept_a_zero_b() -> None:
    """The flat slice is a representable market state, not a broken fit."""
    assert make_params(b=0.0).b == 0.0


def test_params_accept_a_negative_a_with_a_positive_minimum() -> None:
    """a is a level, not a variance: the square-root term lifts the curve above it everywhere.

    Rejecting a negative a on sight would refuse admissible slices, which is why the invariant
    is stated on the minimum of the curve instead.
    """
    admissible = make_params(a=-0.01, b=0.10, rho=-0.35, sigma=0.20)
    assert admissible.min_total_variance > 0


# --- FreeParams and the round trip


@pytest.mark.parametrize(
    ("a", "b", "rho", "m", "sigma"),
    [
        (0.04, 0.10, -0.35, -0.02, 0.20),
        (0.04, 0.10, 0.0, 0.0, 0.20),
        (0.04, 1e-8, 0.999, -0.02, 0.20),
        (0.04, 1e-8, -0.999, -0.02, 1e-8),
        (0.04, 1e6, -0.35, -0.02, 800.0),
        (0.04, 1000.0, 0.5, 3.0, 0.05),
    ],
    ids=["typical", "symmetric", "saturated-rho", "tiny", "huge", "large-b"],
)
def test_free_round_trip_is_the_identity(
    a: float, b: float, rho: float, m: float, sigma: float
) -> None:
    """to_free then from_free returns the same slice, including where the naive inverses break.

    ``log(expm1(b))`` overflows for a large b and ``tanh(atanh(rho))`` collapses onto exactly 1
    for a rho near the edge -- which would come back as an *invalid* SVIParams, not merely an
    imprecise one. The tolerance is a relative one because the trip is through transcendental
    functions; it is not licence for a wrong branch, which misses by orders of magnitude.
    """
    original = make_params(a=a, b=b, rho=rho, m=m, sigma=sigma)
    restored = SVIParams.from_free(original.to_free())

    assert restored.a == original.a
    assert restored.m == original.m
    assert restored.b == pytest.approx(original.b, rel=1e-12)
    assert restored.rho == pytest.approx(original.rho, rel=1e-12)
    assert restored.sigma == pytest.approx(original.sigma, rel=1e-12)


def test_free_round_trip_of_a_zero_b_returns_a_negligible_b() -> None:
    """The one documented inexactness: softplus has no zero in its image.

    b = 0 is a legal slice with no finite pre-image, so to_free floors it and the trip returns
    the smallest positive normal instead. Asserted rather than hidden, because a caller
    comparing warm-start parameters for equality needs to know this is the exception.
    """
    restored = SVIParams.from_free(make_params(b=0.0).to_free())
    assert restored.b > 0
    assert restored.b == pytest.approx(0.0, abs=1e-300)


def test_from_free_never_produces_a_saturated_rho() -> None:
    """tanh returns exactly 1.0 past ~19.06, which its own constructor would then reject.

    An optimiser pushing hard on the skew must not be able to build an invalid slice out of a
    perfectly finite iterate, so the clamp is what keeps the reparameterisation total.
    """
    assert abs(SVIParams.from_free(make_free_params(rho_raw=50.0)).rho) < 1
    assert abs(SVIParams.from_free(make_free_params(rho_raw=-50.0)).rho) < 1


def test_from_free_never_produces_a_zero_sigma() -> None:
    """softplus underflows to exactly 0.0 for a very negative raw value; sigma must stay > 0."""
    assert SVIParams.from_free(make_free_params(sigma_raw=-800.0)).sigma > 0


def test_from_free_does_not_overflow_on_a_large_raw_value() -> None:
    """The textbook softplus computes exp(b_raw) first and returns infinity from ~709 up."""
    assert math.isfinite(SVIParams.from_free(make_free_params(b_raw=1000.0)).b)


def test_from_free_rejects_a_negative_minimum_variance() -> None:
    """The one constraint the free space cannot make structural, because it couples a.

    a is unconstrained by design, so an optimiser can always propose a level low enough to sink
    the curve below zero. That is not a reparameterisation bug, it is a proposal with no
    meaning, and it fails at the constructor.
    """
    with pytest.raises(ValueError, match="minimum total variance"):
        SVIParams.from_free(make_free_params(a=-5.0))


@pytest.mark.parametrize("field", ["a", "b_raw", "rho_raw", "m", "sigma_raw"])
@pytest.mark.parametrize("bad", NOT_FINITE)
def test_free_params_reject_a_non_finite_field(field: str, bad: float) -> None:
    """Finiteness is the only rule the free space has; anything more would constrain it."""
    with pytest.raises(ValueError, match="finite"):
        replace_field(make_free_params(), field, bad)


def test_free_params_accept_values_no_slice_would_have() -> None:
    """A raw of -900 maps to a b of 0: extreme, admissible, and reachable by a gradient step."""
    assert FreeParams(a=0.0, b_raw=-900.0, rho_raw=0.0, m=0.0, sigma_raw=-900.0).b_raw == -900.0


def test_min_positive_is_the_documented_floor() -> None:
    """The floor is part of the published behaviour of to_free, so it is pinned here."""
    assert MIN_POSITIVE > 0
    assert make_params(b=0.0).to_free().b_raw == pytest.approx(
        SVIParams(a=0.04, b=MIN_POSITIVE, rho=-0.35, m=-0.02, sigma=0.20).to_free().b_raw
    )


# --- SVISlice


def test_slice_rejects_a_naive_expiry() -> None:
    """Slices are ordered and matched across snapshots by subtracting datetimes."""
    with pytest.raises(ValueError, match="timezone-aware"):
        replace(make_slice(), expiry=NAIVE)


@pytest.mark.parametrize("bad", [0.0, -1.0, *NOT_FINITE])
def test_slice_rejects_an_invalid_tenor(bad: float) -> None:
    """An expired slice has nothing to annualise: w / 0 is not a volatility."""
    with pytest.raises(ValueError, match="tenor"):
        replace(make_slice(), tenor_years=bad)


@pytest.mark.parametrize("field", ["k_min", "k_max"])
@pytest.mark.parametrize("bad", NOT_FINITE)
def test_slice_rejects_a_non_finite_validity_domain(field: str, bad: float) -> None:
    with pytest.raises(ValueError, match="finite"):
        replace_field(make_slice(), field, bad)


@pytest.mark.parametrize(("k_min", "k_max"), [(0.5, 0.5), (0.6, -0.6), (0.0, -0.1)])
def test_slice_rejects_an_empty_validity_domain(k_min: float, k_max: float) -> None:
    """A band of one point or less is not a fit, and code interpolating inside it is lost."""
    with pytest.raises(ValueError, match="validity domain"):
        make_slice(k_min=k_min, k_max=k_max)


def test_slice_does_not_clamp_evaluation_to_its_validity_domain() -> None:
    """The band is published so a consumer can judge; enforcing it would hide extrapolation.

    A vol asked for outside the quoted strikes is an extrapolation of the functional form, and
    saying so is the caller's decision, not a silent clamp to the edge of the band.
    """
    fitted = make_slice(k_min=-0.6, k_max=0.6)
    assert math.isfinite(fitted.params.total_variance(5.0))


# --- SVISurface


def test_surface_rejects_slices_out_of_tenor_order() -> None:
    """Everything walking the tenor axis -- interpolation, calendar checks -- assumes it."""
    far_first = (
        make_slice(expiry=FAR, tenor_years=FAR_TENOR),
        make_slice(expiry=NEAR, tenor_years=NEAR_TENOR),
    )
    with pytest.raises(ValueError, match="increasing in tenor"):
        make_surface(slices=far_first)


def test_surface_rejects_two_slices_at_the_same_tenor() -> None:
    """Ordering is strict: equal tenors leave the pair's order undefined."""
    tied = (
        make_slice(expiry=NEAR, tenor_years=NEAR_TENOR),
        make_slice(expiry=FAR, tenor_years=NEAR_TENOR),
    )
    with pytest.raises(ValueError, match="increasing in tenor"):
        make_surface(slices=tied)


def test_surface_rejects_a_duplicated_expiry() -> None:
    """Checked apart from the tenor ordering, which a rounding error would let it pass.

    Two entries for one expiry are two different answers to the same question, and whichever
    the consumer reaches first is arbitrary.
    """
    duplicated = (
        make_slice(expiry=NEAR, tenor_years=NEAR_TENOR),
        make_slice(expiry=NEAR, tenor_years=NEAR_TENOR + 1e-9),
    )
    with pytest.raises(ValueError, match="unique expiries"):
        make_surface(slices=duplicated)


def test_surface_rejects_an_empty_collection() -> None:
    with pytest.raises(ValueError, match="at least one slice"):
        SVISurface(slices=())


def test_surface_accepts_a_calendar_arbitrage_between_its_slices() -> None:
    """ADR-008 accepts this risk deliberately: raw SVI fits each slice on its own.

    Total variance cannot shrink with maturity, and nothing in an independent per-slice fit
    prevents it from doing so. The violation is measured by durrleman.calendar_violation and
    reported; refusing to build the surface would delete the measurement along with the
    problem, and would also delete the honest report of what v1 does not guarantee.
    """
    rich_near = make_slice(expiry=NEAR, tenor_years=NEAR_TENOR, params=make_params(a=0.10))
    cheap_far = make_slice(expiry=FAR, tenor_years=FAR_TENOR, params=make_params(a=0.02))

    # Guards the point: without this the test would pass on an arbitrage-free pair.
    assert rich_near.params.total_variance(0.0) > cheap_far.params.total_variance(0.0)

    surface = make_surface(slices=(rich_near, cheap_far))
    assert surface.slices == (rich_near, cheap_far)


def test_surface_accepts_a_single_slice() -> None:
    """A one-expiry chain is a real market state, and no ordering rule can fail on it."""
    assert len(make_surface(slices=(make_slice(),)).slices) == 1


# --- Architecture: every type here is a frozen value object


def test_params_are_immutable() -> None:
    with pytest.raises(FrozenInstanceError):
        make_params().rho = 0.1  # type: ignore[misc]


def test_free_params_are_immutable() -> None:
    with pytest.raises(FrozenInstanceError):
        make_free_params().rho_raw = 0.1  # type: ignore[misc]


def test_slice_is_immutable() -> None:
    with pytest.raises(FrozenInstanceError):
        make_slice().tenor_years = 1.0  # type: ignore[misc]


def test_surface_is_immutable() -> None:
    with pytest.raises(FrozenInstanceError):
        make_surface().slices = ()  # type: ignore[misc]


def test_params_compare_by_value() -> None:
    """Warm start compares the previous parameters against the new ones by equality."""
    assert make_params() == make_params()
