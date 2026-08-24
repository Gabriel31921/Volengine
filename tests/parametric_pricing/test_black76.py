from __future__ import annotations

import math
from itertools import pairwise

import pytest

from volengine.parametric_pricing.domain.black76 import (
    VOL_TOLERANCE,
    OptionKindP,
    implied_vol,
    price,
    vega,
)
from volengine.parametric_pricing.domain.errors import CalibrationError, NoImpliedVolError

FORWARD = 60_000.0
STRIKE = 60_000.0
TENOR = 0.25
VOL = 0.65

BOTH_KINDS = [OptionKindP.CALL, OptionKindP.PUT]
UNUSABLE = [0.0, -1.0, float("nan"), float("inf"), float("-inf")]

# From a deep put wing to a deep call wing, at a quarter-year tenor.
ACROSS_THE_SMILE = [30_000.0, 45_000.0, 60_000.0, 80_000.0, 120_000.0]

HEALTHY_VEGA = 1e-6
"""Below this, a Newton step built from a price residual is 1e20 or worse."""


# --- builders


def call(vol: float = VOL, strike: float = STRIKE, discount: float = 1.0) -> float:
    return price(FORWARD, strike, TENOR, vol, OptionKindP.CALL, discount)


def put(vol: float = VOL, strike: float = STRIKE, discount: float = 1.0) -> float:
    return price(FORWARD, strike, TENOR, vol, OptionKindP.PUT, discount)


# --- the identity that ties the two sides together


@pytest.mark.parametrize("strike", ACROSS_THE_SMILE)
def test_a_call_and_a_put_satisfy_put_call_parity(strike: float) -> None:
    """C - P = D * (F - K) is an arbitrage identity, not a model result: it must hold to
    rounding for every strike, or one of the two branches has a sign or a CDF argument wrong.
    """
    assert call(strike=strike) - put(strike=strike) == pytest.approx(FORWARD - strike, rel=1e-12)


def test_put_call_parity_survives_discounting() -> None:
    """The discount multiplies the whole premium, so it multiplies the parity gap too. A branch
    that applied it to only one leg would still return plausible prices.
    """
    discount = 0.97
    gap = call(strike=55_000.0, discount=discount) - put(strike=55_000.0, discount=discount)
    assert gap == pytest.approx(discount * (FORWARD - 55_000.0), rel=1e-12)


# --- price: shape


@pytest.mark.parametrize("kind", BOTH_KINDS)
def test_price_is_strictly_increasing_in_vol(kind: OptionKindP) -> None:
    """Monotonicity is what makes the inversion well posed: without it the bracket in
    implied_vol would be meaningless and a target could have two roots.
    """
    prices = [price(FORWARD, STRIKE, TENOR, vol, kind) for vol in (0.05, 0.2, 0.5, 1.0, 2.0, 5.0)]
    assert all(lower < higher for lower, higher in pairwise(prices))


def test_price_approaches_the_intrinsic_value_as_vol_vanishes() -> None:
    """The lower bound implied_vol checks against is a limit of this function, not a separate
    convention, so the two have to agree.
    """
    assert call(vol=1e-6, strike=50_000.0) == pytest.approx(FORWARD - 50_000.0, rel=1e-9)


def test_a_call_approaches_the_discounted_forward_as_vol_explodes() -> None:
    """The ceiling implied_vol rejects targets above. As vol grows the strike leg is worth
    nothing and what is left is a forward contract.
    """
    assert call(vol=100.0) == pytest.approx(FORWARD, rel=1e-9)


def test_a_put_approaches_the_discounted_strike_as_vol_explodes() -> None:
    assert put(vol=100.0) == pytest.approx(STRIKE, rel=1e-9)


def test_the_discount_factor_moves_the_price() -> None:
    """Anti-vacuous guard: several tests here pass a discount and would still pass if the
    argument were ignored entirely.
    """
    assert call(discount=0.9) == pytest.approx(0.9 * call(discount=1.0), rel=1e-12)


def test_a_worthless_wing_still_prices_above_zero() -> None:
    """`not 0.0` is True and so is a truthiness test on a premium; nothing in this module may
    treat a very small price as a missing one.
    """
    assert call(vol=0.4, strike=400_000.0) > 0.0


# --- vega


@pytest.mark.parametrize("strike", ACROSS_THE_SMILE)
def test_vega_matches_a_central_finite_difference_of_price(strike: float) -> None:
    """The one test that pins the constant. Vega weights the calibration loss, so a wrong
    factor -- a missing sqrt(T), phi(d2) instead of phi(d1) -- would tilt every fit while every
    price stayed correct, and nothing else here would notice.
    """
    h = 1e-5
    numeric = (
        price(FORWARD, strike, TENOR, VOL + h, OptionKindP.CALL)
        - price(FORWARD, strike, TENOR, VOL - h, OptionKindP.CALL)
    ) / (2.0 * h)
    assert vega(FORWARD, strike, TENOR, VOL) == pytest.approx(numeric, rel=1e-6)


def test_vega_matches_a_finite_difference_of_the_put_too() -> None:
    """Guards the claim in the signature: vega takes no kind because a call and a put share it.
    Differencing the *put* is what proves the shared number is right for both.
    """
    h = 1e-5
    numeric = (put(vol=VOL + h) - put(vol=VOL - h)) / (2.0 * h)
    assert vega(FORWARD, STRIKE, TENOR, VOL) == pytest.approx(numeric, rel=1e-6)


def test_vega_under_discounting_scales_with_the_discount() -> None:
    assert vega(FORWARD, STRIKE, TENOR, VOL, discount=0.9) == pytest.approx(
        0.9 * vega(FORWARD, STRIKE, TENOR, VOL), rel=1e-12
    )


def test_vega_collapses_in_the_deep_wing() -> None:
    """The premise of the bisection safeguard: if vega stayed healthy out here, a bare Newton
    would be enough and the bracket would be dead code.
    """
    assert vega(FORWARD, 600_000.0, 0.02, 0.8) < HEALTHY_VEGA


# --- implied_vol: the round trip


@pytest.mark.parametrize("kind", BOTH_KINDS)
@pytest.mark.parametrize("z_score", [-3.0, -1.5, 0.0, 1.5, 3.0])
@pytest.mark.parametrize("vol", [0.15, 0.65, 1.8])
def test_the_inversion_recovers_the_vol_the_price_was_built_from(
    z_score: float, vol: float, kind: OptionKindP
) -> None:
    """The property the module exists for, swept across both wings, both sides and a wide range
    of volatilities: price -> implied_vol -> the same vol.

    Strikes are placed in units of total standard deviation rather than at fixed levels, so the
    sweep means the same thing at every volatility. A fixed 30,000 strike is a mild wing at
    180% vol and eight standard deviations away at 15%, where the option has no time value left
    to invert -- see the test below, which is that case on purpose.
    """
    strike = FORWARD * math.exp(z_score * vol * math.sqrt(TENOR))
    target = price(FORWARD, strike, TENOR, vol, kind)
    assert implied_vol(target, FORWARD, strike, TENOR, kind) == pytest.approx(vol, rel=1e-6)


def test_a_deep_in_the_money_option_with_no_representable_time_value_has_no_implied_vol() -> None:
    """Eight standard deviations in the money, the time value is around 1e-16 of the intrinsic
    and disappears into the rounding of a 30,000 premium, so the price *is* the intrinsic in
    double precision and the open lower bound rejects it.

    Documented in NoImpliedVolError as the expected fate of deep in-the-money quotes. The right
    behaviour is to say so and let the caller drop the quote: any vol at all reproduces this
    price to the last bit, so returning one would be inventing information.
    """
    assert call(vol=0.15, strike=30_000.0) == 30_000.0
    with pytest.raises(NoImpliedVolError, match="intrinsic value"):
        implied_vol(call(vol=0.15, strike=30_000.0), FORWARD, 30_000.0, TENOR, OptionKindP.CALL)


@pytest.mark.parametrize("tenor", [1.0 / 365.0, 0.25, 2.0])
def test_the_inversion_recovers_the_vol_across_tenors(tenor: float) -> None:
    """The bracket is 40 / sqrt(T), so it is the tenor that decides how wide the search is."""
    target = price(FORWARD, 70_000.0, tenor, 0.9, OptionKindP.CALL)
    assert implied_vol(target, FORWARD, 70_000.0, tenor, OptionKindP.CALL) == pytest.approx(
        0.9, rel=1e-6
    )


def test_the_inversion_is_accurate_to_the_stated_tolerance() -> None:
    """VOL_TOLERANCE is a published promise about the answer, not an internal detail."""
    target = call(vol=0.4123456789)
    recovered = implied_vol(target, FORWARD, STRIKE, TENOR, OptionKindP.CALL)
    assert abs(recovered - 0.4123456789) < VOL_TOLERANCE


def test_the_discount_factor_moves_the_implied_vol() -> None:
    """Anti-vacuous guard for the inversion: inverting a discounted premium as if it were
    undiscounted must not quietly return the same vol.
    """
    target = call(vol=0.5, discount=0.9)
    assert implied_vol(target, FORWARD, STRIKE, TENOR, OptionKindP.CALL, 0.9) == pytest.approx(
        0.5, rel=1e-6
    )
    assert implied_vol(target, FORWARD, STRIKE, TENOR, OptionKindP.CALL) != pytest.approx(
        0.5, rel=1e-3
    )


# --- implied_vol: the deep wing, which is why the safeguard exists

DEEP_WING_STRIKE = 600_000.0
DEEP_WING_TENOR = 0.02
DEEP_WING_VOL = 0.8


def deep_wing_target() -> float:
    return price(FORWARD, DEEP_WING_STRIKE, DEEP_WING_TENOR, DEEP_WING_VOL, OptionKindP.CALL)


def test_a_deep_wing_price_still_inverts() -> None:
    """Ten times out of the money at a one-week tenor: the option is worth about 1e-87 and vega
    is about 1e-85, which is the classic Newton divergence for implied volatility.
    """
    recovered = implied_vol(
        deep_wing_target(), FORWARD, DEEP_WING_STRIKE, DEEP_WING_TENOR, OptionKindP.CALL
    )
    assert recovered == pytest.approx(DEEP_WING_VOL, rel=1e-6)


def test_a_bare_newton_really_does_diverge_there() -> None:
    """Guards the test above. Without this, the bisection safeguard could be deleted and the
    deep-wing round trip might still pass on a case Newton happened to survive.
    """
    vol = 0.5
    residual = (
        price(FORWARD, DEEP_WING_STRIKE, DEEP_WING_TENOR, vol, OptionKindP.CALL)
        - deep_wing_target()
    )
    slope = vega(FORWARD, DEEP_WING_STRIKE, DEEP_WING_TENOR, vol)
    assert vol - residual / slope > 1e100


# --- implied_vol: why the CDF is erfc-based


def norm_cdf_via_erf(x: float) -> float:
    """The textbook spelling the module deliberately does not use."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def call_price_via_erf(strike: float, tenor_years: float, vol: float) -> float:
    total_stdev = vol * math.sqrt(tenor_years)
    d1 = (math.log(FORWARD / strike) + 0.5 * total_stdev * total_stdev) / total_stdev
    return FORWARD * norm_cdf_via_erf(d1) - strike * norm_cdf_via_erf(d1 - total_stdev)


def test_the_erf_form_underflows_to_exactly_zero_in_the_deep_wing() -> None:
    """The premise of the CDF choice: erf saturates at -1 past about six standard deviations,
    so 1 + erf(...) cancels to exactly zero and the price disappears.
    """
    assert call_price_via_erf(DEEP_WING_STRIKE, DEEP_WING_TENOR, DEEP_WING_VOL) == 0.0


def test_the_erfc_form_keeps_the_same_price_representable() -> None:
    """Guards the test above from the other side: the price is small, not absent, and a zero
    would be an uninvertible quote thrown away for no reason.
    """
    assert 0.0 < deep_wing_target() < 1e-80


def test_a_price_the_erf_form_would_lose_still_inverts() -> None:
    """The whole point: with the naive CDF this quote prices at zero, fails the intrinsic bound
    and is dropped -- and that happens to about half the strikes on a crypto chain.
    """
    recovered = implied_vol(
        deep_wing_target(), FORWARD, DEEP_WING_STRIKE, DEEP_WING_TENOR, OptionKindP.CALL
    )
    assert recovered == pytest.approx(DEEP_WING_VOL, rel=1e-6)


# --- implied_vol: prices no volatility reproduces


@pytest.mark.parametrize("kind", BOTH_KINDS)
def test_a_target_at_the_intrinsic_value_has_no_implied_vol(kind: OptionKindP) -> None:
    """The bound is open: the intrinsic value is the vol -> 0 limit, reached by no positive
    volatility at all.
    """
    intrinsic = max(FORWARD - STRIKE, 0.0) if kind is OptionKindP.CALL else 0.0
    with pytest.raises(NoImpliedVolError, match="intrinsic value"):
        implied_vol(intrinsic, FORWARD, STRIKE, TENOR, kind)


def test_a_target_below_the_intrinsic_value_has_no_implied_vol() -> None:
    """Routine, not exceptional: a mid from a crossed or stale book lands here regularly."""
    with pytest.raises(NoImpliedVolError, match="intrinsic value"):
        implied_vol(9_000.0, FORWARD, 50_000.0, TENOR, OptionKindP.CALL)


def test_the_intrinsic_bound_accounts_for_the_discount() -> None:
    """Under discounting the floor is D * (F - K), so a target between the two would raise on
    an undiscounted implementation and invert fine here.
    """
    assert implied_vol(9_900.0, FORWARD, 50_000.0, TENOR, OptionKindP.CALL, 0.98) > 0.0
    with pytest.raises(NoImpliedVolError, match="intrinsic value"):
        implied_vol(9_700.0, FORWARD, 50_000.0, TENOR, OptionKindP.CALL, 0.98)


def test_a_call_at_the_forward_has_no_implied_vol() -> None:
    """No volatility makes a call worth more than the forward it is written on."""
    with pytest.raises(NoImpliedVolError, match="ceiling"):
        implied_vol(FORWARD, FORWARD, STRIKE, TENOR, OptionKindP.CALL)


def test_a_call_above_the_forward_has_no_implied_vol() -> None:
    with pytest.raises(NoImpliedVolError, match="ceiling"):
        implied_vol(FORWARD * 1.5, FORWARD, STRIKE, TENOR, OptionKindP.CALL)


def test_a_put_above_the_strike_has_no_implied_vol() -> None:
    """The put's ceiling is the strike, not the forward: the most it can pay is the whole
    strike, when the underlying finishes at zero.
    """
    with pytest.raises(NoImpliedVolError, match="ceiling"):
        implied_vol(STRIKE * 1.1, FORWARD, STRIKE, TENOR, OptionKindP.PUT)


def test_a_zero_price_has_no_implied_vol_rather_than_a_value_error() -> None:
    """A worthless quote is a market condition the caller drops, not a construction bug. The
    two are different exception types on purpose.
    """
    with pytest.raises(NoImpliedVolError):
        implied_vol(0.0, FORWARD, 120_000.0, TENOR, OptionKindP.CALL)


def test_a_negative_price_has_no_implied_vol() -> None:
    with pytest.raises(NoImpliedVolError):
        implied_vol(-1.0, FORWARD, 120_000.0, TENOR, OptionKindP.CALL)


def test_a_target_just_inside_the_ceiling_still_inverts() -> None:
    """The bound is checked as an open interval, so a price one part in ten thousand below the
    ceiling must produce a very large vol rather than an error.
    """
    assert implied_vol(FORWARD * 0.9999, FORWARD, STRIKE, TENOR, OptionKindP.CALL) > 1.0


# --- guards: construction bugs, which are ValueError and not NoImpliedVolError


@pytest.mark.parametrize("bad", UNUSABLE)
def test_price_rejects_an_unusable_forward(bad: float) -> None:
    with pytest.raises(ValueError, match="forward"):
        price(bad, STRIKE, TENOR, VOL, OptionKindP.CALL)


@pytest.mark.parametrize("bad", UNUSABLE)
def test_price_rejects_an_unusable_strike(bad: float) -> None:
    with pytest.raises(ValueError, match="strike"):
        price(FORWARD, bad, TENOR, VOL, OptionKindP.CALL)


@pytest.mark.parametrize("bad", UNUSABLE)
def test_price_rejects_an_unusable_tenor(bad: float) -> None:
    """A zero tenor is an expired option, not a priceable one, and it would divide by zero two
    lines later.
    """
    with pytest.raises(ValueError, match="tenor"):
        price(FORWARD, STRIKE, bad, VOL, OptionKindP.CALL)


@pytest.mark.parametrize("bad", UNUSABLE)
def test_price_rejects_an_unusable_vol(bad: float) -> None:
    """NaN is the one that matters: `float("nan") <= 0` is False, so it walks through any guard
    written with the ordering test alone and comes back out as a NaN premium.
    """
    with pytest.raises(ValueError, match="volatility"):
        price(FORWARD, STRIKE, TENOR, bad, OptionKindP.CALL)


@pytest.mark.parametrize("bad", UNUSABLE)
def test_price_rejects_an_unusable_discount(bad: float) -> None:
    with pytest.raises(ValueError, match="discount factor"):
        price(FORWARD, STRIKE, TENOR, VOL, OptionKindP.CALL, bad)


@pytest.mark.parametrize("bad", UNUSABLE)
def test_vega_rejects_an_unusable_vol(bad: float) -> None:
    with pytest.raises(ValueError, match="volatility"):
        vega(FORWARD, STRIKE, TENOR, bad)


@pytest.mark.parametrize("bad", UNUSABLE)
def test_vega_rejects_an_unusable_tenor(bad: float) -> None:
    with pytest.raises(ValueError, match="tenor"):
        vega(FORWARD, STRIKE, bad, VOL)


@pytest.mark.parametrize("bad", UNUSABLE)
def test_the_inversion_rejects_an_unusable_forward(bad: float) -> None:
    with pytest.raises(ValueError, match="forward"):
        implied_vol(1_000.0, bad, STRIKE, TENOR, OptionKindP.CALL)


@pytest.mark.parametrize("bad", UNUSABLE)
def test_the_inversion_rejects_an_unusable_tenor(bad: float) -> None:
    with pytest.raises(ValueError, match="tenor"):
        implied_vol(1_000.0, FORWARD, STRIKE, bad, OptionKindP.CALL)


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_the_inversion_rejects_a_non_finite_target_price(bad: float) -> None:
    """A NaN premium is the absence of a price, not a price no vol reproduces. Reporting it as
    NoImpliedVolError would let a quote-dropping loop swallow a broken feed silently.
    """
    with pytest.raises(ValueError, match="target price"):
        implied_vol(bad, FORWARD, STRIKE, TENOR, OptionKindP.CALL)


def test_an_underflowing_total_standard_deviation_falls_back_to_the_intrinsic() -> None:
    """Pathological but reachable from a caller: vol and tenor small enough that their product
    underflows to zero. The vol -> 0 limit is the right answer, and it keeps a bare
    ZeroDivisionError from escaping the context.
    """
    tiny = 1e-300
    assert price(FORWARD, 50_000.0, tiny, tiny, OptionKindP.CALL) == pytest.approx(10_000.0)
    assert vega(FORWARD, 50_000.0, tiny, tiny) == 0.0


# --- architecture


def test_the_option_kind_values_are_explicit_strings() -> None:
    """House rule: never auto(). The value is read by humans debugging a fit and must not change
    when a member is renamed.
    """
    assert OptionKindP.CALL.value == "CALL"
    assert OptionKindP.PUT.value == "PUT"
    assert isinstance(OptionKindP.CALL, str)


def test_the_inversion_error_is_a_calibration_error() -> None:
    """It has to be catchable as this context's own hierarchy, or the guarantee that one except
    clause covers the context is worth nothing.
    """
    assert issubclass(NoImpliedVolError, CalibrationError)
