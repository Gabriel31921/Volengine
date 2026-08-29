"""Tests for Risk's own Black-76.

Deliberately importing nothing from ``tests/risk/builders.py``: this module has no value objects
and takes only floats, so a builder would add an import of every other module in the context to a
test file that needs none of them. The numbers below are the same forward, strike and tenor the
Pricing twin is tested at, which makes a side-by-side reading of the two files worth doing.
"""

from __future__ import annotations

import math
from itertools import pairwise

import pytest

from volengine.risk.domain import pricing
from volengine.risk.domain.pricing import OptionKindR, price

FORWARD = 60_000.0
STRIKE = 60_000.0
TENOR = 0.25
VOL = 0.65

BOTH_KINDS = [OptionKindR.CALL, OptionKindR.PUT]
UNUSABLE = [0.0, -1.0, float("nan"), float("inf"), float("-inf")]

# From a deep put wing to a deep call wing, at a quarter-year tenor.
ACROSS_THE_SMILE = [30_000.0, 45_000.0, 60_000.0, 80_000.0, 120_000.0]


# --- builders


def call(vol: float = VOL, strike: float = STRIKE, discount: float = 1.0) -> float:
    return price(FORWARD, strike, TENOR, vol, OptionKindR.CALL, discount)


def put(vol: float = VOL, strike: float = STRIKE, discount: float = 1.0) -> float:
    return price(FORWARD, strike, TENOR, vol, OptionKindR.PUT, discount)


# --- the identities that tie the two sides together


@pytest.mark.parametrize("strike", ACROSS_THE_SMILE)
def test_a_call_and_a_put_satisfy_put_call_parity(strike: float) -> None:
    """C - P = D * (F - K) is an arbitrage identity, not a model result: it must hold to rounding
    at every strike, or one of the two branches has a sign or a CDF argument wrong.

    It is also the cheapest guard this file has on a wing: parity ties the deep out-of-the-money
    branch to the deep in-the-money one, so a CDF that collapsed in the left tail would break the
    identity at 30,000 and 120,000 while the at-the-money case stayed perfect.
    """
    assert call(strike=strike) - put(strike=strike) == pytest.approx(FORWARD - strike, rel=1e-12)


def test_put_call_parity_survives_discounting() -> None:
    """The discount multiplies the whole premium, so it multiplies the parity gap too. A branch
    that applied it to only one leg would still return plausible prices.
    """
    discount = 0.97
    gap = call(strike=55_000.0, discount=discount) - put(strike=55_000.0, discount=discount)
    assert gap == pytest.approx(discount * (FORWARD - 55_000.0), rel=1e-12)


@pytest.mark.parametrize("strike", ACROSS_THE_SMILE)
def test_a_call_equals_the_put_with_forward_and_strike_swapped(strike: float) -> None:
    """Black-76 is symmetric under exchanging F and K together with the side: d1(K, F) = -d2(F, K).
    Independent of parity, and it pins the *arguments* of the two CDF calls in the put branch --
    a put written with d1 and d2 transposed would still satisfy parity at the money.
    """
    assert price(FORWARD, strike, TENOR, VOL, OptionKindR.CALL) == pytest.approx(
        price(strike, FORWARD, TENOR, VOL, OptionKindR.PUT), rel=1e-12
    )


def test_an_at_the_money_call_matches_the_closed_form_erf_identity() -> None:
    """At F = K the whole formula collapses to D * F * erf(sqrt(w) / (2 * sqrt(2))), because
    2 * N(x) - 1 is erf(x / sqrt(2)). An independent expression of the same number, so it pins the
    constant inside ``_norm_cdf`` rather than only the shape of the price.
    """
    expected = 0.97 * FORWARD * math.erf(VOL * math.sqrt(TENOR) / (2.0 * math.sqrt(2.0)))
    assert call(discount=0.97) == pytest.approx(expected, rel=1e-12)


# --- shape


@pytest.mark.parametrize("kind", BOTH_KINDS)
def test_the_price_is_strictly_increasing_in_vol(kind: OptionKindR) -> None:
    """Monotonicity in vol is what makes a vega meaningful: ``valuation.py`` bumps this argument
    symmetrically and divides by the step, so a formula that was flat or non-monotone here would
    report a zero or negative vega on a long option and no other test would notice.
    """
    prices = [price(FORWARD, STRIKE, TENOR, vol, kind) for vol in (0.05, 0.2, 0.5, 1.0, 2.0, 5.0)]
    assert all(lower < higher for lower, higher in pairwise(prices))


def test_the_price_approaches_the_intrinsic_value_as_vol_vanishes() -> None:
    """The vol -> 0 limit is the discounted intrinsic value. A book valued with a collapsing
    volatility must converge to what it would settle for, not to zero and not to a rounding of it.
    """
    assert call(vol=1e-6, strike=50_000.0) == pytest.approx(FORWARD - 50_000.0, rel=1e-9)
    assert put(vol=1e-6, strike=70_000.0) == pytest.approx(70_000.0 - FORWARD, rel=1e-9)


def test_an_out_of_the_money_option_is_worthless_as_vol_vanishes() -> None:
    """The other half of the same limit, and the one that would hide a sign error: an
    out-of-the-money option has no intrinsic value at all, so its vol -> 0 limit is zero.
    """
    assert call(vol=1e-6, strike=70_000.0) == pytest.approx(0.0, abs=1e-9)


def test_the_discount_factor_scales_the_price_exactly_linearly() -> None:
    """The discount is a multiplicative constant, which is the property ``valuation.py`` leans on
    when it claims that leaving it at 1.0 costs the Design 7.3 comparison nothing: it scales the
    value and every bumped revaluation, hence every greek, by the same factor.

    Doubles as an anti-vacuous guard for every other test that passes a discount: if the argument
    were ignored entirely, both sides of the tests above would move together and stay equal.
    """
    assert call(discount=0.9) == pytest.approx(0.9 * call(discount=1.0), rel=1e-12)
    assert put(discount=0.9) == pytest.approx(0.9 * put(discount=1.0), rel=1e-12)


def test_an_underflowing_total_standard_deviation_falls_back_to_the_intrinsic() -> None:
    """Pathological but reachable: a vol and a tenor each positive and finite whose product still
    underflows to zero. Both pass the guards individually, so the division would be by zero and a
    bare ZeroDivisionError -- a builtin, invisible to any ``except RiskError`` -- would escape the
    context. The vol -> 0 limit is both the correct answer and the safe one.
    """
    tiny = 1e-300
    assert price(FORWARD, 50_000.0, tiny, tiny, OptionKindR.CALL) == pytest.approx(10_000.0)
    assert price(FORWARD, 70_000.0, tiny, tiny, OptionKindR.PUT) == pytest.approx(10_000.0)


# --- the deep wing, which is why the CDF is written with erfc


DEEP_WING_STRIKE = 600_000.0
DEEP_WING_TENOR = 0.02
DEEP_WING_VOL = 0.8


def norm_cdf_via_erf(x: float) -> float:
    """The textbook spelling the module deliberately does not use."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def call_price_via_erf(strike: float, tenor_years: float, vol: float) -> float:
    """Exactly what ``price`` computes for a call, with the naive CDF substituted in."""
    total_stdev = vol * math.sqrt(tenor_years)
    d1 = (math.log(FORWARD / strike) + 0.5 * total_stdev * total_stdev) / total_stdev
    return FORWARD * norm_cdf_via_erf(d1) - strike * norm_cdf_via_erf(d1 - total_stdev)


def test_a_deep_wing_option_still_prices_above_zero() -> None:
    """Ten times out of the money at a one-week tenor: d1 is about -20, so the price runs through
    N(-20) and lands around 2.3e-89. Small, but perfectly real and perfectly representable.

    **A zero here would not look like a bug**, which is the entire reason this test exists. The
    option is genuinely nearly worthless, so a report showing 0.00 against it reads as correct to
    anyone who glances at it -- and the position would then carry a delta, a gamma and a vega of
    exactly zero as well, because a price pinned at zero does not move when ``valuation.py`` bumps
    it. An unhedged tail would be reported as no risk at all, indistinguishable from a flat book,
    and on a crypto chain that is about half the strikes.
    """
    premium = price(FORWARD, DEEP_WING_STRIKE, DEEP_WING_TENOR, DEEP_WING_VOL, OptionKindR.CALL)
    assert 1e-90 < premium < 1e-88


def test_the_naive_erf_form_would_price_that_same_option_at_exactly_zero() -> None:
    """Guards the test above from being vacuous. Without it, ``_norm_cdf`` could be rewritten with
    the textbook ``0.5 * (1 + erf(x / sqrt(2)))`` and a reader would have no evidence that the
    choice ever mattered.

    erf saturates at -1.0 once its argument passes about -6, so ``1 + erf(...)`` cancels
    catastrophically and returns exactly 0.0 for both legs: N(-9) comes out as 0.0 instead of
    1.128e-19.
    """
    assert norm_cdf_via_erf(-9.0) == 0.0
    exact = 0.5 * math.erfc(9.0 / math.sqrt(2.0))
    assert exact == pytest.approx(1.1285884e-19, rel=1e-6, abs=1e-24)
    assert call_price_via_erf(DEEP_WING_STRIKE, DEEP_WING_TENOR, DEEP_WING_VOL) == 0.0


def test_a_deep_wing_price_still_responds_to_a_vol_bump() -> None:
    """The consequence spelled out: a live price moves when the vol does, so the wing has a vega.
    Under the naive CDF both revaluations would be 0.0 and the difference a clean zero.
    """

    def wing(vol: float) -> float:
        return price(FORWARD, DEEP_WING_STRIKE, DEEP_WING_TENOR, vol, OptionKindR.CALL)

    assert wing(DEEP_WING_VOL + 0.01) > wing(DEEP_WING_VOL - 0.01) > 0.0


def test_a_deep_wing_put_still_prices_above_zero() -> None:
    """The left wing runs through the other branch, which calls the CDF with negated arguments. A
    fix applied to only one of the two would leave this one at zero.
    """
    assert price(FORWARD, 6_000.0, DEEP_WING_TENOR, DEEP_WING_VOL, OptionKindR.PUT) > 0.0


# --- guards: construction bugs, which are ValueError and never RiskError


@pytest.mark.parametrize("bad", UNUSABLE)
def test_the_price_rejects_an_unusable_forward(bad: float) -> None:
    with pytest.raises(ValueError, match="forward"):
        price(bad, STRIKE, TENOR, VOL, OptionKindR.CALL)


@pytest.mark.parametrize("bad", UNUSABLE)
def test_the_price_rejects_an_unusable_strike(bad: float) -> None:
    with pytest.raises(ValueError, match="strike"):
        price(FORWARD, bad, TENOR, VOL, OptionKindR.CALL)


@pytest.mark.parametrize("bad", UNUSABLE)
def test_the_price_rejects_an_unusable_tenor(bad: float) -> None:
    """A zero tenor is an expired position rather than a priceable one -- ``SurfaceView.tenor_of``
    raises ``ExpiredPositionError`` before it can reach here -- and it would divide by zero anyway.
    """
    with pytest.raises(ValueError, match="tenor"):
        price(FORWARD, STRIKE, bad, VOL, OptionKindR.CALL)


@pytest.mark.parametrize("bad", UNUSABLE)
def test_the_price_rejects_an_unusable_vol(bad: float) -> None:
    """NaN is the member of that list that matters: ``float("nan") <= 0`` is False, so it walks
    through any guard written with the ordering test alone, comes back out as a NaN premium and
    poisons the sum in ``RiskReport.total_value`` -- one bad interpolation, and a whole book's
    total reads ``nan``.
    """
    with pytest.raises(ValueError, match="volatility"):
        price(FORWARD, STRIKE, TENOR, bad, OptionKindR.CALL)


@pytest.mark.parametrize("bad", UNUSABLE)
def test_the_price_rejects_an_unusable_discount(bad: float) -> None:
    with pytest.raises(ValueError, match="discount factor"):
        price(FORWARD, STRIKE, TENOR, VOL, OptionKindR.CALL, bad)


def test_a_nan_really_would_survive_an_ordering_guard_written_alone() -> None:
    """The premise of every guard in the module, asserted once so the ``isfinite``-first spelling
    is not mistaken for belt and braces: with ``x <= 0`` alone, a NaN vol is accepted.
    """
    assert (float("nan") <= 0) is False
    assert math.isnan(float("nan") * math.sqrt(TENOR))


# --- architecture


def test_the_option_kind_values_are_explicit_strings() -> None:
    """House rule: never auto(). The value is what a human reads in a risk report and must not
    change when a member is renamed.
    """
    assert OptionKindR.CALL.value == "CALL"
    assert OptionKindR.PUT.value == "PUT"
    assert isinstance(OptionKindR.CALL, str)


def test_risk_pricing_carries_no_inversion_and_no_greeks() -> None:
    """The asymmetry with the Pricing twin, kept standing by a test. Risk goes vol -> price only:
    it is handed a volatility by the surface and asked what the position is worth, so an
    ``implied_vol`` here would re-derive its own input. And its greeks are finite differences
    through the interpolated grid in ``valuation.py`` (Design 7.4) -- an analytic ``vega`` in this
    module would be a second, quietly different number that misses the interpolation entirely.
    """
    assert not hasattr(pricing, "implied_vol")
    assert not hasattr(pricing, "vega")
    assert not hasattr(pricing, "delta")
