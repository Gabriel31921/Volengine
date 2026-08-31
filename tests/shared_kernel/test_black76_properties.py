"""Black-76 stated as laws over a whole family of contracts, not as a table of examples.

``test_black76.py`` pins the formula at chosen points: parity at one strike, the left tail at one
argument, one inversion of one premium. Those are the cases somebody thought of. This module says
the same things quantified over every forward, tenor, volatility and side inside a stated band, and
hands the search for the bad case to hypothesis -- which is the only way a closed form gets tested
the way a closed form is used, since the engine will invert millions of premiums nobody wrote down
in advance.

**The band is part of the statement.** The inversion recovers a volatility only where the premium
depends on one: a contract twelve standard deviations in the money is worth its intrinsic value to
the last bit a float carries, its vega is zero to the same precision, and no solver can read a
volatility out of a number that no longer varies with it. So the round-trip property is quantified
over strikes within three standard deviations of the forward, and the test right after it shows
that the qualifier is load-bearing rather than decorative -- an unbounded version of the same
property fails, and it should.
"""

from __future__ import annotations

import math

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from volengine.shared_kernel.domain.black76 import implied_vol, price

FORWARDS = st.floats(min_value=1e2, max_value=1e6)
"""From a hundred to a million, which spans an ETH forward and a BTC one with room either side."""

TENORS = st.floats(min_value=1.0 / 365.0, max_value=3.0)
"""One day to three years: the shortest expiry a venue lists and the longest anyone quotes."""

VOLS = st.floats(min_value=0.05, max_value=3.0)
"""Five per cent to three hundred: an index in a coma and a coin mid-liquidation."""

STANDARD_DEVIATIONS = st.floats(min_value=-3.0, max_value=3.0)
"""Where the strike sits, in units of the contract's own total standard deviation ``vol * sqrt(T)``.

Drawn in this coordinate rather than as an absolute strike on purpose: a strike is far from the
money only relative to how far the underlying can travel before expiry, and a fixed band in
log-moneyness would be the whole chain for a one-day option and a single strike for a three-year
one.
"""

SIDES = st.booleans()
"""``is_call``. The kernel takes a bool and imposes no vocabulary on its callers (ADR-014)."""


def strike_at(forward: float, tenor_years: float, vol: float, standard_deviations: float) -> float:
    """The strike that many standard deviations away from the forward, in log-moneyness."""
    return forward * math.exp(standard_deviations * vol * math.sqrt(tenor_years))


@settings(max_examples=400)
@given(
    forward=FORWARDS,
    tenor_years=TENORS,
    vol=VOLS,
    standard_deviations=STANDARD_DEVIATIONS,
    is_call=SIDES,
)
def test_an_inversion_returns_the_volatility_that_priced_it(
    forward: float, tenor_years: float, vol: float, standard_deviations: float, is_call: bool
) -> None:
    """``implied_vol(price(sigma)) == sigma``: the law both calibrations rest on.

    Every mid in a chain becomes a volatility through this inversion before anything fits it, so
    an error here is an error in every residual the engine computes, uniformly and invisibly.
    """
    strike = strike_at(forward, tenor_years, vol, standard_deviations)
    premium = price(forward, strike, tenor_years, vol, is_call)

    recovered = implied_vol(premium, forward, strike, tenor_years, is_call)

    assert recovered == pytest.approx(vol, abs=1e-6)


def test_eight_standard_deviations_out_the_volatility_is_no_longer_recoverable() -> None:
    """The guard on the band above: three standard deviations is a real qualifier.

    A put eight standard deviations in the money is worth its intrinsic value to every digit a
    float has left, so its premium no longer varies with volatility and the inversion returns
    whatever its bracket collapsed onto -- here two and a half thousandths of a volatility point
    away from the truth, which is two thousand times the tolerance the property above holds to.
    Further out still the premium reaches intrinsic exactly and the kernel refuses outright; it is
    this quieter band, where it answers and is wrong, that makes the qualifier necessary.

    Stated as a test rather than left to a docstring, because a property quantified over a band is
    only honest if somebody has checked that the band is where the property stops.
    """
    forward, tenor_years, vol = 100.0, 1.0, 0.2
    strike = strike_at(forward, tenor_years, vol, standard_deviations=8.0)
    premium = price(forward, strike, tenor_years, vol, is_call=False)

    recovered = implied_vol(premium, forward, strike, tenor_years, is_call=False)

    assert abs(recovered - vol) > 1e-3


@settings(max_examples=200)
@given(
    forward=FORWARDS,
    tenor_years=TENORS,
    vol=VOLS,
    standard_deviations=STANDARD_DEVIATIONS,
    is_call=SIDES,
    extra_vol=st.floats(min_value=0.01, max_value=1.0),
)
def test_the_premium_rises_with_the_volatility(
    forward: float,
    tenor_years: float,
    vol: float,
    standard_deviations: float,
    is_call: bool,
    extra_vol: float,
) -> None:
    """Strict monotonicity in volatility, which is what makes the inversion's root unique.

    The docstring of ``implied_vol`` argues uniqueness from a positive vega; this is that argument
    as an executable statement, over the same band the inversion is used on.
    """
    strike = strike_at(forward, tenor_years, vol, standard_deviations)

    cheaper = price(forward, strike, tenor_years, vol, is_call)
    dearer = price(forward, strike, tenor_years, vol + extra_vol, is_call)

    assert dearer > cheaper


@settings(max_examples=200)
@given(
    forward=FORWARDS,
    tenor_years=TENORS,
    vol=VOLS,
    standard_deviations=STANDARD_DEVIATIONS,
    discount=st.floats(min_value=0.5, max_value=1.0),
)
def test_a_call_less_a_put_is_the_discounted_forward_less_the_strike(
    forward: float, tenor_years: float, vol: float, standard_deviations: float, discount: float
) -> None:
    """Put-call parity everywhere, not only at the one strike the example test pins.

    The relative tolerance is against the forward rather than against the difference: the two
    premiums are each of the order of the forward and their difference can be a rounding error
    away from zero at the money, where an absolute bound stated in the difference's own units
    would be asking floating point for more digits than it has.
    """
    strike = strike_at(forward, tenor_years, vol, standard_deviations)

    call = price(forward, strike, tenor_years, vol, is_call=True, discount=discount)
    put = price(forward, strike, tenor_years, vol, is_call=False, discount=discount)

    assert call - put == pytest.approx(discount * (forward - strike), abs=1e-9 * forward)
