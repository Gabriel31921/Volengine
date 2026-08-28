from __future__ import annotations

import math
from datetime import timedelta

import pytest

from tests.risk.builders import (
    FORWARDS,
    K_AXIS,
    NOW,
    TENORS,
    make_bumps,
    make_flat_view,
    make_position,
    make_view,
    smile_vol,
)
from tests.support import replace_field
from volengine.risk.domain.errors import ExpiredPositionError
from volengine.risk.domain.interpolation import implied_vol
from volengine.risk.domain.pricing import OptionKindR, price
from volengine.risk.domain.risk_report import PositionRisk
from volengine.risk.domain.surface_view import SurfaceView
from volengine.risk.domain.valuation import DOWN_BUMP_FLOOR, BumpSpec, position_risk, value

TENOR = TENORS[1]
FORWARD = FORWARDS[1]
"""The default position's node: the three-month expiry, and the forward at it.

``make_position`` expires at ``EXPIRIES[1]`` and strikes at ``FORWARDS[1]``, so it lands exactly on
a grid node in both directions -- log-moneyness zero, tenor on the axis. Everything asserted below
against a closed form leans on that: no interpolation is hiding in the base valuation, so a
mismatch is the valuation's fault and not the grid's.
"""

UNUSABLE = [0.0, -1.0, float("nan"), float("inf"), float("-inf")]

FD_TOLERANCE = 1e-4
"""How far a bumped delta may sit from the analytic one, on the flat surface, at a 1% bump.

Justified from the bump rather than guessed. A central difference carries a truncation error of
``(h^2 / 6) * d3V/dF3``; with ``h = 604`` and the third derivative of an at-the-money call of this
size around ``-5e-10``, that is about ``3e-5``, and the measured error below is ``-3.05e-5``. The
tolerance is three times it, which is loose enough not to be brittle and tight enough to be a real
assertion: the same delta computed on the *smiled* surface is off by ``2e-2``, two hundred times
this bound, so a test that accidentally used the wrong surface could not pass.
"""


# --- builders local to this file


def atm_risk(
    view: SurfaceView | None = None,
    quantity: float = 10.0,
    discount: float = 1.0,
    bumps: BumpSpec | None = None,
) -> PositionRisk:
    """The default at-the-money line, valued. Kept out of the way of what each test is saying."""
    return position_risk(
        make_view() if view is None else view,
        make_position(quantity=quantity),
        make_bumps() if bumps is None else bumps,
        discount,
    )


def strike_at(k: float) -> float:
    """The strike whose log-moneyness against the three-month forward is exactly ``k``."""
    return FORWARD * math.exp(k)


def norm_cdf(x: float) -> float:
    """The standard normal CDF, independently of ``pricing``: ``erfc``, never ``1 + erf``."""
    return 0.5 * math.erfc(-x / math.sqrt(2.0))


def flat_view_matching(view: SurfaceView, strike: float) -> SurfaceView:
    """A smile-free surface carrying exactly the volatility ``view`` shows at ``strike``.

    The control for every vacuity guard below. Comparing a smiled surface against
    ``make_flat_view()`` at its default 65% would conflate two effects -- the level of the
    volatility and its slope -- and the level is the uninteresting one. Matching the level leaves
    the slope as the only difference there is.
    """
    vol = implied_vol(view, math.log(strike / FORWARD), TENOR)
    return make_flat_view(vol=vol)


# --- the base valuation


def test_the_value_of_an_at_the_money_call_matches_a_hand_computed_black_76_price() -> None:
    """At the money forward the two normal terms collapse: with ``F == K``, ``d1 = -d2 = s / 2``
    and the premium is exactly ``F * erf(s / (2 sqrt 2))``. That closed form goes nowhere near the
    module under test, so it pins the whole chain -- tenor, forward, moneyness, interpolation,
    price -- against arithmetic rather than against another implementation.
    """
    vol = smile_vol(TENOR, 0.0)
    total_stdev = vol * math.sqrt(TENOR)
    expected = 10.0 * FORWARD * math.erf(total_stdev / (2.0 * math.sqrt(2.0)))

    assert value(make_view(), make_position()) == pytest.approx(expected, rel=1e-12)
    assert expected == pytest.approx(75_981.410, rel=1e-8)


def test_the_reported_vol_is_the_one_the_surface_interpolates_at_the_position() -> None:
    """The line reports the volatility it actually priced with, which is what makes a report
    auditable: a reader can look up ``k`` and the tenor on the published grid and find this number.
    """
    risk = position_risk(make_view(), make_position(), make_bumps())

    assert risk.vol == implied_vol(make_view(), 0.0, TENOR)


def test_value_and_position_risk_agree_on_the_value() -> None:
    """Two entry points, one number. They perform the same five steps, and a divergence would mean
    a report whose lines do not add up to what the same surface says a position is worth.
    """
    risk = position_risk(make_view(), make_position(), make_bumps())

    assert risk.value == value(make_view(), make_position())


def test_the_value_scales_linearly_in_the_quantity() -> None:
    """Value is quantity times a premium and nothing else, so three lots are worth three times one
    lot exactly -- not approximately, since it is the same multiplication.
    """
    one = value(make_view(), make_position(quantity=1.0))

    assert value(make_view(), make_position(quantity=3.0)) == pytest.approx(3.0 * one, rel=1e-15)


def test_a_short_position_flips_the_sign_of_the_value_and_of_every_greek() -> None:
    """Direction is recorded in one place, the signed quantity, and every number on the line is
    multiplied by it. A branch on the sign anywhere in the valuation would show up here as a greek
    that failed to turn over.
    """
    long_leg = atm_risk(quantity=10.0)
    short_leg = atm_risk(quantity=-10.0)

    for field in ("value", "delta", "gamma", "vega"):
        assert getattr(short_leg, field) == pytest.approx(-getattr(long_leg, field), rel=1e-15)


def test_a_flattened_leg_reports_exactly_zero_value_and_greeks() -> None:
    """A quantity of zero is legal -- a leg flattened intraday stays in the file -- and the honest
    line for it is zero everywhere, with a volatility that is still the market's.
    """
    risk = atm_risk(quantity=0.0)

    assert (risk.value, risk.delta, risk.gamma, risk.vega) == (0.0, 0.0, 0.0, 0.0)
    assert risk.vol > 0.0


# --- the limits every delta has to satisfy


def test_the_delta_of_a_deep_in_the_money_call_approaches_the_discount() -> None:
    """Far enough in the money the option is a forward contract: it will be exercised in every
    state anyone can price, so its value is ``D * (F - K)`` and its sensitivity to the forward is
    ``D``. Asserted with a discount of 0.97 rather than 1.0 so that a valuation ignoring the
    discount factor in the bumped legs could not pass by returning 1.
    """
    position = make_position(strike=strike_at(-3.0), quantity=1.0)
    risk = position_risk(make_view(), position, make_bumps(), 0.97)

    assert risk.delta == pytest.approx(0.97, abs=1e-6)


def test_the_delta_of_a_deep_out_of_the_money_call_approaches_zero() -> None:
    """Nine standard deviations out, the option moves with the forward by about 4e-20 -- small, and
    emphatically not zero. A CDF written as ``1 + erf`` would have deleted this wing outright, so
    the assertion is two-sided on purpose: tiny, and strictly positive.
    """
    position = make_position(strike=strike_at(3.0), quantity=1.0)
    risk = position_risk(make_view(), position, make_bumps())

    assert 0.0 < risk.delta < 1e-6


def test_gamma_is_strictly_positive_for_a_long_option() -> None:
    """Convexity in the forward: both a call and a put gain from a move in either direction, so a
    long option has positive gamma whichever side it is. A sign error in the second difference
    would report a long option as short convexity, which is the worst possible hedge.
    """
    for kind in (OptionKindR.CALL, OptionKindR.PUT):
        risk = position_risk(make_view(), make_position(kind=kind), make_bumps())
        assert risk.gamma > 0.0


def test_vega_is_strictly_positive_for_a_long_option() -> None:
    """Black-76 is strictly increasing in volatility on both sides -- put-call parity has no
    volatility in it, so the two vegas are identical -- and a long position inherits that.
    """
    for kind in (OptionKindR.CALL, OptionKindR.PUT):
        risk = position_risk(make_view(), make_position(kind=kind), make_bumps())
        assert risk.vega > 0.0


def test_vega_is_reported_per_unit_of_vol_and_not_per_vol_point() -> None:
    """Both conventions are in daily use and they differ by a factor of a hundred, so the one this
    module chose is pinned against the analytic ``F phi(d1) sqrt(T)`` rather than left to a reader
    to infer. Ten at-the-money BTC calls come out near 118,000 currency units per unit of vol; a
    per-point convention would report 1,180 and look just as plausible on the page.
    """
    vol = smile_vol(TENOR, 0.0)
    total_stdev = vol * math.sqrt(TENOR)
    d1 = 0.5 * total_stdev
    pdf = math.exp(-0.5 * d1 * d1) / math.sqrt(2.0 * math.pi)
    analytic = 10.0 * FORWARD * pdf * math.sqrt(TENOR)

    assert atm_risk().vega == pytest.approx(analytic, rel=1e-5)
    assert analytic == pytest.approx(118_162.0, rel=1e-5)


# --- the bump machinery, pinned against a closed form


def test_the_delta_on_a_flat_surface_matches_the_analytic_black_76_delta() -> None:
    """With no smile there is nothing for the moneyness to pick up as the forward moves, so
    sticky-moneyness and sticky-strike coincide and the bumped delta must reproduce ``N(d1)``. This
    is the control that says the machinery itself -- five revaluations, a central difference and a
    quantity -- is right, before any test attributes a discrepancy to the smile.
    """
    analytic = norm_cdf(0.5 * 0.65 * math.sqrt(TENOR))
    risk = position_risk(make_flat_view(), make_position(quantity=1.0), make_bumps())

    assert risk.delta == pytest.approx(analytic, abs=FD_TOLERANCE)


def test_halving_the_bump_quarters_the_error_in_the_delta() -> None:
    """The claim behind :data:`FD_TOLERANCE` is that the central difference is second-order
    accurate, and second order is a statement about how the error *moves*, not about its size.
    Halving the bump must therefore divide the error by about four -- if it only halved it, the
    difference would secretly be one-sided and the tolerance above would be off by orders of
    magnitude at any other bump size.
    """
    analytic = norm_cdf(0.5 * 0.65 * math.sqrt(TENOR))
    position = make_position(quantity=1.0)
    view = make_flat_view()
    coarse = position_risk(view, position, make_bumps(forward_rel=0.01)).delta - analytic
    fine = position_risk(view, position, make_bumps(forward_rel=0.005)).delta - analytic

    assert coarse / fine == pytest.approx(4.0, rel=0.05)


def test_the_smile_slope_enters_the_delta() -> None:
    """The vacuity guard for every assertion above, and the test that makes sticky-moneyness
    observable. The same position is valued twice: once on the smiled surface, once on a flat one
    carrying exactly the volatility the smiled one shows at that strike. The base values are
    identical by construction, so the level of the volatility is ruled out as an explanation -- and
    the deltas still differ by 0.02, which is the smile's slope seen through vega. Under
    sticky-strike the two would agree, so this number is the price of the convention.
    """
    strike = strike_at(0.15)
    position = make_position(strike=strike, quantity=1.0)
    smiled = position_risk(make_view(), position, make_bumps())
    flat = position_risk(flat_view_matching(make_view(), strike), position, make_bumps())

    assert smiled.value == pytest.approx(flat.value, rel=1e-12)
    assert smiled.vol == pytest.approx(flat.vol, rel=1e-12)
    assert smiled.delta - flat.delta == pytest.approx(-0.0198, abs=1e-3)


def test_the_matched_flat_surface_really_is_flat_where_the_smiled_one_is_not() -> None:
    """Guarding the guard. The test above proves nothing if the surface it calls smiled happens to
    be flat in the region the bump explores, so the two volatilities either side of the bumped
    forward are compared directly: on the smiled grid they differ, on the matched flat one they do
    not, and the bumped moneyness stays inside the axis where that is a real interpolation rather
    than the clamp.
    """
    strike = strike_at(0.15)
    flat = flat_view_matching(make_view(), strike)
    k_up = math.log(strike / (FORWARD * 1.01))
    k_down = math.log(strike / (FORWARD * 0.99))

    assert K_AXIS[0] < k_up < k_down < K_AXIS[-1]
    assert implied_vol(make_view(), k_up, TENOR) != implied_vol(make_view(), k_down, TENOR)
    assert implied_vol(flat, k_up, TENOR) == pytest.approx(implied_vol(flat, k_down, TENOR))


def test_gamma_at_a_node_strike_carries_the_kink_of_the_interpolation() -> None:
    """A documented limitation, pinned so it stays visible. The grid is piecewise linear in total
    variance, so it has a kink at every node; the default position strikes exactly on one, and the
    two forward bumps straddle it. The second difference therefore measures the curvature of the
    *interpolation* on top of the curvature of the option, and comes out about twice the value the
    same position shows on a flat surface at the same volatility level. This is the concrete
    content of the module docstring's claim that these greeks are not Pricing's AD greeks: nothing
    is broken, and a comparison against an analytic gamma has to know it.
    """
    smiled = atm_risk()
    flat = position_risk(flat_view_matching(make_view(), 60_400.0), make_position(), make_bumps())

    assert smiled.gamma / flat.gamma == pytest.approx(2.10, abs=0.05)


# --- the volatility floor


def low_vol_view() -> SurfaceView:
    """A surface at half a volatility point: below any bump this context would be configured with.

    Not a market anyone quotes, and that is the point -- it is the only way to reach the branch,
    and the branch exists because a grid this quiet is representable and would otherwise send a
    negative volatility into the pricer.
    """
    return make_view(
        total_variance=tuple(tuple(0.005 * 0.005 * tenor for _ in K_AXIS) for tenor in TENORS)
    )


def test_a_vol_bump_wider_than_the_vol_would_be_unpriceable_without_the_floor() -> None:
    """The vacuity guard for the two tests below: they are only meaningful if the unclamped
    down-bump really is a volatility the pricer refuses. It is -- ``0.005 - 0.01`` is negative, and
    ``price`` rejects it rather than returning a plausible number.
    """
    with pytest.raises(ValueError, match="volatility"):
        price(FORWARD, 60_400.0, TENOR, 0.005 - 0.01, OptionKindR.CALL)


def test_the_down_bumped_vol_is_floored_rather_than_allowed_below_zero() -> None:
    """The bump is wider than the whole volatility, and the line still reports: a positive vega, a
    finite value, and the unbumped volatility -- visibly smaller than the configured bump, which is
    how a reader can tell this line's vega was taken one-sided.
    """
    risk = position_risk(low_vol_view(), make_position(), make_bumps())

    assert risk.vol == pytest.approx(0.005)
    assert risk.vol < make_bumps().vol_abs
    assert risk.vega > 0.0
    assert math.isfinite(risk.value)


def test_the_floored_vega_is_the_quotient_over_the_span_actually_used() -> None:
    """Dividing by ``2e`` when only part of that span was taken would understate the vega by the
    amount clipped -- here by about a third. The reported number is the difference quotient over
    the two points that were really evaluated, which is what keeps it a derivative estimate of the
    function rather than of an interval nobody visited.
    """
    vol_up = 0.005 + 0.01
    vol_down = DOWN_BUMP_FLOOR * 0.005
    up = price(FORWARD, 60_400.0, TENOR, vol_up, OptionKindR.CALL)
    down = price(FORWARD, 60_400.0, TENOR, vol_down, OptionKindR.CALL)
    expected = 10.0 * (up - down) / (vol_up - vol_down)

    risk = position_risk(low_vol_view(), make_position(), make_bumps())

    assert risk.vega == pytest.approx(expected, rel=1e-12)
    assert risk.vega != pytest.approx(10.0 * (up - down) / 0.02, rel=1e-6)


# --- the discount factor


def test_the_discount_scales_the_value_and_every_greek_identically() -> None:
    """It is a multiplicative constant on the price, so it survives every difference quotient
    unchanged. That is the reason leaving it at 1.0 costs the producer comparison nothing: both
    sides carry the same missing factor and the ratios between the greeks are untouched.
    """
    plain = atm_risk()
    discounted = atm_risk(discount=0.97)

    for field in ("value", "delta", "gamma", "vega"):
        assert getattr(discounted, field) == pytest.approx(0.97 * getattr(plain, field), rel=1e-12)


# --- the one condition that has no number behind it


@pytest.mark.parametrize("offset", [timedelta(0), timedelta(seconds=-1), timedelta(days=-30)])
def test_a_position_expiring_at_or_before_the_snapshot_raises(offset: timedelta) -> None:
    """``ExpiredPositionError`` comes out of ``SurfaceView.tenor_of`` and is deliberately not
    caught here: an expired option has no time value to interpolate, and reporting a zero would be
    indistinguishable from an option that is merely worthless. The boundary is inclusive -- an
    option expiring at the snapshot instant has a tenor of exactly zero, which no volatility
    divides into.
    """
    position = make_position(expiry=NOW + offset)

    with pytest.raises(ExpiredPositionError):
        value(make_view(), position)
    with pytest.raises(ExpiredPositionError):
        position_risk(make_view(), position, make_bumps())


# --- the bump specification


def test_a_bump_spec_has_no_defaults() -> None:
    """An architecture test. Bump sizes are configuration (ADR-012), built at the composition root
    from a TOML file; a default here would be a policy nobody chose that quietly becomes the one
    every report uses.
    """
    with pytest.raises(TypeError):
        BumpSpec()  # type: ignore[call-arg]


@pytest.mark.parametrize("bad", UNUSABLE)
def test_an_unusable_forward_bump_is_refused(bad: float) -> None:
    with pytest.raises(ValueError, match="forward bump"):
        replace_field(make_bumps(), "forward_rel", bad)


@pytest.mark.parametrize("bad", [1.0, 1.5, 100.0])
def test_a_forward_bump_of_a_hundred_percent_or_more_is_refused(bad: float) -> None:
    """The down-bumped forward is ``F * (1 - forward_rel)``: at one it is zero and past it is
    negative, and ``ln(K / F)`` stops existing. Refused here, where the message can name the bump,
    rather than three calls down where it would surface as a bare domain error from ``math.log``.
    """
    with pytest.raises(ValueError, match="strictly between 0 and 1"):
        replace_field(make_bumps(), "forward_rel", bad)


@pytest.mark.parametrize("bad", UNUSABLE)
def test_an_unusable_vol_bump_is_refused(bad: float) -> None:
    with pytest.raises(ValueError, match="vol bump"):
        replace_field(make_bumps(), "vol_abs", bad)


def test_a_vol_bump_larger_than_one_is_accepted() -> None:
    """No upper bound on the volatility bump, unlike the forward one: it is a poor bump, handled by
    the floor, and its admissible range would otherwise depend on a surface the constructor cannot
    see.
    """
    assert BumpSpec(forward_rel=0.01, vol_abs=2.0).vol_abs == 2.0
