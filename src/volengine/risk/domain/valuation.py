"""What a position is worth on a surface, and how that worth moves. Design 7.4.

This module is where the five other modules of the context meet, and the sequence it performs is
the whole of it: the expiry becomes a tenor (``surface_view``), the tenor gives a forward
(``surface_view``), the strike becomes a log-moneyness against that forward, the moneyness and the
tenor give a volatility (``interpolation``), and the volatility gives a price (``pricing``), which
is multiplied by the signed size (``portfolio``) and reported as a line (``risk_report``). Nothing
else happens here. Every step is somebody else's rule; this file is the order they go in.

**The greeks are bumped, not differentiated, and the bump goes through the surface.** For delta and
gamma the forward is moved by ``h = F * forward_rel`` and the option is *revalued from scratch*:
``k = ln(K / F')`` is recomputed at the moved forward and the volatility re-interpolated at that
new moneyness. That is a modelling choice with a name -- **sticky-moneyness** -- and it has to be
stated rather than discovered, because the alternative is equally defensible and gives a different
hedge. Under sticky-moneyness the smile is nailed to the forward: as the market moves, the
volatility of a *fixed strike* changes, because that strike now sits at a different moneyness.
Under **sticky-strike** the smile is nailed to the strike ladder and the volatility of a fixed
strike is what stays put. To first order the two deltas differ by exactly one term::

    delta_moneyness  =  delta_black  -  vega * (dsigma/dk) / F
    delta_strike     =  delta_black

so the gap is the smile slope seen through vega, and it is not small: on the test surface here, a
call 15% out of the money in log-moneyness comes out at 0.360 under sticky-moneyness against 0.380
under sticky-strike, a 5% difference in the hedge ratio for the same position on the same day. At
the money on a symmetric smile the slope vanishes and the two conventions agree to second order,
which is precisely why the difference has to be measured off the money or not at all. A desk that
believes the other convention would hedge differently, and would be right to; this engine picks
sticky-moneyness because that is what the grid it is handed actually parameterises -- the published
surface is indexed by ``k``, not by strike, so holding the moneyness fixed is the only reading that
does not invent information the producer never published.

**These greeks are deliberately different from Pricing's AD greeks (Design 7.4).** The parametric
context differentiates an analytic SVI slice with JAX and gets a derivative that is exact up to
floating point. What comes out of this module is a finite difference over a *bilinearly
interpolated grid*, and it inherits two error sources the AD version does not have. The first is
the truncation error of the bump itself, ``O(h^2)`` for every central difference below. The second
is structural and much more interesting: the interpolated surface is piecewise linear in total
variance, so it has **kinks at the nodes**, and a second difference straddling one reports the
curvature of the interpolation on top of the curvature of the option. The at-the-money gamma of the
test surface is twice the analytic value for exactly that reason, and the test that pins it is
there to keep the fact visible rather than to hide it. None of this is a defect to be corrected.
The two numbers answer different questions -- *how does the model curve* against *how does the
published grid curve* -- and the comparison between them is one of the analyses the project is for.
Making them agree would mean either differentiating a surface Risk was never given in closed form,
or importing another context to get it, which rule 6 forbids and rule 3 forbids again.

**Conventions, both of which have a common alternative and are therefore stated.** Every greek here
is already multiplied by ``quantity``: they are portfolio quantities, not per-option ones, so a
short position reports negative delta with no branch anywhere in this file. And vega is **per unit
of volatility**, i.e. per 1.00 of vol and not per vol point: a surface whose volatility moves by one
point moves the position by a hundredth of the reported number. The bump size is unrelated to the
convention -- bumping by ``0.01`` and dividing by ``0.02`` gives the same per-unit derivative as
bumping by ``0.0001`` -- which is what makes the reported number comparable across configurations.

``discount`` is a multiplicative constant on the price and on nothing else, so it scales the value
and all three greeks identically, and none of the *ratios* between them depends on it. That is why
leaving it at its honest default of ``1.0`` costs the producer-to-producer comparison of Design 7.3
nothing at all: both sides are scaled by the same missing factor, and the difference between two
surfaces is what that report is about.

**What this module deliberately does not offer.** No theta: bumping the valuation date moves
``ts_snapshot``, hence every tenor on the axis and the freshness verdict with them, so it is a
different ``SurfaceView`` rather than a bump, and manufacturing one here would put a
carry assumption in the domain. No rho: the rate enters only through ``discount``, so the
sensitivity to it is exactly ``-T`` times the value -- a multiplication the caller can do, not
something worth two revaluations. And nothing here catches ``ExpiredPositionError``: a position the
surface has already passed has no honest number, and swallowing that into a zero would put a
silently wrong line in a report someone hedges from.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from volengine.risk.domain.interpolation import implied_vol
from volengine.risk.domain.portfolio import Position
from volengine.risk.domain.pricing import price
from volengine.risk.domain.risk_report import PositionRisk
from volengine.risk.domain.surface_view import SurfaceView

DOWN_BUMP_FLOOR = 1e-6
"""Fraction of the unbumped volatility the down-bumped volatility is floored at.

``pricing.price`` refuses a volatility of zero or less -- correctly, since that is the intrinsic
value limit rather than a quote -- so a vega bump larger than the volatility itself has to land
somewhere strictly positive. This is where.

**A fraction rather than an absolute constant, and the reason is an ordering that must not break.**
The floor has to be strictly below the unbumped volatility *whatever its magnitude*, because the
difference quotient divides by ``vol_up - vol_down`` and a floor sitting above the up-bump would
return a vega with the wrong sign -- a plausible number, in the right units, pointing the hedge the
wrong way. An absolute floor cannot promise that: on a grid whose volatility is smaller than the
constant, ``max(vol - eps, floor)`` returns something *larger* than ``vol + eps``. A fraction of the
volatility itself is ordered by construction and needs no case analysis.

The value only has to be small enough that the floored point is the zero-volatility limit for
practical purposes: at a millionth of the volatility the option is worth its intrinsic value to
about ten significant figures. It is a property of the numerical method and so a module constant,
explicitly **not** TOML configuration -- ADR-012 governs business thresholds, and an operator who
changed this would only be choosing a worse derivative.
"""


@dataclass(frozen=True, slots=True)
class BumpSpec:
    """How far to move the forward and the volatility when differencing.

    **No defaults, on purpose.** Bump sizes are configuration (ADR-012): they are a trade between
    truncation error, which falls as ``h^2``, and cancellation error, which grows as the two
    revaluations approach each other and their difference loses significant digits. Where the
    optimum sits depends on the market -- a 1% forward bump is a small move in BTC and a large one
    in a rates book -- so the number belongs in the TOML file the composition root reads, and a
    default here would be a policy nobody chose that quietly becomes the one everybody uses.

    Two independent sizes rather than one, because the two bumps are dimensionally different. A
    forward bump is naturally relative: 1% of 60,000 and 1% of 3,000 are the same *market move*,
    and an absolute one would be a different move in every market. A volatility bump is naturally
    absolute: one volatility point is one volatility point at 20% and at 200%, and a relative bump
    would make the reported vega depend on the level of the surface it was measured on.
    """

    forward_rel: float
    """Relative bump of the forward, as a decimal: ``0.01`` is one percent. In ``(0, 1)``.

    Strictly positive because a zero bump divides by zero, and strictly **below one** because the
    down-bumped forward is ``F * (1 - forward_rel)``: at one the forward reaches zero, which is not
    a market, and past it goes negative and ``ln(K / F)`` stops existing. The upper bound is not in
    the sketch this module was written against and is added deliberately -- without it the failure
    is a bare ``ValueError`` from ``math.log`` three calls down, with nothing in the message to say
    that a bump size was the thing misconfigured.
    """

    vol_abs: float
    """Absolute bump of the interpolated volatility: ``0.01`` is one volatility point. Positive.

    Positive and finite, with no upper bound: a bump wider than the volatility itself is legal and
    handled, by the floor in :data:`DOWN_BUMP_FLOOR`. It is a poor bump -- the quotient it produces
    is a secant from the zero-volatility limit rather than a derivative -- but it is a
    configuration choice with a documented consequence, not a construction error, and refusing it
    would make the admissible range of this field depend on a surface the constructor cannot see.
    """

    def __post_init__(self) -> None:
        # Finiteness first and the bad cases joined with `or`: `float("nan") <= 0` is `False`, so
        # a NaN bump written the other way round walks through into every difference quotient and
        # comes back out as a NaN greek that every ordering guard downstream accepts.
        if not math.isfinite(self.forward_rel) or not 0 < self.forward_rel < 1:
            raise ValueError(
                f"The relative forward bump must lie strictly between 0 and 1, got "
                f"{self.forward_rel}"
            )
        if not math.isfinite(self.vol_abs) or self.vol_abs <= 0:
            raise ValueError(
                f"The absolute vol bump must be positive and finite, got {self.vol_abs}"
            )


def _revalue(
    view: SurfaceView,
    position: Position,
    forward: float,
    tenor_years: float,
    discount: float,
) -> float:
    """Price one option at a given forward, taking its volatility from the surface there.

    The single place the sticky-moneyness assumption is implemented, which is why it is one
    function called four times rather than four inlined copies: ``k`` is derived from the forward
    passed in, so a caller that moves the forward automatically moves the moneyness and gets a
    re-interpolated volatility. Bumping the forward while holding the volatility fixed -- the
    sticky-strike convention -- would be the same three lines with ``k`` hoisted out of them, and
    keeping the derivation inside is what makes the choice a property of this function instead of a
    detail a later edit could reverse by accident.

    Returns the price of **one** option; the quantity is applied by the callers, once, so that the
    scaling convention lives in one place too.
    """
    k = math.log(position.strike / forward)
    vol = implied_vol(view, k, tenor_years)
    return price(forward, position.strike, tenor_years, vol, position.kind, discount)


def value(view: SurfaceView, position: Position, discount: float = 1.0) -> float:
    """What the position is worth on this surface: ``quantity`` times the option premium.

    Signed, because the quantity is: a short position is worth a negative number, which is what
    makes the report's total a portfolio value rather than a sum of magnitudes. Zero quantity is
    legal and returns exactly ``0.0``, the honest report on a leg that was flattened intraday.

    The volatility is read at the position's own moneyness against the forward at its own tenor, so
    two positions on the same strike at different expiries are valued off different points of the
    surface even though they share a strike -- which is the entire reason the moneyness axis exists
    and the reason the conversion cannot be done in the portfolio file.

    Args:
        view: The surface, as this context holds it: total variance on a moneyness-by-tenor grid,
            with the forwards and expiries the grid was measured against.
        position: The contract and the signed size held in it.
        discount: ``exp(-r * T)`` from expiry back to today, defaulting to ``1.0``. Undiscounted is
            exact on an inverse crypto book, and is the honest default rather than a hidden rate
            assumption; see the module docstring on why the choice does not affect comparisons.

    Returns:
        The position value, in the currency the forward and the strike are quoted in.

    Raises:
        ExpiredPositionError: If the position expired at or before the instant the surface
            describes. Raised by ``SurfaceView.tenor_of`` and deliberately not caught: there is no
            number here, and a zero would be indistinguishable from a worthless option.
        ValueError: If ``discount`` is not positive and finite, or if the surface produced a
            volatility ``pricing.price`` refuses. Both are construction bugs upstream.
    """
    tenor_years = view.tenor_of(position.expiry)
    forward = view.forward_at(tenor_years)
    return position.quantity * _revalue(view, position, forward, tenor_years, discount)


def position_risk(
    view: SurfaceView,
    position: Position,
    bumps: BumpSpec,
    discount: float = 1.0,
) -> PositionRisk:
    """Value the position and difference it: one report line, five revaluations.

    The base valuation is the sequence :func:`value` performs. The three greeks are then finite
    differences around it, and the way each one is taken is a decision documented at length in the
    module docstring; in summary:

    * **delta**, ``quantity * (V(F + h) - V(F - h)) / (2h)`` with ``h = F * forward_rel``. Central
      rather than one-sided because the leading error term cancels: a central difference is
      ``O(h^2)`` against ``O(h)`` for the same two revaluations, and on the surface here that is
      the difference between agreeing with the analytic delta to 3e-5 and to 3e-3.
    * **gamma**, ``quantity * (V(F + h) - 2 V(F) + V(F - h)) / h^2``, reusing the same two bumped
      prices, so the whole line costs five revaluations rather than seven.
    * **vega**, ``quantity * (V(vol + e) - V(vol - e)) / (vol_up - vol_down)`` with
      ``e = vol_abs``, holding the forward -- and therefore the moneyness -- fixed. It is a
      parallel shift of **one point** of the surface, not of the surface: the volatility at every
      other moneyness and tenor is untouched, so this is not the vega of a parallel surface shift
      and would not be, even if every position in the book were bumped at once.

    **The vega denominator is the span actually used, not ``2e``.** They are the same number
    whenever ``vol - e`` is positive, which is every ordinary case. When it is not, the down-bump
    is floored at :data:`DOWN_BUMP_FLOOR` times the volatility and the two evaluation points stop
    being symmetric -- so dividing by ``2e`` would report a quotient over a span that was never
    taken and understate the vega by the amount that was clipped. Dividing by the real span keeps
    the number a difference quotient of the function it was evaluated on. What it stops being is
    *central*: with the lower point at essentially zero volatility, it is a secant from the
    intrinsic value up to ``vol + e``, one-sided in effect, first-order accurate, and biased by
    the convexity of the price in volatility across the whole span. A very low volatility point is
    the only place this happens, it is visible in the report as a ``vol`` smaller than the
    configured bump, and it is a better answer than the alternatives -- refusing to report the
    line, or letting a non-positive volatility reach the pricer.

    Args:
        view: The surface to value against.
        position: The contract and the signed size held in it.
        bumps: How far to move the forward and the volatility. Configuration, from the composition
            root; see :class:`BumpSpec`.
        discount: ``exp(-r * T)``, defaulting to ``1.0``. It multiplies the value and all three
            greeks identically.

    Returns:
        The valued line: the volatility actually used, the position value, and delta, gamma and
        vega already scaled by ``quantity``.

    Raises:
        ExpiredPositionError: If the position expired at or before the surface's ``ts_snapshot``.
        ValueError: If ``discount`` is not positive and finite, or if any bumped price is not
            computable. Finiteness of the five reported numbers is not re-checked here: it is
            ``PositionRisk``'s own invariant, enforced in its constructor, and duplicating the
            check would create a second place that has to agree about what a usable report line is.
    """
    tenor_years = view.tenor_of(position.expiry)
    forward = view.forward_at(tenor_years)
    k = math.log(position.strike / forward)
    vol = implied_vol(view, k, tenor_years)
    base = price(forward, position.strike, tenor_years, vol, position.kind, discount)

    step = forward * bumps.forward_rel
    # Both revaluations go through `_revalue`, so both re-interpolate the volatility at the
    # moneyness the bumped forward implies. That is the sticky-moneyness assumption, and it is the
    # only line in this function where it could have been made differently.
    up = _revalue(view, position, forward + step, tenor_years, discount)
    down = _revalue(view, position, forward - step, tenor_years, discount)

    vol_up = vol + bumps.vol_abs
    # Ordered by construction: the floor is a fraction of `vol`, which is strictly positive, so
    # `vol_down` is strictly positive and strictly below `vol_up` for any bump size at all.
    vol_down = max(vol - bumps.vol_abs, DOWN_BUMP_FLOOR * vol)
    vega_up = price(forward, position.strike, tenor_years, vol_up, position.kind, discount)
    vega_down = price(forward, position.strike, tenor_years, vol_down, position.kind, discount)

    return PositionRisk(
        position=position,
        vol=vol,
        value=position.quantity * base,
        delta=position.quantity * (up - down) / (2.0 * step),
        gamma=position.quantity * (up - 2.0 * base + down) / (step * step),
        vega=position.quantity * (vega_up - vega_down) / (vol_up - vol_down),
    )
