"""Scalar Black-76 for valuation: the one place in this context that turns a volatility into money.

Risk is handed a surface and a book and asked what the book is worth. The surface answers in
volatility -- that is what a calibration produces and what an interpolation of the grid recovers
(Design 7.1) -- and a report is denominated in currency, so somewhere between the two there has to
be a pricing formula. This is it, and it is the whole of it::

    C = D * (F * N(d1) - K * N(d2))
    P = D * (K * N(-d2) - F * N(-d1))
    d1 = (log(F / K) + w / 2) / sqrt(w),  d2 = d1 - sqrt(w),  w = vol^2 * T

Black-76 is Black-Scholes written on the **forward** rather than the spot. A forward contract costs
nothing to enter, so the drift disappears and no interest rate survives anywhere except the single
``discount`` factor that carries the premium back to today. Every convention that made the forward
what it is -- the venue's expiry rule, its underlying reference, its carry (ADR-002) -- was resolved
upstream, so by the time a number arrives here it is a homogeneous forward, tenor, strike and vol,
and there is no way left to ask which exchange produced it.

**A twin of ``parametric_pricing/domain/black76.py``, not a copy of it, and not debt.** No context
imports another context (rule 6), and the domain does not import the contracts at all (rule 3), so
the alternative to writing these three lines twice is a shared pricing library on the boundary --
the canonical-model anti-pattern that bounded contexts exist to prevent. What makes the duplication
right here rather than merely permitted is that the two modules are not the same function used
twice. They point in opposite directions:

* Pricing runs **price to volatility**. Its subject is inversion, and the file is mostly the
  machinery inversion needs: no-arbitrage bounds, an analytic bracket, a vega to step Newton along,
  a bisection safeguard for the wings where Newton diverges. All of that exists to serve a
  calibration, and none of it has a caller in Risk.
* Risk runs **volatility to price**. It never inverts anything: the vol is given to it by the
  surface, and asking Risk to recover a vol from a premium would mean asking it to re-derive the
  input it was handed. So this module carries the formula and nothing else -- no ``implied_vol``, no
  ``_intrinsic`` bound, no solver, no tolerance constant.

The shared part is therefore the definition of Black-76, which is a page of a textbook and does not
change; the unshared part is the majority of both files. Coupling the two through an import would
tie the valuation of a book to the lifecycle of a calibrator: when Pricing vectorises its Black-76
in JAX for the fit (F3), Risk would either follow it into a dependency rule 3 bars or become the
reason the calibrator cannot move. There is a smaller benefit too. Two independent spellings of the
same formula that agree to the last bit are a real cross-check, and the producer-versus-producer
comparison of Design 7.3 is only meaningful while the valuation side is not literally the code that
did the fitting.

**No vega here, and no greeks at all.** Risk's greeks are finite differences over the interpolated
grid, computed in ``valuation.py`` (Design 7.4): a bump goes through the surface, the vol is
re-interpolated, and the option is repriced. An analytic vega in this module would be a second and
quietly different number -- it would hold the vol fixed where the bump does not, and it would miss
the interpolation's kinks at the nodes, which are exactly what Design 7.4 wants measured against
Pricing's AD greeks. A parameter no code path reads gets deleted rather than documented, and the
same goes for a function.

**Pure ``math``, deliberately.** numpy is permitted in ``*/domain/`` and would buy nothing: a report
prices a handful of positions and revalues each one five times, so the work is a few dozen
multiplications, and an array-based version would add the truth-value-of-an-array trap for no
measurable gain. scipy, jax and torch are barred outright by rule 3, which is what keeps this layer
testable with no GPU, no JIT warm-up and no optimiser anywhere in sight.
"""

from __future__ import annotations

import math
from enum import StrEnum

_INV_SQRT_2 = 1.0 / math.sqrt(2.0)


class OptionKindR(StrEnum):
    """Side of the option contract, in the Risk context's own vocabulary.

    A third spelling of the same idea, next to Pricing's ``OptionKindP`` and the published
    ``OptionKind`` in ``contracts/``. Rule 6 forbids importing the first and rule 3 the second, and
    the duplication is the architecture doing its job rather than debt: a single canonical enum on
    the boundary would make every context's vocabulary hostage to every other's, so that renaming a
    member for Market Data's benefit would edit the language Risk reports in. The ``R`` suffix keeps
    the spellings legible side by side inside ``application/acl.py``, the only module allowed to see
    both.

    The three enums happen to carry the same two members today, and that is not a promise. This one
    describes a **position** somebody holds; Pricing's describes a quote being inverted. If Risk
    ever books a future or a perpetual as a position, this is where that member appears, and it has
    no business turning up in a calibrator's vocabulary.

    Explicit string values, never ``auto()``: house rule, because the value is what a human reads in
    a risk report and must not change silently when a member is renamed.
    """

    CALL = "CALL"
    PUT = "PUT"


def _norm_cdf(x: float) -> float:
    """Standard normal CDF, computed as ``0.5 * erfc(-x / sqrt(2))``.

    The textbook spelling ``0.5 * (1 + erf(x / sqrt(2)))`` is mathematically identical and
    numerically unusable in the left tail. ``erf`` saturates at ``-1.0`` once its argument passes
    about -6, so the sum ``1 + erf(...)`` cancels catastrophically and then returns **exactly**
    zero: ``N(-9)`` comes out as ``0.0`` instead of 1.128e-19, and ``N(-20)`` has no chance at all.
    ``erfc`` computes that same tail directly, never forms the cancelling sum, and stays accurate
    down to about 1e-300.

    In this context the consequence is a valuation that is silently wrong rather than a quote that
    is silently dropped. A far out-of-the-money option is worth a very small but perfectly real
    number, and with the naive form the whole deep wing of a book -- about half the strikes on a
    crypto chain -- would price at exactly ``0.0``. Worse, it would price at zero *and stay there
    under a bump*, so the delta, gamma and vega of every wing position would come back as clean
    zeros: an unhedged tail reported as no risk at all, which is the failure mode a risk report
    exists to prevent. Nothing downstream could tell that from a genuinely flat position.

    This is written down under recurring traps in ``CLAUDE.md``, and ``test_pricing.py`` ships the
    test that fails on the naive spelling.
    """
    return 0.5 * math.erfc(-x * _INV_SQRT_2)


def _require_positive_finite(value: float, what: str) -> None:
    """Reject anything that is not a usable positive number, as a plain ``ValueError``.

    Value validation stays ``ValueError`` throughout this repo. A NaN tenor or a negative strike
    reaching this function is a bug in whatever built the position or read the grid, not a market
    condition anyone catches and recovers from -- and the ``RiskError`` hierarchy is reserved for
    the second kind, so that a caller draping ``except RiskError`` around a report cannot swallow a
    broken portfolio file one position at a time.

    The condition joins the *bad* cases with ``or`` and tests ``isfinite`` first, because
    ``float("nan") <= 0`` is ``False``: a NaN walks straight through any ordering guard on its own,
    and a NaN premium would then propagate through the sum in ``RiskReport.total_value`` and poison
    a whole book's total. Writing this as an ``and`` of the good cases is the mistake that keeps
    coming back in this codebase.
    """
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"The {what} must be positive and finite, got {value}")


def price(
    forward: float,
    strike: float,
    tenor_years: float,
    vol: float,
    kind: OptionKindR,
    discount: float = 1.0,
) -> float:
    """Black-76 price of a European option on a forward.

    The premium is a function of the forward, the strike, the total variance ``vol^2 * T`` and a
    discount factor, and of nothing else. There is no rate and no carry in the signature because the
    drift is already inside ``forward``, which is what makes this the natural formula for options
    that settle against a future.

    Args:
        forward: Forward price of the underlying for delivery at expiry, read off the surface's own
            forward curve by ``SurfaceView.forward_at``.
        strike: Strike of the position being valued, in the same units as the forward.
        tenor_years: Time to expiry in years, recovered from the grid by ``SurfaceView.tenor_of``.
            Convention-free by then: the grid states the daycount at its own nodes (ADR-002), so
            this function never learns which one produced the number.
        vol: Black-76 implied volatility, annualised, as a decimal (``0.65``, not ``65``), as
            interpolated from the surface by ``interpolation.implied_vol``.
        kind: Call or put.
        discount: ``exp(-r * T)``, the factor from expiry back to today. Defaults to 1, and the
            default is the honest one rather than a hidden rate assumption: no producer in this
            engine computes a discount factor today, it is exact on an inverse crypto book where
            premium and settlement share a numeraire, and being a plain multiplicative constant it
            scales every value and every greek identically -- so leaving it at 1 costs the
            producer-to-producer comparison of Design 7.3 nothing at all.

    Returns:
        The premium **per unit**, in the same units as the forward, and non-negative by
        construction. Scaling by ``Position.quantity`` -- including its sign, since a short position
        is worth a negative number -- happens in ``valuation.py``, because a signed quantity is a
        property of the book and not of the contract this function prices.

    Raises:
        ValueError: If any argument is not positive and finite. ``vol`` included: zero volatility is
            a limit rather than a quote, and this function is fed a vol interpolated off a grid
            whose every node is strictly positive, so a non-positive one arriving here means the
            interpolation is broken and the right response is to say so loudly rather than to return
            an intrinsic value that looks like a price.
    """
    _require_positive_finite(forward, "forward")
    _require_positive_finite(strike, "strike")
    _require_positive_finite(tenor_years, "tenor in years")
    _require_positive_finite(vol, "volatility")
    _require_positive_finite(discount, "discount factor")

    total_stdev = vol * math.sqrt(tenor_years)
    if total_stdev == 0.0:
        # Only reachable when vol and tenor are both small enough that their product underflows
        # (around 1e-300 each), which the guards above cannot catch because both are individually
        # positive and finite. The mathematically correct answer is the vol -> 0 limit, the
        # discounted intrinsic value, and returning it keeps a bare ZeroDivisionError from escaping
        # this context two lines below as a builtin nobody's `except RiskError` would see.
        if kind is OptionKindR.CALL:
            return discount * max(forward - strike, 0.0)
        return discount * max(strike - forward, 0.0)

    d1 = (math.log(forward / strike) + 0.5 * total_stdev * total_stdev) / total_stdev
    d2 = d1 - total_stdev

    if kind is OptionKindR.CALL:
        return discount * (forward * _norm_cdf(d1) - strike * _norm_cdf(d2))
    return discount * (strike * _norm_cdf(-d2) - forward * _norm_cdf(-d1))
