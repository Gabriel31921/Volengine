"""Scalar Black-76 for the Neural Surface context: the fourth copy, and the one that hurts.

**This is a deliberate duplicate of ``parametric_pricing/domain/black76.py``, line for line where
the mathematics is concerned.** No context imports another context (rule 6), and this one has to
invert a mid: ``TrainingSample.implied_vol`` is explicitly *our own* volatility rather than the
venue's, because a network trained on an exchange's published IV would learn that exchange's
model, its forward and its expiry convention -- and being a universal approximator it would learn
the mismatch perfectly and never show it in a residual. A five-parameter SVI slice cannot absorb
an arbitrary distortion; a network can, which makes the rule stricter here than one context over.

There are now four Black-76s in this repo and they are not four copies of one thing. Pricing's is
the full model with the inversion. Risk's carries only ``price``, because Risk is handed a
volatility by the surface and never asked for one. Market Data will grow a partial one in F2 for
``IV_DIVERGENCE``. This one is the full model again, because the inversion is exactly what it is
here for, and it is the closest of the four to a straight copy.

**What the duplication endangers, and how that is guarded.**
``TrainingSample.weight`` and ``SliceTask.weights`` are built from vega and the relative spread,
and Design 6.5 compares the two producers on surfaces fitted to the same market -- a comparison
that means nothing if the two engines silently weight that market differently. Two implementations
of one formula is exactly how that happens: not through a visible bug, but through one of them
being improved. The guard is a cross-context test in ``tests/`` -- which is subject to none of the
import rules, and is therefore the one place the two implementations can be put side by side and
required to agree. If this module and ``black76`` ever diverge numerically, that test fails.

Everything else about it is its parametric twin's docstrings, and they are reproduced rather than
summarised on purpose: a reader here must not have to open another context to learn why the normal
CDF is written with ``erfc``, or why the Newton step is abandoned below a vega floor.
"""

from __future__ import annotations

import math
from enum import StrEnum

from volengine.neural_surface.domain.errors import NoImpliedVolError

_INV_SQRT_2 = 1.0 / math.sqrt(2.0)
_INV_SQRT_2PI = 1.0 / math.sqrt(2.0 * math.pi)

_MAX_TOTAL_STDEV = 40.0
"""Upper bracket for the inversion, expressed in total standard deviations ``vol * sqrt(T)``.

At ``sqrt(w) = 40`` the call has ``d2 ~ -20`` and ``d1 ~ +20``, so ``N(d2)`` is around 1e-89 and
``N(d1)`` is 1 to well under one ulp: the price has reached its ``vol -> infinity`` asymptote
exactly, in double precision. Any target strictly below that asymptote by even one ulp is
therefore bracketed by ``(0, 40 / sqrt(T))``, which is why the inversion needs no expanding
search -- the bracket is analytic, and a search loop is one more thing that can fail to
terminate.
"""

VOL_TOLERANCE = 1e-8
"""Convergence tolerance of the inversion, **in volatility**: 1e-8 is a ten-thousandth of a
basis point of vol, far below any spread the surface is fitted to.

Stated in vol rather than in price on purpose: a price tolerance means something different at
every strike -- a cent is loose at the money and absurd in a wing where the option is worth
1e-87 -- while "the answer is right to 1e-8 of vol" is the same statement everywhere and is
directly comparable to the number the caller actually wanted.

Not a business threshold, so deliberately not TOML config (ADR-012). Cadence, admissibility and
acceptance describe how this engine chooses to treat a market and belong in a file an operator
can edit; this is a property of the root-finder, and an operator who loosened it would only be
choosing to publish a worse number.
"""

_MAX_ITERATIONS = 100
"""Iteration budget of the inversion, chosen so that reaching it still means *converged*.

Bisection alone halves the bracket every step, so it needs at most ``log2(width / tolerance)``
steps: the widest bracket this module ever builds is about 4e6 (a one-hour tenor), and
``log2(4e6 / 1e-8)`` is under 49. Newton only ever gets there faster. The budget is therefore a
guard against a coding error in the loop, not a give-up point -- which is why the function
returns its estimate after the loop instead of raising, and why ``NoImpliedVolError`` has no
"did not converge" sibling.
"""

_MIN_VEGA = 1e-12
"""Below this vega the Newton step is abandoned in favour of bisection.

Deep in the wings vega collapses towards zero -- ``exp(-d1^2 / 2)`` reaches 1e-230 at four
standard deviations of total variance -- and dividing the price residual by it produces steps of
1e140. That is the textbook Newton divergence for implied volatility, and it is not a corner
case here: half of a crypto chain is far out of the money. Even where the division does not
overflow, the quotient amplifies the rounding error already in the residual by 1e12, so the step
is meaningless before it is dangerous.
"""


class OptionKindN(StrEnum):
    """Side of the option contract, in the Neural Surface context's own vocabulary.

    The **fifth** spelling of the same idea, beside Market Data's ``OptionKindD``, Parametric
    Pricing's ``OptionKindP``, Risk's ``OptionKindR`` and the published ``OptionKind`` in
    ``contracts/``. The duplication is the architecture working rather than debt: no context
    imports another context (rule 6), and the domain does not import the contracts at all
    (rule 3). Unifying them into one canonical enum would put a shared type on the boundary and
    make every context's vocabulary hostage to every other's -- the exact anti-pattern the bounded
    contexts exist to prevent. The ``N`` suffix keeps the two spellings legible side by side inside
    ``application/acl.py``, which is the only module allowed to see both.

    Explicit string values, never ``auto()``: house rule, because a value that is written
    anywhere a human reads it must not change silently when a member is renamed.
    """

    CALL = "CALL"
    PUT = "PUT"


def _norm_cdf(x: float) -> float:
    """Standard normal CDF, computed as ``0.5 * erfc(-x / sqrt(2))``.

    The textbook spelling ``0.5 * (1 + erf(x / sqrt(2)))`` is mathematically identical and
    numerically unusable in the left tail. ``erf`` saturates at ``-1.0`` once its argument passes
    about -6, so the sum ``1 + erf(...)`` cancels catastrophically and then returns **exactly**
    zero: ``N(-9)`` comes out as ``0.0`` instead of 1.128e-19, and ``N(-20)`` has no chance at
    all. ``erfc`` computes that same tail directly and never forms the cancelling sum, so it
    stays accurate down to 1e-300.

    This is not a numerical nicety, it decides which quotes the engine can use. A far
    out-of-the-money option is worth a very small but perfectly real number, and a zero price is
    uninvertible -- ``implied_vol`` would reject it as "at or below intrinsic" and the quote
    would be dropped. With ``erf`` the whole deep wing of every chain silently disappears, which
    on a crypto book is about half the strikes and precisely the region where the smile carries
    the most information about the tails.
    """
    return 0.5 * math.erfc(-x * _INV_SQRT_2)


def _norm_pdf(x: float) -> float:
    """Standard normal density. Underflows to ``0.0`` past roughly 38 standard deviations, which
    is the correct answer: the true value there is below 1e-300 and vega is genuinely nil.
    """
    return _INV_SQRT_2PI * math.exp(-0.5 * x * x)


def _require_positive_finite(value: float, what: str) -> None:
    """Reject anything that is not a usable positive number, as a plain ``ValueError``.

    Value validation stays ``ValueError`` throughout this repo: a non-positive strike or a NaN
    tenor is a construction bug upstream, not a market condition anyone catches and recovers
    from. ``NoImpliedVolError`` is reserved for the one genuine market outcome -- a price that no
    volatility reproduces -- so that a caller dropping uninvertible quotes in a loop cannot
    accidentally swallow a broken feed with the same ``except``.

    The condition joins the *bad* cases with ``or`` and tests ``isfinite`` first, because
    ``float("nan") <= 0`` is ``False``: a NaN walks straight through any ordering guard, and
    writing this as ``isfinite(x) and x > 0`` inverted is the mistake that keeps coming back.
    """
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"The {what} must be positive and finite, got {value}")


def _require_contract(forward: float, strike: float, tenor_years: float, discount: float) -> None:
    """The four quantities that describe *which* option is being priced, validated once."""
    _require_positive_finite(forward, "forward")
    _require_positive_finite(strike, "strike")
    _require_positive_finite(tenor_years, "tenor in years")
    _require_positive_finite(discount, "discount factor")


def _intrinsic(forward: float, strike: float, kind: OptionKindN, discount: float) -> float:
    """Discounted intrinsic value: the ``vol -> 0`` limit, and the lower no-arbitrage bound."""
    if kind is OptionKindN.CALL:
        return discount * max(forward - strike, 0.0)
    return discount * max(strike - forward, 0.0)


def _ceiling(forward: float, strike: float, kind: OptionKindN, discount: float) -> float:
    """The ``vol -> infinity`` limit, and the upper no-arbitrage bound.

    A call can never be worth more than the discounted forward -- as vol grows, ``N(d1) -> 1``
    and ``N(d2) -> 0``, so the option converges to a forward contract with the strike leg worth
    nothing. A put is bounded by the discounted strike for the mirror reason: the most it can pay
    is the whole strike, when the underlying finishes at zero.
    """
    if kind is OptionKindN.CALL:
        return discount * forward
    return discount * strike


def price(
    forward: float,
    strike: float,
    tenor_years: float,
    vol: float,
    kind: OptionKindN,
    discount: float = 1.0,
) -> float:
    """Black-76 price of a European option on a forward.

    The premium is a function of the forward, the strike, the total variance ``vol^2 * T`` and a
    discount factor, and of nothing else. In particular there is no rate and no carry: the drift
    is already inside ``forward``, which is what makes this the natural formula for a market
    whose options settle against a future.

    Args:
        forward: Forward price of the underlying for delivery at expiry, from Market Data.
        strike: Strike of the contract, in the same units as the forward.
        tenor_years: Time to expiry in years, under the market's own day count. Computed before
            the boundary precisely so this function stays convention-free.
        vol: Black-76 implied volatility, annualised, as a decimal (``0.65``, not ``65``).
        kind: Call or put.
        discount: ``exp(-r * T)``, the factor from expiry back to today. Defaults to 1, which is
            exact on an inverse crypto book where premium and settlement share a numeraire, and
            is the honest default rather than a hidden rate assumption.

    Returns:
        The premium, in the same units as the forward. Non-negative by construction.

    Raises:
        ValueError: If any argument is not positive and finite. Note that ``vol`` must be
            strictly positive: zero volatility is a limit, not a quote, and callers who want the
            limit want the intrinsic value, which is a subtraction rather than a model.
    """
    _require_contract(forward, strike, tenor_years, discount)
    _require_positive_finite(vol, "volatility")

    total_stdev = vol * math.sqrt(tenor_years)
    if total_stdev == 0.0:
        # Only reachable when vol and tenor are both small enough that their product underflows
        # (around 1e-300 each). The mathematically correct answer is the vol -> 0 limit, and
        # returning it here keeps a ZeroDivisionError from escaping the context as a bare builtin
        # from three lines below.
        return _intrinsic(forward, strike, kind, discount)

    d1 = (math.log(forward / strike) + 0.5 * total_stdev * total_stdev) / total_stdev
    d2 = d1 - total_stdev

    if kind is OptionKindN.CALL:
        return discount * (forward * _norm_cdf(d1) - strike * _norm_cdf(d2))
    return discount * (strike * _norm_cdf(-d2) - forward * _norm_cdf(-d1))


def vega(
    forward: float,
    strike: float,
    tenor_years: float,
    vol: float,
    discount: float = 1.0,
) -> float:
    """Sensitivity of the premium to volatility, ``dPrice / dVol``.

    **No ``kind`` argument, and that is a statement rather than an omission.** Put-call parity
    says ``C - P = D * (F - K)``, whose right-hand side has no volatility in it at all, so
    differentiating gives ``dC/dVol == dP/dVol`` exactly: a call and a put on the same strike
    have the identical vega. Accepting a side here would invite a caller to believe it mattered,
    and would leave a parameter no code path reads -- which this repo deletes rather than
    documents.

    ``vega = D * F * phi(d1) * sqrt(T)``, equivalently ``D * K * phi(d2) * sqrt(T)``; the two
    forms agree identically, which is itself a useful check on ``d1`` and ``d2``.

    It has two jobs downstream. It is the derivative the Newton iteration in ``implied_vol``
    steps along, and it is the natural weight for the calibration loss: fitting in price space
    without dividing by vega would let a fat at-the-money premium dominate a whole slice, because
    an error of one vol point is worth hundreds of currency units there and pennies in the wings.

    Raises:
        ValueError: If any argument is not positive and finite.
    """
    _require_contract(forward, strike, tenor_years, discount)
    _require_positive_finite(vol, "volatility")

    root_t = math.sqrt(tenor_years)
    total_stdev = vol * root_t
    if total_stdev == 0.0:
        # Same underflow corner as in `price`. The vol -> 0 limit of vega is zero: at no
        # volatility, an infinitesimal change in it moves nothing.
        return 0.0

    d1 = (math.log(forward / strike) + 0.5 * total_stdev * total_stdev) / total_stdev
    return discount * forward * _norm_pdf(d1) * root_t


def implied_vol(
    target_price: float,
    forward: float,
    strike: float,
    tenor_years: float,
    kind: OptionKindN,
    discount: float = 1.0,
) -> float:
    """Invert Black-76: find the volatility that reproduces ``target_price``.

    This is the function the whole calibration rests on. SVI is fitted in total-variance space,
    so every mid in the chain has to become a volatility first, and it has to become *our*
    volatility -- inverted from our forward with our tenor -- rather than the venue's published
    IV (Design 4.5).

    **Why the answer is unique.** Black-76 is strictly increasing in volatility: vega is
    ``D * F * phi(d1) * sqrt(T)``, which is positive for every finite ``d1``. The price therefore
    sweeps the open interval from the discounted intrinsic value at ``vol -> 0`` up to
    ``D * F`` (call) or ``D * K`` (put) at ``vol -> infinity``, hitting every value in between
    exactly once. A target inside that interval has one root; a target outside it has none at
    all, and that is checked first, before any iteration, because no amount of solver effort can
    conjure a root that does not exist.

    **Why Newton alone is not enough.** Newton converges quadratically near the root and diverges
    spectacularly away from it: in the wings vega is 1e-230, so the first step is 1e140 and the
    iteration never comes back. The loop therefore carries a bracket ``(lo, hi)`` that is
    tightened from the sign of every residual it computes, and any Newton proposal that leaves
    the bracket -- or that is built on a vega below ``_MIN_VEGA`` -- is replaced by the bracket's
    midpoint. Bisection on a bracketed monotone function halves the interval every iteration
    unconditionally, so termination is a property of the construction rather than a hope, which
    is exactly why ``NoImpliedVolError`` has no "did not converge" case for the caller to handle.

    Args:
        target_price: Observed premium to invert, typically a mid. Same units as the forward.
        forward: Forward price for delivery at expiry.
        strike: Strike of the contract.
        tenor_years: Time to expiry in years.
        kind: Call or put.
        discount: ``exp(-r * T)``; must be the same factor the target was quoted under.

    Returns:
        The volatility that reproduces ``target_price`` to within ``VOL_TOLERANCE``.

    Raises:
        ValueError: If ``forward``, ``strike``, ``tenor_years`` or ``discount`` is not positive
            and finite, or if ``target_price`` is not finite. A NaN premium is not a price no
            volatility reproduces, it is the absence of a price: reporting it as
            ``NoImpliedVolError`` would let a caller's quote-dropping loop swallow a broken feed
            one quote at a time and never notice the feed was broken.
        NoImpliedVolError: If ``target_price`` lies at or outside the open interval between the
            discounted intrinsic value and the no-arbitrage ceiling. Routine rather than
            exceptional -- a mid from a crossed or stale book falls below intrinsic regularly,
            and a deep in-the-money quote sits close enough to a bound that one tick of noise
            crosses it. The caller drops the quote from the slice; it does not repair it.
    """
    _require_contract(forward, strike, tenor_years, discount)
    if not math.isfinite(target_price):
        raise ValueError(f"The target price must be finite, got {target_price}")

    floor = _intrinsic(forward, strike, kind, discount)
    ceiling = _ceiling(forward, strike, kind, discount)
    if target_price <= floor:
        raise NoImpliedVolError(
            f"A {kind.value} at strike {strike} priced at {target_price} is at or below its "
            f"intrinsic value of {floor}: no volatility reproduces it"
        )
    if target_price >= ceiling:
        raise NoImpliedVolError(
            f"A {kind.value} at strike {strike} priced at {target_price} is at or above its "
            f"no-arbitrage ceiling of {ceiling}: no volatility reproduces it"
        )

    lo = 0.0
    hi = _MAX_TOTAL_STDEV / math.sqrt(tenor_years)

    # Brenner-Subrahmanyam: at the money the price is about 0.4 * F * vol * sqrt(T), which
    # inverts to this. It is only a seed -- exact at the money, badly wrong in the wings -- and a
    # bad seed costs iterations, never correctness, because the bracket rules every step.
    guess = math.sqrt(2.0 * math.pi / tenor_years) * (target_price / discount) / forward
    if not math.isfinite(guess) or guess <= lo or guess >= hi:
        guess = 0.5 * (lo + hi)
    vol = guess

    for _ in range(_MAX_ITERATIONS):
        residual = price(forward, strike, tenor_years, vol, kind, discount) - target_price
        # The function is increasing, so the sign of the residual says which side of the root we
        # are on and the bracket can only ever shrink.
        if residual < 0.0:
            lo = vol
        else:
            hi = vol
        if hi - lo < VOL_TOLERANCE:
            break

        slope = vega(forward, strike, tenor_years, vol, discount)
        newton = vol - residual / slope if slope > _MIN_VEGA else math.inf
        # A Newton proposal outside the bracket has left the region where the tangent means
        # anything; the midpoint always lies inside and always halves the remaining interval.
        candidate = newton if lo < newton < hi else 0.5 * (lo + hi)

        if abs(candidate - vol) < VOL_TOLERANCE:
            return candidate
        vol = candidate

    # Reached only by the `hi - lo` break, or by exhausting a budget sized well above the
    # bisection worst case. Either way the bracket is tighter than the tolerance and the midpoint
    # is the best available estimate -- there is no failure to report here.
    return 0.5 * (lo + hi)
