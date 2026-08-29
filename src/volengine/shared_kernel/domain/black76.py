"""Scalar Black-76, the closed form itself, owned jointly by every context that prices an option.

Black-76 is Black-Scholes rewritten on the **forward** instead of the spot. The underlying of the
formula is a forward contract, which costs nothing to enter, so the drift term disappears and no
interest rate survives anywhere in it except the single ``discount`` factor that carries the
premium back to today::

    C = D * (F * N(d1) - K * N(d2))
    P = D * (K * N(-d2) - F * N(-d1))
    d1 = (log(F / K) + w / 2) / sqrt(w),  d2 = d1 - sqrt(w),  w = vol^2 * T

**Why this is in the shared kernel and the quote models are not.** The Shared Kernel is DDD's one
sanctioned exception to "no context imports another": a small, jointly owned piece that several
contexts depend on, bought at the price that changing it requires every owner's agreement. That
price is only worth paying for something that passes three tests at once, and this module is the
statement of what they are:

1. **It is a fact, not a policy.** It has a right answer independent of who is asking. No
   threshold, no business rule, nothing an operator could reasonably want to tune -- so nothing
   that belongs in the TOML of ADR-012.
2. **It carries no vocabulary.** Primitives in, primitives out. The side of the contract arrives
   as ``is_call: bool`` and never as an enum, precisely so that Market Data's ``OptionKindD``,
   Pricing's ``OptionKindP``, Neural Surface's ``OptionKindN``, Risk's ``OptionKindR`` and the
   published ``OptionKind`` can all keep their own spelling. Sharing the arithmetic must not
   become sharing the model; a shared enum on this boundary would make every context's language
   hostage to every other's, which is the anti-pattern the bounded contexts exist to prevent.
3. **It is frozen by nature.** It changes only if it was *wrong*, and then every caller wants the
   fix. That is the exact opposite of ``QuoteChain`` or ``SnapshotPolicy``, whose whole job is to
   evolve under one owner's judgement.

``QuoteObservation`` and ``QuoteData`` fail the first test outright and are duplicated on purpose
for it. ``d1 = (log(F/K) + w/2)/sqrt(w)`` fails none of the three: mathematics does not fork.

**What the duplication actually cost.** This module was extracted after the same formula had been
written four times -- Pricing, Neural Surface, Risk, with Market Data's inversion still to come in
F2 -- and the danger was never the typing. It was that a numerical fix would land in one copy and
not the others: the ``erfc`` spelling of the normal CDF below is written down under recurring
traps in ``CLAUDE.md`` *because it kept reappearing*, once per copy. Design 6.5 compares two
producers fitted to one market, and both weight their quotes by vega; two implementations of one
vega is exactly how that comparison stops meaning anything, not through a visible bug but through
one of them being improved.

**Pure ``math``, deliberately.** Rule 1 confines the shared kernel to the standard library, which
this satisfies with room to spare, and three independent reasons would have forced the same choice
anyway:

1. Cost. ``scipy.stats.norm.cdf`` measures at roughly 19 us per scalar call in this environment
   against 0.035 us for the ``math.erfc`` form below -- about 550x -- because every call walks the
   generic ``rv_continuous`` machinery. The inversion loop runs per quote per snapshot, so that
   factor lands on the hot path of the whole engine.
2. Testability. The rules in ``*/domain/`` have to be checkable without a GPU, a JIT warm-up or an
   optimiser, which is the point of the import rules that keep them where they are.
3. Reference status. The vectorised JAX Black-76 written in F3 cannot live here -- rule 1 and
   ADR-011 both forbid it, and a differentiable batched implementation is a genuine
   reimplementation rather than a reuse -- so it will be tested *against this module*. That is the
   remaining duplication in the system, and extracting this one turns it from an accidental copy
   into a deliberate one with an oracle. A reference has to be simple and must not drift when a
   dependency changes its numerics in a patch release.

Each context wraps these functions in a module of its own -- ``parametric_pricing/domain/black76``,
``neural_surface/domain/pricing``, ``risk/domain/pricing`` -- which owns its enum, its error
hierarchy, and a docstring saying what a wrong answer would mean *there*. Those docstrings differ
for a real reason: in Pricing a price that underflows to zero is an uninvertible quote silently
dropped from a slice, and in Risk it is a whole wing of a book reported as no risk at all.
"""

from __future__ import annotations

import math

_INV_SQRT_2 = 1.0 / math.sqrt(2.0)
_INV_SQRT_2PI = 1.0 / math.sqrt(2.0 * math.pi)

_MAX_TOTAL_STDEV = 40.0
"""Upper bracket for the inversion, expressed in total standard deviations ``vol * sqrt(T)``.

At ``sqrt(w) = 40`` the call has ``d2 ~ -20`` and ``d1 ~ +20``, so ``N(d2)`` is around 1e-89 and
``N(d1)`` is 1 to well under one ulp: the price has reached its ``vol -> infinity`` asymptote
exactly, in double precision. Any target strictly below that asymptote by even one ulp is
therefore bracketed by ``(0, 40 / sqrt(T))``, which is why the inversion needs no expanding
search -- the bracket is analytic, and a search loop is one more thing that can fail to terminate.
"""

VOL_TOLERANCE = 1e-8
"""Convergence tolerance of the inversion, **in volatility**: 1e-8 is a ten-thousandth of a basis
point of vol, far below any spread a surface is fitted to.

Stated in vol rather than in price on purpose: a price tolerance means something different at
every strike -- a cent is loose at the money and absurd in a wing where the option is worth 1e-87
-- while "the answer is right to 1e-8 of vol" is the same statement everywhere and is directly
comparable to the number the caller actually wanted.

Not a business threshold, so deliberately not TOML config (ADR-012), and its presence here is the
second admission test in the module docstring being applied: cadence, admissibility and acceptance
describe how this engine chooses to treat a market and belong in a file an operator can edit, while
this is a property of the root-finder, and an operator who loosened it would only be choosing to
publish a worse number.
"""

_MAX_ITERATIONS = 100
"""Iteration budget of the inversion, chosen so that reaching it still means *converged*.

Bisection alone halves the bracket every step, so it needs at most ``log2(width / tolerance)``
steps: the widest bracket this module ever builds is about 4e6 (a one-hour tenor), and
``log2(4e6 / 1e-8)`` is under 49. Newton only ever gets there faster. The budget is therefore a
guard against a coding error in the loop, not a give-up point -- which is why the function returns
its estimate after the loop instead of raising, and why the error below has no "did not converge"
sibling.
"""

_MIN_VEGA = 1e-12
"""Below this vega the Newton step is abandoned in favour of bisection.

Deep in the wings vega collapses towards zero -- ``exp(-d1^2 / 2)`` reaches 1e-230 at four standard
deviations of total variance -- and dividing the price residual by it produces steps of 1e140. That
is the textbook Newton divergence for implied volatility, and it is not a corner case here: half of
a crypto chain is far out of the money. Even where the division does not overflow, the quotient
amplifies the rounding error already in the residual by 1e12, so the step is meaningless before it
is dangerous.
"""


class PriceNotInvertibleError(Exception):
    """No volatility reproduces the given price: it lies at or outside the no-arbitrage interval.

    The shared kernel raises its own exception rather than any context's, because it belongs to no
    context and rule 1 confines it to the standard library anyway. Every wrapper catches this and
    re-raises its own ``NoImpliedVolError`` from it, which keeps each context's error hierarchy
    whole -- a caller draping ``except CalibrationError`` around a slice must still catch the one
    genuine market outcome, and must still not catch a construction bug.

    Routine rather than exceptional. A mid from a crossed or stale book falls below intrinsic
    regularly, and a deep in-the-money quote sits close enough to a bound that one tick of noise
    crosses it. The caller drops the quote; it does not repair it.
    """


def norm_cdf(x: float) -> float:
    """Standard normal CDF, computed as ``0.5 * erfc(-x / sqrt(2))``.

    The textbook spelling ``0.5 * (1 + erf(x / sqrt(2)))`` is mathematically identical and
    numerically unusable in the left tail. ``erf`` saturates at ``-1.0`` once its argument passes
    about -6, so the sum ``1 + erf(...)`` cancels catastrophically and then returns **exactly**
    zero: ``N(-9)`` comes out as ``0.0`` instead of 1.128e-19, and ``N(-20)`` has no chance at all.
    ``erfc`` computes that same tail directly, never forms the cancelling sum, and stays accurate
    down to about 1e-300.

    This is not a numerical nicety, and it is the single strongest argument for this module
    existing. A far out-of-the-money option is worth a very small but perfectly real number, and
    with the naive form the whole deep wing of every chain silently disappears -- about half the
    strikes on a crypto book, and precisely the region where the smile carries the most information
    about the tails. Downstream that shows up as two different disasters depending on who called:
    in Pricing a zero price is uninvertible and the quote is dropped, in Risk it is a position
    priced at zero that *stays* at zero under a bump, so the delta, gamma and vega of every wing
    position come back as clean zeros and an unhedged tail is reported as no risk at all.

    Written down under recurring traps in ``CLAUDE.md``, where it earned its place by reappearing
    once per copy of this formula.
    """
    return 0.5 * math.erfc(-x * _INV_SQRT_2)


def norm_pdf(x: float) -> float:
    """Standard normal density. Underflows to ``0.0`` past roughly 38 standard deviations, which is
    the correct answer: the true value there is below 1e-300 and vega is genuinely nil.
    """
    return _INV_SQRT_2PI * math.exp(-0.5 * x * x)


def _require_positive_finite(value: float, what: str) -> None:
    """Reject anything that is not a usable positive number, as a plain ``ValueError``.

    Value validation stays ``ValueError`` throughout this repo: a non-positive strike or a NaN
    tenor is a construction bug upstream, not a market condition anyone catches and recovers from.
    :class:`PriceNotInvertibleError` is reserved for the one genuine market outcome, so that a
    caller dropping uninvertible quotes in a loop cannot accidentally swallow a broken feed with
    the same ``except``.

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


def intrinsic(forward: float, strike: float, is_call: bool, discount: float) -> float:
    """Discounted intrinsic value: the ``vol -> 0`` limit, and the lower no-arbitrage bound."""
    if is_call:
        return discount * max(forward - strike, 0.0)
    return discount * max(strike - forward, 0.0)


def ceiling(forward: float, strike: float, is_call: bool, discount: float) -> float:
    """The ``vol -> infinity`` limit, and the upper no-arbitrage bound.

    A call can never be worth more than the discounted forward -- as vol grows, ``N(d1) -> 1`` and
    ``N(d2) -> 0``, so the option converges to a forward contract with the strike leg worth nothing.
    A put is bounded by the discounted strike for the mirror reason: the most it can pay is the
    whole strike, when the underlying finishes at zero.
    """
    if is_call:
        return discount * forward
    return discount * strike


def price(
    forward: float,
    strike: float,
    tenor_years: float,
    vol: float,
    is_call: bool,
    discount: float = 1.0,
) -> float:
    """Black-76 price of a European option on a forward.

    The premium is a function of the forward, the strike, the total variance ``vol^2 * T`` and a
    discount factor, and of nothing else. In particular there is no rate and no carry: the drift is
    already inside ``forward``, which is what makes this the natural formula for a market whose
    options settle against a future.

    Args:
        forward: Forward price of the underlying for delivery at expiry. Built upstream by Market
            Data, which is where the venue's expiry convention, its underlying reference and its
            carry live (ADR-002); by the time a number reaches this function it is a homogeneous
            forward and there is no way left to ask which exchange it came from.
        strike: Strike of the contract, in the same units as the forward.
        tenor_years: Time to expiry in years, under the market's own day count. Computed before the
            boundary precisely so this function stays convention-free.
        vol: Black-76 implied volatility, annualised, as a decimal (``0.65``, not ``65``).
        is_call: ``True`` for a call, ``False`` for a put. A bool rather than an enum, so that this
            module imposes no vocabulary on the contexts that share it -- see the module docstring.
        discount: ``exp(-r * T)``, the factor from expiry back to today. Defaults to 1, which is
            exact on an inverse crypto book where premium and settlement share a numeraire, and is
            the honest default rather than a hidden rate assumption.

    Returns:
        The premium **per unit**, in the same units as the forward, and non-negative by
        construction. Scaling by a signed position quantity is the caller's business, because a
        signed quantity is a property of a book and not of the contract this function prices.

    Raises:
        ValueError: If any argument is not positive and finite. ``vol`` included: zero volatility
            is a limit rather than a quote, and callers who want the limit want the intrinsic
            value, which is a subtraction rather than a model.
    """
    _require_contract(forward, strike, tenor_years, discount)
    _require_positive_finite(vol, "volatility")

    total_stdev = vol * math.sqrt(tenor_years)
    if total_stdev == 0.0:
        # Only reachable when vol and tenor are both small enough that their product underflows
        # (around 1e-300 each), which the guards above cannot catch because both are individually
        # positive and finite. The mathematically correct answer is the vol -> 0 limit, and
        # returning it here keeps a ZeroDivisionError from escaping as a bare builtin from three
        # lines below -- where no caller's `except <Context>Error` would ever see it.
        return intrinsic(forward, strike, is_call, discount)

    d1 = (math.log(forward / strike) + 0.5 * total_stdev * total_stdev) / total_stdev
    d2 = d1 - total_stdev

    if is_call:
        return discount * (forward * norm_cdf(d1) - strike * norm_cdf(d2))
    return discount * (strike * norm_cdf(-d2) - forward * norm_cdf(-d1))


def vega(
    forward: float,
    strike: float,
    tenor_years: float,
    vol: float,
    discount: float = 1.0,
) -> float:
    """Sensitivity of the premium to volatility, ``dPrice / dVol``.

    **No side argument, and that is a statement rather than an omission.** Put-call parity says
    ``C - P = D * (F - K)``, whose right-hand side has no volatility in it at all, so
    differentiating gives ``dC/dVol == dP/dVol`` exactly: a call and a put on the same strike have
    the identical vega. Accepting a side here would invite a caller to believe it mattered, and
    would leave a parameter no code path reads -- which this repo deletes rather than documents.

    ``vega = D * F * phi(d1) * sqrt(T)``, equivalently ``D * K * phi(d2) * sqrt(T)``; the two forms
    agree identically, which is itself a useful check on ``d1`` and ``d2``.

    It has two jobs downstream. It is the derivative the Newton iteration in :func:`implied_vol`
    steps along, and it is the natural weight for a calibration loss: fitting in price space
    without dividing by vega would let a fat at-the-money premium dominate a whole slice, because
    an error of one vol point is worth hundreds of currency units there and pennies in the wings.
    That second job is why this function is shared rather than copied. Two producers are compared
    on one market in Design 6.5, and the comparison holds the weights constant; two implementations
    of this formula is exactly how that stops being true.

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
    return discount * forward * norm_pdf(d1) * root_t


def implied_vol(
    target_price: float,
    forward: float,
    strike: float,
    tenor_years: float,
    is_call: bool,
    discount: float = 1.0,
) -> float:
    """Invert Black-76: find the volatility that reproduces ``target_price``.

    This is the function both calibrations rest on. Every mid in a chain has to become a volatility
    before it can be fitted, and it has to become *our* volatility -- inverted from our forward
    with our tenor -- rather than the venue's published IV (Design 4.5).

    **Why the answer is unique.** Black-76 is strictly increasing in volatility: vega is
    ``D * F * phi(d1) * sqrt(T)``, which is positive for every finite ``d1``. The price therefore
    sweeps the open interval from the discounted intrinsic value at ``vol -> 0`` up to ``D * F``
    (call) or ``D * K`` (put) at ``vol -> infinity``, hitting every value in between exactly once.
    A target inside that interval has one root; a target outside it has none at all, and that is
    checked first, before any iteration, because no amount of solver effort can conjure a root that
    does not exist.

    **Why Newton alone is not enough.** Newton converges quadratically near the root and diverges
    spectacularly away from it: in the wings vega is 1e-230, so the first step is 1e140 and the
    iteration never comes back. The loop therefore carries a bracket ``(lo, hi)`` that is tightened
    from the sign of every residual it computes, and any Newton proposal that leaves the bracket --
    or that is built on a vega below ``_MIN_VEGA`` -- is replaced by the bracket's midpoint.
    Bisection on a bracketed monotone function halves the interval every iteration unconditionally,
    so termination is a property of the construction rather than a hope, which is exactly why
    :class:`PriceNotInvertibleError` has no "did not converge" case for the caller to handle.

    Args:
        target_price: Observed premium to invert, typically a mid. Same units as the forward.
        forward: Forward price for delivery at expiry.
        strike: Strike of the contract.
        tenor_years: Time to expiry in years.
        is_call: ``True`` for a call, ``False`` for a put.
        discount: ``exp(-r * T)``; must be the same factor the target was quoted under.

    Returns:
        The volatility that reproduces ``target_price`` to within :data:`VOL_TOLERANCE`.

    Raises:
        ValueError: If ``forward``, ``strike``, ``tenor_years`` or ``discount`` is not positive and
            finite, or if ``target_price`` is not finite. A NaN premium is not a price no
            volatility reproduces, it is the absence of a price: reporting it as
            :class:`PriceNotInvertibleError` would let a caller's quote-dropping loop swallow a
            broken feed one quote at a time and never notice the feed was broken.
        PriceNotInvertibleError: If ``target_price`` lies at or outside the open interval between
            the discounted intrinsic value and the no-arbitrage ceiling.
    """
    _require_contract(forward, strike, tenor_years, discount)
    if not math.isfinite(target_price):
        raise ValueError(f"The target price must be finite, got {target_price}")

    side = "CALL" if is_call else "PUT"
    floor = intrinsic(forward, strike, is_call, discount)
    cap = ceiling(forward, strike, is_call, discount)
    if target_price <= floor:
        raise PriceNotInvertibleError(
            f"A {side} at strike {strike} priced at {target_price} is at or below its "
            f"intrinsic value of {floor}: no volatility reproduces it"
        )
    if target_price >= cap:
        raise PriceNotInvertibleError(
            f"A {side} at strike {strike} priced at {target_price} is at or above its "
            f"no-arbitrage ceiling of {cap}: no volatility reproduces it"
        )

    lo = 0.0
    hi = _MAX_TOTAL_STDEV / math.sqrt(tenor_years)

    # Brenner-Subrahmanyam: at the money the price is about 0.4 * F * vol * sqrt(T), which inverts
    # to this. It is only a seed -- exact at the money, badly wrong in the wings -- and a bad seed
    # costs iterations, never correctness, because the bracket rules every step.
    guess = math.sqrt(2.0 * math.pi / tenor_years) * (target_price / discount) / forward
    if not math.isfinite(guess) or guess <= lo or guess >= hi:
        guess = 0.5 * (lo + hi)
    vol = guess

    for _ in range(_MAX_ITERATIONS):
        residual = price(forward, strike, tenor_years, vol, is_call, discount) - target_price
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

    # Reached only by the `hi - lo` break, or by exhausting a budget sized well above the bisection
    # worst case. Either way the bracket is tighter than the tolerance and the midpoint is the best
    # available estimate -- there is no failure to report here.
    return 0.5 * (lo + hi)
