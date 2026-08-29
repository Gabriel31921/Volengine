"""Black-76 in this context's own vocabulary: the inversion the training samples are built from.

This module used to be a near-literal 384-line copy of ``parametric_pricing/domain/black76.py``,
and its own docstring called it "the fourth copy, and the one that hurts". The mathematics now
lives once, in ``shared_kernel/domain/black76.py``: the Shared Kernel is DDD's sanctioned exception
to "no context imports another", reserved for pieces that are a *fact* rather than a policy, carry
no vocabulary, and change only if they were wrong -- three tests the closed form passes and a quote
model fails. What stays here is what is genuinely this context's: :class:`OptionKindN`, the
:class:`~volengine.neural_surface.domain.errors.NoImpliedVolError` that keeps this context's error
hierarchy whole, and the reason the inversion is done at all.

**The network is trained on our own implied volatilities, never the venue's.**
``TrainingSample.implied_vol`` is inverted from the mid with our forward and our tenor, and the
rule is stricter here than one context over: a network trained on an exchange's published IV would
be learning that exchange's model, its forward and its expiry convention -- and being a universal
approximator it would learn the mismatch *perfectly* and never show it in a residual. Five SVI
parameters cannot absorb an arbitrary distortion; an MLP can.

**What the old duplication endangered, and why sharing is the real guard.**
``TrainingSample.weight`` and ``SliceTask.weights`` are both built from vega and the relative
spread, and Design 6.5 compares the two producers on surfaces fitted to the same market -- a
comparison that means nothing if the two engines silently weight that market differently. Two
implementations of one formula is exactly how that happens: not through a visible bug, but through
one of them being improved. A cross-context test in ``tests/`` used to be the only place the copies
could be required to agree; with one implementation the agreement is structural, and that test now
guards what is still duplicated, which is the ACL's out-of-the-money rule and its spread discount
-- prose rather than arithmetic, and the easier kind to let drift.
"""

from __future__ import annotations

from enum import StrEnum

from volengine.neural_surface.domain.errors import NoImpliedVolError
from volengine.shared_kernel.domain import black76 as kernel
from volengine.shared_kernel.domain.black76 import VOL_TOLERANCE, PriceNotInvertibleError

__all__ = ["VOL_TOLERANCE", "OptionKindN", "implied_vol", "price", "vega"]


class OptionKindN(StrEnum):
    """Side of the option contract, in the Neural Surface context's own vocabulary.

    One spelling of the same idea among several, beside Market Data's ``OptionKindD``, Parametric
    Pricing's ``OptionKindP``, Risk's ``OptionKindR`` and the published ``OptionKind`` in
    ``contracts/``. The duplication is the architecture working rather than debt: no context
    imports another context (rule 6), and the domain does not import the contracts at all (rule 3).
    Unifying them into one canonical enum would put a shared type on the boundary and make every
    context's vocabulary hostage to every other's -- the exact anti-pattern the bounded contexts
    exist to prevent.

    **This is precisely what the shared kernel does not take.** The functions below hand it an
    ``is_call: bool`` and translate on the way in, so that the arithmetic can be shared without the
    language being shared. Formula in the kernel, vocabulary in the context, one line of mapping
    between them.

    The ``N`` suffix keeps the spellings legible side by side inside ``application/acl.py``, which
    is the only module allowed to see both. Explicit string values, never ``auto()``: house rule,
    because a value that is written anywhere a human reads it must not change silently when a
    member is renamed.
    """

    CALL = "CALL"
    PUT = "PUT"


def price(
    forward: float,
    strike: float,
    tenor_years: float,
    vol: float,
    kind: OptionKindN,
    discount: float = 1.0,
) -> float:
    """Black-76 price of a European option on a forward. See the shared kernel for the formula.

    Args:
        forward: Forward price of the underlying for delivery at expiry, from Market Data.
        strike: Strike of the contract, in the same units as the forward.
        tenor_years: Time to expiry in years, under the market's own day count. Computed before the
            boundary precisely so this function stays convention-free.
        vol: Black-76 implied volatility, annualised, as a decimal (``0.65``, not ``65``).
        kind: Call or put, in this context's vocabulary.
        discount: ``exp(-r * T)``, the factor from expiry back to today. Defaults to 1, which is
            exact on an inverse crypto book where premium and settlement share a numeraire, and is
            the honest default rather than a hidden rate assumption.

    Returns:
        The premium, in the same units as the forward. Non-negative by construction.

    Raises:
        ValueError: If any argument is not positive and finite. Note that ``vol`` must be strictly
            positive: zero volatility is a limit, not a quote, and callers who want the limit want
            the intrinsic value, which is a subtraction rather than a model.
    """
    return kernel.price(forward, strike, tenor_years, vol, kind is OptionKindN.CALL, discount)


def vega(
    forward: float,
    strike: float,
    tenor_years: float,
    vol: float,
    discount: float = 1.0,
) -> float:
    """Sensitivity of the premium to volatility, ``dPrice / dVol``.

    **No ``kind`` argument, and that is a statement rather than an omission.** Put-call parity says
    ``C - P = D * (F - K)``, whose right-hand side has no volatility in it, so a call and a put on
    the same strike have the identical vega. The kernel's own signature omits the side for the same
    reason, so there is nothing to translate here.

    In this context it is the weight of a training sample: one gradient step spans every point of a
    snapshot, and without vega the fat at-the-money premiums would own the step. The parametric
    producer weights the same market with the same formula, out of the same module, which is what
    makes Design 6.5's comparison of the two mean anything.

    Raises:
        ValueError: If any argument is not positive and finite.
    """
    return kernel.vega(forward, strike, tenor_years, vol, discount)


def implied_vol(
    target_price: float,
    forward: float,
    strike: float,
    tenor_years: float,
    kind: OptionKindN,
    discount: float = 1.0,
) -> float:
    """Invert Black-76: find the volatility that reproduces ``target_price``.

    Every mid destined for a :class:`~volengine.neural_surface.domain.training_batch.TrainingSample`
    passes through here, because the sample stores an implied volatility and derives the total
    variance the network actually outputs. The kernel explains why the root is unique and why the
    bracketed Newton always terminates.

    Args:
        target_price: Observed premium to invert, typically a mid. Same units as the forward.
        forward: Forward price for delivery at expiry.
        strike: Strike of the contract.
        tenor_years: Time to expiry in years.
        kind: Call or put, in this context's vocabulary.
        discount: ``exp(-r * T)``; must be the same factor the target was quoted under.

    Returns:
        The volatility that reproduces ``target_price`` to within ``VOL_TOLERANCE``.

    Raises:
        ValueError: If ``forward``, ``strike``, ``tenor_years`` or ``discount`` is not positive and
            finite, or if ``target_price`` is not finite. A NaN premium is not a price no
            volatility reproduces, it is the absence of a price: reporting it as
            ``NoImpliedVolError`` would let a caller's quote-dropping loop swallow a broken feed
            one quote at a time and never notice the feed was broken.
        NoImpliedVolError: If ``target_price`` lies at or outside the open interval between the
            discounted intrinsic value and the no-arbitrage ceiling. Routine rather than
            exceptional -- a mid from a crossed or stale book falls below intrinsic regularly, and
            a deep in-the-money quote sits close enough to a bound that one tick of noise crosses
            it. The caller drops the quote from the batch; it does not repair it.

            Translated from the kernel's ``PriceNotInvertibleError`` rather than allowed to escape
            as it is, because a caller draping ``except NeuralSurfaceError`` around a batch must
            still catch the one genuine market outcome. An exception from the shared kernel belongs
            to no context and would sail straight through that guard.
    """
    try:
        return kernel.implied_vol(
            target_price, forward, strike, tenor_years, kind is OptionKindN.CALL, discount
        )
    except PriceNotInvertibleError as exc:
        raise NoImpliedVolError(str(exc)) from exc
