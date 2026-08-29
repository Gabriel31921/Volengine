"""Black-76 in this context's own vocabulary: the one place here that turns a volatility into a
price, and a price back into a volatility.

The mathematics is not written here. It lives in ``shared_kernel/domain/black76.py``, which is the
Shared Kernel -- DDD's sanctioned exception to "no context imports another", reserved for pieces
that are a *fact* rather than a policy, carry no vocabulary, and change only if they were wrong.
The closed form passes all three; a quote model passes none of them, which is why
``QuoteObservation`` and ``QuoteData`` stay duplicated and this does not. What remains in this
module is everything that *is* this context's: :class:`OptionKindP`, the
:class:`~volengine.parametric_pricing.domain.errors.NoImpliedVolError` that keeps the calibration's
error hierarchy whole, and the reasons a wrong answer would matter here.

That reason is the calibration itself. **SVI is fitted to volatilities this context inverts from
the mid** (Design 4.5). An exchange's published IV is the output of the exchange's own model, its
own forward and its own expiry convention; fitting SVI to it would be fitting our model to someone
else's model rather than to the market, and every convention mismatch would enter the surface
disguised as a smile feature. The venue's IV is ingested as a comparison column, never as an input.

The consequence of a numerical slip is therefore specific here: a price that underflows to zero is
*uninvertible*, so :func:`implied_vol` rejects it as at-or-below intrinsic and the ACL drops the
quote from its slice. The whole deep wing of a chain would disappear without a single error being
raised. The shared kernel's ``norm_cdf`` docstring is where that is defended.
"""

from __future__ import annotations

from enum import StrEnum

from volengine.parametric_pricing.domain.errors import NoImpliedVolError
from volengine.shared_kernel.domain import black76 as kernel
from volengine.shared_kernel.domain.black76 import VOL_TOLERANCE, PriceNotInvertibleError

__all__ = ["VOL_TOLERANCE", "OptionKindP", "implied_vol", "price", "vega"]


class OptionKindP(StrEnum):
    """Side of the option contract, in the Parametric Pricing context's own vocabulary.

    One spelling of the same idea among several, next to Market Data's ``OptionKindD``, Neural
    Surface's ``OptionKindN``, Risk's ``OptionKindR`` and the published ``OptionKind`` in
    ``contracts/``, and the duplication is the architecture working rather than debt: no context
    imports another context (rule 6), and the domain does not import the contracts at all (rule 3).
    Unifying them into one canonical enum would put a shared type on the boundary and make every
    context's vocabulary hostage to every other's -- the exact anti-pattern the bounded contexts
    exist to prevent.

    **This is precisely what the shared kernel does not take.** The functions below hand it an
    ``is_call: bool`` and translate on the way in, so that the arithmetic can be shared without the
    language being shared. Formula in the kernel, vocabulary in the context, one line of mapping
    between them: that boundary is the whole design, and moving this enum down would erase it.

    The ``P`` suffix keeps the spellings legible side by side inside ``application/acl.py``, which
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
    kind: OptionKindP,
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
    return kernel.price(forward, strike, tenor_years, vol, kind is OptionKindP.CALL, discount)


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

    In this context it is the weight of the calibration loss: fitting in price space without
    dividing by vega would let a fat at-the-money premium dominate a whole slice, because an error
    of one vol point is worth hundreds of currency units there and pennies in the wings. The
    neural producer weights the same market with the same formula, out of the same module, which is
    what makes Design 6.5's comparison of the two mean anything.

    Raises:
        ValueError: If any argument is not positive and finite.
    """
    return kernel.vega(forward, strike, tenor_years, vol, discount)


def implied_vol(
    target_price: float,
    forward: float,
    strike: float,
    tenor_years: float,
    kind: OptionKindP,
    discount: float = 1.0,
) -> float:
    """Invert Black-76: find the volatility that reproduces ``target_price``.

    This is the function the whole calibration rests on -- SVI is fitted in total-variance space,
    so every mid in the chain has to become one of *our* volatilities first. The kernel explains
    why the root is unique and why the bracketed Newton always terminates.

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
            it. The caller drops the quote from the slice; it does not repair it.

            Translated from the kernel's ``PriceNotInvertibleError`` rather than allowed to escape
            as it is, because a caller draping ``except CalibrationError`` around a slice must
            still catch the one genuine market outcome. An exception from the shared kernel belongs
            to no context and would sail straight through that guard.
    """
    try:
        return kernel.implied_vol(
            target_price, forward, strike, tenor_years, kind is OptionKindP.CALL, discount
        )
    except PriceNotInvertibleError as exc:
        raise NoImpliedVolError(str(exc)) from exc
