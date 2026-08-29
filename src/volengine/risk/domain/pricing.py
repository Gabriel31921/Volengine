"""Black-76 in this context's own vocabulary: the one place here that turns a volatility into money.

Risk is handed a surface and a book and asked what the book is worth. The surface answers in
volatility -- that is what a calibration produces and what an interpolation of the grid recovers
(Design 7.1) -- and a report is denominated in currency, so somewhere between the two there has to
be a pricing formula. That formula lives once, in ``shared_kernel/domain/black76.py``. What lives
here is what is genuinely this context's: :class:`OptionKindR`, and the half of Black-76 Risk
actually uses.

**Half, and the omissions are the point.** The two directions are not the same function used twice:

* Pricing runs **price to volatility**. Its subject is inversion, and most of its file is the
  machinery inversion needs -- no-arbitrage bounds, an analytic bracket, a vega to step Newton
  along, a bisection safeguard for the wings where Newton diverges. All of that serves a
  calibration, and none of it has a caller in Risk.
* Risk runs **volatility to price**. It never inverts anything: the vol is given to it by the
  surface, and asking Risk to recover a vol from a premium would mean asking it to re-derive the
  input it was handed. So this module exposes :func:`price` and nothing else. The kernel's
  ``implied_vol``, ``intrinsic`` and ``ceiling`` exist and are deliberately not re-exported: a
  context's module is its vocabulary, and offering a solver nobody here calls would invite a future
  caller to invert a mark rather than trust the surface it was handed.

**No vega here, and no greeks at all.** Risk's greeks are finite differences over the interpolated
grid, computed in ``valuation.py`` (Design 7.4): a bump goes through the surface, the vol is
re-interpolated, and the option is repriced. Re-exporting the kernel's analytic vega would put a
second and quietly different number within reach -- it holds the vol fixed where the bump does not,
and it misses the interpolation's kinks at the nodes, which are exactly what Design 7.4 wants
measured against Pricing's AD greeks. A parameter no code path reads gets deleted rather than
documented, and the same goes for a function.

**Why this delegates rather than duplicating, which reverses an earlier decision here.** The
formula used to be written out in this file, on the argument that rule 6 forbids importing
Pricing's and that a shared pricing library on the boundary is the canonical-model anti-pattern.
Half of that still holds and half of it does not. Rule 6 does forbid importing Pricing's -- and
still does; nothing here imports another context. But the shared kernel is not the boundary: it is
DDD's sanctioned exception, admitted only for something that is a *fact* rather than a policy,
carries no vocabulary, and changes only if it was wrong. ``d1 = (log(F/K) + w/2)/sqrt(w)`` passes
all three, and the anti-pattern the old argument feared is avoided concretely, by the kernel taking
``is_call: bool`` instead of an enum: the arithmetic is shared, the language is not, and
:class:`OptionKindR` never leaves this context.

The second half of the old argument -- that sharing would tie the valuation of a book to the
lifecycle of a calibrator, and drag Risk into JAX when Pricing vectorises in F3 -- is answered by
where the shared code sits. Rule 1 confines the shared kernel to the standard library, so the
vectorised Black-76 of F3 *cannot* live there; it is a genuine reimplementation that will be tested
against the kernel rather than replacing it, and this module's dependency does not move an inch.

One thing is genuinely lost, and it is worth naming rather than glossing: two independent spellings
of one formula that agree to the last bit were a real cross-check, and there is now one spelling to
be wrong. The trade was made because the failure that check guards against is far rarer than the
one duplication causes -- a numerical fix landing in one copy and not the others, which is how the
``erfc`` spelling of the normal CDF came to be written down under recurring traps in ``CLAUDE.md``
in the first place, once per copy.

**Pure ``math``, deliberately.** numpy is permitted in ``*/domain/`` and would buy nothing: a report
prices a handful of positions and revalues each one five times, so the work is a few dozen
multiplications, and an array-based version would add the truth-value-of-an-array trap for no
measurable gain. scipy, jax and torch are barred outright by rule 3, which is what keeps this layer
testable with no GPU, no JIT warm-up and no optimiser anywhere in sight.
"""

from __future__ import annotations

from enum import StrEnum

from volengine.shared_kernel.domain import black76 as kernel

__all__ = ["OptionKindR", "price"]


class OptionKindR(StrEnum):
    """Side of the option contract, in the Risk context's own vocabulary.

    One spelling of the same idea among several, next to Pricing's ``OptionKindP``, Neural
    Surface's ``OptionKindN``, Market Data's ``OptionKindD`` and the published ``OptionKind`` in
    ``contracts/``. Rule 6 forbids importing another context's and rule 3 the contract's, and the
    duplication is the architecture doing its job rather than debt: a single canonical enum on the
    boundary would make every context's vocabulary hostage to every other's, so that renaming a
    member for Market Data's benefit would edit the language Risk reports in.

    **This is precisely what the shared kernel does not take.** :func:`price` hands it an
    ``is_call: bool`` and translates on the way in, so that the arithmetic can be shared without the
    language being shared. Formula in the kernel, vocabulary in the context, one line of mapping
    between them: that boundary is the whole design, and moving this enum down would erase it.

    The enums happen to carry the same two members today, and that is not a promise. This one
    describes a **position** somebody holds; Pricing's describes a quote being inverted. If Risk
    ever books a future or a perpetual as a position, this is where that member appears, and it has
    no business turning up in a calibrator's vocabulary. The ``R`` suffix keeps the spellings
    legible side by side inside ``application/acl.py``, the only module allowed to see both.

    Explicit string values, never ``auto()``: house rule, because the value is what a human reads in
    a risk report and must not change silently when a member is renamed.
    """

    CALL = "CALL"
    PUT = "PUT"


def price(
    forward: float,
    strike: float,
    tenor_years: float,
    vol: float,
    kind: OptionKindR,
    discount: float = 1.0,
) -> float:
    """Black-76 price of a European option on a forward. See the shared kernel for the formula.

    Args:
        forward: Forward price of the underlying for delivery at expiry, read off the surface's own
            forward curve by ``SurfaceView.forward_at``.
        strike: Strike of the position being valued, in the same units as the forward.
        tenor_years: Time to expiry in years, recovered from the grid by ``SurfaceView.tenor_of``.
            Convention-free by then: the grid states the daycount at its own nodes (ADR-002), so
            this function never learns which one produced the number.
        vol: Black-76 implied volatility, annualised, as a decimal (``0.65``, not ``65``), as
            interpolated from the surface by ``interpolation.implied_vol``.
        kind: Call or put, in this context's vocabulary.
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
    return kernel.price(forward, strike, tenor_years, vol, kind is OptionKindR.CALL, discount)
