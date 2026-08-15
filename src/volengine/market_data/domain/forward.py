"""Construction of the forward and the cross-check between its two routes (ADR-002).

The forward is the price of the underlying for delivery at a given expiry, and it is the
anchor of the whole surface: log-forward-moneyness is ``k = log(K / F)``, so every strike in
the slice is measured from it. A forward wrong by a fraction of a percent shifts every ``k``
at once and tilts the smile, and it does so without making any single quote look suspicious --
which is what makes it dangerous. Nothing downstream can detect the error, because downstream
only ever sees the moneyness, never the number it was divided by.

That is why ``MarketConventions.forward_method`` *declares* the route instead of leaving it
implicit, and why a second, independent route is computed anyway and its disagreement
published as ``QualityBlock.forward_crosscheck_error``. The two routes share no input: one is
the venue's own underlying reference, the other is inferred from option premiums we already
hold. When independent measurements of the same quantity disagree, the fault is in our
bookkeeping, not in the market.

Three pure functions: one per ``ForwardMethod`` route that this engine implements (see
``market_conventions``), plus the cross-check. No state, no clock -- the forward is a
function of the quotes it is built from, so building it twice from the same quotes must give
the same number on a live session and on a replay of it (ADR-004).
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from statistics import median

type ParityPairs = Sequence[tuple[float, float, float]]
"""``(strike, call_mid, put_mid)`` at strikes where **both** sides are quoted.

Triples rather than three parallel sequences on purpose: parallel sequences have to be kept
aligned by the caller, and a single misaligned index pairs a call with another strike's put and
produces a forward that is wrong by the strike spacing rather than obviously broken. Unpack
them by name -- ``for strike, call_mid, put_mid in pairs`` -- and never index the tuple.
"""


def forward_from_provider(underlying_price: float) -> float:
    """The ``PROVIDER_UNDERLYING`` route: take the reference the venue publishes.

    On Deribit the ticker carries the synchronous future or perpetual price for that expiry,
    i.e. an actually traded forward rather than one derived from a carry model. There is
    nothing to compute, and that is the point of the route: no rate curve, no dividend
    schedule, no assumption of ours standing between the market and the anchor.

    The function exists despite being one guard because the guard is the whole content. The
    forward is the denominator of every ``log(K / F)`` in the slice, so "positive and finite"
    has to hold somewhere; declared here it holds once, in the place the method is named,
    instead of being re-derived at each call site and eventually forgotten at one of them.

    Raises:
        ValueError: If ``underlying_price`` is not positive and finite. A venue that publishes
            a zero, a negative or a NaN underlying has published nothing usable, and carrying
            it forward would poison every moneyness in the expiry rather than one quote.
    """
    if not math.isfinite(underlying_price) or underlying_price <= 0:
        raise ValueError(
            f"The provider underlying price must be positive and finite, got {underlying_price}"
        )
    return underlying_price


def forward_from_parity(pairs: ParityPairs, discount: float = 1.0) -> float:
    """The ``PUT_CALL_PARITY`` route: infer the forward from the options themselves.

    Put-call parity holds for European options by static replication -- long a call, short a
    put at the same strike pays exactly ``S_T - K`` at expiry, which is a forward contract --
    so::

        C - P = D * (F - K)      =>      F = K + (C - P) / D

    with ``D = exp(-r * T)`` the discount factor to expiry. It is an arbitrage identity, not a
    model: no volatility, no distribution, no assumption about how the underlying moves enters
    it. That independence is exactly what makes it a legitimate cross-check on the provider
    route rather than a second opinion from the same source.

    On Deribit ``D`` is effectively 1 and the default reflects it: the options are on futures
    and settle in the same numeraire as the premium, so there is no cash leg to discount. The
    parameter stays because the identity does not stop being true elsewhere, and a market with
    a real rate curve would pass its own factor rather than fork the function.

    The result is the **median** of the per-strike implied forwards, deliberately not the mean.
    Parity is exact in theory and noisy in practice: mids are midpoints of spreads, the two
    legs are not observed at the same instant, and ``C - P`` is a difference of two numbers far
    larger than itself, so spread noise lands on it undiluted. One stale or crossed quote moves
    a mean by an unbounded amount and there is nothing in the output to say it happened; the
    median simply ignores a minority of bad pairs. Callers pass the strikes nearest the money,
    where both sides are genuinely two-sided and ``C - P`` is least dominated by spread.

    Args:
        pairs: ``(strike, call_mid, put_mid)`` at strikes with both sides quoted.
        discount: Discount factor to expiry, ``exp(-r * T)``. Positive and finite; 1 means no
            discounting.

    Raises:
        ValueError: If ``pairs`` is empty, since there is nothing to infer a forward from and
            any fallback value would be a guess wearing a measurement's clothes; if ``discount``
            is not positive and finite; if any strike is not positive and finite or any premium
            is not finite; or if the implied forward is not positive and finite. A chain whose
            own quotes imply a negative forward is malformed rather than merely suspicious --
            ``log(K / F)`` has no value there, so there is no degraded answer to return.
    """
    if len(pairs) == 0:
        raise ValueError("Put-call parity needs at least one strike quoted on both sides")
    if not math.isfinite(discount) or discount <= 0:
        raise ValueError(f"The discount factor must be positive and finite, got {discount}")

    implied: list[float] = []
    for strike, call_mid, put_mid in pairs:
        if not math.isfinite(strike) or strike <= 0:
            raise ValueError(f"A parity strike must be positive and finite, got {strike}")
        if not math.isfinite(call_mid) or not math.isfinite(put_mid):
            raise ValueError(
                f"The parity premiums at strike {strike} must be finite, "
                f"got call={call_mid} put={put_mid}"
            )
        # F = K + (C - P) / D. Only the premium difference is discounted; a paren around the
        # division that swallowed the strike would give a number that still looks like a
        # forward, which is precisely the kind of error this whole module exists to catch.
        implied.append(strike + (call_mid - put_mid) / discount)

    forward = median(implied)
    if not math.isfinite(forward) or forward <= 0:
        raise ValueError(
            f"Put-call parity implies a forward of {forward}, which is not a tradeable price"
        )
    return forward


def crosscheck_error(primary: float, secondary: float) -> float:
    """Relative disagreement between the two independent routes to the forward.

    ``abs(primary - secondary) / primary``, published as
    ``QualityBlock.forward_crosscheck_error``. This is the sanity check on ADR-002: the two
    routes share no input, so a disagreement cannot be blamed on the market. It means our
    conventions are wrong -- the wrong expiry instant, the wrong underlying reference, a
    numeraire mismatch, or premiums read on one side of the book and mids on the other.

    Relative rather than absolute because the number has to be comparable across markets and
    across time: 50 units of disagreement is a catastrophe on a 5,000 index and noise on a
    60,000 coin, and a single threshold in TOML (ADR-012) cannot be written against an absolute
    scale.

    The magnitude is unsigned, and not merely as a convenience. ``QualityBlock`` rejects a
    negative ``forward_crosscheck_error`` at construction, so returning a signed residual would
    turn a routine disagreement into a ``ValueError`` two layers away, inside the ACL, with
    nothing at the raise site to explain why. Direction is not lost information worth that: the
    question the quality block answers is *how far apart*, and which route is higher is a
    diagnostic for whoever reads both numbers.

    Args:
        primary: Forward from the market's declared method. It is the denominator, so it sets
            the scale the disagreement is measured against.
        secondary: Forward from the independent route.

    Raises:
        ValueError: If ``primary`` is not positive and finite -- there is no scale to divide
            by, and a zero denominator would surface as an unattributable ``ZeroDivisionError``
            -- or if ``secondary`` is not finite, which would silently return a NaN that the
            published contract would happily accept as "not negative".
    """
    if not math.isfinite(primary) or primary <= 0:
        raise ValueError(f"The primary forward must be positive and finite, got {primary}")
    if not math.isfinite(secondary):
        raise ValueError(f"The secondary forward must be finite, got {secondary}")
    return abs(primary - secondary) / primary
