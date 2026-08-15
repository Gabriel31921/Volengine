"""Quality rules over quotes, slices and the surface -- pure functions, no state, no clock.

Ingestion **flags**; the calibrator **weights or excludes**. That split is the reason this
module returns markers instead of a filtered chain: hard no-arbitrage filtering is a model
criterion, not a data criterion, and a rule that silently deleted a quote here would be making
a modelling decision on the calibrator's behalf, invisibly and one layer too early.

Only two things are rejected outright, by raising: inputs that make a rule meaningless (a
non-positive forward, a naive ``now``, two quotes at the same strike). Everything a market can
legitimately do -- crossed books, zero bids, absurd wings -- comes back as a flag.

Every function is pure. ``now`` is a parameter rather than a clock read, so a rule gives the
same answer on a live session and on a replay of it (ADR-004), and a test needs no fixture
beyond the values it is asserting about.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from itertools import pairwise
from typing import Final

from volengine.market_data.domain.option_quote import OptionKindD, QuoteObservation

BASIS_POINTS_PER_UNIT: Final[float] = 10_000.0
"""One basis point is 0.0001, so a full vol point of disagreement is 100 bp."""

type SliceQuotes = Sequence[tuple[float, OptionKindD, float]]
"""``(strike, kind, mid)`` for the quotes of one expiry, mids already resolved.

Quotes with no mid -- a one-sided book -- are dropped by the caller: the shape rules compare
prices, and there is no price to compare. They are not flagged here either, because a missing
mid is the absence of an observation rather than a bad one.
"""


class QuoteFlagD(StrEnum):
    """Advisory markers this module raises, in the domain's own vocabulary.

    Mirrors ``contracts.market_snapshot.QuoteFlag`` member for member, and is deliberately not
    the same enum: these values are read while debugging a chain, those are a wire format that
    must never change silently. ``application/acl.py`` maps between the two.
    """

    STALE = "STALE"
    WIDE_SPREAD = "WIDE_SPREAD"
    CROSSED = "CROSSED"
    LOW_SIZE = "LOW_SIZE"
    EXTREME_MONEYNESS = "EXTREME_MONEYNESS"
    IV_DIVERGENCE = "IV_DIVERGENCE"
    SLICE_MONOTONICITY = "SLICE_MONOTONICITY"
    SLICE_CONVEXITY = "SLICE_CONVEXITY"


@dataclass(frozen=True, slots=True)
class AdmissibilityThresholds:
    """Where each rule draws its line. Configuration, never constants in code (ADR-012).

    Loaded per market from TOML, so the same chain can be degraded under one deployment and
    clean under another -- which is the point: the right threshold for Deribit's wings is not
    the right threshold for SPX, and that difference is an operational decision, not a code
    change.
    """

    max_spread_rel: float
    """Widest ``(ask - bid) / mid`` still considered a usable quote. Positive and finite."""

    max_age_seconds: float
    """Seconds since the venue last updated a quote before it counts as stale. Positive."""

    moneyness_range: tuple[float, float]
    """Bounds on log-forward-moneyness ``k = log(K / F)``, as ``(lo, hi)`` with ``lo < hi``.

    A fixed range means different things at different tenors: a one-day option's entire smile
    lives inside a fraction of it, while a one-year option overflows it. Standardising by
    ``sqrt(tenor)`` would fix that at the cost of needing a vol to standardise with; the flat
    range is kept for v1 because it matches the published grid (ADR-011), and it is expected
    to be retuned once real Deribit wings are in hand (risk 1 of section 11).
    """

    max_iv_divergence_bp: float
    """Disagreement with the venue's own implied vol, in basis points, before flagging."""

    convexity_tolerance: float
    """Slack allowed on the butterfly test, in premium units. Non-negative; 0 means none.

    Indispensable in practice: convexity is a second difference, and second differences
    amplify microstructure noise brutally. With no tolerance at all, a live chain flags almost
    every triple and the marker stops carrying information.
    """

    min_size: float
    """Smallest resting size on either side that still counts. Non-negative; 0 disables it."""

    def __post_init__(self) -> None:
        for name, value in (
            ("max_spread_rel", self.max_spread_rel),
            ("max_age_seconds", self.max_age_seconds),
            ("max_iv_divergence_bp", self.max_iv_divergence_bp),
        ):
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"The {name} must be positive and finite, got {value}")

        # Zero is a legitimate setting for both of these: it is how a deployment turns the
        # rule off. Only negatives are impossible.
        for name, value in (
            ("convexity_tolerance", self.convexity_tolerance),
            ("min_size", self.min_size),
        ):
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"The {name} must be non-negative and finite, got {value}")

        lo, hi = self.moneyness_range
        if not math.isfinite(lo) or not math.isfinite(hi):
            raise ValueError(f"The moneyness range must be finite, got {self.moneyness_range}")
        if lo >= hi:
            raise ValueError(
                f"The moneyness range must be non-empty, got lo={lo} not below hi={hi}"
            )


def flag_quote(
    observation: QuoteObservation,
    strike: float,
    forward: float,
    own_iv: float | None,
    thresholds: AdmissibilityThresholds,
    now: datetime,
) -> tuple[QuoteFlagD, ...]:
    """Judge one quote against every rule, returning every marker it earns.

    The rules are independent and none of them excludes another: a wide, stale quote in the
    far wing comes back with three flags. They are emitted in a fixed order so that the result
    is comparable between runs and a test can assert on the tuple itself.

    Args:
        observation: The quote as observed, microstructure intact.
        strike: Absolute strike of the instrument.
        forward: Forward for this expiry, the reference moneyness is measured from.
        own_iv: Implied vol inverted from the mid with our own conventions, or ``None`` when
            the inversion failed or was not attempted. Never the venue's number -- comparing
            the venue's vol against itself would make the divergence rule vacuous.
        thresholds: Where each rule draws its line.
        now: Instant the judgement is made at. Timezone-aware.

    Raises:
        ValueError: If ``forward`` is not positive and finite, or ``now`` is naive. Both make
            a rule meaningless rather than merely suspicious, so they are errors, not flags.
    """
    if not math.isfinite(forward) or forward <= 0:
        raise ValueError(f"The forward must be positive and finite, got {forward}")
    if now.tzinfo is None:
        raise ValueError(f"now must be timezone-aware, got naive {now}")

    flags: list[QuoteFlagD] = []
    bid, ask = observation.bid, observation.ask

    # Strictly greater: bid == ask is a locked market, unusual but perfectly tradeable.
    if bid is not None and ask is not None and bid > ask:
        flags.append(QuoteFlagD.CROSSED)

    # A crossed book has a negative spread_rel and therefore never trips this rule, which is
    # right: the crossing is already reported, and a negative spread is not a wide one. A
    # missing spread_rel means there is no width to judge, not that the width is bad.
    spread_rel = observation.spread_rel
    if spread_rel is not None and spread_rel > thresholds.max_spread_rel:
        flags.append(QuoteFlagD.WIDE_SPREAD)

    # Measured from ts_exchange, never ts_local: staleness is how long the venue has left the
    # quote untouched. Using our own arrival time would report our transport latency as a dead
    # market. A negative age is the venue's clock running ahead of ours -- skew, not staleness,
    # and it falls through without flagging.
    age_seconds = (now - observation.ts_exchange).total_seconds()
    if age_seconds > thresholds.max_age_seconds:
        flags.append(QuoteFlagD.STALE)

    # The thinner side decides: a quote is only as good as the leg you would struggle to fill.
    if min(observation.bid_size, observation.ask_size) < thresholds.min_size:
        flags.append(QuoteFlagD.LOW_SIZE)

    # k = log(K / F): 0 is at-the-money forward, positive is out of the money for a call.
    # Working in log-forward-moneyness rather than in the raw strike is what makes expiries
    # and underlyings comparable -- 5% above the forward is the same k on BTC at 60,000 and on
    # SPX at 5,000.
    log_moneyness = math.log(strike / forward)
    lo, hi = thresholds.moneyness_range
    if log_moneyness < lo or log_moneyness > hi:
        flags.append(QuoteFlagD.EXTREME_MONEYNESS)

    # isfinite is checked explicitly rather than left to the comparison: abs(nan - x) > t is
    # False, so a failed inversion would sail through looking like agreement.
    exchange_iv = observation.exchange_iv
    if (
        own_iv is not None
        and exchange_iv is not None
        and math.isfinite(own_iv)
        and math.isfinite(exchange_iv)
        and abs(own_iv - exchange_iv) * BASIS_POINTS_PER_UNIT > thresholds.max_iv_divergence_bp
    ):
        flags.append(QuoteFlagD.IV_DIVERGENCE)

    return tuple(flags)


def flag_slice(quotes: SliceQuotes, thresholds: AdmissibilityThresholds) -> tuple[QuoteFlagD, ...]:
    """Judge the shape of one expiry's smile.

    Both rules are real arbitrages: violating either one means a portfolio exists that takes
    money in today and can never lose. They are still only flagged, because at mid prices a
    violation is usually microstructure noise rather than free money, and deciding which is a
    model judgement.

    Calls and puts are two separate curves and are checked separately; mixing them
    manufactures violations out of nothing.

    Raises:
        ValueError: If two quotes of the same kind share a strike. That is a malformed chain
            rather than a suspicious market, and it would divide by zero in the butterfly test.
    """
    calls = _prices_by_strike(quotes, OptionKindD.CALL)
    puts = _prices_by_strike(quotes, OptionKindD.PUT)

    flags: list[QuoteFlagD] = []

    # One flag even if both curves and a dozen triples are broken: this marks the slice, and
    # appending per violation would put the same member in the tuple several times.
    if _breaks_monotonicity(calls, OptionKindD.CALL) or _breaks_monotonicity(puts, OptionKindD.PUT):
        flags.append(QuoteFlagD.SLICE_MONOTONICITY)

    tolerance = thresholds.convexity_tolerance
    if _breaks_convexity(calls, tolerance) or _breaks_convexity(puts, tolerance):
        flags.append(QuoteFlagD.SLICE_CONVEXITY)

    return tuple(flags)


def flag_surface(slices: Sequence[tuple[float, SliceQuotes]]) -> tuple[QuoteFlagD, ...]:
    """Declared, and deliberately empty. The calendar check does not belong to ingestion.

    The rule it would implement is that total variance ``w = sigma^2 * T`` must be
    non-decreasing in ``T`` at fixed log-moneyness: variance is *accumulated* uncertainty, and
    accumulated uncertainty cannot shrink. If a three-month option implied less total variance
    than a one-month one at the same ``k``, you would buy the long one, sell the short one and
    hold a calendar arbitrage.

    Ingestion cannot answer that question honestly. "At fixed ``k``" requires ``sigma(k, T)``
    at moneyness values two expiries have in common, and quotes sit at scattered strikes that
    do not line up across expiries. Bridging the gap means interpolating, interpolating is a
    modelling choice, and modelling choices belong to the calibrator -- which is exactly why
    ``Design.md`` calls this a soft flag.

    So the check lives downstream, as ``parametric_pricing.domain.durrleman.calendar_violation``
    over fitted ``SVISlice`` pairs (F2-04), where there is a surface to evaluate at any ``k``.
    ADR-008 already commits to *measuring and reporting* calendar arbitrage between
    independently fitted slices rather than preventing it, and that report is where it surfaces.
    A cruder duplicate here would have to be deleted when the real one lands.

    ``QuoteFlagD`` therefore has no calendar member: a flag no producer can raise is a lie
    about the model, and the ACL author would have nothing to map it to.

    Args:
        slices: ``(tenor_years, quotes)`` per expiry, ordered by tenor. Accepted so that the
            signature is the one F2 would need, and ignored.

    Returns:
        Always empty.
    """
    return ()


def _prices_by_strike(quotes: SliceQuotes, kind: OptionKindD) -> list[tuple[float, float]]:
    """``(strike, mid)`` of one kind, ascending by strike -- the order both rules assume.

    Sorting here rather than trusting the caller is not defensive coding: a chain is assembled
    from a dictionary keyed by instrument, so its iteration order is arbitrary, and an
    unsorted comparison of neighbours reports violations that are pure accident.
    """
    selected = sorted((strike, mid) for strike, quote_kind, mid in quotes if quote_kind is kind)
    for (near_strike, _), (far_strike, _) in pairwise(selected):
        if near_strike == far_strike:
            raise ValueError(f"Two {kind} quotes share strike {near_strike} in the same slice")
    return selected


def _breaks_monotonicity(prices: Sequence[tuple[float, float]], kind: OptionKindD) -> bool:
    """A call at a higher strike pays off in strictly fewer states, so it cannot cost more.

    Formally, calls are non-increasing in strike and puts non-decreasing. If ``C(K1) < C(K2)``
    for ``K1 < K2`` you buy the cheap one, sell the dear one, bank the difference today and
    hold a payoff that is never negative at expiry.

    Equality between neighbours is allowed: two strikes both worth nothing are equal, not
    disordered, and a strict comparison would flag every dead wing in the chain.
    """
    for (_, near_price), (_, far_price) in pairwise(prices):
        if kind is OptionKindD.CALL and near_price < far_price:
            return True
        if kind is OptionKindD.PUT and near_price > far_price:
            return True
    return False


def _breaks_convexity(prices: Sequence[tuple[float, float]], tolerance: float) -> bool:
    """The butterfly test: option price must be a convex function of strike.

    Buy one at ``K1``, sell two at ``K2``, buy one at ``K3``. The payoff is a tent peaking at
    ``K2``, worth zero outside ``[K1, K3]`` and never negative -- so the structure cannot cost
    less than nothing. With evenly spaced strikes that is ``C(K1) - 2*C(K2) + C(K3) >= 0``, a
    discrete second derivative. Unevenly spaced, the middle price must sit below the chord
    joining its neighbours, which is the general form used here.

    Fewer than three quotes cannot form a triple, and the loop simply does not run.
    """
    for (k_low, price_low), (k_mid, price_mid), (k_high, price_high) in zip(
        prices, prices[1:], prices[2:], strict=False
    ):
        chord = ((k_high - k_mid) * price_low + (k_mid - k_low) * price_high) / (k_high - k_low)
        if price_mid - chord > tolerance:
            return True
    return False
