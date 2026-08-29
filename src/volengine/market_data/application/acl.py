"""The one module in Market Data that speaks both languages (rule 5).

The domain works on ``ChainSnapshot`` and knows nothing about ``contracts/`` (rule 3). This
module is where an internal view becomes a published ``MarketSnapshot``, and it is the only
place in the context allowed to build a DTO. Everything above it is business rules that survive
a rename of the published language; everything below it is a wire format that survives a rewrite
of the rules.

Four jobs, and each of them is a translation the domain deliberately refused to do:

* **Two vocabularies of the same idea.** ``OptionKindD.CALL`` becomes ``OptionKind.CALL`` and
  ``QuoteFlagD.STALE`` becomes ``QuoteFlag.STALE``. The mappings below are written out member by
  member rather than bridged with ``QuoteFlag(flag.value)``, which would work today and would
  keep working silently on the day the two enums stop agreeing. Spelled out, the coupling is
  visible, ``mypy`` checks it, and a member added to one side and not the other is a
  ``KeyError`` in a test rather than a flag that quietly stops crossing.

* **Reconciling the two clocks.** ``ChainSnapshot.ts_exchange`` is deliberately not clamped
  upstream, so a venue running ahead lands in the future. See
  :func:`reconcile_exchange_instant`.

* **Identity.** A ``ChainSnapshot`` has no ``snapshot_id``, because inventing one is a published
  language concern and randomness in the domain would break replay (ADR-004). It is minted here,
  from a sequence number the use case owns.

* **The forward cross-check.** ``QualityBlock.forward_crosscheck_error`` is the sanity check on
  ADR-002, and it is computed here because the published quality block is what it exists for. It
  is the only number in this module that is calculated rather than copied, and it calls the
  domain's own ``forward`` functions to do it.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from datetime import datetime
from typing import Final

from volengine.contracts.events import ChainCompositionChanged, SnapshotReady
from volengine.contracts.market_snapshot import (
    MarketSnapshot,
    OptionKind,
    QualityBlock,
    QuoteData,
    QuoteFlag,
    SliceData,
)
from volengine.market_data.domain.admissibility import QuoteFlagD
from volengine.market_data.domain.forward import crosscheck_error, forward_from_parity
from volengine.market_data.domain.option_quote import InstrumentId, OptionKindD
from volengine.market_data.domain.quote_chain import ChainSlice, ChainSnapshot
from volengine.market_data.domain.snapshot_policy import QualityAssessment

_KINDS: Final[Mapping[OptionKindD, OptionKind]] = {
    OptionKindD.CALL: OptionKind.CALL,
    OptionKindD.PUT: OptionKind.PUT,
}
"""Domain side to wire side. ``"CALL"`` is what a human debugging a chain reads; ``"C"`` is what
travels. Neither spelling may be derived from the other, which is why this is a table."""

_FLAGS: Final[Mapping[QuoteFlagD, QuoteFlag]] = {
    QuoteFlagD.STALE: QuoteFlag.STALE,
    QuoteFlagD.WIDE_SPREAD: QuoteFlag.WIDE_SPREAD,
    QuoteFlagD.CROSSED: QuoteFlag.CROSSED,
    QuoteFlagD.LOW_SIZE: QuoteFlag.LOW_SIZE,
    QuoteFlagD.EXTREME_MONEYNESS: QuoteFlag.EXTREME_MONEYNESS,
    QuoteFlagD.IV_DIVERGENCE: QuoteFlag.IV_DIVERGENCE,
    QuoteFlagD.SLICE_MONOTONICITY: QuoteFlag.SLICE_MONOTONICITY,
    QuoteFlagD.SLICE_CONVEXITY: QuoteFlag.SLICE_CONVEXITY,
}
"""Every domain flag and where it lands. The two enums mirror each other member for member
today; the day one grows a member the other lacks, this table is what fails."""


def clock_skew_seconds(snapshot: ChainSnapshot) -> float:
    """How far the venue's clock sits ahead of ours, in seconds. Negative means behind.

    Signed, unlike almost every other measurement in this engine, because the two directions mean
    different things and the operator has to be able to tell them apart. A venue *behind* us is
    ordinarily transport latency, which is what we should be measuring anyway. A venue *ahead* of
    us is a clock disagreement outright, since no message can be received before it was sent, and
    it is the one that breaks arithmetic downstream.

    Exposed separately from :func:`reconcile_exchange_instant` so that the use case can gauge it
    on every snapshot, whether or not it crossed the tolerance. A reconciliation that only became
    visible once it fired would leave a venue drifting towards the threshold invisible until the
    day it arrived.
    """
    return (snapshot.ts_exchange - snapshot.ts_local).total_seconds()


def reconcile_exchange_instant(
    ts_exchange: datetime, ts_local: datetime, max_skew_seconds: float
) -> datetime:
    """The instant to publish as ``ts_exchange``, with an unbelievable venue clock discarded.

    ``ChainSnapshot.ts_exchange`` is the freshest stamp the venue put on anything we hold, and the
    domain deliberately does not clamp it: the skew is real information and hiding it there would
    be the wrong place. Here it has to be resolved, because a DTO further down has an opinion
    about the order of two instants -- ``CalibratedSurface`` refuses ``ts_calibrated <
    ts_snapshot``, so a venue one second fast would make every surface built from this snapshot
    fail construction, and an entire market would go dark for as long as the skew lasted.

    **Within the tolerance the venue's stamp is kept, in both directions.** A venue slightly
    behind is our own transport latency, which is exactly what ``QuoteData.age_seconds`` and the
    freshness policy downstream are supposed to see; overwriting it with our arrival time would
    report every market as perfectly fresh and delete the measurement. A venue slightly ahead is
    ordinary clock jitter, and the surface contract's equality case absorbs it.

    **Outside the tolerance, in either direction, the venue's clock is not evidence about
    anything** and ``ts_local`` is used instead. That is a deliberate choice of which failure to
    prefer, and both directions were considered:

    * A venue far *ahead* publishes snapshots stamped in the future. Keeping the stamp breaks the
      calibration contract as described above.
    * A venue far *behind* -- a misconfigured clock rather than a slow link -- publishes snapshots
      that are born looking stale. Keeping the stamp is worse than the first case rather than
      better: nothing fails loudly, the freshness policy of Design 7.2 simply reports ``DEGRADED``
      and then ``REJECT`` for a market whose data is arriving perfectly well, and the diagnosis
      sits three contexts away from the cause.

    Neither branch drops a snapshot. That was the requirement this function was written against:
    the worst outcome available here is discarding good market data, and falling back to the
    instant we can vouch for costs nothing but a timestamp we had no reason to believe.

    Args:
        ts_exchange: Freshest venue stamp among the observations held. Timezone-aware.
        ts_local: When we assembled the view, from the injected clock (ADR-004). Timezone-aware.
        max_skew_seconds: How far apart the two clocks may be before the venue's is disbelieved.
            Positive and finite; configuration, never a constant (ADR-012). A deployment that
            trusts its venue absolutely sets it high, and one talking to a known-bad clock sets it
            low; there is no defensible value to hard-code, because it is a statement about a
            venue rather than about this code.

    Returns:
        ``ts_exchange`` when the two clocks agree within the tolerance, ``ts_local`` otherwise.

    Raises:
        ValueError: If ``max_skew_seconds`` is not positive and finite. Finiteness first and the
            bad cases joined with ``or``: ``float("nan") <= 0`` is ``False``, and a NaN tolerance
            would make every comparison below ``False`` and silently disable the reconciliation
            entirely -- the exact failure this function exists to prevent, reintroduced by its own
            configuration.
    """
    if not math.isfinite(max_skew_seconds) or max_skew_seconds <= 0:
        raise ValueError(
            f"The maximum clock skew must be positive and finite, got {max_skew_seconds}"
        )
    if abs((ts_exchange - ts_local).total_seconds()) > max_skew_seconds:
        return ts_local
    return ts_exchange


def to_market_snapshot(
    snapshot: ChainSnapshot,
    snapshot_id: str,
    quality: QualityAssessment,
    max_skew_seconds: float,
) -> MarketSnapshot:
    """Translate an internal chain view into the published snapshot.

    A field-by-field copy everywhere it can be one. The three places it cannot are the clock
    reconciliation above, the forward cross-check below, and the crossed-market spread described
    in :func:`_to_slice_data`.

    Args:
        snapshot: The frozen view the aggregate produced.
        snapshot_id: Identity to publish under, minted by the use case. See
            :func:`build_snapshot_id` on why it is not generated here and not random.
        quality: The snapshotting policy's verdict on the same stats. Passed in rather than
            recomputed, because the policy owns that judgement and this module owns none.
        max_skew_seconds: Clock tolerance -- see :func:`reconcile_exchange_instant`.

    Returns:
        The published snapshot, ready for ``SnapshotReady``.

    Raises:
        ValueError: If any DTO below refuses its own invariants. Not caught and not translated:
            a chain that produced an unpublishable snapshot is a bug in the domain or in this
            translation, not a market condition, and the constructor's message names the field.
    """
    return MarketSnapshot(
        snapshot_id=snapshot_id,
        market_id=snapshot.market_id,
        ts_exchange=reconcile_exchange_instant(
            snapshot.ts_exchange, snapshot.ts_local, max_skew_seconds
        ),
        ts_local=snapshot.ts_local,
        underlying=snapshot.underlying,
        slices=tuple(_to_slice_data(chain_slice) for chain_slice in snapshot.slices),
        quality=QualityBlock(
            coverage_ratio=snapshot.stats.coverage_ratio,
            max_age_seconds=snapshot.stats.max_age_seconds,
            n_quotes_admissible=snapshot.stats.n_quotes_admissible,
            n_quotes_total=snapshot.stats.n_quotes_total,
            forward_crosscheck_error=forward_crosscheck_error(snapshot),
            degraded=quality.degraded,
        ),
    )


def _to_slice_data(chain_slice: ChainSlice) -> SliceData:
    """One expiry, published.

    **The crossed-market spread is clamped to zero here, and the flag is what survives.**
    ``ChainQuote.spread_rel`` is negative on a crossed book on purpose -- the sign is the evidence
    -- while ``QuoteData`` refuses a negative spread outright, so the two types genuinely
    disagree and the disagreement has to be resolved somewhere. It is resolved in favour of the
    contract, for two reasons. The published spread is the basis of the calibration weights, and a
    negative weight would ask an optimiser to move *away* from a quote, which is not a thing
    anyone means. And the evidence is not lost: ``QuoteFlag.CROSSED`` is already on the quote,
    which is the channel the published language provides for exactly this fact, and a consumer
    reading a spread of zero next to a ``CROSSED`` flag is not being misled about anything.
    """
    return SliceData(
        expiry=chain_slice.expiry,
        tenor_years=chain_slice.tenor_years,
        forward=chain_slice.forward,
        quotes=tuple(
            QuoteData(
                strike=quote.strike,
                kind=_KINDS[quote.kind],
                mid=quote.mid,
                spread_rel=max(0.0, quote.spread_rel),
                age_seconds=quote.age_seconds,
                flags=tuple(_FLAGS[flag] for flag in quote.flags),
                exchange_iv=quote.exchange_iv,
            )
            for quote in chain_slice.quotes
        ),
        flags=tuple(_FLAGS[flag] for flag in chain_slice.flags),
    )


def forward_crosscheck_error(snapshot: ChainSnapshot) -> float | None:
    """Worst disagreement, across expiries, between the two independent routes to the forward.

    ADR-002's sanity check, and the only computed number in this module. The chain's forward comes
    from the venue's own underlying reference; put-call parity infers a second one from premiums
    we already hold, sharing no input with the first. When two independent measurements of one
    quantity disagree, the fault is in our bookkeeping rather than in the market.

    **The maximum over the expiries, not the mean.** ``QualityBlock`` carries one number for the
    whole snapshot, and a mean would let a single badly broken expiry disappear into a dozen
    healthy ones -- which is the one case the check exists to surface. The same reasoning as
    ``max_age_seconds`` beside it: a quality block reports the worst case, because that is what a
    consumer has to decide against.

    ``None`` when no expiry offered a single strike quoted on both sides, which is routine early
    in a session and on a thin market. Deliberately not ``0.0``: zero is perfect agreement, and
    reporting "we could not look" as "we looked and found nothing wrong" is the kind of
    fabrication the published contract's ``| None`` exists to refuse.

    Crossed and stale quotes are fed to parity along with everything else, and the median inside
    ``forward_from_parity`` is what absorbs them. Filtering them here would mean this module
    forming a view about which quotes count, which is the calibrator's decision, not ingestion's.
    """
    worst: float | None = None
    for chain_slice in snapshot.slices:
        parity = _parity_forward(chain_slice)
        if parity is None:
            continue
        error = crosscheck_error(primary=chain_slice.forward, secondary=parity)
        worst = error if worst is None else max(worst, error)
    return worst


def _parity_forward(chain_slice: ChainSlice) -> float | None:
    """The put-call parity forward for one expiry, or ``None`` if no strike has both sides.

    Every two-sided strike is used rather than only those nearest the money. The median already
    provides the robustness that selecting near-the-money strikes was meant to provide, and any
    cutoff would be one more threshold to configure for a check whose whole purpose is to be a
    second opinion computed differently from the first.

    ``forward_from_parity`` raises on a set of pairs whose median implies a non-positive forward.
    That is a malformed chain rather than a suspicious one -- but it must not take down a snapshot
    that is otherwise publishable, because this number is a diagnostic and the quality block's
    contract already admits "could not be computed". So the failure is folded into ``None``, which
    is the same answer the absent-pairs case gives and means the same thing: no cross-check this
    cycle.
    """
    calls = {
        quote.strike: quote.mid for quote in chain_slice.quotes if quote.kind is OptionKindD.CALL
    }
    puts = {
        quote.strike: quote.mid for quote in chain_slice.quotes if quote.kind is OptionKindD.PUT
    }
    pairs = [(strike, calls[strike], puts[strike]) for strike in sorted(calls.keys() & puts.keys())]
    if not pairs:
        return None
    try:
        return forward_from_parity(pairs)
    except ValueError:
        return None


def build_snapshot_id(market_id: str, sequence: int) -> str:
    """Mint the identity a snapshot is published under: ``"BTC-DERIBIT:00000042"``.

    **Deterministic, never random**, and that is the whole reason this exists as a function rather
    than as a ``uuid4()`` inside the constructor call. ADR-004 promises that replaying a recording
    reproduces the session exactly, and every surface, failure and risk figure downstream carries
    this string: a random id would make two runs of one recording incomparable line by line, while
    proving nothing that a counter does not.

    The sequence belongs to the use case, which is the only object that knows how many snapshots
    this market has emitted. Deriving the id from the timestamp instead would collide under a
    ``ManualClock`` held still across two emissions, which is a fixture this repo writes on
    purpose, and a heartbeat snapshot in a dead-quiet market repeats ``ts_exchange`` exactly.

    Raises:
        ValueError: If ``market_id`` is empty or ``sequence`` is negative. Both would produce an
            id that the published contract accepts and that no operator can trace.
    """
    if not market_id:
        raise ValueError("The market id must not be empty")
    if sequence < 0:
        raise ValueError(f"The snapshot sequence must not be negative, got {sequence}")
    return f"{market_id}:{sequence:08d}"


def to_snapshot_ready(snapshot: MarketSnapshot) -> SnapshotReady:
    """Wrap a published snapshot in the event that carries it.

    One line, and it lives here rather than in the use case for the reason rule 5 exists: an
    event is a DTO, and the moment a use case constructs one directly, the set of modules that
    know the published language stops being "the ACL" and starts being "whoever needed an event
    that day". The rule is only enforceable while it is absolute, and the cost of keeping it
    absolute is this function.
    """
    return SnapshotReady(snapshot=snapshot)


def to_composition_changed(
    market_id: str, ts: datetime, instruments: frozenset[InstrumentId]
) -> ChainCompositionChanged:
    """Announce the live instrument set, as strings the bus can carry.

    ``InstrumentId`` is a domain object and cannot cross a boundary (rule 3), so each one is
    rendered by :func:`instrument_key`. The set is sorted on the way out: a ``frozenset`` has no
    order, and an event whose payload permuted between two runs of one recording would defeat
    ADR-004 for a reason that has nothing to do with the market.

    The whole set travels, never a delta -- ADR-013, and the bus conflates (ADR-003), so a
    consumer that missed the previous event could not reconstruct the universe from increments.
    """
    return ChainCompositionChanged(
        market_id=market_id,
        ts=ts,
        instruments=tuple(sorted(instrument_key(one) for one in instruments)),
    )


def instrument_key(instrument: InstrumentId) -> str:
    """Render one instrument as ``"BTC|2026-08-27T08:00:00+00:00|60000|CALL"``.

    **Not the venue's symbol.** ``"BTC-27AUG26-60000-C"`` is Deribit's encoding, the adapter
    parsed it away on purpose, and reconstructing it here would put a venue's format back in the
    published language and make this event unusable for any other venue. This is the engine's own
    spelling: the four fields that *are* the identity, in a fixed order, with the expiry as an
    ISO-8601 instant rather than a date -- the time of day is part of the identity (ADR-002), and
    two contracts expiring on one calendar day at different hours are different instruments.

    ``{:g}`` on the strike so that a round number reads as ``60000`` rather than ``60000.0``,
    which is what an operator grepping a log expects to find.
    """
    return (
        f"{instrument.underlying}|{instrument.expiry.isoformat()}|"
        f"{instrument.strike:g}|{instrument.kind.value}"
    )
