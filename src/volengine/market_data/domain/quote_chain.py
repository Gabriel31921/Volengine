"""The live option chain for one underlying: the context's aggregate root.

One instance per market, and the only mutable object in this domain. Everything else here is
frozen, because everything else describes a moment; this describes a market that is still
moving. All state changes go through the root -- nothing reaches inside -- which is what makes
"the chain is consistent" a statement that can be true at every instant rather than a hope.

The boundary is the whole underlying rather than one expiry, because the invariants that
matter cross expiries: total variance must not fall as tenor grows, and in crypto the forward
arrives on the same ticker stream as the quotes. Split per expiry, those checks would fall
*between* aggregates, where there is no instant at which they can be evaluated consistently.

The cost of that choice is that this object is not thread-safe and does not pretend to be. One
asyncio task owns one chain; concurrency between markets is concurrency between chains.

``snapshot()`` returns a ``ChainSnapshot``, a domain type -- never a contract DTO. The domain
does not know the published language, and ``application/acl.py`` translates. That is what lets
the published contract change without touching a line of the rules below.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime

from volengine.market_data.domain.admissibility import (
    AdmissibilityThresholds,
    QuoteFlagD,
    flag_quote,
    flag_slice,
)
from volengine.market_data.domain.market_conventions import MarketConventions
from volengine.market_data.domain.option_quote import (
    InstrumentId,
    OptionKindD,
    QuoteObservation,
    QuoteUpdate,
)
from volengine.shared_kernel.domain.instants import require_aware


@dataclass(frozen=True, slots=True)
class ChainQuote:
    """One quote as it enters a snapshot: reduced to numbers, judged, and frozen."""

    strike: float
    kind: OptionKindD
    mid: float
    """Positive and finite. A quote without a usable mid never becomes a ChainQuote."""

    spread_rel: float
    """``(ask - bid) / mid``. Negative on a crossed book -- the sign is the evidence."""

    age_seconds: float
    """Seconds since the venue last updated it, measured at the snapshot instant.

    Clamped at zero. A venue clock running ahead of ours yields a negative difference, which
    is skew rather than a quote from the future, and the published ``QualityBlock`` rejects a
    negative age outright.
    """

    flags: tuple[QuoteFlagD, ...]
    exchange_iv: float | None

    def __post_init__(self) -> None:
        if self.mid <= 0 or not math.isfinite(self.mid):
            raise ValueError(f"The mid must be positive and finite, got {self.mid}")
        if self.age_seconds < 0:
            raise ValueError(f"The age cannot be negative, got {self.age_seconds}")


@dataclass(frozen=True, slots=True)
class ChainSlice:
    """Every quote of one expiry, with the two numbers that make them comparable.

    ``tenor_years`` and ``forward`` are resolved here, before the boundary, so that a
    calibrator downstream receives homogeneous numbers and never learns which market they came
    from (ADR-002).
    """

    expiry: datetime
    tenor_years: float
    forward: float
    quotes: tuple[ChainQuote, ...]
    flags: tuple[QuoteFlagD, ...]

    def __post_init__(self) -> None:
        if self.tenor_years <= 0 or not math.isfinite(self.tenor_years):
            raise ValueError(f"The tenor must be positive and finite, got {self.tenor_years}")
        if self.forward <= 0 or not math.isfinite(self.forward):
            raise ValueError(f"The forward must be positive and finite, got {self.forward}")


@dataclass(frozen=True, slots=True)
class ChainStats:
    """How much of the chain is actually usable right now.

    Computed rather than accumulated, so it always describes the instant it was asked about.
    The snapshotting policy reads this to decide whether emitting is worth it at all, and the
    ACL turns it into the published ``QualityBlock``.
    """

    n_instruments_known: int
    """Instruments the chain believes exist: discovered, plus any that have ticked.

    The denominator of coverage. Without a discovery feed it degenerates to "whatever has
    ticked", and coverage would then be 1.0 by construction -- a number that looks reassuring
    and measures nothing.
    """

    n_quotes_total: int
    """Observations held, usable or not. Never more than ``n_instruments_known``.

    True by construction in ``QuoteChain`` -- ``apply`` registers the instrument it records,
    and ``set_live_instruments`` drops the observations of whatever died -- but stated as an
    invariant because consumers divide by ``n_instruments_known``. Without it, the admissible
    ratio the snapshotting policy compares against its threshold is not bounded by 1, and a
    hand-built ``ChainStats`` could report a chain as better than complete.
    """

    n_quotes_admissible: int
    """Observations with a usable mid and not a single flag.

    A deliberately strict definition: anything less would need a policy about which flags are
    forgivable, and that policy belongs to the calibrator, not to the count.
    """

    max_age_seconds: float
    """Age of the oldest observation. The worst case, not the average, and never negative."""

    coverage_ratio: float
    """Fraction of known instruments carrying a usable mid. In ``[0, 1]``."""

    def __post_init__(self) -> None:
        if not 0 <= self.coverage_ratio <= 1:
            raise ValueError(f"The coverage must be in [0, 1], got {self.coverage_ratio}")
        if self.max_age_seconds < 0:
            raise ValueError(f"The maximum age cannot be negative, got {self.max_age_seconds}")
        if self.n_quotes_admissible > self.n_quotes_total:
            raise ValueError(
                f"Admissible quotes cannot exceed the total, got "
                f"{self.n_quotes_admissible} > {self.n_quotes_total}"
            )
        if self.n_quotes_total > self.n_instruments_known:
            raise ValueError(
                f"Quotes cannot exceed the instruments known to exist, got "
                f"{self.n_quotes_total} > {self.n_instruments_known}"
            )


@dataclass(frozen=True, slots=True)
class ChainSnapshot:
    """A consistent, disconnected view of the chain at one instant.

    Disconnected is the operative word: the chain keeps mutating while a calibration runs, and
    a live view would change under the consumer mid-fit. Everything here is frozen and shares
    nothing with the aggregate that produced it.

    It carries no ``snapshot_id`` and no ``schema_version``. Both are published-language
    concerns, invented by the ACL as the snapshot crosses the boundary -- which also keeps
    randomness out of the domain, and is part of why a recorded session replays identically.
    """

    market_id: str
    underlying: str

    ts_exchange: datetime
    """Freshest exchange stamp among the observations held. The economically meaningful one.

    It is *not* clamped to ``ts_local``: when a venue's clock runs ahead of ours this lands in
    the future, and that skew is real information rather than something to hide. The ACL is
    where the two are reconciled, because it is the DTO that has an opinion about their order.
    """

    ts_local: datetime
    """When we assembled the view, from the injected clock (ADR-004)."""

    slices: tuple[ChainSlice, ...]
    """Expiries with a usable forward and at least one usable quote, ascending by expiry."""

    stats: ChainStats

    def __post_init__(self) -> None:
        expiries = [chain_slice.expiry for chain_slice in self.slices]
        if expiries != sorted(set(expiries)):
            raise ValueError("The slices must be sorted by expiry, without duplicates")


class QuoteChain:
    """Last known state per instrument, and the rules for turning it into a snapshot.

    Deliberately dumb on the way in and careful on the way out: ``apply`` only records, while
    every judgement -- staleness, moneyness, the shape of a smile -- happens in ``snapshot``,
    evaluated against a single instant. Flags computed on arrival would be stale before they
    were read, since age advances on its own and a quote's neighbours keep changing.
    """

    def __init__(self, conventions: MarketConventions, thresholds: AdmissibilityThresholds) -> None:
        self._conventions = conventions
        self._thresholds = thresholds
        self._observations: dict[InstrumentId, QuoteObservation] = {}
        self._live: set[InstrumentId] = set()
        self._forwards: dict[datetime, float] = {}
        self._baseline_mids: dict[InstrumentId, float] = {}

    # --- writes

    def apply(self, update: QuoteUpdate) -> None:
        """Record the latest state of one instrument, replacing whatever was there.

        A full replacement rather than a merge, because the ticker republishes the whole top
        of book on every tick. An instrument that has never been discovered is registered on
        the spot: discovery polls on its own schedule, and a strike born between two polls
        still trades. Dropping its quotes to keep a tidy inventory would lose real data.

        Raises:
            ValueError: If the update belongs to a different underlying. One chain covers one
                underlying, and mixing two would corrupt every slice-level rule silently.
        """
        instrument = update.instrument
        if instrument.underlying != self._conventions.underlying:
            raise ValueError(
                f"This chain covers {self._conventions.underlying}, "
                f"got an update for {instrument.underlying}"
            )

        self._observations[instrument] = update.observation
        self._live.add(instrument)

        # Per expiry, not global: the venue publishes the synchronous future for *that*
        # expiry, and in contango the March and June forwards differ by percent, not by noise.
        if update.underlying_price is not None:
            self._forwards[instrument.expiry] = update.underlying_price

    def set_live_instruments(self, instruments: Iterable[InstrumentId]) -> None:
        """Replace the set of instruments believed to exist, and forget the ones that died.

        Takes the whole set rather than an added/removed delta, for the same reason ADR-013
        put the full set in ``ChainCompositionChanged``: deltas accumulate instead of
        replacing, so missing one leaves the chain permanently wrong. This is idempotent --
        applying it twice, or applying only the last of five, reaches the same state.

        Raises:
            ValueError: If any instrument belongs to a different underlying.
        """
        live = set(instruments)
        for instrument in live:
            if instrument.underlying != self._conventions.underlying:
                raise ValueError(
                    f"This chain covers {self._conventions.underlying}, "
                    f"got {instrument.underlying} in the live set"
                )

        self._live = live
        for dead in self._observations.keys() - live:
            del self._observations[dead]
            self._baseline_mids.pop(dead, None)

        surviving_expiries = {instrument.expiry for instrument in live}
        for expiry in self._forwards.keys() - surviving_expiries:
            del self._forwards[expiry]

    def reset_move_baseline(self) -> None:
        """Mark the current mids as the reference future moves are measured against.

        Called by the use case right after it emits. Keeping one baseline instead of a history
        is what bounds the memory of a process that runs for hours: the snapshotting policy
        only ever asks "how far since the last time I published", never about an arbitrary
        past instant.
        """
        self._baseline_mids = {
            instrument: mid
            for instrument, observation in self._observations.items()
            if (mid := _usable_mid(observation)) is not None
        }

    # --- reads

    def max_relative_move_since_baseline(self) -> float:
        """Largest relative mid move since the baseline, as a fraction. Zero if nothing moved.

        The material-change half of the snapshotting policy: a chain that has not moved is not
        worth republishing, however long the cadence timer has been running.

        Instruments absent from the baseline do not count as a move. A newly born strike is a
        composition change, reported by its own event, and treating it as an infinite move
        would make every discovery force a snapshot.
        """
        largest = 0.0
        for instrument, baseline in self._baseline_mids.items():
            observation = self._observations.get(instrument)
            if observation is None or baseline <= 0:
                continue
            current = _usable_mid(observation)
            if current is None:
                continue
            largest = max(largest, abs(current - baseline) / baseline)
        return largest

    def stats(self, now: datetime) -> ChainStats:
        """Coverage and freshness at ``now``, without building the slices.

        Costs the same as a snapshot, because admissibility depends on the forward and on a
        quote's neighbours and cannot be judged instrument by instrument. At one snapshot per
        second over a few hundred instruments that is not worth optimising away.
        """
        return _stats_from(self._build_slices(now), self._observations, self._live, now)

    def snapshot(self, now: datetime) -> ChainSnapshot:
        """Freeze a consistent view of the chain as of ``now``.

        Expired expiries are dropped -- venues keep them listed until settlement, and there is
        no honest tenor for them. So are expiries with no forward yet: without it there is
        neither moneyness nor a comparable price, and publishing the strikes raw would hand a
        calibrator numbers it cannot place.

        Raises:
            ValueError: If ``now`` is naive.
        """
        require_aware(now, "now")
        slices = self._build_slices(now)
        return ChainSnapshot(
            market_id=self._conventions.market_id,
            underlying=self._conventions.underlying,
            ts_exchange=self._latest_exchange_stamp(default=now),
            ts_local=now,
            slices=slices,
            stats=_stats_from(slices, self._observations, self._live, now),
        )

    # --- internals

    def _build_slices(self, now: datetime) -> tuple[ChainSlice, ...]:
        require_aware(now, "now")
        by_expiry: dict[datetime, list[tuple[InstrumentId, QuoteObservation]]] = {}
        for instrument, observation in self._observations.items():
            if self._conventions.is_expired(instrument.expiry, now):
                continue
            by_expiry.setdefault(instrument.expiry, []).append((instrument, observation))

        slices: list[ChainSlice] = []
        for expiry in sorted(by_expiry):
            forward = self._forwards.get(expiry)
            if forward is None:
                continue

            quotes = self._quotes_for(by_expiry[expiry], forward, now)
            if not quotes:
                continue

            # Two quotes of the same kind cannot share a strike here: that would be one
            # InstrumentId, hence one dictionary key. flag_slice's duplicate guard protects
            # callers that build a slice by hand, not this one.
            slices.append(
                ChainSlice(
                    expiry=expiry,
                    tenor_years=self._conventions.tenor_years(expiry, now),
                    forward=forward,
                    quotes=quotes,
                    flags=flag_slice(
                        [(quote.strike, quote.kind, quote.mid) for quote in quotes],
                        self._thresholds,
                    ),
                )
            )
        return tuple(slices)

    def _quotes_for(
        self,
        entries: list[tuple[InstrumentId, QuoteObservation]],
        forward: float,
        now: datetime,
    ) -> tuple[ChainQuote, ...]:
        quotes: list[ChainQuote] = []
        for instrument, observation in sorted(entries, key=lambda e: (e[0].strike, e[0].kind)):
            mid = _usable_mid(observation)
            spread_rel = observation.spread_rel
            if mid is None or spread_rel is None:
                continue
            quotes.append(
                ChainQuote(
                    strike=instrument.strike,
                    kind=instrument.kind,
                    mid=mid,
                    spread_rel=spread_rel,
                    age_seconds=_age_seconds(observation, now),
                    # own_iv is None until Market Data can invert a mid with its own
                    # Black-76. The pricing context has one, but importing it would be a
                    # context depending on a context, so the inversion gets duplicated here
                    # in F2 -- intentionally, as the two mean different things. Until then
                    # IV_DIVERGENCE stays dormant.
                    flags=flag_quote(
                        observation=observation,
                        strike=instrument.strike,
                        forward=forward,
                        own_iv=None,
                        thresholds=self._thresholds,
                        now=now,
                    ),
                    exchange_iv=observation.exchange_iv,
                )
            )
        return tuple(quotes)

    def _latest_exchange_stamp(self, default: datetime) -> datetime:
        stamps = [observation.ts_exchange for observation in self._observations.values()]
        return max(stamps) if stamps else default


def _usable_mid(observation: QuoteObservation) -> float | None:
    """The mid, when there is one worth publishing.

    A one-sided book has none, and a market bid and offered at zero has one that is worth
    nothing: both are absences of an observation rather than bad observations, so neither is
    flagged and neither reaches a slice.
    """
    mid = observation.mid
    if mid is None or not math.isfinite(mid) or mid <= 0:
        return None
    return mid


def _age_seconds(observation: QuoteObservation, now: datetime) -> float:
    """Seconds since the venue's stamp, floored at zero to absorb a clock running ahead."""
    return max(0.0, (now - observation.ts_exchange).total_seconds())


def _stats_from(
    slices: tuple[ChainSlice, ...],
    observations: dict[InstrumentId, QuoteObservation],
    live: set[InstrumentId],
    now: datetime,
) -> ChainStats:
    published = [quote for chain_slice in slices for quote in chain_slice.quotes]
    ages = [_age_seconds(observation, now) for observation in observations.values()]
    return ChainStats(
        n_instruments_known=len(live),
        n_quotes_total=len(observations),
        n_quotes_admissible=sum(1 for quote in published if not quote.flags),
        max_age_seconds=max(ages) if ages else 0.0,
        coverage_ratio=len(published) / len(live) if live else 0.0,
    )
