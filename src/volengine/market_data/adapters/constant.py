"""A fixed option chain, priced from one volatility: the walking skeleton's feed.

The first implementation of ``MarketDataProvider`` in the engine, and the least one that proves
the port is usable. There is no venue, no socket, no symbol parsing and no reconnection: a chain
of strikes and expiries is built once at construction and republished on a timer, which is exactly
the shape a ticker channel has -- a **full replacement** of one instrument's top of book, never a
delta -- with everything that makes a real feed hard taken away.

**The premiums are real Black-76 prices, and that is the whole point.** A provider that invented
plausible-looking numbers would be inverted by ``parametric_pricing``'s ACL into whatever
volatilities those numbers happened to imply -- or, far more often, into nothing at all, because a
premium below intrinsic admits no implied volatility and the quote is silently dropped. Pricing
the chain forwards from a single ``vol`` means the inversion has a right answer to recover, the
whole chain survives it, and the flat calibrator downstream fits it to within floating-point
noise. That is what makes the walking skeleton *walk* rather than compose and publish nothing.
It is the same trick D-7 asks of F2-03's synthetic generator, one step simpler: known parameters
in, so a known surface comes out.

**It quotes both legs of every strike.** Only the out-of-the-money one is fitted, but the twin is
what lets Market Data's put-call parity cross-check compute a second, independent forward -- so
the quality block of the published snapshot is exercised rather than left at ``None``.

**Timestamps come from the wall clock, not from the engine's ``Clock`` port.** The registry hands
a provider only its ``MarketConfig`` (ADR-022), so there is no clock to inject, and the analogy
holds anyway: a venue stamps its own messages and the engine reconciles the difference
(``max_skew_seconds``, ADR-021). A deterministic replay does not come from this adapter reading a
different clock, it comes from ``RecordedProvider`` in F3-B replaying recorded stamps. The ``now``
argument exists so a test can pin the chain to an instant; ``docs/SEAMS.md`` records the gap.
"""

from __future__ import annotations

import asyncio
import math
from collections.abc import AsyncIterator, Callable, Sequence
from datetime import UTC, datetime, timedelta

from volengine.market_data.domain.market_conventions import MarketConventions
from volengine.market_data.domain.option_quote import (
    InstrumentId,
    OptionKindD,
    QuoteObservation,
    QuoteUpdate,
)
from volengine.shared_kernel.domain.black76 import price

FORWARD = 60_000.0
"""Underlying reference published with every update, in the quote currency."""

VOL = 0.60
"""The one volatility the whole chain is priced from. Annualised, as a decimal."""

LOG_MONEYNESS = (-0.2, -0.1, 0.0, 0.1, 0.2)
"""Where the strikes sit, as ``ln(K / F)``. Wide enough to be a smile, narrow enough that every
leg -- including the in-the-money one -- stays comfortably invertible."""

TENOR_DAYS = (30.0, 90.0, 365.0)
"""Three expiries, so the published surface has a tenor axis a risk report can interpolate on
rather than a single slice everything is extrapolated from."""

SPREAD_REL = 0.02
"""Full bid-ask spread as a fraction of the mid. Two per cent is tight enough to earn no
``WIDE_SPREAD`` flag under any threshold anyone would configure, and wide enough to be a spread."""

SIZE = 10.0
"""Size resting on both sides, in contracts. Above any plausible ``min_size``."""

LATENCY = timedelta(milliseconds=8)
"""Fixed gap between ``ts_exchange`` and ``ts_local``, standing in for transport latency. A
constant rather than zero so that the two instants are never accidentally interchangeable."""

CYCLES = 5
"""How many times the chain is republished before the stream ends.

Finite, so ``volengine run`` over this provider terminates instead of idling forever -- the
walking skeleton has to be something a person can watch end. More than one because a single burst
arrives inside one cadence window, and the snapshot policy would emit exactly once: it is the
repetition across time, not the number of quotes, that exercises the cadence, the warm start and
the conflating bus.
"""

INTERVAL_SECONDS = 1.0
"""Wall-clock gap between two cycles. Sized to a typical ``cadence_seconds`` so that each cycle
has a chance of producing a snapshot; a shorter one only makes the policy refuse more often."""


def _utc_now() -> datetime:
    """The current instant, aware. Never ``utcnow()``, which returns a naive value."""
    return datetime.now(UTC)


class ConstantProvider:
    """A ``MarketDataProvider`` that republishes one fixed chain a fixed number of times.

    Satisfies the port structurally: no inheritance, three methods, and nothing in this module
    that the domain could import back. The chain is computed once, in the constructor, so every
    cycle yields the same premiums for the same instruments and only the timestamps advance --
    which is what makes it *constant*, and what makes the surface it produces flat and boring by
    construction.
    """

    def __init__(
        self,
        conventions: MarketConventions,
        forward: float = FORWARD,
        vol: float = VOL,
        log_moneyness: Sequence[float] = LOG_MONEYNESS,
        tenor_days: Sequence[float] = TENOR_DAYS,
        spread_rel: float = SPREAD_REL,
        size: float = SIZE,
        cycles: int = CYCLES,
        interval_seconds: float = INTERVAL_SECONDS,
        now: Callable[[], datetime] = _utc_now,
    ) -> None:
        """Build the chain against the venue's conventions and the instant it is built at.

        Args:
            conventions: The market this feed pretends to be. Its ``underlying`` names every
                instrument -- ``QuoteChain`` refuses an update for any other -- its
                ``expiry_time_utc`` places the expiries, and its day count turns them into the
                tenors the premiums are computed at.
            forward: Underlying reference published with every update.
            vol: The volatility every premium is computed from.
            log_moneyness: The strike ladder, as ``ln(K / F)``.
            tenor_days: Calendar days from now to each expiry, before it is snapped to the
                venue's expiry time.
            spread_rel: Full spread as a fraction of the mid, split evenly around it.
            size: Size on both sides of every quote.
            cycles: How many times the whole chain is republished.
            interval_seconds: Wall-clock gap between cycles.
            now: Where the timestamps come from. Injected so a test can pin them; see the module
                docstring on why it is not the ``Clock`` port.

        Raises:
            ValueError: On any argument that could not describe a chain. Plain ``ValueError`` and
                not a ``MarketDataError``: these are construction bugs in the wiring, not market
                conditions anybody catches and recovers from.
        """
        _require_positive_finite(forward, "forward")
        _require_positive_finite(vol, "volatility")
        # Non-negative rather than positive: a zero size is a legal quote and is exactly what
        # raises `LOW_SIZE`, so refusing it here would make one admissibility rule unreachable.
        if not math.isfinite(size) or size < 0:
            raise ValueError(f"The size must be non-negative and finite, got {size}")
        if not log_moneyness:
            raise ValueError("The chain needs at least one strike")
        if not tenor_days:
            raise ValueError("The chain needs at least one expiry")
        if not math.isfinite(spread_rel) or not 0 < spread_rel < 2:
            raise ValueError(
                f"The relative spread must lie strictly inside (0, 2), got {spread_rel}"
            )
        if cycles < 1:
            raise ValueError(f"The provider must publish at least one cycle, got {cycles}")
        if not math.isfinite(interval_seconds) or interval_seconds < 0:
            raise ValueError(
                f"The interval between cycles must be non-negative, got {interval_seconds}"
            )

        self._forward = forward
        self._size = size
        self._cycles = cycles
        self._interval_seconds = interval_seconds
        self._now = now
        self._closed = False
        self._quotes = _chain(
            conventions=conventions,
            start=now(),
            forward=forward,
            vol=vol,
            log_moneyness=log_moneyness,
            tenor_days=tenor_days,
            spread_rel=spread_rel,
        )

    @property
    def instruments(self) -> tuple[InstrumentId, ...]:
        """Every instrument this feed quotes, in the order it publishes them."""
        return tuple(instrument for instrument, _, _ in self._quotes)

    async def discover(self) -> tuple[InstrumentId, ...]:
        """The full live set, which here never changes.

        Fixed for the lifetime of the provider on purpose. The chain is built once so that
        ``discover`` and ``stream`` cannot disagree: a set recomputed against a later instant
        would place its expiries on different dates from the quotes already in the chain, and
        ``QuoteChain`` would then report instruments it has never been quoted for as missing.
        A venue that lists a new strike is F2-03's problem, not this one's.
        """
        return self.instruments

    def stream(self) -> AsyncIterator[QuoteUpdate]:
        """Open the feed. A plain ``def`` returning the iterator, as the port documents."""
        return self._stream()

    async def _stream(self) -> AsyncIterator[QuoteUpdate]:
        """Republish the chain ``cycles`` times, then end.

        The timestamps are read per update rather than per cycle, so the ages the admissibility
        rules measure are the real ones. ``close`` is checked between updates, which is what makes
        the promise in the port -- that closing ends the stream -- true rather than aspirational.
        """
        for cycle in range(self._cycles):
            if cycle > 0 and self._interval_seconds > 0:
                await asyncio.sleep(self._interval_seconds)
            for instrument, bid, ask in self._quotes:
                if self._closed:
                    return
                yield self._update(instrument, bid, ask)

    async def close(self) -> None:
        """Stop the stream at the next update. Idempotent, as the port requires."""
        self._closed = True

    def _update(self, instrument: InstrumentId, bid: float, ask: float) -> QuoteUpdate:
        ts_exchange = self._now()
        return QuoteUpdate(
            instrument=instrument,
            observation=QuoteObservation(
                bid=bid,
                ask=ask,
                bid_size=self._size,
                ask_size=self._size,
                ts_exchange=ts_exchange,
                ts_local=ts_exchange + LATENCY,
                # No `exchange_iv`. The venue's own volatility is an independent cross-check, and
                # a synthetic feed publishing the number its own premiums were built from would
                # make `IV_DIVERGENCE` agree with itself by construction -- a check that cannot
                # fail is worse than an absent one, because it reads like a check.
                exchange_iv=None,
            ),
            underlying_price=self._forward,
        )


_KINDS_OTM_FIRST = {
    False: (OptionKindD.PUT, OptionKindD.CALL),
    True: (OptionKindD.CALL, OptionKindD.PUT),
}
"""Which leg of a strike is quoted first, keyed by whether the strike is at or above the forward.

At the forward exactly the call goes first, matching the tie-break the pricing ACL already makes:
at ``k = 0`` the two legs have identical time value and identical vega, so the choice cannot
matter -- having the same rule on both sides of the boundary is what stops it from mattering by
accident.
"""


def _chain(
    conventions: MarketConventions,
    start: datetime,
    forward: float,
    vol: float,
    log_moneyness: Sequence[float],
    tenor_days: Sequence[float],
    spread_rel: float,
) -> tuple[tuple[InstrumentId, float, float], ...]:
    """Every instrument with the two sides it is quoted at, ordered by expiry then strike.

    Ordered rather than arbitrary because the snapshot policy emits on the *first* update it
    sees: with the near expiry's lowest strike first, the very first snapshot of a session rests
    on an out-of-the-money put, which is the best-conditioned quote in the chain to invert. A
    dictionary-order chain would make the first surface of every run depend on the hash seed.
    """
    quotes: list[tuple[InstrumentId, float, float]] = []
    for days in sorted(tenor_days):
        expiry = conventions.expiry_instant((start + timedelta(days=days)).date())
        tenor_years = conventions.tenor_years(expiry, start)
        for k in sorted(log_moneyness):
            strike = forward * math.exp(k)
            # The out-of-the-money leg first: a put below the forward, a call above it. Same
            # reason as the sort -- the first quote of a session is the one the first snapshot
            # rests on, and the out-of-the-money side is where an inversion is well conditioned.
            for kind in _KINDS_OTM_FIRST[strike >= forward]:
                premium = price(
                    forward=forward,
                    strike=strike,
                    tenor_years=tenor_years,
                    vol=vol,
                    is_call=kind is OptionKindD.CALL,
                )
                quotes.append(
                    (
                        InstrumentId(
                            underlying=conventions.underlying,
                            expiry=expiry,
                            strike=strike,
                            kind=kind,
                        ),
                        premium * (1.0 - spread_rel / 2.0),
                        premium * (1.0 + spread_rel / 2.0),
                    )
                )
    return tuple(quotes)


def _require_positive_finite(value: float, what: str) -> None:
    """Reject anything that is not a usable positive number, NaN included.

    ``isfinite`` first and the bad cases joined with ``or``: ``float("nan") <= 0`` is ``False``,
    so a NaN walks straight through an ordering guard written the other way round -- and a NaN
    forward would produce a whole chain of NaN premiums that every downstream constructor would
    have to catch one at a time.
    """
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"The {what} must be positive and finite, got {value}")
