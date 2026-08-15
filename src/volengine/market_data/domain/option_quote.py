"""Quotes as the Market Data domain sees them: observations, not published data.

These types are internal to the context and are never allowed to cross a boundary. Their
published counterparts in ``contracts/market_snapshot.py`` look similar and are deliberately
not the same types: ``QuoteObservation`` keeps the microstructure a market maker reasons with
-- two sides, sizes, both timestamps -- while ``QuoteData`` publishes a datum that has already
been reduced to a mid and a relative spread. Unifying them into one canonical quote is the
anti-pattern the boundary exists to prevent, so the duplication is the point, not debt.

The domain does not import ``contracts/`` at all: ``QuoteChain.snapshot()`` returns an
internal ``ChainSnapshot`` and the ACL translates it. The ``D`` suffix on the enum below is
there for the one module that does see both -- ``application/acl.py``, whose whole job is the
translation. Two spellings sitting side by side in that mapping read better than aliasing one
of them away at the import, and the suffix makes it obvious which side of the boundary a value
came from at every call site.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum

from volengine.shared_kernel.domain.instants import require_aware


class OptionKindD(StrEnum):
    """Side of the option contract, in the domain's own vocabulary.

    Spelled out in full here, unlike the published ``"C"`` / ``"P"``, because these values are
    read by humans debugging the chain rather than written to a wire. The ACL maps between the
    two; nothing else is allowed to depend on either spelling.
    """

    CALL = "CALL"
    PUT = "PUT"


@dataclass(frozen=True, slots=True)
class InstrumentId:
    """Identity of one option contract, independent of how any venue spells it.

    This is what the chain is keyed by, so it has to be hashable and compared by value: two
    ``InstrumentId`` built from the same four fields are the same instrument, and updates for
    them must land on the same slot. That is what ``frozen=True`` buys, and why the venue's
    symbol string (``"BTC-27JUN25-80000-C"``) is parsed away at the adapter rather than kept
    as the identity -- a symbol is a venue's encoding, and it changes when the venue feels
    like it.
    """

    underlying: str
    """Symbol of the underlying, e.g. ``"BTC"``. Non-empty and not just whitespace."""

    expiry: datetime
    """Exact expiry instant, timezone-aware.

    The instant, not the date: the venue's expiry time of day is part of the identity, and
    resolving it is the conventions' job (ADR-002). Two contracts expiring on the same
    calendar day at different times are different instruments.
    """

    strike: float
    """Absolute strike in the underlying's quote currency. Positive and finite."""

    kind: OptionKindD
    """Call or put. Same strike and expiry on both sides are two distinct instruments."""

    def __post_init__(self) -> None:
        if not self.underlying.strip():
            raise ValueError(f"The underlying can't be empty, got {self.underlying}")
        require_aware(self.expiry, "expiry")
        if not math.isfinite(self.strike) or not self.strike > 0:
            raise ValueError(
                f"The strike for the option must be positive and finite, got {self.strike}"
            )


@dataclass(frozen=True, slots=True)
class QuoteObservation:
    """Top of book for one instrument at one instant, as observed.

    Faithful to what the venue said, including the awkward states: one-sided books, zero
    sizes, a crossed market. None of that is repaired here. Ingestion flags, the calibrator
    weights or excludes -- so this object records, and judgement happens downstream with the
    thresholds that live in configuration (ADR-012).
    """

    bid: float | None
    """Best bid premium, or ``None`` when nobody is bidding. Non-negative and finite.

    ``None`` and ``0.0`` are different states and must not be collapsed: a deep out-of-the-
    money option genuinely bid at zero is information, an empty side is the absence of it.
    """

    ask: float | None
    """Best ask premium, or ``None`` when nobody is offering. Non-negative and finite."""

    bid_size: float
    """Size resting at the bid, in the venue's contract units. Non-negative and finite.

    A float rather than an int because venues quote fractional contract sizes. Zero is legal
    and meaningful -- it is what raises ``LOW_SIZE`` -- so never test it for truthiness.
    """

    ask_size: float
    """Size resting at the ask, same units and same rules as ``bid_size``."""

    ts_exchange: datetime
    """When the venue says this state held. Timezone-aware.

    The economically meaningful timestamp, and the one staleness is measured against.
    """

    ts_local: datetime
    """When we received it. Timezone-aware.

    Its distance from ``ts_exchange`` is our own transport latency, an operational quantity
    that must not be confused with the age of the quote. Under replay it comes from the
    injected clock (ADR-004), which is what makes a recorded session reproduce exactly.

    Both timestamps are required to be aware because staleness is a subtraction of datetimes,
    and mixing an aware one with a naive one raises ``TypeError`` at the worst possible moment.
    """

    exchange_iv: float | None
    """Implied vol the venue published, or ``None``. Strictly positive when present.

    Never an input to the fit: the venue computed it with its own conventions, which are not
    necessarily ours. It is kept as an independent cross-check, and a material disagreement
    with our own inversion is what raises ``IV_DIVERGENCE``.
    """

    def __post_init__(self) -> None:
        if self.bid is not None and (not math.isfinite(self.bid) or self.bid < 0):
            raise ValueError("The bid value must be positive or 0 if it exists")
        if self.ask is not None and (not math.isfinite(self.ask) or self.ask < 0):
            raise ValueError("The ask value must be positive or 0 if it exists")
        if not math.isfinite(self.bid_size) or self.bid_size < 0:
            raise ValueError(f"The bid_size value must be positive or 0, got {self.bid_size}")
        if not math.isfinite(self.ask_size) or self.ask_size < 0:
            raise ValueError(f"The ask_size value must be positive or 0, got {self.ask_size}")
        require_aware(self.ts_exchange, "ts_exchange")
        require_aware(self.ts_local, "ts_local")
        if self.exchange_iv is not None and (
            not math.isfinite(self.exchange_iv) or self.exchange_iv <= 0
        ):
            raise ValueError("The exchange_iv value must be positive if it exists")

    @property
    def mid(self) -> float | None:
        """Arithmetic mid of the two sides, or ``None`` if either side is missing.

        ``None`` propagates on purpose: there is no defensible mid for a one-sided book, and
        inventing one -- last trade, the single live side, a stale mid -- would launder a gap
        in the data into a number the calibrator cannot tell apart from a real observation.
        """
        return (
            self.bid + (self.ask - self.bid) / 2
            if self.bid is not None and self.ask is not None
            else None
        )

    @property
    def spread_rel(self) -> float | None:
        """Full bid-ask spread as a fraction of mid, ``(ask - bid) / mid``.

        This is the microstructure noise scale of the quote and the natural basis for the
        calibration weights: a wide quote deserves less influence on the fit than a tight one.
        The published ``QuoteData.spread_rel`` is this same quantity.

        Returns ``None`` when there is no usable mid, which covers both a one-sided book and a
        market bid and offered at zero. The result is **negative on a crossed market** rather
        than clamped or rejected: crossing is a real state worth flagging (``CROSSED``), and
        hiding the sign here would destroy the evidence.
        """
        mid = self.mid
        if mid is None or mid <= 0:
            return None
        assert self.bid is not None and self.ask is not None
        return (self.ask - self.bid) / mid


@dataclass(frozen=True, slots=True)
class QuoteUpdate:
    """What we know about one instrument as of one exchange message.

    This is the unit ``MarketDataProvider.stream()`` emits and ``QuoteChain.apply()``
    consumes. It is a full replacement of the state of one instrument, never a delta: the
    ticker channel republishes the whole top of book on every tick (see 4.5), so there is
    nothing to accumulate and no ordering to reconstruct. A message lost to a reconnection
    costs freshness, never correctness -- which is what lets the chain survive a resubscribe.
    """

    instrument: InstrumentId
    """Which contract this refers to, already parsed out of the venue's symbol string."""

    observation: QuoteObservation
    """Market state for that contract at ``observation.ts_exchange``."""

    underlying_price: float | None
    """Underlying reference the venue published with this quote, or ``None``.

    Optional because not every ticker message carries one, and the chain keeps the last value
    it saw rather than discarding an otherwise usable update. On Deribit this is the price of
    the synchronous future or perpetual, and it is the primary route to the forward; put-call
    parity is the independent cross-check that feeds ``forward_crosscheck_error``.
    """

    def __post_init__(self) -> None:
        if self.underlying_price is None:
            return
        if not math.isfinite(self.underlying_price) or self.underlying_price <= 0:
            raise ValueError(
                f"The underlying price must be positive if it exists, got {self.underlying_price}"
            )
