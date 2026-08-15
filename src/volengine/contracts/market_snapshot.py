"""Published market data contract: the snapshot Market Data puts on the bus.

This module is part of the published language between bounded contexts. It carries data and
nothing else: no pricing, no greeks, no interpolation (ADR-001). Consumers read the numbers
and apply their own methods to them.

Everything convention-dependent -- ``tenor_years``, ``forward``, the exact expiry instant --
is resolved upstream by ``MarketConventions`` (ADR-002), so a calibrator receives homogeneous
numbers and never learns which market produced them. The Market Data domain does not know
this module: it works on its own ``ChainSnapshot``, and ``market_data/application/acl.py``
translates that into the DTOs defined here.

``to_dict`` emits primitives only, and ``from_dict`` rebuilds enums and datetimes, because
the bus is not allowed to assume shared memory (ADR-003): every payload must survive a
``json.dumps`` round trip. ``schema_version`` is checked on read, and a payload newer than
this build understands is refused rather than silently misread.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from itertools import pairwise
from typing import Any, Final

from volengine.shared_kernel.domain.instants import require_aware

SCHEMA_VERSION: Final[int] = 1


class OptionKind(StrEnum):
    """Side of the option contract.

    The member values are the wire format and must never change when a member is renamed,
    which is why they are spelled out instead of using ``auto()``.
    """

    CALL = "C"
    PUT = "P"


class QuoteFlag(StrEnum):
    """Advisory quality marker attached to a quote or to a whole slice.

    Flags are raised by ingestion and are purely descriptive: nothing is dropped here. It is
    the calibrator that decides whether a flagged quote is down-weighted or excluded, because
    hard no-arbitrage filtering is a model criterion, not a data criterion. The thresholds
    that trigger each flag are configuration, not constants (ADR-012).

    The member values are the wire format; see :class:`OptionKind`.
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
class QuoteData:
    """One published option quote, already stripped of exchange conventions.

    This is deliberately not the same type as the ingestion-side ``QuoteObservation``, which
    keeps the microstructure detail (bid, ask, sizes, per-side timestamps). A quote that has
    crossed the boundary is a datum, not an observation, and the two must not be unified.
    """

    strike: float
    """Absolute strike, in the currency the underlying is quoted in. Positive and finite."""

    kind: OptionKind
    """Call or put. Both sides of a strike may be present; the calibrator chooses."""

    mid: float
    """Mid premium, in the market's premium currency. Positive and finite."""

    spread_rel: float
    """Bid-ask spread as a fraction of mid, ``(ask - bid) / mid``. Zero for a locked market.

    This is the microstructure noise scale of the quote, and the natural basis for the
    calibration weights: a wide quote deserves less influence on the fit than a tight one.
    """

    age_seconds: float
    """Seconds since the exchange last updated this quote, measured at the snapshot instant.

    Computed from ``ts_exchange``, so it does not include our own transport latency. The
    staleness threshold that raises ``STALE`` is configuration (ADR-012), not a constant.
    """

    flags: tuple[QuoteFlag, ...]
    """Quality markers for this quote. Advisory only -- see :class:`QuoteFlag`."""

    exchange_iv: float | None
    """Implied vol published by the exchange, when it publishes one; ``None`` otherwise.

    A cross-check, never an input to the fit: it is computed with the exchange's own
    conventions, which are not necessarily ours. A material disagreement with our own
    inversion is what raises ``IV_DIVERGENCE``.
    """

    def __post_init__(self) -> None:
        if self.strike <= 0 or not math.isfinite(self.strike):
            raise ValueError(f"The strike must be positive, got {self.strike}")
        if self.mid <= 0 or not math.isfinite(self.mid):
            raise ValueError(f"The mid must be positive, got {self.mid}")
        if self.spread_rel < 0 or not math.isfinite(self.spread_rel):
            raise ValueError(f"The spread_rel must be positive or 0, got {self.spread_rel}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "strike": self.strike,
            "kind": self.kind.value,
            "mid": self.mid,
            "spread_rel": self.spread_rel,
            "age_seconds": self.age_seconds,
            "flags": [flag.value for flag in self.flags],
            "exchange_iv": self.exchange_iv,
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> QuoteData:
        return cls(
            strike=raw["strike"],
            kind=OptionKind(raw["kind"]),
            mid=raw["mid"],
            spread_rel=raw["spread_rel"],
            age_seconds=raw["age_seconds"],
            flags=tuple(QuoteFlag(flag) for flag in raw["flags"]),
            exchange_iv=raw["exchange_iv"],
        )


@dataclass(frozen=True, slots=True)
class SliceData:
    """All quotes sharing one expiry, plus the two numbers that make them comparable.

    A slice is the unit a calibrator fits: raw SVI is parameterised per slice (ADR-008), so
    the tenor and the forward have to travel with the quotes rather than being looked up.
    """

    expiry: datetime
    """Exact expiry instant, timezone-aware.

    Resolved from the market's conventions before the boundary (ADR-002), including the
    venue's expiry time of day -- 08:00 UTC on Deribit is not the close of a cash session.
    """

    tenor_years: float
    """Year fraction from the snapshot to expiry, under the market's daycount (ADR-002).

    Positive and finite. It is *not* recoverable as ``expiry - ts_exchange`` divided by a
    constant: ACT/365 fixed and a 252-business-day count give different numbers for the same
    two instants. That resolution happened upstream precisely so no consumer redoes it.
    """

    forward: float
    """Forward price of the underlying for this expiry, same currency as ``strike``.

    Positive and finite. Built by the market's own method -- put-call parity, or carry from
    spot with rates and dividends -- which is a domain concept, not an adapter detail
    (ADR-002). It is the reference the log-moneyness axis is measured against, so a wrong
    forward shifts the whole smile without any individual quote looking wrong.
    """

    quotes: tuple[QuoteData, ...]
    """Quotes for this expiry. No ordering is imposed, and coverage may be incomplete."""

    flags: tuple[QuoteFlag, ...]
    """Markers that are properties of the slice as a whole rather than of a single quote.

    ``SLICE_MONOTONICITY`` and ``SLICE_CONVEXITY`` live here: a lone quote cannot violate
    either one, only its relationship with its neighbours can.
    """

    def __post_init__(self) -> None:
        if self.tenor_years <= 0 or not math.isfinite(self.tenor_years):
            raise ValueError(f"The tenor years must be positive, got {self.tenor_years}")
        if self.forward <= 0 or not math.isfinite(self.forward):
            raise ValueError(f"The forward must be positive, got {self.forward}")
        require_aware(self.expiry, "expiry")

    def to_dict(self) -> dict[str, Any]:
        return {
            "expiry": self.expiry.isoformat(),
            "tenor_years": self.tenor_years,
            "forward": self.forward,
            "quotes": [quote.to_dict() for quote in self.quotes],
            "flags": [flag.value for flag in self.flags],
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> SliceData:
        return cls(
            expiry=datetime.fromisoformat(raw["expiry"]),
            tenor_years=raw["tenor_years"],
            forward=raw["forward"],
            quotes=tuple(QuoteData.from_dict(quote) for quote in raw["quotes"]),
            flags=tuple(QuoteFlag(flag) for flag in raw["flags"]),
        )


@dataclass(frozen=True, slots=True)
class QualityBlock:
    """How trustworthy this snapshot is, published alongside it rather than inferred.

    The consumer must be able to judge the input without re-deriving the judgement: a
    calibration that succeeded on a degraded snapshot is a different event from one that
    succeeded on a clean one, and only this block makes the two distinguishable downstream.
    """

    coverage_ratio: float
    """Fraction of the expected quote grid that was actually populated. In ``[0, 1]``."""

    max_age_seconds: float
    """Age of the oldest quote in the snapshot -- the worst case, not the average."""

    n_quotes_admissible: int
    """Quotes that passed ingestion's admissibility checks. Never more than the total."""

    n_quotes_total: int
    """Quotes received for this snapshot, admissible or not."""

    forward_crosscheck_error: float | None
    """Disagreement between the forward from put-call parity and the one built from
    conventions, or ``None`` when it could not be computed (no usable call-put pair).

    This is the sanity check on ADR-002: the two routes to the forward are independent, so
    when they diverge it is our conventions that are wrong, not the market.
    """

    degraded: bool
    """True when the snapshot was published despite failing at least one acceptance rule.

    Publishing it anyway is intentional -- a degraded snapshot is more useful than no
    snapshot, as long as it says so. The thresholds that decide this are TOML configuration
    (ADR-012), so the same snapshot can be degraded under one deployment and not another.
    """

    def __post_init__(self) -> None:
        if not 0 <= self.coverage_ratio <= 1:
            raise ValueError(
                f"The coverage ratio must be between 0 and 1, got {self.coverage_ratio}"
            )

        if self.max_age_seconds < 0:
            raise ValueError(
                f"The maximum quote age cannot be negative, got {self.max_age_seconds}"
            )

        if self.n_quotes_admissible < 0:
            raise ValueError(
                "The number of admissible quotes cannot be negative, "
                f"got {self.n_quotes_admissible}"
            )

        if self.n_quotes_total < 0:
            raise ValueError(
                f"The total number of quotes cannot be negative, got {self.n_quotes_total}"
            )

        if self.n_quotes_admissible > self.n_quotes_total:
            raise ValueError(
                "The number of admissible quotes must be less than or equal to the total "
                f"number of quotes ({self.n_quotes_admissible} > {self.n_quotes_total})."
            )

        if self.forward_crosscheck_error is not None and self.forward_crosscheck_error < 0:
            raise ValueError(
                "The forward cross-check error cannot be negative, "
                f"got {self.forward_crosscheck_error}"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "coverage_ratio": self.coverage_ratio,
            "max_age_seconds": self.max_age_seconds,
            "n_quotes_admissible": self.n_quotes_admissible,
            "n_quotes_total": self.n_quotes_total,
            "forward_crosscheck_error": self.forward_crosscheck_error,
            "degraded": self.degraded,
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> QualityBlock:
        return cls(
            coverage_ratio=raw["coverage_ratio"],
            max_age_seconds=raw["max_age_seconds"],
            n_quotes_admissible=raw["n_quotes_admissible"],
            n_quotes_total=raw["n_quotes_total"],
            forward_crosscheck_error=raw["forward_crosscheck_error"],
            degraded=raw["degraded"],
        )


@dataclass(frozen=True, slots=True)
class MarketSnapshot:
    """A consistent view of one market's option chain at one instant.

    This is the payload of ``SnapshotReady`` and the sole input a calibrator receives. It is
    a snapshot in the strict sense: the quotes were consistent with each other at
    ``ts_exchange``, not assembled from whatever arrived over some window.
    """

    snapshot_id: str
    """Identity of this snapshot, stable across the bus and across recordings.

    Every surface, failure and risk figure derived from it carries this id, which is what
    makes a replay attributable: the bus conflates and drops messages by design (ADR-003),
    so position in a stream proves nothing about which input produced which output.
    """

    market_id: str
    """The market that produced it -- venue plus the convention set applied. Non-empty.

    Not the same thing as ``underlying``: the same underlying on two venues is two markets,
    with different expiry times and daycounts, and their numbers are not interchangeable.
    """

    ts_exchange: datetime
    """Exchange-side instant this snapshot refers to. Timezone-aware.

    This is the timestamp that means something economically, and the one staleness is
    measured from. Awareness is enforced because staleness is a subtraction of datetimes,
    and mixing a naive one with an aware one raises ``TypeError`` at the worst moment.
    """

    ts_local: datetime
    """Instant we assembled the snapshot locally. Timezone-aware.

    Its distance from ``ts_exchange`` is our transport and processing latency, which is an
    operational metric, not a market one. Under replay it is driven by the injected clock
    (ADR-004), which is what makes a recorded session reproduce exactly.
    """

    underlying: str
    """Symbol of the underlying, e.g. ``"BTC"``."""

    slices: tuple[SliceData, ...]
    """Expiries present in this snapshot, strictly increasing by ``expiry``.

    Strict ordering is an invariant, not a convenience: duplicate expiries would mean the
    chain was assembled from two inconsistent reads of the same market.
    """

    quality: QualityBlock
    """Trustworthiness of this snapshot -- see :class:`QualityBlock`."""

    schema_version: int = SCHEMA_VERSION
    """Wire format version, present from day one and checked by :meth:`from_dict`."""

    def __post_init__(self) -> None:
        if not self.market_id:
            raise ValueError(f"The market id of the snapshot must exist, got {self.market_id}")
        require_aware(self.ts_exchange, "ts_exchange")
        require_aware(self.ts_local, "ts_local")
        if not all(a.expiry < b.expiry for a, b in pairwise(self.slices)):
            raise ValueError("slices must be sorted by expiry")

    def to_dict(self) -> dict[str, Any]:
        return {
            "snapshot_id": self.snapshot_id,
            "market_id": self.market_id,
            "ts_exchange": self.ts_exchange.isoformat(),
            "ts_local": self.ts_local.isoformat(),
            "underlying": self.underlying,
            "slices": [SliceData.to_dict(slice_data) for slice_data in self.slices],
            "quality": QualityBlock.to_dict(self.quality),
            "schema_version": self.schema_version,
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> MarketSnapshot:
        """Rebuild a snapshot from its primitive form, refusing a version we cannot read.

        A payload older than this build is accepted -- fields we know how to default are
        defaulted. A payload newer is rejected outright, because the alternative is reading
        an unknown format under an old interpretation and publishing plausible nonsense.

        Raises:
            ValueError: If ``schema_version`` is newer than this build supports.
        """
        version = raw.get("schema_version", 1)
        if version > SCHEMA_VERSION:
            raise ValueError(
                f"cannot read schema version {version}, this build supports {SCHEMA_VERSION}"
            )

        return cls(
            snapshot_id=raw["snapshot_id"],
            market_id=raw["market_id"],
            ts_exchange=datetime.fromisoformat(raw["ts_exchange"]),
            ts_local=datetime.fromisoformat(raw["ts_local"]),
            underlying=raw["underlying"],
            slices=tuple(SliceData.from_dict(slice_data) for slice_data in raw["slices"]),
            quality=QualityBlock.from_dict(raw["quality"]),
            schema_version=version,
        )
