from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from itertools import pairwise
from typing import Any, Final

SCHEMA_VERSION: Final[int] = 1


class OptionKind(StrEnum):
    CALL = "C"
    PUT = "P"


class QuoteFlag(StrEnum):
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
    strike: float
    kind: OptionKind
    mid: float
    spread_rel: float
    age_seconds: float
    flags: tuple[QuoteFlag, ...]
    exchange_iv: float | None

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
    expiry: datetime
    tenor_years: float
    forward: float
    quotes: tuple[QuoteData, ...]
    flags: tuple[QuoteFlag, ...]

    def __post_init__(self) -> None:
        if self.tenor_years <= 0 or not math.isfinite(self.tenor_years):
            raise ValueError(f"The tenor years must be positive, got {self.tenor_years}")
        if self.forward <= 0 or not math.isfinite(self.forward):
            raise ValueError(f"The forward must be positive, got {self.forward}")
        _require_aware(self.expiry, "expiry")

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
    coverage_ratio: float
    max_age_seconds: float
    n_quotes_admissible: int
    n_quotes_total: int
    forward_crosscheck_error: float | None
    degraded: bool

    def __post_init__(self) -> None:
        if not 0 <= self.coverage_ratio <= 1:
            raise ValueError(
                f"The coverage ratio must be between 0 and 1, got {self.coverage_ratio}"
            )
        if self.n_quotes_admissible > self.n_quotes_total:
            raise ValueError(
                "The number of admissible quotes must be less or equal than the total quotes, ",
                f"they are, respectively: {self.n_quotes_admissible} and {self.n_quotes_total}",
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
    snapshot_id: str
    market_id: str
    ts_exchange: datetime
    ts_local: datetime
    underlying: str
    slices: tuple[SliceData, ...]
    quality: QualityBlock
    schema_version: int = SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not self.market_id:
            raise ValueError(f"The market id of the snapshot must exist, got {self.market_id}")
        _require_aware(self.ts_exchange, "ts_exchange")
        _require_aware(self.ts_local, "ts_local")
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


def _require_aware(ts: datetime, field: str) -> None:
    if ts.tzinfo is None:
        raise ValueError(f"{field} must be timezone-aware, got naive {ts}")
