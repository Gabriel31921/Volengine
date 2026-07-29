from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from itertools import pairwise
from typing import Any, Final

SCHEMA_VERSION: Final[int] = 1


class SurfaceStatus(StrEnum):
    OK = "OK"
    DEGRADED = "DEGRADED"
    STALE_REPUBLISH = "STALE_REPUBLISH"


@dataclass(frozen=True, slots=True)
class VolGrid:
    log_moneyness: tuple[float, ...]
    tenors: tuple[float, ...]
    vols: tuple[tuple[float, ...], ...]

    def __post_init__(self) -> None:
        tenors_n = len(self.tenors)
        log_moneyness_n = len(self.log_moneyness)
        if tenors_n == 0:
            raise ValueError("The grid must have at least one tenor")
        if log_moneyness_n == 0:
            raise ValueError("The grid must have at least one moneyness node")
        if not all(math.isfinite(k) for k in self.log_moneyness):
            raise ValueError(f"The moneyness axis must be finite, got {self.log_moneyness}")
        if not all(math.isfinite(t) for t in self.tenors):
            raise ValueError(f"The tenor axis must be finite, got {self.tenors}")
        if not all(a < b for a, b in pairwise(self.log_moneyness)):
            raise ValueError("The moneyness axis must be a strictly increasing sequence")
        if not all(a < b for a, b in pairwise(self.tenors)):
            raise ValueError("The tenor axis must be a strictly increasing sequence")
        if tenors_n != len(self.vols):
            raise ValueError(
                "There should be one smile per tenor, got "
                f"{len(self.vols)} smiles for {tenors_n} tenors"
            )
        for i, smile in enumerate(self.vols):
            if len(smile) != log_moneyness_n:
                raise ValueError(
                    "Every smile must have one vol per moneyness node, got "
                    f"{len(smile)} vols for {log_moneyness_n} nodes at tenor {self.tenors[i]}"
                )
            for j, vol in enumerate(smile):
                if vol <= 0 or not math.isfinite(vol):
                    raise ValueError(
                        f"The volatility must be positive and finite, got {vol} at "
                        f"tenor {self.tenors[i]} and moneyness {self.log_moneyness[j]}"
                    )

        # No monotonicity or convexity check on the vols on purpose: no-arbitrage is a model
        # criterion, not a data one. It belongs to the calibrators, as a soft penalty in the
        # loss and as a hard gate before publishing (ADR-010). This contract only guarantees
        # that the structure is coherent.

    def to_dict(self) -> dict[str, Any]:
        return {
            "log_moneyness": list(self.log_moneyness),
            "tenors": list(self.tenors),
            "vols": [list(smile) for smile in self.vols],
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> VolGrid:
        return cls(
            log_moneyness=tuple(raw["log_moneyness"]),
            tenors=tuple(raw["tenors"]),
            vols=tuple(tuple(smile) for smile in raw["vols"]),
        )


@dataclass(frozen=True, slots=True)
class FitMetrics:
    rmse_vol_bp: float
    max_err_vol_bp: float
    n_quotes_used: int
    n_iterations: int
    duration_ms: float

    def __post_init__(self) -> None:
        # Zero is a legitimate value for every error and cost here: a noiseless synthetic
        # chain fits perfectly, a warm start can converge in zero extra iterations, and a
        # sub-microsecond calibration rounds to 0.0 ms. Only negatives are impossible.
        if self.rmse_vol_bp < 0 or not math.isfinite(self.rmse_vol_bp):
            raise ValueError(f"The RMSE must be non-negative and finite, got {self.rmse_vol_bp}")
        if self.max_err_vol_bp < 0 or not math.isfinite(self.max_err_vol_bp):
            raise ValueError(
                f"The maximum vol bp error must be non-negative and finite, "
                f"got {self.max_err_vol_bp}"
            )
        if self.rmse_vol_bp > self.max_err_vol_bp:
            raise ValueError(
                "The RMSE error can't be greater than the maximum error, they are, "
                f"respectively: {self.rmse_vol_bp}, {self.max_err_vol_bp}"
            )
        if self.n_quotes_used <= 0:
            raise ValueError(
                f"The number of quotes used must be positive, got {self.n_quotes_used}"
            )
        if self.n_iterations < 0:
            raise ValueError(
                f"The number of iterations must be non-negative, got {self.n_iterations}"
            )
        if self.duration_ms < 0 or not math.isfinite(self.duration_ms):
            raise ValueError(
                f"The fitting duration (ms) must be non-negative and finite, got {self.duration_ms}"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "rmse_vol_bp": self.rmse_vol_bp,
            "max_err_vol_bp": self.max_err_vol_bp,
            "n_quotes_used": self.n_quotes_used,
            "n_iterations": self.n_iterations,
            "duration_ms": self.duration_ms,
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> FitMetrics:
        return cls(
            rmse_vol_bp=raw["rmse_vol_bp"],
            max_err_vol_bp=raw["max_err_vol_bp"],
            n_quotes_used=raw["n_quotes_used"],
            n_iterations=raw["n_iterations"],
            duration_ms=raw["duration_ms"],
        )


@dataclass(frozen=True, slots=True)
class CalibratedSurface:
    surface_id: str
    source_snapshot_id: str
    market_id: str
    ts_snapshot: datetime
    ts_calibrated: datetime
    producer_id: str  # "svi-scipy", "svi-jax", "neural-torch"

    grid: VolGrid
    fit: FitMetrics
    status: SurfaceStatus
    producer_meta: (
        Mapping[str, float] | None
    )  # Risk musn't depend on this, here goes something as SVI parameters
    schema_version: int = SCHEMA_VERSION

    def __post_init__(self) -> None:
        for name, value in (
            ("surface_id", self.surface_id),
            ("source_snapshot_id", self.source_snapshot_id),
            ("market_id", self.market_id),
            ("producer_id", self.producer_id),
        ):
            if not value:
                raise ValueError(f"The {name} must not be empty")
        _require_aware(self.ts_snapshot, "ts_snapshot")
        _require_aware(self.ts_calibrated, "ts_calibrated")
        # Equality is allowed: under a ManualClock the whole pipeline runs within one tick.
        if self.ts_calibrated < self.ts_snapshot:
            raise ValueError(
                "The surface can't be calibrated before the market was observed, snapshot "
                f"and calibration times are: {self.ts_snapshot}, {self.ts_calibrated}"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "surface_id": self.surface_id,
            "source_snapshot_id": self.source_snapshot_id,
            "market_id": self.market_id,
            "ts_snapshot": self.ts_snapshot.isoformat(),
            "ts_calibrated": self.ts_calibrated.isoformat(),
            "producer_id": self.producer_id,
            "grid": self.grid.to_dict(),
            "fit": self.fit.to_dict(),
            "status": self.status.value,
            "producer_meta": None if self.producer_meta is None else dict(self.producer_meta),
            "schema_version": self.schema_version,
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> CalibratedSurface:
        version = raw.get("schema_version", 1)
        if version > SCHEMA_VERSION:
            raise ValueError(
                f"cannot read schema version {version}, this build supports {SCHEMA_VERSION}"
            )
        meta = raw["producer_meta"]
        return cls(
            surface_id=raw["surface_id"],
            source_snapshot_id=raw["source_snapshot_id"],
            market_id=raw["market_id"],
            ts_snapshot=datetime.fromisoformat(raw["ts_snapshot"]),
            ts_calibrated=datetime.fromisoformat(raw["ts_calibrated"]),
            producer_id=raw["producer_id"],
            grid=VolGrid.from_dict(raw["grid"]),
            fit=FitMetrics.from_dict(raw["fit"]),
            status=SurfaceStatus(raw["status"]),
            producer_meta=None if meta is None else dict(meta),
            schema_version=version,
        )


def _require_aware(ts: datetime, field: str) -> None:
    if ts.tzinfo is None:
        raise ValueError(f"{field} must be timezone-aware, got naive {ts}")
