from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from volengine.contracts.calibrated_surface import CalibratedSurface
from volengine.contracts.market_snapshot import MarketSnapshot


@dataclass(frozen=True, slots=True)
class SnapshotReady:
    snapshot: MarketSnapshot


@dataclass(frozen=True, slots=True)
class SurfaceCalibrated:
    surface: CalibratedSurface


@dataclass(frozen=True, slots=True)
class ChainCompositionChanged:
    market_id: str
    ts: datetime
    instruments: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class CalibrationFailed:
    market_id: str
    source_snapshot_id: str
    producer_id: str
    reason: str
    ts: datetime


type Event = SnapshotReady | SurfaceCalibrated | ChainCompositionChanged | CalibrationFailed
