from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Strike:
    value: float

    def __post_init__(self) -> None:
        if not math.isfinite(self.value):
            raise ValueError(f"Strike must exist, got {self.value}")
        if self.value <= 0.0:
            raise ValueError(f"Strike must be positive, got {self.value}")


@dataclass(frozen=True, slots=True)
class Tenor:
    years: float

    def __post_init__(self) -> None:
        if not math.isfinite(self.years):
            raise ValueError(f"Tenor must exist, got {self.years}")
        if self.years <= 0.0:
            raise ValueError(f"Tenor must be greater than 0, got {self.years}")


@dataclass(frozen=True, slots=True)
class Moneyness:
    log_forward: float

    def __post_init__(self) -> None:
        if not math.isfinite(self.log_forward):
            raise ValueError(f"Moneyness must be finite, got {self.log_forward}")
