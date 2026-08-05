from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from typing import Protocol


class Clock(Protocol):
    def now(self) -> datetime:
        """Current instant, always timezone-aware UTC."""
        ...

    async def sleep(self, seconds: float) -> None:
        """Wait, or advance simulated time"""
        ...


class SystemClock:
    def now(self) -> datetime:
        return datetime.now(UTC)

    async def sleep(self, seconds: float) -> None:
        await asyncio.sleep(seconds)


class ManualClock:
    def __init__(self, instant: datetime) -> None:
        if instant.tzinfo is None:
            raise ValueError(f"The initial moment must be timezone-aware, got naive {instant}")
        self._now = instant

    def now(self) -> datetime:
        return self._now

    def advance(self, seconds: float) -> None:
        self._now += timedelta(seconds=seconds)

    async def sleep(self, seconds: float) -> None:
        self.advance(seconds)
        await asyncio.sleep(0)


class SimulatedClock:
    def __init__(self, instant: datetime) -> None:
        if instant.tzinfo is None:
            raise ValueError(f"The initial moment must be timezone-aware, got naive {instant}")
        self._now = instant

    def now(self) -> datetime:
        return self._now

    def set(self, time: datetime) -> None:
        if time.tzinfo is None:
            raise ValueError(f"The initial moment must be timezone-aware, got naive {time}")
        self._now = time

    async def sleep(self, _seconds: float) -> None:
        await asyncio.sleep(0)
