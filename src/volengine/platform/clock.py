"""Time as an injected dependency, so a recorded session replays exactly (ADR-004).

Every timestamp in this system is load-bearing. Staleness, latency, freshness policy and
calibration cadence are all subtractions of instants, so code that calls ``datetime.now()``
directly is code whose behaviour cannot be reproduced, cannot be tested at a chosen moment,
and cannot be replayed from a recording. Reading the clock is therefore a port, and the
composition root decides which implementation the contexts get.

Three implementations, for three different jobs: real time in production, time the test
drives by hand, and time a recording drives. They differ only in where ``now()`` comes from
and in what ``sleep()`` means -- which is the whole point, because a use case must not be able
to tell them apart.

Everything here returns timezone-aware UTC, always. ``datetime.utcnow()`` returns a *naive*
datetime despite its name, and subtracting a naive instant from an aware one raises
``TypeError``; the constructors reject naive input rather than let that surface later.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from typing import Protocol


class Clock(Protocol):
    """What a context needs from time: read it, and wait.

    Each context declares its own copy of this protocol; the implementations below satisfy it
    structurally, with no import in either direction. Two methods only -- a richer clock would
    tempt a use case into scheduling logic that belongs to the composition root.
    """

    def now(self) -> datetime:
        """Current instant, always timezone-aware UTC."""
        ...

    async def sleep(self, seconds: float) -> None:
        """Wait, or advance simulated time.

        Deliberately the same call for both. A use case that paused differently under replay
        would no longer be the code being tested.
        """
        ...


class SystemClock:
    """Real time. The production implementation."""

    def now(self) -> datetime:
        return datetime.now(UTC)

    async def sleep(self, seconds: float) -> None:
        await asyncio.sleep(seconds)


class ManualClock:
    """Time the test moves by hand: ``sleep`` advances the clock instead of waiting.

    A pause of an hour costs nothing and happens instantly, so cadence, staleness and timeout
    behaviour become ordinary assertions rather than slow, flaky ones. It also lets the whole
    pipeline run inside a single instant -- which is why ``CalibratedSurface`` permits
    ``ts_calibrated == ts_snapshot``.

    Args:
        instant: Starting time. Must be timezone-aware.
    """

    def __init__(self, instant: datetime) -> None:
        if instant.tzinfo is None:
            raise ValueError(f"The initial moment must be timezone-aware, got naive {instant}")
        self._now = instant

    def now(self) -> datetime:
        return self._now

    def advance(self, seconds: float) -> None:
        """Move time forward. The test's own lever, outside the ``Clock`` protocol."""
        self._now += timedelta(seconds=seconds)

    async def sleep(self, seconds: float) -> None:
        """Advance instantly, then yield once so other coroutines actually get to run.

        The ``asyncio.sleep(0)`` is not decoration: without it this coroutine never suspends,
        and a use case waiting on it would starve every other task on the loop.
        """
        self.advance(seconds)
        await asyncio.sleep(0)


class SimulatedClock:
    """Time driven by a recording: the replay sets it from each event's own timestamp.

    The difference from ``ManualClock`` is who is in charge. There, the test advances time by
    a chosen amount; here, the recorded data dictates it, so the engine relives the exact
    instants of the original session and a replay reproduces the original decisions rather
    than merely similar ones.

    ``sleep`` therefore does nothing but yield -- waiting would be meaningless when time only
    moves on the next recorded event.

    Args:
        instant: Starting time, normally the first event's timestamp. Must be aware.
    """

    def __init__(self, instant: datetime) -> None:
        if instant.tzinfo is None:
            raise ValueError(f"The initial moment must be timezone-aware, got naive {instant}")
        self._now = instant

    def now(self) -> datetime:
        return self._now

    def set(self, time: datetime) -> None:
        """Place the clock at a recorded instant. Not part of the ``Clock`` protocol.

        Only the replay driver calls this. Going backwards is not prevented here, because the
        ordering guarantee belongs to the recording, not to the clock.
        """
        if time.tzinfo is None:
            raise ValueError(f"The initial moment must be timezone-aware, got naive {time}")
        self._now = time

    async def sleep(self, _seconds: float) -> None:
        await asyncio.sleep(0)
