from __future__ import annotations

import asyncio
import time
from datetime import UTC, datetime, timedelta

import pytest

from volengine.platform.clock import Clock, ManualClock, SimulatedClock, SystemClock

TS = datetime(2026, 7, 27, 8, 0, tzinfo=UTC)
NAIVE = datetime(2026, 7, 27, 8, 0)


# --- port conformance
# Structural typing means nothing inherits from Clock, so the only place the shape is
# checked is wherever an implementation is assigned to a Clock. These annotations are
# that place. They only bite once `tests` is added to the mypy paths; until then they
# document the intent, and pipeline.py will enforce it for real in F1-07.


def test_every_implementation_satisfies_the_clock_port() -> None:
    clocks: list[Clock] = [SystemClock(), ManualClock(TS), SimulatedClock(TS)]

    assert all(callable(clock.now) for clock in clocks)


# --- SystemClock


def test_system_clock_returns_an_aware_utc_instant() -> None:
    now = SystemClock().now()

    assert now.tzinfo is not None
    assert now.utcoffset() == timedelta(0)


async def test_system_clock_sleep_actually_waits() -> None:
    started = time.perf_counter()

    await SystemClock().sleep(0.02)

    assert time.perf_counter() - started >= 0.015


# --- ManualClock


def test_manual_clock_rejects_a_naive_start() -> None:
    with pytest.raises(ValueError, match="timezone-aware"):
        ManualClock(NAIVE)


def test_manual_clock_does_not_move_on_its_own() -> None:
    clock = ManualClock(TS)

    assert clock.now() == TS
    assert clock.now() == TS


def test_manual_clock_advances_exactly_what_it_is_told() -> None:
    clock = ManualClock(TS)

    clock.advance(90.5)

    assert clock.now() == TS + timedelta(seconds=90.5)


async def test_manual_clock_sleep_advances_simulated_time() -> None:
    clock = ManualClock(TS)
    started = time.perf_counter()

    await clock.sleep(3600.0)

    assert clock.now() == TS + timedelta(hours=1)
    assert time.perf_counter() - started < 0.05  # an hour of simulated time, no real waiting


async def test_manual_clock_sleep_yields_to_the_event_loop() -> None:
    """This is why `await asyncio.sleep(0)` is inside ManualClock.sleep.

    A coroutine that never awaits anything monopolises the single thread: no other task
    gets a turn. Here the watcher records the clock every time it is scheduled. If sleep
    yields, the two tasks interleave and the watcher sees several distinct instants. If
    the yield is removed, the ticker runs to completion first and the watcher only ever
    sees the final one.
    """
    clock = ManualClock(TS)
    observed: list[datetime] = []

    async def ticker() -> None:
        for _ in range(3):
            await clock.sleep(60.0)

    async def watcher() -> None:
        for _ in range(3):
            observed.append(clock.now())
            await asyncio.sleep(0)

    await asyncio.gather(ticker(), watcher())

    assert len(set(observed)) > 1, "the two tasks never interleaved: sleep did not yield"


# --- SimulatedClock


def test_simulated_clock_rejects_a_naive_start() -> None:
    with pytest.raises(ValueError, match="timezone-aware"):
        SimulatedClock(NAIVE)


def test_simulated_clock_set_replaces_the_instant() -> None:
    clock = SimulatedClock(TS)

    clock.set(TS + timedelta(days=1))

    assert clock.now() == TS + timedelta(days=1)


def test_simulated_clock_set_rejects_a_naive_instant() -> None:
    """set() is the door recorded timestamps come through in F3-B: the least trusted input."""
    clock = SimulatedClock(TS)

    with pytest.raises(ValueError, match="timezone-aware"):
        clock.set(NAIVE)


async def test_simulated_clock_sleep_does_not_move_time() -> None:
    """Under replay the recording drives the clock, not whoever is sleeping."""
    clock = SimulatedClock(TS)

    await clock.sleep(3600.0)

    assert clock.now() == TS
