"""The subscription loop: what it drives, what it survives, and how it stops.

Small surface, and every test here is about one of the three decisions the class makes -- it
handles, it counts, and it refuses to die on a bad event while still dying on a cancellation.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime

import pytest

from tests.support import RecordingMetrics
from volengine.contracts.events import ChainCompositionChanged, Event
from volengine.platform.bus import InProcessConflatingBus
from volengine.platform.metrics import NullMetricsSink
from volengine.platform.runner import BusRunner

TOPIC = "chain.composition"


def an_event(market_id: str = "BTC-DERIBIT") -> Event:
    return ChainCompositionChanged(
        market_id=market_id,
        ts=datetime(2026, 7, 27, 12, 0, tzinfo=UTC),
        instruments=(),
    )


async def drive(runner: BusRunner, ticks: int = 4) -> None:
    """Run the loop long enough for the pending events to be consumed, then stop it.

    Cancellation is how a runner is stopped, so this is also the shutdown path under test. The
    ``sleep(0)`` yields let the loop's own task run; there is no wall-clock waiting anywhere.
    """
    task = asyncio.create_task(runner.run())
    for _ in range(ticks):
        await asyncio.sleep(0)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task


async def test_an_event_reaches_the_handler() -> None:
    bus = InProcessConflatingBus(NullMetricsSink())
    subscription = bus.subscribe(TOPIC, "svi")
    seen: list[Event] = []

    async def handler(event: Event) -> None:
        seen.append(event)

    bus.publish(TOPIC, an_event())
    await drive(BusRunner(subscription, handler, RecordingMetrics(), "svi"))

    assert len(seen) == 1


async def test_a_handled_event_is_counted() -> None:
    bus = InProcessConflatingBus(NullMetricsSink())
    subscription = bus.subscribe(TOPIC, "svi")
    metrics = RecordingMetrics()

    async def handler(event: Event) -> None:
        return None

    bus.publish(TOPIC, an_event())
    await drive(BusRunner(subscription, handler, metrics, "svi"))

    assert "runner.handled" in metrics.counter_names()


async def test_a_failing_handler_does_not_end_the_loop() -> None:
    """One unanticipated event must not take a market off the air permanently."""
    bus = InProcessConflatingBus(NullMetricsSink())
    subscription = bus.subscribe(TOPIC, "svi")
    seen: list[Event] = []

    async def handler(event: Event) -> None:
        seen.append(event)
        if len(seen) == 1:
            raise RuntimeError("nobody anticipated this")

    task = asyncio.create_task(BusRunner(subscription, handler, RecordingMetrics(), "svi").run())
    bus.publish(TOPIC, an_event())
    for _ in range(4):
        await asyncio.sleep(0)
    bus.publish(TOPIC, an_event("ETH-DERIBIT"))
    for _ in range(4):
        await asyncio.sleep(0)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert len(seen) == 2


async def test_a_failing_handler_is_counted_apart_from_a_successful_one() -> None:
    """A chronically failing consumer has to be visible rather than merely quiet."""
    bus = InProcessConflatingBus(NullMetricsSink())
    subscription = bus.subscribe(TOPIC, "svi")
    metrics = RecordingMetrics()

    async def handler(event: Event) -> None:
        raise RuntimeError("boom")

    bus.publish(TOPIC, an_event())
    await drive(BusRunner(subscription, handler, metrics, "svi"))

    assert "runner.handler_failed" in metrics.counter_names()
    assert "runner.handled" not in metrics.counter_names()


async def test_a_cancelled_runner_stops() -> None:
    """Cancellation is the shutdown path, not an error, so it must not be swallowed."""
    bus = InProcessConflatingBus(NullMetricsSink())
    subscription = bus.subscribe(TOPIC, "svi")

    async def handler(event: Event) -> None:
        return None

    await drive(BusRunner(subscription, handler, RecordingMetrics(), "svi"))


def test_an_unnamed_runner_is_refused() -> None:
    bus = InProcessConflatingBus(NullMetricsSink())

    async def handler(event: Event) -> None:
        return None

    with pytest.raises(ValueError, match="named"):
        BusRunner(bus.subscribe(TOPIC, "svi"), handler, RecordingMetrics(), "")
