from __future__ import annotations

import asyncio
from datetime import UTC, datetime

from volengine.contracts.events import CalibrationFailed, Event
from volengine.platform.bus import EventBus, InProcessConflatingBus, Subscription

TS = datetime(2026, 7, 27, 8, 0, tzinfo=UTC)
TOPIC = "snapshot.BTC-DERIBIT"
OTHER_TOPIC = "surface.BTC-DERIBIT.svi-scipy"


class RecordingMetricsSink:
    """Test double. Satisfies MetricsSink structurally and remembers every call."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, str, float, dict[str, str]]] = []

    def gauge(self, name: str, value: float, **tags: str) -> None:
        self.calls.append(("gauge", name, value, tags))

    def counter(self, name: str, value: int = 1, **tags: str) -> None:
        self.calls.append(("counter", name, value, tags))

    def timing(self, name: str, ms: float, **tags: str) -> None:
        self.calls.append(("timing", name, ms, tags))

    def count_of(self, name: str) -> int:
        return sum(int(value) for kind, metric, value, _ in self.calls if metric == name)

    def tags_of(self, name: str) -> list[dict[str, str]]:
        return [tags for _, metric, _, tags in self.calls if metric == name]


def tick(marker: str) -> CalibrationFailed:
    """A cheap event whose `reason` carries a marker, so tests can tell them apart."""
    return CalibrationFailed(
        market_id="BTC-DERIBIT",
        source_snapshot_id="01JZQ0S9K7",
        producer_id="svi-scipy",
        reason=marker,
        ts=TS,
    )


def marker_of(event: Event) -> str:
    assert isinstance(event, CalibrationFailed)
    return event.reason


def make_bus() -> tuple[InProcessConflatingBus, RecordingMetricsSink]:
    metrics = RecordingMetricsSink()
    return InProcessConflatingBus(metrics), metrics


# --- port conformance


def test_the_bus_and_its_subscriptions_satisfy_their_ports() -> None:
    bus: EventBus = InProcessConflatingBus(RecordingMetricsSink())
    subscription: Subscription = bus.subscribe(TOPIC, "calibrator")

    assert callable(subscription.receive)


# --- delivery


async def test_a_subscriber_receives_what_was_published_to_its_topic() -> None:
    bus, _ = make_bus()
    subscription = bus.subscribe(TOPIC, "calibrator")

    bus.publish(TOPIC, tick("first"))

    assert marker_of(await subscription.receive()) == "first"


async def test_an_event_published_before_receive_is_not_lost() -> None:
    """asyncio.Event.wait() returns immediately when the flag is already up.

    Otherwise every event published while the consumer was busy elsewhere would deadlock it.
    """
    bus, _ = make_bus()
    subscription = bus.subscribe(TOPIC, "calibrator")
    bus.publish(TOPIC, tick("published-first"))

    await asyncio.sleep(0)  # let the loop run; nobody is waiting on the mailbox yet

    assert marker_of(await subscription.receive()) == "published-first"


async def test_every_subscriber_of_a_topic_gets_its_own_copy() -> None:
    bus, _ = make_bus()
    calibrator = bus.subscribe(TOPIC, "calibrator")
    risk = bus.subscribe(TOPIC, "risk")

    bus.publish(TOPIC, tick("fan-out"))

    assert marker_of(await calibrator.receive()) == "fan-out"
    assert marker_of(await risk.receive()) == "fan-out"


async def test_topics_are_isolated() -> None:
    bus, _ = make_bus()
    snapshots = bus.subscribe(TOPIC, "calibrator")
    surfaces = bus.subscribe(OTHER_TOPIC, "risk")

    bus.publish(TOPIC, tick("only-for-snapshots"))

    assert marker_of(await snapshots.receive()) == "only-for-snapshots"
    with_timeout = asyncio.wait_for(surfaces.receive(), timeout=0.05)
    try:
        await with_timeout
    except TimeoutError:
        pass
    else:  # pragma: no cover - only runs when the isolation is broken
        raise AssertionError("an event leaked into another topic")


def test_publishing_to_a_topic_with_no_subscribers_is_a_no_op() -> None:
    bus, metrics = make_bus()

    bus.publish("nobody.listens.here", tick("into-the-void"))

    # Still counted: publishing into the void means the pipeline wiring is wrong, and this
    # counter is the only thing that would say so.
    assert metrics.count_of("bus.published") == 1


async def test_receive_blocks_again_after_the_mailbox_is_drained() -> None:
    """The flag has to be cleared on read, or the second receive returns an empty slot."""
    bus, _ = make_bus()
    subscription = bus.subscribe(TOPIC, "calibrator")
    bus.publish(TOPIC, tick("only-one"))

    await subscription.receive()

    try:
        await asyncio.wait_for(subscription.receive(), timeout=0.05)
    except TimeoutError:
        pass
    else:  # pragma: no cover - only runs when the flag is not cleared
        raise AssertionError("receive returned twice for a single published event")


# --- conflation (ADR-003)


async def test_conflation_keeps_only_the_most_recent_event() -> None:
    """The core claim, with no timing involved: five publishes, one delivery, the latest wins."""
    bus, metrics = make_bus()
    subscription = bus.subscribe(TOPIC, "slow-calibrator")

    for index in range(5):
        bus.publish(TOPIC, tick(f"tick-{index}"))

    assert marker_of(await subscription.receive()) == "tick-4"
    assert metrics.count_of("bus.dropped") == 4


async def test_a_slow_consumer_never_processes_a_stale_event() -> None:
    """A producer twice as fast as its consumer. Whatever the interleaving, the consumer
    only ever moves forward: it never processes an event older than one it already saw.
    """
    bus, _ = make_bus()
    subscription = bus.subscribe(TOPIC, "slow-calibrator")
    processed: list[int] = []

    async def producer() -> None:
        for index in range(8):
            bus.publish(TOPIC, tick(f"tick-{index}"))
            await asyncio.sleep(0)

    async def slow_consumer() -> None:
        for _ in range(4):
            event = await subscription.receive()
            processed.append(int(marker_of(event).removeprefix("tick-")))
            await asyncio.sleep(0)  # stands in for the calibration

    await asyncio.gather(producer(), slow_consumer())

    assert processed == sorted(processed), f"processed a stale event: {processed}"
    assert len(set(processed)) == len(processed), f"processed the same event twice: {processed}"


async def test_drops_are_attributed_to_the_subscriber_that_fell_behind() -> None:
    bus, metrics = make_bus()
    slow = bus.subscribe(TOPIC, "slow-calibrator")
    fast = bus.subscribe(TOPIC, "fast-risk")

    bus.publish(TOPIC, tick("first"))
    await fast.receive()  # only the fast one drains its mailbox
    bus.publish(TOPIC, tick("second"))

    assert metrics.tags_of("bus.dropped") == [
        {"topic": TOPIC, "subscriber": "slow-calibrator"}
    ]
    assert marker_of(await slow.receive()) == "second"


def test_published_is_counted_once_per_publish_and_tagged_with_the_topic() -> None:
    bus, metrics = make_bus()
    bus.subscribe(TOPIC, "calibrator")
    bus.subscribe(TOPIC, "risk")

    bus.publish(TOPIC, tick("one-publish"))

    assert metrics.count_of("bus.published") == 1, "counted per subscriber instead of per publish"
    assert metrics.tags_of("bus.published") == [{"topic": TOPIC}]
