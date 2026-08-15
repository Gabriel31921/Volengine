"""In-process event bus with conflation (ADR-003).

Contexts are forbidden from importing each other, so they need an intermediary. The rates
involved are wildly mismatched: the exchange pushes ticker updates every 100 ms while a
calibration takes milliseconds to seconds, so a fast producer will routinely outrun a slow
consumer.

The answer here is a **mailbox of size one with last-write-wins semantics**. Each subscriber
holds exactly one pending event; a second arrival overwrites the first. Dropping messages is
the intended behaviour, not a defect -- a stale snapshot has no value once a newer one exists,
and processing it would cost latency to produce an answer about a market that has moved. An
unbounded queue would let the backlog grow without limit and leave the system busy computing
obsolete results; conflation bounds the lag by construction.

The consequence every consumer must respect: **you cannot assume you saw every message.**
Anything needing correlation uses ``source_snapshot_id``, never arrival order or a count.
Drops are not silent -- each one increments ``bus.dropped``, tagged by topic and subscriber,
so a chronically outrun consumer is visible rather than merely slow.

The interface deliberately does not assume shared memory. Everything it carries is
serializable, which is what would let this be swapped for a real broker without touching a
context -- and what forces the contracts to survive a ``json.dumps`` hop today.
"""

from __future__ import annotations

import asyncio
from typing import Protocol

from volengine.contracts.events import Event
from volengine.platform.metrics import MetricsSink


class Subscription(Protocol):
    """One consumer's end of a topic."""

    async def receive(self) -> Event:
        """Wait for the next event and consume it.

        Returns the *most recent* event, not the oldest pending one -- there is only ever one
        slot. Between two calls, any number of events may have been published and dropped.
        """
        ...


class EventBus(Protocol):
    """What a context needs from a bus, and nothing more.

    A ``Protocol``, so a context depends on the shape rather than on the implementation: a
    test can pass a list-backed fake with no asyncio in sight, and this stays swappable for a
    real broker. Publishing is synchronous and non-blocking by design -- a producer must never
    be slowed down by a slow consumer, which is the same reason the mailbox conflates.
    """

    def subscribe(self, topic: str, subscriber: str) -> Subscription:
        """Open a mailbox on a topic.

        ``subscriber`` is a label, not an identity: it tags the metrics so a drop can be
        attributed to a specific consumer. Two subscribers on one topic each get their own
        mailbox and both see every event they manage to keep up with.
        """
        ...

    def publish(self, topic: str, event: Event) -> None:
        """Deliver to every current subscriber of a topic, overwriting anything pending.

        Never blocks and never fails on a full mailbox -- there is no such thing here.
        Publishing to a topic with no subscribers is legal and does nothing.
        """
        ...


class _Mailbox:
    """A one-slot, last-write-wins inbox for a single subscriber.

    The conflation itself. A slot plus an ``asyncio.Event`` used as a flag, rather than a
    ``Queue(maxsize=1)``, because a bounded queue makes the *producer* wait when it is full --
    exactly the coupling this is meant to remove. Here the producer always wins and the loser
    is the obsolete event.

    Private to this module: consumers see only the ``Subscription`` protocol, which it
    satisfies structurally.
    """

    def __init__(self, topic: str, subscriber: str, metrics: MetricsSink) -> None:
        self._slot: Event | None = None
        self._ready = asyncio.Event()
        self._topic = topic
        self._subscriber = subscriber
        self._metrics = metrics

    def put(self, event: Event) -> None:
        """Overwrite the slot and raise the flag, counting the event we just destroyed.

        A non-empty slot means the consumer never got to the previous event, so ``bus.dropped``
        is incremented before it is lost. That counter is the only evidence conflation
        happened -- without it, a consumer falling permanently behind looks identical to a
        quiet market.
        """
        if self._slot is not None:
            self._metrics.counter("bus.dropped", topic=self._topic, subscriber=self._subscriber)
        self._slot = event
        self._ready.set()

    async def receive(self) -> Event:
        """Wait for the flag, lower it, take the contents and empty the slot -- in that order.

        The order matters. Clearing the flag *before* reading the slot means a ``put`` racing
        this call raises it again and the next ``receive`` returns immediately; clearing it
        after would discard that signal and the consumer would block on an event that is
        already sitting in the slot.

        The final check is a guard on that invariant, not defensive coding: a raised flag over
        an empty slot is impossible unless the protocol has been broken, and failing loudly
        beats returning ``None`` into a consumer typed to expect an ``Event``.
        """
        await self._ready.wait()
        self._ready.clear()
        event = self._slot
        self._slot = None
        if event is None:
            raise RuntimeError("mailbox flagged as ready with an empty slot")
        return event


class InProcessConflatingBus:
    """The single-process implementation: a dict of topics to mailboxes.

    Kafka or Redis would be out of all proportion at this scale, and would add operational
    weight that teaches nothing about the actual problem. What matters is that the *interface*
    stays honest about not sharing memory, so the day this is outgrown, the swap is confined
    to the composition root.

    Not thread-safe, and not meant to be: everything runs on one asyncio event loop, and the
    heavy numerical work is pushed to per-producer thread pools (ADR-005) that communicate
    back through the loop rather than through this bus.

    Satisfies :class:`EventBus` structurally, without inheriting from it.
    """

    def __init__(self, metrics: MetricsSink) -> None:
        self._metrics = metrics
        self._mailboxes: dict[str, list[_Mailbox]] = {}

    def subscribe(self, topic: str, subscriber: str) -> Subscription:
        """Create a mailbox for this subscriber and register it against the topic.

        Topics are created implicitly on first use, so no declaration step is needed and no
        subscription can fail. Subscriptions are permanent for the process lifetime: the
        contexts are wired once at startup and live as long as the engine does.
        """
        mailbox = _Mailbox(topic=topic, subscriber=subscriber, metrics=self._metrics)
        self._mailboxes.setdefault(topic, []).append(mailbox)
        return mailbox

    def publish(self, topic: str, event: Event) -> None:
        """Hand the event to every mailbox on the topic, then return.

        Synchronous, but not blocking: each ``put`` is a slot assignment and a flag, so the
        producer's cost is bounded by the number of subscribers and never by how fast they
        consume. No subscriber is a no-op beyond the counter -- ``bus.published`` is
        incremented regardless, so a topic nobody listens to is visible in the metrics rather
        than invisible in the code.
        """
        self._metrics.counter("bus.published", topic=topic)
        for mailbox in self._mailboxes.get(topic, ()):
            mailbox.put(event=event)
