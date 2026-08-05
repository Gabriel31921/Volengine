from __future__ import annotations

import asyncio
from typing import Protocol

from volengine.contracts.events import Event
from volengine.platform.metrics import MetricsSink


class Subscription(Protocol):
    async def receive(self) -> Event: ...


class EventBus(Protocol):
    def subscribe(self, topic: str, subscriber: str) -> Subscription: ...

    def publish(self, topic: str, event: Event) -> None: ...


class _Mailbox:
    def __init__(self, topic: str, subscriber: str, metrics: MetricsSink) -> None:
        self._slot: Event | None = None
        self._ready = asyncio.Event()
        self._topic = topic
        self._subscriber = subscriber
        self._metrics = metrics

    def put(self, event: Event) -> None:
        if self._slot is not None:
            self._metrics.counter("bus.dropped", topic=self._topic, subscriber=self._subscriber)
        self._slot = event
        self._ready.set()

    async def receive(self) -> Event:
        """
        Each line, in order, first we wait for the flag to go up, then we put it down,
        we take the contents (value) and we empty the mailbox.
        """
        await self._ready.wait()
        self._ready.clear()
        event = self._slot
        self._slot = None
        if event is None:
            raise RuntimeError("mailbox flagged as ready with an empty slot")
        return event


class InProcessConflatingBus:
    def __init__(self, metrics: MetricsSink) -> None:
        self._metrics = metrics
        self._mailboxes: dict[str, list[_Mailbox]] = {}

    def subscribe(self, topic: str, subscriber: str) -> Subscription:
        mailbox = _Mailbox(topic=topic, subscriber=subscriber, metrics=self._metrics)
        self._mailboxes.setdefault(topic, []).append(mailbox)
        return mailbox

    def publish(self, topic: str, event: Event) -> None:
        self._metrics.counter("bus.published", topic=topic)
        for mailbox in self._mailboxes.get(topic, ()):
            mailbox.put(event=event)
