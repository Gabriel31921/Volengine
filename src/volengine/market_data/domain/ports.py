"""What the Market Data domain needs from the outside world, said in its own words.

These are the *driven* ports of this context: the domain declares the shape of the
collaborators it depends on, and the composition root supplies something that fits. A Deribit
websocket client, a synthetic generator and a recorded replay are three implementations of
``MarketDataProvider``; none of them is imported here, and none of them ever will be. That is
the direction of the dependency arrow in a hexagon -- the outside knows about the inside, not
the reverse -- and it is what makes ingestion testable without a network and reproducible from
a recording.

Everything below is a ``typing.Protocol``, so conformance is **structural**: an adapter is
never asked to inherit from anything, it simply has the right methods. Nothing here is
``@runtime_checkable``, deliberately. ``isinstance`` against a runtime-checkable Protocol only
compares *member names*, so a ``stream`` that returns the wrong element type or a ``now`` with
extra required arguments would pass it happily. That is worse than no check at all, because it
reads like one. The real guard is ``mypy --strict`` at the two places the shape actually
matters -- the composition root, where the concrete object is assigned to the port -- plus a
call through the port in the tests.

``Clock`` and ``MetricsSink`` are duplicated on purpose across contexts rather than imported
from ``platform/``. See their docstrings; the short version is that a port describes a need,
and a need belongs to whoever has it.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from datetime import datetime
from typing import Protocol

from volengine.market_data.domain.option_quote import InstrumentId, QuoteUpdate


class MarketDataProvider(Protocol):
    """The source of quotes: an inventory call, a push stream, and a way to stop.

    Two shapes of interaction, because venues offer two. ``discover`` is pull and answers
    "what exists right now"; ``stream`` is push and answers "what just changed". Keeping them
    separate is what lets the chain distinguish an instrument that died from one that has
    merely gone quiet -- a distinction no single stream of updates can express.

    Implementations live in ``market_data/adapters/``. This module knows nothing about
    websockets, REST, JSON, reconnection or authentication, and must not learn.
    """

    async def discover(self) -> tuple[InstrumentId, ...]:
        """List every instrument the venue currently considers live.

        The REST-style inventory call, made once at startup and then re-run periodically:
        an option chain is not a fixed set. Strikes are born when the underlying moves into
        them and die at every expiry, so a set captured at startup is wrong within hours.

        Returns the **full live set, never a delta**, for the same reason ADR-013 put the whole
        set into ``ChainCompositionChanged``: a delta has to be accumulated, and accumulation
        means one missed message leaves the chain permanently wrong with no way to notice.
        A full set is idempotent -- applying it twice, or applying only the last of five,
        reaches the same state. Its result is what feeds ``QuoteChain.set_live_instruments``,
        and through the ACL, the ``ChainCompositionChanged`` event.

        ``async`` because every real implementation goes over the network. Awaiting it must not
        be assumed to be cheap: the caller decides the rediscovery cadence, from configuration
        (ADR-012), never this port.
        """
        ...

    def stream(self) -> AsyncIterator[QuoteUpdate]:
        """Open the push feed of quote updates and hand back an async iterator over it.

        Note the signature: a plain ``def`` returning an ``AsyncIterator``, not an
        ``async def``. That is what an async generator function looks like from the outside,
        and it means an implementation can be written as ``async def stream(self)`` with
        ``yield`` in the body -- calling it returns the iterator immediately, without an
        ``await``, and the work happens as the consumer iterates. Typing it as
        ``async def ... -> AsyncIterator[...]`` would instead demand a coroutine that must be
        awaited *before* iterating, which is not the shape an async generator has.

        Each ``QuoteUpdate`` is a **full replacement** of one instrument's top of book, not an
        increment. The adapter subscribes to a ticker channel that republishes the entire top
        of book on every tick, so there is nothing to accumulate and no ordering to
        reconstruct. The consequence is the property this whole design leans on: a message lost
        to a reconnection costs **freshness, never correctness**. The chain survives a
        resubscribe, and staleness -- which is measured, flagged and acted on downstream --
        is the only trace such a gap leaves.

        The iterator is expected to be long-lived and to outlast individual connections: an
        adapter that reconnects internally keeps yielding into the same iterator rather than
        ending it. It terminates when the provider is closed, or when a finite source such as a
        replay runs out of recorded events.
        """
        ...

    async def close(self) -> None:
        """Release whatever the implementation is holding, and end ``stream``.

        A websocket adapter owns a connection, a task and a socket; the composition root has to
        be able to shut those down deterministically at the end of a run instead of relying on
        interpreter teardown. It is also what makes a replay *finish*: closing is the signal
        that ends the iterator, so a driver can await the consumer loop and know it will
        return.

        Expected to be idempotent -- closing an already-closed provider is not an error --
        because shutdown paths overlap, and a shutdown that raises is a shutdown that leaks.
        """
        ...


class Clock(Protocol):
    """Time, as this context needs it: read the current instant, and wait.

    Declared here rather than imported from ``platform/`` on purpose, and the duplication is
    the design, not debt. ``market_data/domain/`` states the need; ``platform.clock.SystemClock``
    happens to meet it. Neither module imports the other, in either direction -- structural
    typing is what connects them, and the connection is made exactly once, in the composition
    root, where a ``SystemClock``, a ``ManualClock`` or a ``SimulatedClock`` is passed to a use
    case annotated with *this* protocol.

    What that buys: the domain of this context has zero dependency on ``platform/``, so the
    import rule that forbids it is a rule tooling can check rather than one review has to
    remember. And the ports can diverge. If ingestion ever needs a ``deadline()`` that
    calibration has no use for, it is added here, and only implementations wired into *this*
    context have to grow it. A single shared ``Clock`` would force that method on every context
    at once -- which is how a shared kernel quietly becomes a god object.

    Two methods, and no more, for the same reason ``platform.clock`` keeps two: a richer clock
    invites use cases into scheduling logic that belongs to the composition root.

    Time is a port at all because ADR-004 requires a recorded session to replay exactly. Every
    quantity that matters here -- staleness, transport latency, snapshot cadence -- is a
    subtraction of instants, so a module calling ``datetime.now()`` directly is a module whose
    behaviour cannot be reproduced, tested at a chosen moment, or replayed.
    """

    def now(self) -> datetime:
        """Current instant, always timezone-aware UTC.

        Aware, never naive: subtracting a naive datetime from an aware one raises ``TypeError``,
        and the domain compares this against ``ts_exchange`` constantly. ``datetime.utcnow()``
        returns a naive value despite its name and is banned everywhere in this repo.
        """
        ...

    async def sleep(self, seconds: float) -> None:
        """Wait for a duration, or advance simulated time by it.

        Deliberately the same call under real time, hand-driven time and replay. A use case
        that paused differently under replay would no longer be the code being tested.
        """
        ...


class MetricsSink(Protocol):
    """Where this context's observations go, without it knowing where that is.

    ``logging`` is banned in the domain, and this is the replacement. A logger forces a module
    to pick a format, a level and a destination -- three decisions that have nothing to do with
    ingesting quotes and that a test then has to work around. A sink lets this layer state
    *what* it observed and stop: quotes applied, quotes rejected, staleness, book width,
    forward cross-check error.

    Declared by this context for the same reason as ``Clock``, and satisfied structurally by
    ``platform.metrics.LoggingMetricsSink`` and ``NullMetricsSink`` with no import in either
    direction.

    Tags are the dimensions a measurement is filtered by later -- underlying, expiry, venue,
    reason -- and they are ``str`` so that any backend can carry them. The numbers are the
    experiment: whether ingestion keeps up, and how the two calibrators compare, is answered
    from these series, so they are structured from day one rather than parsed back out of log
    prose.
    """

    def gauge(self, name: str, value: float, **tags: str) -> None:
        """A value that goes up and down: the age of the oldest live quote in the chain."""
        ...

    def counter(self, name: str, value: int = 1, **tags: str) -> None:
        """A value that only grows: how many updates were rejected as inadmissible."""
        ...

    def timing(self, name: str, ms: float, **tags: str) -> None:
        """A duration in milliseconds: exchange timestamp to applied in the chain."""
        ...
