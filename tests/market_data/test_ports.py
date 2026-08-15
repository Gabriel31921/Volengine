"""Architectural tests for the driven ports of the Market Data context.

Nothing here asserts business behaviour -- the ports have no behaviour. What these tests keep
standing is a decision: the domain declares what it needs, and the outside world satisfies it
*structurally*, with no inheritance and no import from ``market_data/domain/`` towards
``platform/``. Because nothing inherits from a ``Protocol``, the only places that shape is ever
checked are (a) an assignment of a concrete object to a variable annotated with the port and
(b) a call made through such a variable. This file is both, deliberately: the annotations are
what mypy will catch drift with, and the calls are what pytest catches it with today.

Test modules are free to import ``platform/`` -- the import rules constrain production code.
That freedom is what lets this file prove, here rather than at startup, that the composition
root will actually be able to wire the platform implementations into this context.
"""

from __future__ import annotations

import inspect
import logging
import re
from collections.abc import AsyncIterator
from datetime import datetime

import pytest

from tests.market_data.builders import NOW, make_instrument, make_update
from volengine.market_data.domain import ports
from volengine.market_data.domain.option_quote import (
    InstrumentId,
    QuoteUpdate,
)
from volengine.market_data.domain.ports import Clock, MarketDataProvider, MetricsSink
from volengine.platform.clock import ManualClock, SimulatedClock, SystemClock
from volengine.platform.metrics import LoggingMetricsSink, NullMetricsSink


class FakeProvider:
    """A provider adapter with no network in it, written against the port and nothing else.

    Exactly what a synthetic feed or a recorded replay is: three methods. It inherits from
    nothing, which is the point -- if this satisfies ``MarketDataProvider``, so will the
    websocket adapter, and neither had to know the protocol object exists.
    """

    def __init__(self, instruments: tuple[InstrumentId, ...], updates: tuple[QuoteUpdate, ...]):
        self._instruments = instruments
        self._updates = updates
        self.closed = False

    async def discover(self) -> tuple[InstrumentId, ...]:
        return self._instruments

    async def stream(self) -> AsyncIterator[QuoteUpdate]:
        # An async generator function: declared `async def` with a `yield`, its *call* returns
        # an AsyncIterator without being awaited. That is why the port types `stream` as a
        # plain `def` returning an AsyncIterator.
        for update in self._updates:
            yield update

    async def close(self) -> None:
        self.closed = True


def make_provider() -> FakeProvider:
    return FakeProvider(
        instruments=(make_instrument(60_000.0), make_instrument(65_000.0)),
        updates=(make_update(60_000.0), make_update(65_000.0)),
    )


# --- calls made through the ports
# Annotated with the port, never with the concrete class. These are stand-ins for the use
# cases of F1-06, and the reason a drift in a platform signature fails here.


async def read_after_waiting(clock: Clock, seconds: float) -> datetime:
    await clock.sleep(seconds)
    return clock.now()


def record_ingestion(metrics: MetricsSink) -> None:
    metrics.gauge("md.oldest_quote_age_s", 0.4, underlying="BTC")
    metrics.counter("md.quotes_applied", 2, underlying="BTC")
    metrics.counter("md.quotes_rejected", underlying="BTC")  # relies on the default value=1
    metrics.timing("md.apply_ms", 0.12, underlying="BTC")


async def drain(provider: MarketDataProvider) -> list[QuoteUpdate]:
    collected = [update async for update in provider.stream()]
    await provider.close()
    return collected


# --- Clock


@pytest.mark.parametrize(
    "clock",
    [SystemClock(), ManualClock(NOW), SimulatedClock(NOW)],
    ids=["system", "manual", "simulated"],
)
async def test_every_platform_clock_satisfies_the_market_data_clock_port(clock: Clock) -> None:
    now = await read_after_waiting(clock, 0.0)

    assert now.tzinfo is not None


# --- MetricsSink


def test_null_metrics_sink_satisfies_the_market_data_metrics_port() -> None:
    metrics: MetricsSink = NullMetricsSink()

    record_ingestion(metrics)  # a null sink records nothing anywhere; not raising is the behaviour


def test_logging_metrics_sink_emits_every_measurement_made_through_the_port(
    caplog: pytest.LogCaptureFixture,
) -> None:
    metrics: MetricsSink = LoggingMetricsSink(logging.getLogger("test.md.metrics"))

    with caplog.at_level(logging.INFO, logger="test.md.metrics"):
        record_ingestion(metrics)

    assert len(caplog.records) == 4


# --- MarketDataProvider


async def test_a_provider_discovers_the_full_live_instrument_set() -> None:
    provider: MarketDataProvider = make_provider()

    discovered = await provider.discover()

    assert discovered == (make_instrument(60_000.0), make_instrument(65_000.0))


async def test_a_provider_streams_quote_updates_through_the_port() -> None:
    provider: MarketDataProvider = make_provider()

    streamed = await drain(provider)

    assert streamed == [make_update(60_000.0), make_update(65_000.0)]


async def test_closing_a_provider_releases_it() -> None:
    fake = make_provider()
    provider: MarketDataProvider = fake

    await drain(provider)

    assert fake.closed


def test_stream_is_declared_as_a_plain_call_returning_an_async_iterator() -> None:
    # Not a coroutine function: an async generator's call returns the iterator immediately.
    # Typing it `async def -> AsyncIterator[...]` would demand an await before iterating and
    # silently rule out every adapter written as an async generator.
    assert not inspect.iscoroutinefunction(MarketDataProvider.stream)


# --- the module itself


def test_the_ports_module_does_not_import_asyncio() -> None:
    # Import rule 3: */domain/** never imports asyncio. The ports describe async collaborators
    # without depending on a runtime, which is what keeps the domain independent of how the
    # composition root schedules anything. import-linter enforces this repo-wide in F1-09.
    source = inspect.getsource(ports)

    assert re.search(r"^\s*(import asyncio|from asyncio\b)", source, re.MULTILINE) is None


@pytest.mark.parametrize("port", [MarketDataProvider, Clock, MetricsSink])
def test_the_ports_are_not_runtime_checkable(port: type) -> None:
    # Deliberate: isinstance against a runtime-checkable Protocol compares member *names* only,
    # so it would bless an adapter whose signatures are wrong while reading like a real check.
    # Conformance is verified statically, and by the calls made through the ports above.
    with pytest.raises(TypeError, match="runtime_checkable"):
        isinstance(object(), port)
