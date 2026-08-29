"""The ingestion loop: what it emits, in what order, and what it does when the stream ends.

The only asynchronous tests in this context, and they need no event loop management of their own
-- ``asyncio_mode = "auto"`` in ``pyproject.toml`` runs a coroutine test directly. There is still
no waiting anywhere: the provider is a scripted generator and the clock is manual, so the whole
loop runs to completion in microseconds and always in the same order.
"""

from __future__ import annotations

import pytest

from tests.market_data.builders import (
    FORWARD,
    NOW,
    StubProvider,
    make_chain,
    make_instrument,
    make_snapshot_policy,
    make_update,
)
from tests.support import RecordingMetrics
from volengine.contracts.events import ChainCompositionChanged, Event, SnapshotReady
from volengine.market_data.application.build_snapshot import BuildSnapshotUseCase
from volengine.market_data.application.ingest_stream import IngestStreamUseCase
from volengine.market_data.domain.option_quote import OptionKindD, QuoteUpdate
from volengine.platform.clock import ManualClock


def make_loop(
    provider: StubProvider,
    cadence_seconds: float = 1.0,
) -> tuple[IngestStreamUseCase, RecordingMetrics]:
    chain = make_chain()
    clock = ManualClock(NOW)
    metrics = RecordingMetrics()
    build = BuildSnapshotUseCase(
        chain=chain,
        policy=make_snapshot_policy(cadence_seconds=cadence_seconds),
        clock=clock,
        metrics=metrics,
        max_skew_seconds=30.0,
    )
    loop = IngestStreamUseCase(
        provider=provider,
        chain=chain,
        build_snapshot=build,
        clock=clock,
        metrics=metrics,
        market_id="BTC-DERIBIT",
    )
    return loop, metrics


async def drain(loop: IngestStreamUseCase) -> list[Event]:
    return [event async for event in loop.run()]


def two_sided() -> list[QuoteUpdate]:
    """One two-sided strike, which is the least that produces a publishable slice."""
    return [make_update(kind=kind) for kind in (OptionKindD.CALL, OptionKindD.PUT)]


# --- what comes out


async def test_the_universe_is_announced_before_any_quote() -> None:
    """A consumer sizing fixed shapes (ADR-009) cannot use the event after the snapshot it sizes."""
    provider = StubProvider(updates=two_sided(), instruments=[make_instrument()])
    loop, _ = make_loop(provider)

    events = await drain(loop)

    assert isinstance(events[0], ChainCompositionChanged)


async def test_a_publishable_chain_yields_a_snapshot() -> None:
    provider = StubProvider(updates=two_sided(), instruments=[make_instrument()])
    loop, _ = make_loop(provider)

    events = await drain(loop)

    assert any(isinstance(event, SnapshotReady) for event in events)


async def test_the_cadence_holds_inside_a_burst_of_updates() -> None:
    """The clock never advances here, so every tick after the first is inside the cadence."""
    updates = two_sided() + [make_update(bid=0.06, ask=0.064) for _ in range(5)]
    provider = StubProvider(updates=updates, instruments=[make_instrument()])
    loop, _ = make_loop(provider, cadence_seconds=10.0)

    events = await drain(loop)

    assert sum(isinstance(event, SnapshotReady) for event in events) == 1


async def test_an_instrument_born_mid_session_is_announced() -> None:
    """A strike listed between two discovery polls still trades, and the chain accepts it."""
    provider = StubProvider(
        updates=[make_update(strike=FORWARD + 5_000.0)],
        instruments=[make_instrument()],
    )
    loop, _ = make_loop(provider)

    events = await drain(loop)

    assert sum(isinstance(event, ChainCompositionChanged) for event in events) == 2


async def test_a_known_instrument_ticking_again_announces_nothing() -> None:
    """The vacuous-pass guard: every update must not look like a composition change."""
    provider = StubProvider(updates=two_sided() * 3, instruments=[make_instrument()])
    loop, _ = make_loop(provider)

    events = await drain(loop)

    assert sum(isinstance(event, ChainCompositionChanged) for event in events) == 2


async def test_the_second_composition_event_carries_the_whole_set() -> None:
    """Never a delta: a consumer that missed the first event has to be correct from this one."""
    provider = StubProvider(
        updates=[make_update(strike=FORWARD + 5_000.0)],
        instruments=[make_instrument()],
    )
    loop, _ = make_loop(provider)

    events = await drain(loop)
    latest = [event for event in events if isinstance(event, ChainCompositionChanged)][-1]

    assert len(latest.instruments) == 2


# --- lifetime


async def test_the_provider_is_closed_when_the_stream_ends() -> None:
    provider = StubProvider(updates=two_sided(), instruments=[make_instrument()])
    loop, _ = make_loop(provider)

    await drain(loop)

    assert provider.closed is True


async def test_the_provider_is_closed_when_an_update_is_refused() -> None:
    """A wiring bug must fail loudly and still not leak the venue connection."""
    provider = StubProvider(updates=[make_update(underlying="ETH")], instruments=[])
    loop, _ = make_loop(provider)

    with pytest.raises(ValueError, match="chain covers"):
        await drain(loop)

    assert provider.closed is True


async def test_an_update_for_another_underlying_is_not_swallowed() -> None:
    """One chain covers one underlying; mixing two corrupts every slice-level rule silently."""
    provider = StubProvider(updates=[make_update(underlying="ETH")], instruments=[])
    loop, _ = make_loop(provider)

    with pytest.raises(ValueError, match="chain covers"):
        await drain(loop)


# --- configuration


def test_an_empty_market_id_is_refused_at_construction() -> None:
    provider = StubProvider(updates=[], instruments=[])
    chain = make_chain()
    clock = ManualClock(NOW)
    metrics = RecordingMetrics()
    build = BuildSnapshotUseCase(
        chain=chain,
        policy=make_snapshot_policy(),
        clock=clock,
        metrics=metrics,
        max_skew_seconds=30.0,
    )

    with pytest.raises(ValueError, match="market id"):
        IngestStreamUseCase(
            provider=provider,
            chain=chain,
            build_snapshot=build,
            clock=clock,
            metrics=metrics,
            market_id="",
        )


async def test_applied_updates_are_counted() -> None:
    provider = StubProvider(updates=two_sided(), instruments=[make_instrument()])
    loop, metrics = make_loop(provider)

    await drain(loop)

    assert metrics.counter_names().count("marketdata.update.applied") == 2
