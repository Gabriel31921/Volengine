"""The permanent end-to-end over real venue data: the golden fixture of Design §3, replayed.

``tests/fixtures/deribit-btc-2026-09-18.jsonl`` is thirty seconds of two BTC expiries recorded
from the live venue through this engine's own machinery (``tests/fixtures/README.md`` says how).
Everything upstream of the port -- the symbol grammar, the 08:00 UTC expiry, the inverse premium
multiplied by the forward -- is *inside* the file, so what this module proves is that the engine
turns what the Deribit adapter produced into a snapshot the rest of the pipeline can use: two
slices at the venue's forwards, premiums in the strike's currency, a forward cross-check that the
two independent routes agree on, a surface fitted through it and a book valued off that surface.

The assertions are bounds measured on the fixture and written beside the values they defend in
``examples/deribit-live.toml``, which is the configuration this replay runs under -- so the file a
person would take to the venue is the file the fixture is checked against. Two transformations are
applied to it for the bulk of the assertions. The heartbeat and rediscovery are dropped, there being
no venue to ask and no timer that means anything on a recorded clock. And the movement filter is
opened, so the session publishes on every cadence tick and the cross-check and determinism
assertions below have twenty-odd samples rather than a handful.

The example *as written* -- what ``volengine replay`` actually runs since F3-F, with the timers off
and ``max_quiet_seconds`` still in the policy -- has a section of its own at the end. Before F3-F
the replay dropped ``max_quiet_seconds`` outright, and because the first snapshot rests on the first
quote the venue sent and the filter measures only the baseline that snapshot left, the session
published that one snapshot and nothing after it.

Conflation stands (ADR-003): how many of the published snapshots reach a fit depends on how fast
this machine fits, so every assertion about surfaces and reports is about the *last* one -- which
``Pipeline.run`` drains on a finished stream -- and every count is a floor.
"""

from __future__ import annotations

import math
from dataclasses import replace
from itertools import pairwise
from pathlib import Path

import pytest

from tests.entrypoints.builders import RecordingBus, RecordingWriter
from tests.support import RecordingMetrics
from volengine.contracts.calibrated_surface import SurfaceStatus
from volengine.contracts.events import ChainCompositionChanged, SnapshotReady, SurfaceCalibrated
from volengine.contracts.market_snapshot import MarketSnapshot, OptionKind
from volengine.entrypoints.config import AppConfig, MarketConfig, load_config
from volengine.entrypoints.pipeline import build_pipeline, default_adapters, with_replay
from volengine.market_data.adapters.recorded import RecordedProvider, open_recording
from volengine.market_data.application.build_snapshot import BuildSnapshotUseCase
from volengine.market_data.application.ingest_stream import IngestStreamUseCase
from volengine.market_data.domain.quote_chain import QuoteChain
from volengine.market_data.domain.snapshot_policy import SnapshotPolicy
from volengine.platform.clock import SimulatedClock
from volengine.risk.domain.freshness_policy import FreshnessDecision

pytestmark = pytest.mark.e2e

FIXTURE = Path(__file__).parent.parent / "fixtures" / "deribit-btc-2026-09-18.jsonl"
EXAMPLE = Path(__file__).parent.parent.parent / "examples" / "deribit-live.toml"

MAX_CROSSCHECK_ERROR = 0.001
"""Ten basis points between the venue's forward and the one put-call parity implies. Measured on
the fixture: 1.3 to 2.0 basis points across every snapshot, so this is five times the worst case
and still a tenth of a percent -- a numeraire mistake would show as several percent."""

MAX_QUOTE_AGE_SECONDS = 5.0
"""The receipt stamps are real: the loop kept up with the narrowed chain at a median lag of
1.5 s and a worst of 2.7 s, so an age past this would mean a stamp that is not a receipt."""

EXPECTED_SLICES = 2
"""The two expiries the fixture was narrowed to."""


def replay_configuration() -> AppConfig:
    """The example file, as ``volengine replay`` would run it, with the movement filter open."""
    config = load_config(EXAMPLE)
    return replace(
        config,
        markets=tuple(
            replace(
                market,
                snapshot=replace(
                    market.snapshot, max_quiet_seconds=None, material_move_threshold=0.0
                ),
                rediscovery_seconds=None,
            )
            for market in config.markets
        ),
    )


def replay_through_the_engine() -> tuple[RecordingBus, RecordingWriter]:
    """One full run over the fixture: ingestion, the scipy fit, the risk report."""
    import asyncio

    recording = open_recording(FIXTURE)
    clock = SimulatedClock(recording.started_at)
    metrics = RecordingMetrics()
    bus = RecordingBus(metrics)
    writer = RecordingWriter()
    adapters = with_replay(default_adapters(), recording, clock)
    adapters = replace(adapters, writers={**adapters.writers, "console": lambda _risk: writer})
    pipeline = build_pipeline(replay_configuration(), adapters, clock, bus, metrics)
    asyncio.run(pipeline.run())
    return bus, writer


def published_snapshots(market: MarketConfig | None = None) -> list[MarketSnapshot]:
    """The snapshots alone, through the ingestion loop and nothing after it.

    Ingestion is one task on a simulated clock, so this is deterministic where the full run is
    only deterministic up to conflation; it is what the determinism assertion compares against.
    ``market`` defaults to the opened-filter configuration the bulk of this module runs.
    """
    import asyncio

    market = market if market is not None else replay_configuration().markets[0]
    recording = open_recording(FIXTURE)
    clock = SimulatedClock(recording.started_at)
    chain = QuoteChain(market.conventions, market.admissibility)
    metrics = RecordingMetrics()
    build = BuildSnapshotUseCase(
        chain=chain,
        policy=SnapshotPolicy(market.snapshot),
        clock=clock,
        metrics=metrics,
        max_skew_seconds=market.max_skew_seconds,
    )
    loop = IngestStreamUseCase(
        provider=RecordedProvider(recording, clock),
        chain=chain,
        build_snapshot=build,
        clock=clock,
        metrics=metrics,
        market_id=market.market_id,
    )

    async def drain() -> list[MarketSnapshot]:
        return [event.snapshot async for event in loop.run() if isinstance(event, SnapshotReady)]

    return asyncio.run(drain())


@pytest.fixture(scope="module")
def session() -> tuple[RecordingBus, RecordingWriter]:
    """One replay shared by the assertions below; each reads a different end of it."""
    return replay_through_the_engine()


# --- the snapshots


def test_the_fixture_replays_into_snapshots_of_both_expiries(
    session: tuple[RecordingBus, RecordingWriter],
) -> None:
    bus, _ = session
    snapshots = bus.events_of(SnapshotReady)

    assert len(snapshots) >= 20
    last = snapshots[-1].snapshot
    assert [one.expiry.date().isoformat() for one in last.slices] == ["2026-09-25", "2026-12-25"]


def test_the_universe_is_announced_with_the_recorded_instruments(
    session: tuple[RecordingBus, RecordingWriter],
) -> None:
    bus, _ = session
    announced = bus.events_of(ChainCompositionChanged)[0]

    assert len(announced.instruments) == 250
    assert all(key.startswith("BTC|") for key in announced.instruments)


def test_the_two_routes_to_the_forward_agree_on_real_data(
    session: tuple[RecordingBus, RecordingWriter],
) -> None:
    """ADR-002's sanity check on a venue rather than on a generator.

    Both directions are asserted: the error is bounded, and it is *there* -- a chain whose premiums
    were never multiplied by the forward would fail the first, and a cross-check that silently
    became ``None`` would pass it vacuously.
    """
    bus, _ = session
    errors = [
        event.snapshot.quality.forward_crosscheck_error
        for event in bus.events_of(SnapshotReady)
        if not event.snapshot.quality.degraded
    ]

    assert len(errors) >= 20
    assert all(error is not None and 0.0 < error <= MAX_CROSSCHECK_ERROR for error in errors)


def test_the_premiums_are_in_the_strike_currency(
    session: tuple[RecordingBus, RecordingWriter],
) -> None:
    """A call is worth less than the forward and a put less than its strike, in USD, not BTC."""
    bus, _ = session
    last = bus.events_of(SnapshotReady)[-1].snapshot

    for slice_data in last.slices:
        assert slice_data.forward > 10_000.0
        for quote in slice_data.quotes:
            ceiling = slice_data.forward if quote.kind is OptionKind.CALL else quote.strike
            assert 0.0 < quote.mid < ceiling


def test_the_quotes_carry_real_receipt_stamps(
    session: tuple[RecordingBus, RecordingWriter],
) -> None:
    bus, _ = session
    last = bus.events_of(SnapshotReady)[-1].snapshot

    assert last.quality.max_age_seconds < MAX_QUOTE_AGE_SECONDS
    assert last.quality.coverage_ratio >= 0.8
    assert not last.quality.degraded


def test_the_venue_own_volatility_travels_as_auxiliary_data(
    session: tuple[RecordingBus, RecordingWriter],
) -> None:
    bus, _ = session
    last = bus.events_of(SnapshotReady)[-1].snapshot
    published = [
        quote.exchange_iv
        for slice_data in last.slices
        for quote in slice_data.quotes
        if quote.exchange_iv is not None
    ]

    assert len(published) > 100
    assert all(0.1 < iv < 5.0 for iv in published)


def test_two_replays_publish_the_same_snapshots(
    session: tuple[RecordingBus, RecordingWriter],
) -> None:
    """ADR-004 on real data: everything a snapshot carries descends from the file.

    The full run's snapshots against a second replay through the ingestion loop alone -- the same
    file, the same clock, no fit racing anything -- so the comparison also says that nothing the
    calibrators or the bus did in the full run reached back into what was published.
    """
    bus, _ = session
    first = [event.snapshot.to_dict() for event in bus.events_of(SnapshotReady)]
    second = [snapshot.to_dict() for snapshot in published_snapshots()]

    assert len(first) >= 20
    assert first == second


# --- the far end


def test_the_real_chain_fits_and_is_published(
    session: tuple[RecordingBus, RecordingWriter],
) -> None:
    """Both slices accepted under the example's own tuning, at the RMSE it defends."""
    bus, _ = session
    last = bus.events_of(SurfaceCalibrated)[-1].surface
    acceptance = load_config(EXAMPLE).calibration.acceptance

    assert last.status is SurfaceStatus.OK
    assert len(last.grid.tenors) == EXPECTED_SLICES
    assert last.fit.rmse_vol_bp <= acceptance.max_rmse_vol_bp
    assert all(math.isfinite(vol) and 0.1 < vol < 3.0 for row in last.grid.vols for vol in row)


def test_the_surface_is_anchored_on_the_venue_forwards(
    session: tuple[RecordingBus, RecordingWriter],
) -> None:
    bus, _ = session
    last_snapshot = bus.events_of(SnapshotReady)[-1].snapshot
    last_surface = bus.events_of(SurfaceCalibrated)[-1].surface

    assert last_surface.source_snapshot_id == last_snapshot.snapshot_id
    assert last_surface.grid.forwards == tuple(one.forward for one in last_snapshot.slices)


def test_the_book_is_valued_off_the_real_surface(
    session: tuple[RecordingBus, RecordingWriter],
) -> None:
    _, writer = session
    report = writer.reports[-1]

    assert report.freshness is FreshnessDecision.NORMAL
    assert report.positions
    assert 0.2 < report.positions[0].vol < 1.0


# --- the example as `volengine replay` runs it (F3-F)


def example_market() -> MarketConfig:
    """The example's one market, every threshold as the file states it."""
    return load_config(EXAMPLE).markets[0]


def test_a_replay_under_the_example_s_own_policy_keeps_publishing() -> None:
    """The live-replay stall, closed: the heartbeat's rule runs on the recorded ticks.

    Thirty seconds at ``max_quiet_seconds = 5`` cannot be fewer than six snapshots if the rule is
    honoured on every tick; eleven were measured.
    """
    snapshots = published_snapshots(example_market())

    assert len(snapshots) >= 6


def test_no_quiet_period_of_a_replay_outlasts_the_heartbeat() -> None:
    """Gaps between snapshots are bounded by ``max_quiet_seconds`` plus one gap between ticks.

    The rule is evaluated when a quote arrives, so a snapshot can land up to one inter-tick gap
    after the deadline; the fixture's chain ticks many times a second, so a second is generous.
    """
    market = example_market()
    quiet = market.snapshot.max_quiet_seconds
    assert quiet is not None
    stamps = [snapshot.ts_local for snapshot in published_snapshots(market)]

    gaps = [(later - earlier).total_seconds() for earlier, later in pairwise(stamps)]
    assert gaps
    assert max(gaps) <= quiet + 1.0


def test_without_the_heartbeat_s_rule_the_replay_stalls() -> None:
    """The guard on the two above: what the replay did before F3-F, still reproducible.

    Dropping ``max_quiet_seconds`` from the policy leaves the movement filter measuring the
    one-quote baseline of the first snapshot, and the session publishes nothing after it.
    """
    market = example_market()
    stalled = replace(market, snapshot=replace(market.snapshot, max_quiet_seconds=None))

    assert len(published_snapshots(stalled)) == 1


def test_the_replay_pipeline_runs_no_timer_and_publishes_what_ingestion_does() -> None:
    """``build_pipeline(timers=False)``, which is what ``volengine replay`` builds.

    The full engine over the example as written: the snapshots it publishes are the ingestion
    loop's own, byte for byte, and no heartbeat fired -- so every one of them came off a recorded
    tick at a recorded instant, which is what makes them reproducible.
    """
    import asyncio

    recording = open_recording(FIXTURE)
    clock = SimulatedClock(recording.started_at)
    metrics = RecordingMetrics()
    bus = RecordingBus(metrics)
    writer = RecordingWriter()
    adapters = with_replay(default_adapters(), recording, clock)
    adapters = replace(adapters, writers={**adapters.writers, "console": lambda _risk: writer})
    pipeline = build_pipeline(load_config(EXAMPLE), adapters, clock, bus, metrics, timers=False)

    asyncio.run(pipeline.run())

    published = [event.snapshot.to_dict() for event in bus.events_of(SnapshotReady)]
    assert published == [one.to_dict() for one in published_snapshots(example_market())]
    assert "pipeline.heartbeat.emitted" not in metrics.counter_names()
    assert writer.reports
