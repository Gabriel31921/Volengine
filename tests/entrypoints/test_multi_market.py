"""Two markets on one bus, end to end: Design 4.7 and the F3-F proof of the design.

``examples/multi-market.toml`` runs ``BTC-DERIBIT`` and ``BTC-SYNTH`` side by side. Here the venue
is replaced by the golden fixture -- the same substitution ``volengine replay`` makes, through the
same ``RecordedProvider`` -- and the synthetic market is generated as it would be live, with its
origin pinned just after the recording's end so nothing reads the wall clock. Everything between
the two feeds and the writer is the production graph.

**What this module proves is a negative, so it asserts it as an identity.** Adding a second market
must change nothing about the first: the venue's snapshots out of the two-market run are compared,
byte for byte, against the ingestion of the venue alone under the same file. And nothing about the
second market may leak into the first market's surface: every surface is anchored on the snapshot
of its own market, at that market's forwards. That nothing in Parametric Pricing or Risk had to
change for this to hold is visible in the diff that added the example rather than in a test.

Three transformations of the file, each stated where it is made. The timers are off, as in any
replay (``build_pipeline(timers=False)``). The movement filter of the synthetic market is opened, so
its handful of cycles produce a snapshot on every cadence tick rather than on the walk of a seeded
forward. And the synthetic feed is wrapped so that its stamps move the one shared clock forward
(``ClockSettingProvider``): a replay's clock is moved by the recording alone, and a second feed
beside it would otherwise never see its cadence elapse. Its timeline starts one second after the
recording ends, so its stamps only ever push the clock past the venue's. The venue's own snapshots
are read at the recorded instants however the two tasks interleave, because ``RecordedProvider``
places the clock immediately before each quote it hands over -- which is what lets the identity
below be asserted at all.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import replace
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from tests.entrypoints.builders import ClockSettingProvider, RecordingBus, RecordingWriter
from tests.entrypoints.test_deribit_fixture import FIXTURE, published_snapshots
from tests.support import RecordingMetrics
from volengine.contracts.events import SnapshotReady, SurfaceCalibrated
from volengine.entrypoints.config import (
    DERIBIT_PROVIDER,
    SYNTHETIC_PROVIDER,
    AppConfig,
    MarketConfig,
    load_config,
)
from volengine.entrypoints.pipeline import (
    build_pipeline,
    default_adapters,
    snapshot_topic,
    surface_topic,
)
from volengine.market_data.adapters.recorded import RecordedProvider, Recording, open_recording
from volengine.parametric_pricing.adapters.scipy_calibrator import PRODUCER_ID as SCIPY_ID
from volengine.platform.clock import SimulatedClock
from volengine.risk.domain.freshness_policy import FreshnessDecision

pytestmark = pytest.mark.e2e

EXAMPLE = Path(__file__).parent.parent.parent / "examples" / "multi-market.toml"
VENUE = "BTC-DERIBIT"
INVENTED = "BTC-SYNTH"

SYNTHETIC_GAP = timedelta(seconds=1)
"""How long after the recording's last quote the synthetic timeline begins."""

SYNTHETIC_CYCLES = 4
SYNTHETIC_INTERVAL_SECONDS = 0.5
"""Four cycles half a second apart -- of real waiting, since the feed awaits its interval: the
cadence of one second elapses once inside them, so the market publishes its first-quote snapshot
and then one resting on the whole chain."""


def last_instant(recording: Recording) -> datetime:
    """The recording's final ``ts_local``, read straight off the file."""
    stamps = [
        json.loads(line)["ts_local"]
        for line in recording.path.read_text(encoding="utf-8").splitlines()[1:]
        if line.strip()
    ]
    return datetime.fromisoformat(stamps[-1])


def configuration(recording: Recording) -> AppConfig:
    config = load_config(EXAMPLE)
    return replace(config, markets=tuple(_adapted(market, recording) for market in config.markets))


def _adapted(market: MarketConfig, recording: Recording) -> MarketConfig:
    if market.market_id != INVENTED:
        return market
    assert market.synthetic is not None
    return replace(
        market,
        snapshot=replace(market.snapshot, material_move_threshold=0.0),
        synthetic=replace(
            market.synthetic,
            config=replace(
                market.synthetic.config,
                cycles=SYNTHETIC_CYCLES,
                interval_seconds=SYNTHETIC_INTERVAL_SECONDS,
            ),
            start=last_instant(recording) + SYNTHETIC_GAP,
        ),
    )


def run_both_markets() -> tuple[RecordingBus, RecordingWriter]:
    recording = open_recording(FIXTURE)
    clock = SimulatedClock(recording.started_at)
    metrics = RecordingMetrics()
    bus = RecordingBus(metrics)
    writer = RecordingWriter()
    adapters = default_adapters()
    synthetic = adapters.providers[SYNTHETIC_PROVIDER]
    adapters = replace(
        adapters,
        providers={
            DERIBIT_PROVIDER: lambda _market: RecordedProvider(recording, clock),
            SYNTHETIC_PROVIDER: lambda market: ClockSettingProvider(synthetic(market), clock),
        },
        writers={**adapters.writers, "console": lambda _risk: writer},
    )
    pipeline = build_pipeline(configuration(recording), adapters, clock, bus, metrics, timers=False)
    asyncio.run(pipeline.run())
    return bus, writer


@pytest.fixture(scope="module")
def session() -> tuple[RecordingBus, RecordingWriter]:
    return run_both_markets()


def snapshots_of(bus: RecordingBus, market_id: str) -> list[SnapshotReady]:
    return [one for one in bus.events_of(SnapshotReady) if one.snapshot.market_id == market_id]


def surfaces_of(bus: RecordingBus, market_id: str, producer_id: str) -> list[SurfaceCalibrated]:
    return [
        one
        for one in bus.events_of(SurfaceCalibrated)
        if one.surface.market_id == market_id and one.surface.producer_id == producer_id
    ]


# --- the file


def test_the_example_names_two_markets_with_conventions_of_their_own() -> None:
    config = load_config(EXAMPLE)

    assert [market.market_id for market in config.markets] == [VENUE, INVENTED]
    assert [market.provider for market in config.markets] == ["deribit", "synthetic"]
    assert {market.conventions.market_id for market in config.markets} == {VENUE, INVENTED}


# --- routing


def test_both_markets_publish_snapshots_onto_their_own_topics(
    session: tuple[RecordingBus, RecordingWriter],
) -> None:
    bus, _ = session

    for market_id in (VENUE, INVENTED):
        topics = {
            topic
            for topic, event in bus.carried
            if isinstance(event, SnapshotReady) and event.snapshot.market_id == market_id
        }
        assert topics == {snapshot_topic(market_id)}


def test_the_producer_publishes_a_surface_for_every_market_on_its_own_topic(
    session: tuple[RecordingBus, RecordingWriter],
) -> None:
    bus, _ = session

    for market_id in (VENUE, INVENTED):
        published = surfaces_of(bus, market_id, SCIPY_ID)
        assert published, market_id
        assert {
            topic
            for topic, event in bus.carried
            if isinstance(event, SurfaceCalibrated) and event in published
        } == {surface_topic(market_id, SCIPY_ID)}


# --- nothing leaks from one market into the other


def test_the_second_market_changes_nothing_the_venue_publishes(
    session: tuple[RecordingBus, RecordingWriter],
) -> None:
    """The venue's snapshots beside the synthetic market are the venue's snapshots alone."""
    bus, _ = session
    alone = load_config(EXAMPLE).markets[0]

    beside = [one.snapshot.to_dict() for one in snapshots_of(bus, VENUE)]

    assert len(beside) > 1
    assert beside == [one.to_dict() for one in published_snapshots(alone)]


def test_each_surface_is_anchored_on_a_snapshot_of_its_own_market(
    session: tuple[RecordingBus, RecordingWriter],
) -> None:
    """Grids per market: the published grid carries its own market's expiries and forwards."""
    bus, _ = session

    for market_id in (VENUE, INVENTED):
        published = {one.snapshot.snapshot_id: one.snapshot for one in snapshots_of(bus, market_id)}
        for event in surfaces_of(bus, market_id, SCIPY_ID):
            source = published[event.surface.source_snapshot_id]
            fitted = {one.expiry: one.forward for one in source.slices}
            for expiry, forward in zip(
                event.surface.grid.expiries, event.surface.grid.forwards, strict=True
            ):
                assert fitted[expiry] == forward


def test_the_two_markets_are_told_apart_by_their_quotes(
    session: tuple[RecordingBus, RecordingWriter],
) -> None:
    """The guard on the anchoring test: the two markets' grids really are different objects.

    Two markets publishing the same forwards would pass the test above with the routing crossed.
    They do not: the venue's are what it quoted on the day and the invented chain's walk from its
    own ``forward0``. Expiries are no help here, and that is worth knowing -- the synthetic
    seven-day slice lands on the venue's own 25SEP26 weekly, both at 08:00 UTC.
    """
    bus, _ = session
    venue = surfaces_of(bus, VENUE, SCIPY_ID)[-1].surface.grid
    invented = surfaces_of(bus, INVENTED, SCIPY_ID)[-1].surface.grid

    assert set(venue.expiries) & set(invented.expiries)
    assert set(venue.forwards).isdisjoint(invented.forwards)


# --- the far end


def test_the_book_is_valued_once_per_market_and_producer(
    session: tuple[RecordingBus, RecordingWriter],
) -> None:
    _, writer = session

    assert {(report.market_id, report.producer_id) for report in writer.reports} == {
        (VENUE, SCIPY_ID),
        (INVENTED, SCIPY_ID),
    }


def test_the_venue_s_report_is_unaffected_by_the_second_market(
    session: tuple[RecordingBus, RecordingWriter],
) -> None:
    """The venue's fitted book is valued off a fresh surface, exactly as when it runs alone."""
    _, writer = session
    venue = [one for one in writer.reports if (one.market_id, one.producer_id) == (VENUE, SCIPY_ID)]

    assert venue[-1].freshness is FreshnessDecision.NORMAL
    assert venue[-1].positions
    assert 0.2 < venue[-1].positions[0].vol < 1.0
