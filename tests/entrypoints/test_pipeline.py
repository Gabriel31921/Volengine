"""The composition root: what reaches whom, on which topic, and how a run stops.

These are the first tests in the repository that exercise every context at once, which is the
point of the module they cover -- a quote enters through Market Data's provider port and leaves as
a number in a Risk report, having crossed the bus twice as a published DTO. Nothing between those
two ends is faked: the ACLs, the policies and the use cases are the real ones, and the stubs sit
at the edges, where the infrastructure would be.

No waiting anywhere. The clock is manual, the provider is a scripted generator and the stopping
rule is a report count, so every test below runs in milliseconds and always in the same order.
"""

from __future__ import annotations

import asyncio

import pytest

from tests.entrypoints.builders import (
    CALIBRATOR_NAME,
    MARKET_ID,
    BlockingProvider,
    RecordingWriter,
    SlowCalibrator,
    make_adapters,
    make_app_config,
    make_calibration_config,
    make_market_config,
    make_position,
    make_risk_config,
    one_instrument,
    two_sided,
)
from tests.market_data.builders import NOW, StubProvider
from tests.parametric_pricing.builders import StubCalibrator, make_market_snapshot
from tests.risk.builders import make_calibrated_surface
from tests.support import RecordingMetrics
from volengine.contracts.events import (
    CalibrationFailed,
    ChainCompositionChanged,
    SnapshotReady,
    SurfaceCalibrated,
)
from volengine.entrypoints.config import AppConfig, ConfigError
from volengine.entrypoints.pipeline import (
    Adapters,
    Pipeline,
    build_pipeline,
    composition_topic,
    default_adapters,
    failure_topic,
    snapshot_topic,
    surface_topic,
    topic_of,
)
from volengine.parametric_pricing.domain.errors import CalibrationError
from volengine.platform.bus import InProcessConflatingBus
from volengine.platform.clock import ManualClock
from volengine.risk.domain.freshness_policy import FreshnessDecision


def make_pipeline(
    provider: StubProvider | BlockingProvider,
    calibrators: dict[str, StubCalibrator] | None = None,
    max_quiet_seconds: float | None = None,
    material_move_threshold: float = 0.0,
) -> tuple[RecordingWriter, RecordingMetrics, Pipeline]:
    """The whole engine on fakes: one market, one book, whatever calibrators were asked for."""
    held: dict[str, StubCalibrator] = (
        {CALIBRATOR_NAME: StubCalibrator()} if calibrators is None else dict(calibrators)
    )
    writer = RecordingWriter()
    metrics = RecordingMetrics()
    config = make_app_config(
        markets=(
            make_market_config(
                max_quiet_seconds=max_quiet_seconds,
                material_move_threshold=material_move_threshold,
            ),
        ),
        calibration=make_calibration_config(calibrators=tuple(held)),
        risk=make_risk_config(),
    )
    pipeline = build_pipeline(
        config,
        make_adapters(provider, dict(held), writer),
        ManualClock(NOW),
        InProcessConflatingBus(metrics),
        metrics,
    )
    return writer, metrics, pipeline


def build_from(config: AppConfig) -> Pipeline:
    """Build against a registry that knows the stub names and nothing else."""
    return build_pipeline(
        config,
        make_adapters(
            StubProvider(updates=()), {CALIBRATOR_NAME: StubCalibrator()}, RecordingWriter()
        ),
        ManualClock(NOW),
        InProcessConflatingBus(RecordingMetrics()),
        RecordingMetrics(),
    )


# --- routing


def test_a_snapshot_is_routed_by_the_market_it_describes() -> None:
    event = SnapshotReady(snapshot=make_market_snapshot(market_id="ETH-DERIBIT"))

    assert topic_of(event) == snapshot_topic("ETH-DERIBIT")


def test_a_composition_change_is_routed_by_its_market() -> None:
    event = ChainCompositionChanged(market_id=MARKET_ID, ts=NOW, instruments=())

    assert topic_of(event) == composition_topic(MARKET_ID)


def test_a_surface_is_routed_by_market_and_producer() -> None:
    surface = make_calibrated_surface()

    assert topic_of(SurfaceCalibrated(surface=surface)) == surface_topic(
        surface.market_id, surface.producer_id
    )


def test_a_failure_does_not_share_the_surface_topic() -> None:
    """Conflation: on one topic the republished surface would overwrite the failure (ADR-003)."""
    event = CalibrationFailed(
        market_id=MARKET_ID,
        source_snapshot_id="BTC-DERIBIT:00000000",
        producer_id="svi-stub",
        reason="refused",
        ts=NOW,
    )

    assert topic_of(event) == failure_topic(MARKET_ID, "svi-stub")
    assert topic_of(event) != surface_topic(MARKET_ID, "svi-stub")


# --- what the graph does, end to end


async def test_a_quote_becomes_a_report() -> None:
    """The proof the architecture closes: in through the provider port, out through the writer.

    Driven by a calibrator that blocks its thread for a real millisecond, and that is the whole
    point of the fake. A stub returning inside the same event-loop iteration lets the entire chain
    complete before the finite stream ends, so the assertion would hold whatever the run did at
    shutdown -- it would be a statement about the scheduler, not about the pipeline. With the fit
    still on its pool when the last quote arrives, the report exists only if the run settles what
    is in flight before stopping.
    """
    writer, _, pipeline = make_pipeline(
        StubProvider(updates=two_sided(), instruments=one_instrument()),
        calibrators={CALIBRATOR_NAME: SlowCalibrator()},
    )

    await asyncio.wait_for(pipeline.run(), timeout=5.0)

    assert writer.reports


async def test_a_slow_calibration_still_reaches_the_report_goal() -> None:
    """``volengine report`` over a finite chain: the goal is met during the settling, not before.

    The run ends because ingestion ran out, with the fit still on the pool and the goal unmet, so
    a shutdown that cancelled the consumers there would return successfully having written
    nothing -- which the caller cannot tell from a market that produced no report.
    """
    writer, _, pipeline = make_pipeline(
        StubProvider(updates=two_sided(), instruments=one_instrument()),
        calibrators={CALIBRATOR_NAME: SlowCalibrator()},
    )

    await asyncio.wait_for(pipeline.run(max_reports=1), timeout=5.0)

    assert len(writer.reports) == 1


async def test_a_slow_calibration_is_still_published_before_the_run_ends() -> None:
    """The hop between the two: the surface reaches the bus, not only the calibrator's return."""
    _, metrics, pipeline = make_pipeline(
        StubProvider(updates=two_sided(), instruments=one_instrument()),
        calibrators={CALIBRATOR_NAME: SlowCalibrator()},
    )

    await asyncio.wait_for(pipeline.run(), timeout=5.0)

    assert metrics.counter_names().count("runner.handled") == 2


async def test_the_report_is_valued_off_the_producer_that_published() -> None:
    writer, _, pipeline = make_pipeline(
        StubProvider(updates=two_sided(), instruments=one_instrument()),
        calibrators={CALIBRATOR_NAME: StubCalibrator(producer_id="svi-stub")},
    )

    await pipeline.run()

    assert writer.reports[-1].producer_id == "svi-stub"


async def test_a_report_off_a_fresh_surface_carries_positions() -> None:
    """The vacuous-pass guard: a refused report is written too, and holds nothing at all."""
    writer, _, pipeline = make_pipeline(
        StubProvider(updates=two_sided(), instruments=one_instrument())
    )

    await pipeline.run()

    assert writer.reports[-1].freshness is FreshnessDecision.NORMAL
    assert writer.reports[-1].positions


async def test_each_producer_reports_off_its_own_surface() -> None:
    """Design 7.3: one book, two producers, two reports -- never one producer's twice."""
    writer, _, pipeline = make_pipeline(
        StubProvider(updates=two_sided(), instruments=one_instrument()),
        calibrators={
            "svi-a": StubCalibrator(producer_id="svi-a"),
            "svi-b": StubCalibrator(producer_id="svi-b"),
        },
    )

    await pipeline.run()

    assert {report.producer_id for report in writer.reports} == {"svi-a", "svi-b"}


async def test_the_provider_is_closed_when_the_run_ends() -> None:
    provider = StubProvider(updates=two_sided(), instruments=one_instrument())
    _, _, pipeline = make_pipeline(provider)

    await pipeline.run()

    assert provider.closed


async def test_a_blocked_provider_is_closed_when_the_goal_ends_the_run() -> None:
    """Cancellation is how a live session stops, and it must still release the feed."""
    provider = BlockingProvider(updates=two_sided(), instruments=one_instrument())
    _, _, pipeline = make_pipeline(provider)

    await pipeline.run(max_reports=1)

    assert provider.closed


async def test_a_run_stops_once_the_report_goal_is_met() -> None:
    """What ``volengine report`` asks for: the stream never ends and the run returns anyway."""
    writer, _, pipeline = make_pipeline(
        BlockingProvider(updates=two_sided(), instruments=one_instrument())
    )

    await pipeline.run(max_reports=1)

    assert len(writer.reports) == 1


async def test_a_refused_calibration_does_not_stop_the_engine() -> None:
    """A producer that fails is a data point, not an outage (ADR-006, Design 8.2)."""
    writer, metrics, pipeline = make_pipeline(
        StubProvider(updates=two_sided(), instruments=one_instrument()),
        calibrators={CALIBRATOR_NAME: StubCalibrator(failure=CalibrationError("no chance"))},
    )

    await pipeline.run()

    assert writer.reports == []
    assert "runner.handled" in metrics.counter_names()


# --- the heartbeat, which is the seam this task closes


async def test_a_silent_feed_still_produces_a_second_report() -> None:
    """``max_quiet_seconds`` covering a feed that stopped, not only a market that is not moving.

    The movement threshold is set beyond anything the script can produce and the provider stops
    yielding, so the only route to a second snapshot is the timer racing the stream.
    """
    writer, _, pipeline = make_pipeline(
        BlockingProvider(updates=two_sided(), instruments=one_instrument()),
        max_quiet_seconds=10.0,
        material_move_threshold=10.0,
    )

    await pipeline.run(max_reports=2)

    assert len(writer.reports) == 2


async def test_the_heartbeat_waits_out_the_quiet_period() -> None:
    """The guard on the test above: a second report that arrived at once would prove nothing.

    Under a ``ManualClock`` the heartbeat's own ``sleep`` is the only thing moving time, so the gap
    between the two reports is the interval the policy insisted on.
    """
    writer, _, pipeline = make_pipeline(
        BlockingProvider(updates=two_sided(), instruments=one_instrument()),
        max_quiet_seconds=10.0,
        material_move_threshold=10.0,
    )

    await pipeline.run(max_reports=2)

    elapsed = writer.reports[1].ts_report - writer.reports[0].ts_report
    assert elapsed.total_seconds() >= 10.0


async def test_the_heartbeat_is_counted_where_it_fires() -> None:
    """Silence rescued by a timer has to be visible, or the seam closes invisibly."""
    _, metrics, pipeline = make_pipeline(
        BlockingProvider(updates=two_sided(), instruments=one_instrument()),
        max_quiet_seconds=10.0,
        material_move_threshold=10.0,
    )

    await pipeline.run(max_reports=2)

    assert "pipeline.heartbeat.emitted" in metrics.counter_names()


async def test_a_run_without_a_heartbeat_still_ends() -> None:
    """With no timer nothing advances a manual clock, and a finite stream must still finish."""
    writer, _, pipeline = make_pipeline(
        StubProvider(updates=two_sided(), instruments=one_instrument()),
        max_quiet_seconds=None,
    )

    await asyncio.wait_for(pipeline.run(), timeout=5.0)

    assert len(writer.reports) == 1


async def test_a_market_values_only_the_book_written_on_its_underlying() -> None:
    """Nothing downstream can catch this one.

    ``position_risk`` never compares a position's underlying against the surface it is valued on
    -- a ``CalibratedSurface`` carries a market and no underlying (``docs/SEAMS.md``) -- so a book
    handed whole to every market would price the ETH leg off the BTC smile and print a number.
    The second market here never publishes anything, so the only report that exists is BTC's, and
    it must have one line rather than two.
    """
    quoting = StubProvider(updates=two_sided(), instruments=one_instrument())
    silent = StubProvider(updates=())
    writer = RecordingWriter()
    metrics = RecordingMetrics()
    config = make_app_config(
        markets=(
            make_market_config(),
            make_market_config(market_id="ETH-DERIBIT", underlying="ETH", provider="silent"),
        ),
        risk=make_risk_config(positions=(make_position(), make_position(underlying="ETH"))),
    )
    adapters = make_adapters(quoting, {CALIBRATOR_NAME: StubCalibrator()}, writer)
    pipeline = build_pipeline(
        config,
        Adapters(
            providers={**adapters.providers, "silent": lambda _config: silent},
            calibrators=adapters.calibrators,
            writers=adapters.writers,
        ),
        ManualClock(NOW),
        InProcessConflatingBus(metrics),
        metrics,
    )

    await asyncio.wait_for(pipeline.run(), timeout=5.0)

    assert [report.market_id for report in writer.reports] == [MARKET_ID]
    assert len(writer.reports[0].positions) == 1


# --- failures at start-up


def test_an_unregistered_provider_is_refused_at_build_time() -> None:
    with pytest.raises(ConfigError, match="provider adapter"):
        build_from(make_app_config(markets=(make_market_config(provider="deribit-ws"),)))


def test_an_unregistered_calibrator_is_refused_at_build_time() -> None:
    with pytest.raises(ConfigError, match="calibrator adapter"):
        build_from(make_app_config(calibration=make_calibration_config(calibrators=("svi-jax",))))


def test_an_unregistered_writer_is_refused_at_build_time() -> None:
    with pytest.raises(ConfigError, match="writer adapter"):
        build_from(make_app_config(risk=make_risk_config(writer="csv")))


def test_a_position_no_configured_market_quotes_is_refused() -> None:
    """Otherwise it is risk nobody values, and nothing anywhere says so."""
    book = (make_position(), make_position(underlying="ETH"))

    with pytest.raises(ConfigError, match="no configured market"):
        build_from(make_app_config(risk=make_risk_config(positions=book)))


def test_a_market_with_nothing_in_the_book_is_refused() -> None:
    """A market configured with nothing to value is a typo far more often than an intention."""
    markets = (
        make_market_config(),
        make_market_config(market_id="ETH-DERIBIT", underlying="ETH"),
    )

    with pytest.raises(ConfigError, match="holds no position on 'ETH'"):
        build_from(make_app_config(markets=markets))


def test_the_default_registry_holds_the_walking_skeleton_adapters() -> None:
    """The three names a shipped configuration may use, and the seam F1-08 closed.

    Names rather than objects: the factories are what ``build_pipeline`` calls, and asserting on
    what they build here would only repeat the end-to-end test in ``test_walking_skeleton.py``.
    What this pins is the vocabulary a TOML file is allowed to spell.
    """
    adapters = default_adapters()

    assert set(adapters.providers) == {"constant"}
    assert set(adapters.calibrators) == {"flat-vol"}
    assert set(adapters.writers) == {"console"}


async def test_a_report_goal_of_zero_is_a_caller_mistake() -> None:
    _, _, pipeline = make_pipeline(StubProvider(updates=()))

    with pytest.raises(ValueError, match="report goal must be positive"):
        await pipeline.run(max_reports=0)
