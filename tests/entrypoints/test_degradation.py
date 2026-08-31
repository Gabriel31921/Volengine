"""What the engine says when it is running badly, asserted end to end rather than per context.

Every hop of the degraded path is already tested where it lives: ``SnapshotPolicy`` decides that a
half-quoted chain is degraded, ``CalibrateOnSnapshot`` republishes the last good surface when a fit
refuses, ``FreshnessPolicy`` turns an old timestamp into a verdict, and ``ComputeReportUseCase``
turns that verdict into a report with a message. None of those tests can say that the four *reach
each other* -- and that is the whole claim of Design 7.2 and ADR-006: a market that has stopped
being fittable must produce a report that says so, out loud, rather than silence or a stale number
wearing a fresh timestamp.

The degradation is manufactured on purpose and one cause at a time: a chain that is quoting half
its instruments, then a calibrator that fits one cycle and refuses the rest. The clock is manual
and the heartbeat is what moves it -- the same arrangement ``test_pipeline`` uses to reach the
heartbeat at all -- so "the surface is now older than the policy tolerates" happens in
milliseconds instead of in minutes.

**How much older is deliberately never asserted.** That clock turns as fast as the event loop
lets the heartbeat run, and how many quiet periods pass while a fit is out on its thread pool is
a scheduling detail rather than a promise. Every assertion here is one that only grows more true
as time passes: a surface already too old does not become fresh, and a republish does not stop
being a republish.
"""

from __future__ import annotations

from tests.entrypoints.builders import (
    CALIBRATOR_NAME,
    BlockingProvider,
    FlakyCalibrator,
    RecordingBus,
    RecordingWriter,
    make_adapters,
    make_app_config,
    make_calibration_config,
    make_market_config,
    make_risk_config,
    two_sided,
)
from tests.market_data.builders import FORWARD, NOW, StubProvider, make_instrument
from tests.support import RecordingMetrics
from volengine.contracts.calibrated_surface import SurfaceStatus
from volengine.contracts.events import CalibrationFailed, SurfaceCalibrated
from volengine.entrypoints.pipeline import Pipeline, build_pipeline
from volengine.market_data.domain.option_quote import OptionKindD
from volengine.platform.clock import ManualClock
from volengine.risk.domain.freshness_policy import FreshnessDecision, FreshnessPolicy

QUIET_SECONDS = 10.0
"""Configuring a heartbeat at all, which is what creates the task that moves a ``ManualClock``.

The value barely matters: the timer polls at the market's cadence and the movement filter is off,
so an unchanged chain goes out on every tick rather than after this long. What it buys is that
time passes at all -- with no heartbeat the task is never created and a manual clock stands still.
"""

CADENCE_SECONDS = 1.0
"""The builders' own cadence, restated because it is the heartbeat's poll and therefore the step
this clock moves in. One tick is the *smallest* gap there can be between two snapshots."""

WARN_SECONDS = 0.25
REJECT_SECONDS = 0.5
"""A freshness policy narrower than one tick of that clock.

Chosen so that the second report is past refusing however the scheduler ran: its surface is the
first one republished, and a second snapshot cannot exist until the cadence has been met again, so
the age behind it is at least one tick and the verdict cannot be anything but ``REJECT``. A policy
of the thirty and hundred and twenty seconds a deployment would configure would make that verdict
depend on how many ticks happened to fit inside a thread pool round trip.
"""

FOUR_INSTRUMENTS = [
    make_instrument(strike=strike, kind=kind)
    for strike in (FORWARD, FORWARD * 1.1)
    for kind in (OptionKindD.CALL, OptionKindD.PUT)
]
"""A universe of four, of which the feed below quotes two: half a chain, on purpose."""


def build(
    provider: StubProvider | BlockingProvider,
    calibrator: FlakyCalibrator,
    min_coverage_ratio: float = 0.0,
    max_quiet_seconds: float | None = None,
) -> tuple[RecordingWriter, RecordingMetrics, RecordingBus, Pipeline]:
    """The whole engine on one market, with the bus and the writer kept for inspection.

    ``RecordingBus`` rather than the reports alone: a stale republish is a fact about what was
    *published*, and the report only shows its consequence. Both ends are asserted below, in
    separate tests, because they are two different promises.
    """
    writer = RecordingWriter()
    metrics = RecordingMetrics()
    bus = RecordingBus(metrics)
    config = make_app_config(
        markets=(
            make_market_config(
                cadence_seconds=CADENCE_SECONDS,
                max_quiet_seconds=max_quiet_seconds,
                min_coverage_ratio=min_coverage_ratio,
            ),
        ),
        calibration=make_calibration_config(calibrators=(CALIBRATOR_NAME,)),
        risk=make_risk_config(
            freshness=FreshnessPolicy(warn_seconds=WARN_SECONDS, reject_seconds=REJECT_SECONDS)
        ),
    )
    pipeline = build_pipeline(
        config,
        make_adapters(provider, {CALIBRATOR_NAME: calibrator}, writer),
        ManualClock(NOW),
        bus,
        metrics,
    )
    return writer, metrics, bus, pipeline


def half_quoted() -> StubProvider:
    """Four instruments discovered, one strike quoted: coverage of a half."""
    return StubProvider(updates=two_sided(), instruments=FOUR_INSTRUMENTS)


def surfaces(bus: RecordingBus) -> list[SurfaceCalibrated]:
    return bus.events_of(SurfaceCalibrated)


# --- a degraded input can only make a degraded surface


async def test_a_chain_quoting_half_its_instruments_publishes_a_degraded_surface() -> None:
    """The quality block does not travel with a surface, so the status has to carry it.

    A consumer holding a ``CalibratedSurface`` has no way back to the snapshot's coverage ratio.
    If the label were dropped here, a fit of half a chain would arrive downstream looking exactly
    like a fit of a whole one.
    """
    _, _, bus, pipeline = build(
        half_quoted(), FlakyCalibrator(good_cycles=1), min_coverage_ratio=0.9
    )

    await pipeline.run()

    assert [one.surface.status for one in surfaces(bus)] == [SurfaceStatus.DEGRADED]


async def test_the_same_half_chain_is_not_degraded_where_the_policy_does_not_ask_for_more() -> None:
    """The guard on the test above: the label comes from the threshold, not from the feed.

    Same provider, same calibrator, one number changed. Without this, a pipeline that marked
    every surface degraded would pass the previous test and nothing would say so.
    """
    _, _, bus, pipeline = build(
        half_quoted(), FlakyCalibrator(good_cycles=1), min_coverage_ratio=0.0
    )

    await pipeline.run()

    assert [one.surface.status for one in surfaces(bus)] == [SurfaceStatus.OK]


# --- a refused fit republishes what it last had


async def test_a_refused_fit_republishes_the_last_good_surface_as_stale() -> None:
    """ADR-006 end to end: the failure and the republish are two events, and both go out.

    A consumer needs each of them. ``CalibrationFailed`` says this producer could not fit this
    market; the republished surface keeps the risk report alive with numbers that say out loud how
    old they are. Dropping either one loses information the pipeline was built to carry.
    """
    _, _, bus, pipeline = build(
        BlockingProvider(updates=two_sided(), instruments=[make_instrument()]),
        FlakyCalibrator(good_cycles=1),
        max_quiet_seconds=QUIET_SECONDS,
    )

    await pipeline.run(max_reports=2)

    published = surfaces(bus)
    assert [one.surface.status for one in published[:2]] == [
        SurfaceStatus.OK,
        SurfaceStatus.STALE_REPUBLISH,
    ]
    assert bus.events_of(CalibrationFailed)


async def test_a_republished_surface_is_the_earlier_one_relabelled() -> None:
    """It keeps the original instants *and* the original id, because it is the same surface.

    Each of the three is a separate decision (ADR-006). Restamping ``ts_snapshot`` would make a
    surface from ten minutes ago look current, which is the single most dangerous thing this
    engine could do -- and it is why ``SurfaceView`` needs no status field at all: the freshness
    policy already reads the one timestamp that says everything the label would have said, as a
    number that can be compared rather than a word that has to be interpreted. ``ts_calibrated``
    refers to the earlier fit under this status. And a fresh ``surface_id`` would suggest a new
    fit exists, so a consumer correlating results would count one calibration twice.
    """
    _, _, bus, pipeline = build(
        BlockingProvider(updates=two_sided(), instruments=[make_instrument()]),
        FlakyCalibrator(good_cycles=1),
        max_quiet_seconds=QUIET_SECONDS,
    )

    await pipeline.run(max_reports=2)

    good, republished = surfaces(bus)[0].surface, surfaces(bus)[1].surface
    assert republished.ts_snapshot == good.ts_snapshot
    assert republished.ts_calibrated == good.ts_calibrated
    assert republished.surface_id == good.surface_id


# --- and the report says so


async def test_a_surface_older_than_the_policy_tolerates_is_refused_by_name() -> None:
    """The headline business rule of Design 7.2: no valid surface is *said*, not implied.

    The second report is valued against a surface the policy has already given up on -- one tick
    of the cadence is twice the rejection horizon, and a second snapshot cannot exist any sooner
    than that. What the report must not do is publish the old numbers again under a fresh
    timestamp, which is the failure mode that looks healthy from every dashboard.
    """
    writer, _, _, pipeline = build(
        BlockingProvider(updates=two_sided(), instruments=[make_instrument()]),
        FlakyCalibrator(good_cycles=1),
        max_quiet_seconds=QUIET_SECONDS,
    )

    await pipeline.run(max_reports=2)

    refused = writer.reports[1]
    assert refused.freshness is FreshnessDecision.REJECT
    assert refused.positions == ()
    assert refused.message is not None and "no valid surface" in refused.message


async def test_the_refused_report_still_names_the_snapshot_it_refused() -> None:
    """A rejection that dropped the instant behind it would leave nothing to check it against.

    The age *is* the reason, so the stale ``ts_snapshot`` is the evidence and belongs printed next
    to the refusal -- and it is the same instant the earlier report was valued on, which is what
    identifies this as the same surface grown old rather than a second one that arrived broken.

    Nothing here asserts *how* fresh the earlier report was. The heartbeat is the only thing
    moving this clock and it turns as fast as the loop lets it, so how many quiet periods have
    passed by the time the first fit comes back off its thread is a scheduling detail. Every
    assertion below is one that only grows more true as time passes.
    """
    writer, _, _, pipeline = build(
        BlockingProvider(updates=two_sided(), instruments=[make_instrument()]),
        FlakyCalibrator(good_cycles=1),
        max_quiet_seconds=QUIET_SECONDS,
    )

    await pipeline.run(max_reports=2)

    earlier, refused = writer.reports[0], writer.reports[1]
    assert refused.ts_snapshot == earlier.ts_snapshot
    assert refused.ts_report > earlier.ts_report


async def test_the_republish_is_counted_where_it_happens() -> None:
    """Observability is behaviour: a producer that quietly lived off an old fit is an outage.

    ``pricing.surface.republished`` is the only channel that says this happened at all -- the
    report shows an old timestamp, which is also what a quiet market looks like.
    """
    _, metrics, _, pipeline = build(
        BlockingProvider(updates=two_sided(), instruments=[make_instrument()]),
        FlakyCalibrator(good_cycles=1),
        max_quiet_seconds=QUIET_SECONDS,
    )

    await pipeline.run(max_reports=2)

    assert "pricing.surface.republished" in metrics.counter_names()
    assert "pricing.calibration.refused" in metrics.counter_names()
