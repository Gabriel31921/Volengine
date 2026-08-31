"""A feed faster than the fit, run through the real graph: what is lost, and what must not be.

``test_bus.py`` proves the mailbox conflates -- one slot, last write wins, a counter on the event
it destroyed. That is a statement about a class. This one is the statement about the *engine*
ADR-003 was written for: a producer that ticks faster than a calibration can finish must never
make the calibration wait, must never be served a snapshot older than one already published, and
must leave evidence that it happened.

The arrangement is a real one rather than a scripted race. The feed is the synthetic provider on
a fixed cadence, the calibrator is the stub with twenty milliseconds of thread sleep in it -- an
optimistic model of a fit, which in production runs from there into the seconds -- and the clock
is the system's, because conflation is about wall-clock pressure and a manual clock has none.

**Every assertion is one-sided in the safe direction.** A slower machine drops more, never fewer,
and the run is bounded by a feed that ends rather than by a stopping rule that guesses. What the
test would catch is the opposite failure: a bus that queued instead of conflating, or a consumer
that worked its way through a backlog of snapshots the market had already moved past.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from tests.entrypoints.builders import (
    CALIBRATOR_NAME,
    WRITER_NAME,
    RecordingBus,
    RecordingCalibrator,
    RecordingWriter,
    SlowCalibrator,
    make_app_config,
    make_calibration_config,
    make_market_config,
    make_risk_config,
)
from tests.support import RecordingMetrics
from volengine.contracts.events import SnapshotReady
from volengine.entrypoints.config import SYNTHETIC_PROVIDER, SyntheticSettings
from volengine.entrypoints.pipeline import (
    Adapters,
    Pipeline,
    build_pipeline,
    default_adapters,
    snapshot_topic,
)
from volengine.market_data.adapters.synthetic import SVIParamsSpec, SyntheticConfig
from volengine.platform.clock import SystemClock

TENOR = timedelta(days=30)
CYCLES = 8
INTERVAL_SECONDS = 0.005
"""Eight republications of the chain, five milliseconds apart: forty milliseconds of session."""

FIT_SECONDS = 0.02
"""What one fit costs on the pool. Four cycles of the feed, so the consumer cannot keep up."""

CADENCE_SECONDS = 0.001
"""Below the feed's interval, so the policy is not what limits the snapshot rate here."""


def market() -> SyntheticConfig:
    """The smallest ladder the generator allows, republished on a fixed interval.

    Two strikes, because this test is about the *rate* snapshots are produced at and nothing about
    the shape of the fit. Every source of randomness is off for the same reason.
    """
    return SyntheticConfig(
        expiries=(TENOR,),
        true_params={TENOR: SVIParamsSpec(a=0.020, b=0.050, rho=-0.30, m=0.0, sigma=0.20)},
        strikes_per_expiry=2,
        vol_noise_bp=0.0,
        junk_quote_rate=0.0,
        forward_move_rel=0.0,
        jitter_seconds=0.0,
        cycles=CYCLES,
        interval_seconds=INTERVAL_SECONDS,
    )


def build() -> tuple[RecordingCalibrator, RecordingMetrics, RecordingBus, Pipeline]:
    """The engine on the real synthetic feed and a calibrator that takes real milliseconds.

    ``RecordingCalibrator`` around ``SlowCalibrator``: the first says which snapshots were
    actually handed to a fit, which is the only way to see what conflation kept, and the second
    is what makes there be a difference to see.

    The book is the builders' default and nothing here reads the reports. What this test is about
    happens between the snapshot topic and the fit; the far end of the pipeline is asserted in
    ``test_known_truth`` and ``test_determinism``, where the position is written on the expiry the
    feed actually lists.
    """
    calibrator = RecordingCalibrator(SlowCalibrator(seconds=FIT_SECONDS))
    metrics = RecordingMetrics()
    bus = RecordingBus(metrics)
    config = make_app_config(
        markets=(
            make_market_config(
                provider=SYNTHETIC_PROVIDER,
                cadence_seconds=CADENCE_SECONDS,
                synthetic=SyntheticSettings(config=market(), start=datetime.now(UTC)),
            ),
        ),
        calibration=make_calibration_config(calibrators=(CALIBRATOR_NAME,)),
        risk=make_risk_config(),
    )
    writer = RecordingWriter()
    adapters = Adapters(
        providers=default_adapters().providers,
        calibrators={CALIBRATOR_NAME: lambda _config: calibrator},
        writers={WRITER_NAME: lambda _config: writer},
    )
    pipeline = build_pipeline(config, adapters, SystemClock(), bus, metrics)
    return calibrator, metrics, bus, pipeline


def published(bus: RecordingBus) -> list[str]:
    """The snapshot ids that went onto the bus, in order. Derived, so they are comparable."""
    return [event.snapshot.snapshot_id for event in bus.events_of(SnapshotReady)]


def consumed(calibrator: RecordingCalibrator) -> list[str]:
    """The snapshot ids a fit was actually run on."""
    return [task.snapshot_id for task in calibrator.tasks]


async def test_a_fit_slower_than_the_feed_leaves_snapshots_behind() -> None:
    """The conflation itself, in the graph rather than in the mailbox.

    Fewer fits than snapshots is the whole design: the alternative -- a queue -- would have the
    calibrator working through a backlog, reporting on a market that has since moved, and falling
    further behind with every cycle.
    """
    calibrator, _, bus, pipeline = build()

    await pipeline.run()

    assert len(published(bus)) > len(consumed(calibrator)) >= 1


async def test_what_was_dropped_is_counted_against_the_subscriber_that_fell_behind() -> None:
    """Silence and a backlog look identical from outside, so the drop has to be counted.

    ``bus.dropped`` is the only evidence conflation happened at all, and it is tagged with the
    topic and the subscriber precisely so that "the calibrator is behind" can be told from "the
    market is quiet" without reading any code.
    """
    _, metrics, _, pipeline = build()

    await pipeline.run()

    dropped = [tags for name, _, tags in metrics.counters if name == "bus.dropped"]
    assert dropped
    assert all(tags["topic"] == snapshot_topic("BTC-DERIBIT") for tags in dropped)


async def test_every_snapshot_a_fit_ran_on_had_been_published() -> None:
    """Nothing is invented on the way through: the consumer sees a subsequence of the stream.

    Weak on its own and load-bearing beside the test above -- it is what says the missing
    snapshots were *dropped* rather than reordered, duplicated or rebuilt from a stale chain.
    """
    calibrator, _, bus, pipeline = build()

    await pipeline.run()

    stream = published(bus)
    seen = consumed(calibrator)
    assert seen == [one for one in stream if one in set(seen)]


async def test_the_last_snapshot_of_the_session_is_the_one_it_ends_on() -> None:
    """Conflation loses the middle, never the newest -- which is the point of losing the middle.

    A feed that ends leaves its final snapshot in the mailbox, and the run settles rather than
    stopping on the spot, so the last thing fitted is the last thing published. A bus that dropped
    the *new* event instead of the pending one would pass every count above and fail here.
    """
    calibrator, _, bus, pipeline = build()

    await pipeline.run()

    assert consumed(calibrator)[-1] == published(bus)[-1]
