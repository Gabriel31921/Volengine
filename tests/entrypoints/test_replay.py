"""Record a session, replay it twice, and get the same bytes: the F3-B definition of done.

``test_determinism.py`` already showed that one configuration run twice writes one file twice, and
it did so by pinning *both* sources of time from the test -- the feed's origin in its settings and
the engine's clock by hand. ``docs/SEAMS.md`` recorded what that leaves open: a whole run is not
reproducible from anything a person can hand the engine, because the engine's clock is not in the
file. This is the test that closes it. A recording carries the engine's timeline as well as the
feed's, so the only thing a replay is given is a path, and everything else -- the quotes, the
instants, the order -- comes out of it.

**What is asserted is a byte comparison, and that is the strongest available claim.** The engine is
asynchronous, conflates on a one-slot mailbox and fits on a thread pool. Two replays of one file
agreeing to the byte means none of that leaked into the answer: not the scheduler, not the wall
clock, not the pool's latency. The guards below are what stop it from being a comparison of two
empty files -- a valued row is read out of the report, and a session recorded one second earlier
produces different bytes.

The market is the smallest the generator allows, for ``test_determinism.py``'s reason: the first
snapshot arrives on the first quote, so a wider ladder would only add strikes this session never
sees, and the book is written on the strike that first quote carries.
"""

from __future__ import annotations

import csv
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from tests.entrypoints.builders import (
    UNDERLYING,
    make_app_config,
    make_calibration_config,
    make_market_config,
    make_risk_config,
)
from tests.support import RecordingMetrics
from volengine.entrypoints.config import SYNTHETIC_PROVIDER, AppConfig, SyntheticSettings
from volengine.entrypoints.pipeline import (
    build_pipeline,
    default_adapters,
    with_recording,
    with_replay,
)
from volengine.market_data.adapters.recorded import open_recording
from volengine.market_data.adapters.synthetic import (
    SVIParamsSpec,
    SyntheticConfig,
    SyntheticProvider,
)
from volengine.market_data.domain.option_quote import InstrumentId
from volengine.parametric_pricing.adapters.scipy_calibrator import PRODUCER_ID
from volengine.platform.bus import InProcessConflatingBus
from volengine.platform.clock import ManualClock, SimulatedClock
from volengine.risk.domain.portfolio import Position
from volengine.risk.domain.pricing import OptionKindR

pytestmark = pytest.mark.e2e

START = datetime(2026, 8, 27, 12, 0, tzinfo=UTC)
"""When the feed stamps its first quote, and therefore where a replay's clock starts."""

NOW = START + timedelta(seconds=1)
"""Where the *recording* run's clock stands. Only the recording run has one of its own: the replay
reads its instants out of the file, which is the whole point."""

TENOR = timedelta(days=30)
TRUE = SVIParamsSpec(a=0.020, b=0.050, rho=-0.30, m=0.0, sigma=0.20)


def market() -> SyntheticConfig:
    """A two-strike ladder with the noise left on, seeded and pinned."""
    return SyntheticConfig(
        expiries=(TENOR,),
        true_params={TENOR: TRUE},
        strikes_per_expiry=2,
        vol_noise_bp=20.0,
        junk_quote_rate=0.0,
        forward_move_rel=0.0,
        cycles=2,
        interval_seconds=0.0,
    )


def first_quoted(start: datetime) -> InstrumentId:
    """The instrument the feed publishes first, asked of the feed itself."""
    return SyntheticProvider(make_market_config().conventions, market(), start).instruments[0]


def configuration(output: Path, start: datetime = START) -> AppConfig:
    """The whole engine as a file would configure it, with the CSV writer at the far end."""
    quoted = first_quoted(start)
    return make_app_config(
        markets=(
            make_market_config(
                provider=SYNTHETIC_PROVIDER,
                synthetic=SyntheticSettings(config=market(), start=start),
            ),
        ),
        calibration=make_calibration_config(calibrators=(PRODUCER_ID,)),
        risk=make_risk_config(
            writer="csv",
            output_path=output,
            positions=(
                Position(
                    underlying=UNDERLYING,
                    expiry=quoted.expiry,
                    strike=quoted.strike,
                    kind=OptionKindR.CALL,
                    quantity=1.0,
                ),
            ),
        ),
    )


async def record_to(recording: Path, report: Path, start: datetime = START) -> None:
    """One ordinary session with a tap on it, on a clock that does not move."""
    metrics = RecordingMetrics()
    pipeline = build_pipeline(
        configuration(report, start),
        with_recording(default_adapters(), recording),
        ManualClock(NOW),
        InProcessConflatingBus(metrics),
        metrics,
    )
    await pipeline.run()


async def replay_to(recording: Path, report: Path) -> None:
    """One replay, driven end to end by the file: the provider, and the clock behind it.

    The composition is the one ``cli.replay`` performs -- open the recording, place a
    ``SimulatedClock`` at its first instant, swap the providers -- written out here rather than
    invoked through typer, because what is under test is the engine and not the argument parsing.
    """
    session = open_recording(recording)
    clock = SimulatedClock(session.started_at)
    metrics = RecordingMetrics()
    pipeline = build_pipeline(
        configuration(report),
        with_replay(default_adapters(), session, clock),
        clock,
        InProcessConflatingBus(metrics),
        metrics,
    )
    await pipeline.run()


def rows(report: Path) -> list[dict[str, str]]:
    return list(csv.DictReader(report.read_text(encoding="utf-8").splitlines()))


# --- the headline


async def test_two_replays_of_one_recording_write_the_same_bytes(tmp_path: Path) -> None:
    """Same file, same report, twice. The acceptance criterion of this stage, in one assertion."""
    recording = tmp_path / "session.jsonl"
    await record_to(recording, tmp_path / "recorded.csv")

    first, second = tmp_path / "first.csv", tmp_path / "second.csv"
    await replay_to(recording, first)
    await replay_to(recording, second)

    assert first.read_bytes() == second.read_bytes()


async def test_the_replay_writes_a_valued_report(tmp_path: Path) -> None:
    """The guard on the comparison above: two identical files, and not two empty ones.

    The whole chain has to have run for this to hold -- ingestion, the snapshot policy, a real
    ``least_squares`` fit, the grid, the freshness policy, the writer -- off nothing but a path.
    """
    recording = tmp_path / "session.jsonl"
    await record_to(recording, tmp_path / "recorded.csv")
    report = tmp_path / "replayed.csv"

    await replay_to(recording, report)

    assert len(rows(report)) == 1
    assert float(rows(report)[0]["vol"]) > 0.0


async def test_a_session_recorded_one_second_earlier_replays_to_different_bytes(
    tmp_path: Path,
) -> None:
    """The second guard: the replay reads the file rather than reciting the configuration.

    Every instant in a replayed report descends from the recording -- the quote stamps, the
    snapshot instant, the tenor the fit was taken at. Shift the recorded session and the bytes have
    to move, or the test above would pass just as well against a writer that emitted a constant.
    """
    pinned, shifted = tmp_path / "pinned.jsonl", tmp_path / "shifted.jsonl"
    await record_to(pinned, tmp_path / "a.csv")
    await record_to(shifted, tmp_path / "b.csv", start=START - timedelta(seconds=1))

    first, second = tmp_path / "first.csv", tmp_path / "second.csv"
    await replay_to(pinned, first)
    await replay_to(shifted, second)

    assert first.read_bytes() != second.read_bytes()


# --- what the recording holds


async def test_the_recording_holds_every_quote_the_feed_published(tmp_path: Path) -> None:
    """A tap records the session, not a sample of it.

    Two cycles of a two-strike ladder, both legs of each strike: eight quotes, plus the header.
    Counting them is what would catch a tap that recorded only what the snapshot policy kept.
    """
    recording = tmp_path / "session.jsonl"

    await record_to(recording, tmp_path / "report.csv")

    assert len(recording.read_text(encoding="utf-8").splitlines()) == 9


async def test_the_recording_names_the_market_it_was_taken_from(tmp_path: Path) -> None:
    """What ``cli.replay`` narrows the configuration by, so it has to come out of the file."""
    recording = tmp_path / "session.jsonl"
    await record_to(recording, tmp_path / "report.csv")

    session = open_recording(recording)

    assert session.market_id == make_market_config().market_id
    assert session.underlying == UNDERLYING


async def test_the_replayed_clock_ends_inside_the_recorded_session(tmp_path: Path) -> None:
    """The engine really is living on the file's timeline and not on this machine's.

    Stated as a band rather than an instant: what the clock reads at the end is the last recorded
    ``ts_local``, and pinning that exact microsecond would be pinning the feed's jitter draw. What
    matters is that it is inside the recorded session and nowhere near now.
    """
    recording = tmp_path / "session.jsonl"
    await record_to(recording, tmp_path / "report.csv")
    session = open_recording(recording)
    clock = SimulatedClock(session.started_at)

    provider = with_replay(default_adapters(), session, clock).providers[SYNTHETIC_PROVIDER]
    async for _ in provider(configuration(tmp_path / "unused.csv").markets[0]).stream():
        pass

    assert session.started_at <= clock.now() <= session.started_at + timedelta(seconds=1)
