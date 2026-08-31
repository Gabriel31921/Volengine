"""Two runs of one configuration, and the same file out of both -- byte for byte.

This is the F2 definition of done that no single-context test can state: *determinism verified*.
The engine is asynchronous, its fit runs on a thread pool, its bus throws away what a consumer
could not keep up with, and its feed draws random spreads, sizes, latencies and volatility noise.
Run it twice and the same CSV has to come out, or ADR-004's recording and replay -- the whole
premise of F3-B -- is worth nothing, and neither is any comparison between two calibrators over one
recorded session.

**What makes it hold, and it is worth being precise.** Two sources of time feed a run, and this
test is the only place both are pinned to the same instant: the feed's timeline is derived from the
``start`` in its settings, and the engine's is a ``ManualClock`` the test holds. With no heartbeat
configured, nothing advances that clock -- so the snapshot policy's cadence is never met a second
time and the session is exactly one snapshot, one fit, one surface, one report. Everything
downstream of the feed is then a pure function of its seed and its origin, and the thread pool can
take as long as it likes without changing a single byte.

That is also the limit, and ``docs/SEAMS.md`` records it: what is *not* reproducible from a
configuration file is a whole run under a manual clock, because a TOML file carries the feed's
origin and has no way to name the engine's. Here the test supplies both, which is what a replay
driver will do for real in F3-B.

The chain is deliberately the smallest the generator allows. One snapshot arrives on the first
quote, so a wider ladder would only add strikes nothing in this session ever sees -- and the
position is struck where that first quote is, so the number the report carries is one the
generating surface can be asked about directly.
"""

from __future__ import annotations

import csv
import math
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
from volengine.entrypoints.pipeline import build_pipeline, default_adapters
from volengine.market_data.adapters.synthetic import (
    SVIParamsSpec,
    SyntheticConfig,
    SyntheticProvider,
)
from volengine.market_data.domain.option_quote import InstrumentId
from volengine.parametric_pricing.adapters.scipy_calibrator import PRODUCER_ID
from volengine.platform.bus import InProcessConflatingBus
from volengine.platform.clock import ManualClock
from volengine.risk.domain.portfolio import Position
from volengine.risk.domain.pricing import OptionKindR

pytestmark = pytest.mark.e2e

START = datetime(2026, 8, 27, 12, 0, tzinfo=UTC)
"""When the feed starts stamping its quotes. A fixed instant, so the file below is fixed."""

NOW = START + timedelta(seconds=1)
"""Where the engine's clock stands, and stays. A second after the feed's origin, so a snapshot is
always in the past of the fit that reads it -- ``CalibratedSurface`` refuses the other order -- and
comfortably inside the freshness policy the builders configure."""

TENOR = timedelta(days=30)
TRUE = SVIParamsSpec(a=0.020, b=0.050, rho=-0.30, m=0.0, sigma=0.20)
FORWARD = 60_000.0
"""The generating slice and the feed's own initial forward."""


def market() -> SyntheticConfig:
    """A two-strike ladder with the noise left *on*, seeded and pinned.

    The noise is the point. Turning it off would make the run reproducible by having nothing to
    reproduce; leaving it on and fixing the seed is the claim actually being made -- that every
    draw comes from one generator, in a fixed order, from a number written in the configuration.
    """
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
    """The instrument the feed will publish first, asked of the feed itself.

    The one snapshot of this session holds exactly this contract, so it is where the book has to
    be written -- anywhere else and the report would be reading an extrapolation of a fit through
    a single point, which says nothing about the surface that was quoted.
    """
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


async def run_to(output: Path, start: datetime = START) -> None:
    """One session, written to this path, on a clock that does not move."""
    metrics = RecordingMetrics()
    pipeline = build_pipeline(
        configuration(output, start),
        default_adapters(),
        ManualClock(NOW),
        InProcessConflatingBus(metrics),
        metrics,
    )

    await pipeline.run()


async def test_two_runs_of_one_configuration_write_the_same_bytes(tmp_path: Path) -> None:
    """The headline: identical input, identical output, down to the timestamps.

    Two paths rather than one, because the writer appends -- a second run over the same file would
    compare a report against itself concatenated.
    """
    first, second = tmp_path / "first.csv", tmp_path / "second.csv"

    await run_to(first)
    await run_to(second)

    assert first.read_bytes() == second.read_bytes()


async def test_moving_the_feed_by_one_second_moves_the_file(tmp_path: Path) -> None:
    """The guard on the comparison above: it is not two empty files agreeing.

    Everything the engine writes descends from the feed's origin -- the quote stamps, the snapshot
    instant, the tenor the fit was taken at. Shift it and the bytes have to move, or the previous
    test would pass just as well against a writer that emitted a constant.
    """
    pinned, shifted = tmp_path / "pinned.csv", tmp_path / "shifted.csv"

    await run_to(pinned)
    await run_to(shifted, start=START - timedelta(seconds=1))

    assert pinned.read_bytes() != shifted.read_bytes()


async def test_the_session_writes_exactly_one_valued_line(tmp_path: Path) -> None:
    """What is in the file, so that the comparison is over a report rather than over a header.

    One row, because the writer emits one line per valued position and this book holds one. A
    rejected report would be a row too, with an empty ``vol`` -- which is why the value is read
    and not merely counted.
    """
    output = tmp_path / "report.csv"

    await run_to(output)

    rows = list(csv.DictReader(output.read_text(encoding="utf-8").splitlines()))
    assert len(rows) == 1
    assert float(rows[0]["vol"]) > 0.0


async def test_the_line_carries_the_volatility_the_generator_implies_at_that_strike(
    tmp_path: Path,
) -> None:
    """And the number is the market's, not merely a number.

    The single quote of this session sits at the first rung of the ladder, so the fit has one
    residual to satisfy and the report reads it back through the published grid. The tolerance
    covers the noise on that one mid and the interpolation between the two grid nodes it falls
    between; what it does not cover is a different slice, which the guard below shows.
    """
    output = tmp_path / "report.csv"
    quoted = first_quoted(START)

    await run_to(output)

    rows = list(csv.DictReader(output.read_text(encoding="utf-8").splitlines()))
    assert float(rows[0]["vol"]) == pytest.approx(generating_vol(quoted), abs=1.5e-2)


def test_the_at_the_money_volatility_would_have_missed_that_band() -> None:
    """The guard: the tolerance above admits the quoted strike and not the whole smile.

    The ladder's first rung sits a long way down the downside wing, where the generating slice is
    several volatility points above its own minimum. A report that had lost the moneyness axis --
    valued everything at the forward, say -- would land outside the band by an order of magnitude.
    """
    quoted = first_quoted(START)
    tenor_years = (quoted.expiry - NOW).total_seconds() / (365 * 86_400)

    at_the_money = math.sqrt(TRUE.total_variance(0.0) / tenor_years)

    assert abs(generating_vol(quoted) - at_the_money) > 1.5e-2


def generating_vol(instrument: InstrumentId) -> float:
    """The volatility the generating slice implies at that strike, at the engine's own instant."""
    tenor_years = (instrument.expiry - NOW).total_seconds() / (365 * 86_400)
    return TRUE.volatility(math.log(instrument.strike / FORWARD), tenor_years)
