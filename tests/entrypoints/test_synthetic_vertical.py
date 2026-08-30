"""The F2 vertical, end to end: a known surface generated, spoiled, fitted, valued and written.

``test_walking_skeleton.py`` proves the architecture closes on adapters that do no mathematics --
a constant feed and a calibrator that averages. This one runs the same graph on the two adapters
that do: quotes are Black-76 prices of a *known* SVI surface, and the volatility that comes out of
the report is one ``scipy.least_squares`` recovered from them. Any hop that dropped, rescaled or
reinterpreted the surface -- a moneyness axis inverted, a tenor read off the wrong calendar, a fit
handed weights it never asked for -- moves that number, and no single-context test would see it.

**The feed is clean here, on purpose.** No volatility noise, no junk quotes, no forward walk, so
the fit has a right answer it can reach exactly and the tolerance below is about the *wiring*
rather than about an optimiser. Recovering the parameters *through* the noise is the known-truth
test of F2-08, which is a statement about the calibrator and not about the composition root.
"""

from __future__ import annotations

import csv
import math
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from typer.testing import CliRunner

from tests.entrypoints.builders import (
    RecordingWriter,
    make_app_config,
    make_calibration_config,
    make_market_config,
    make_risk_config,
    write_config,
)
from volengine.entrypoints.cli import app
from volengine.entrypoints.config import (
    SYNTHETIC_PROVIDER,
    AppConfig,
    SyntheticSettings,
    load_config,
)
from volengine.entrypoints.pipeline import Adapters, build_pipeline, default_adapters
from volengine.market_data.adapters.synthetic import (
    SVIParamsSpec,
    SyntheticConfig,
    SyntheticProvider,
)
from volengine.parametric_pricing.adapters.scipy_calibrator import PRODUCER_ID
from volengine.platform.bus import InProcessConflatingBus
from volengine.platform.clock import SystemClock
from volengine.platform.metrics import NullMetricsSink
from volengine.risk.domain.freshness_policy import FreshnessDecision
from volengine.risk.domain.portfolio import Position
from volengine.risk.domain.ports import ReportWriter
from volengine.risk.domain.pricing import OptionKindR

pytestmark = pytest.mark.e2e

EXAMPLE = Path(__file__).resolve().parents[2] / "examples" / "synthetic-svi.toml"
"""The configuration shipped for the F2 vertical. Its own header holds the two commands."""

MARKET_ID = "BTC-SYNTH"
UNDERLYING = "BTC"
TENOR = timedelta(days=30)
TRUE = SVIParamsSpec(a=0.020, b=0.050, rho=-0.30, m=0.0, sigma=0.20)
"""The generating slice. At ``k = 0`` and ``m = 0`` its total variance is ``a + b * sigma``, which
is the one closed form this test needs and the reason the position below is struck at the forward.
"""

FORWARD = 60_000.0
"""``SyntheticConfig``'s own initial forward, restated because the position is struck at it: with
no forward walk configured, ``k = ln(K / F)`` is exactly zero for the whole session."""


def clean_market() -> SyntheticConfig:
    """One expiry, seven strikes, and nothing spoiled: an exact surface, quoted exactly.

    Every source of randomness is turned off rather than seeded. A seed makes a run reproducible;
    zero noise makes the *answer* knowable, which is what lets the assertion below be about the
    volatility rather than about a band around it.
    """
    return SyntheticConfig(
        expiries=(TENOR,),
        true_params={TENOR: TRUE},
        strikes_per_expiry=7,
        vol_noise_bp=0.0,
        junk_quote_rate=0.0,
        forward_move_rel=0.0,
        cycles=3,
        interval_seconds=0.05,
    )


def true_vol(tenor_years: float) -> float:
    """The volatility the generator implies at the forward: ``sqrt(w(0) / T)``."""
    return math.sqrt(TRUE.total_variance(0.0) / tenor_years)


def vertical_config(start: datetime, expiry: datetime) -> AppConfig:
    """The whole engine as a third party would configure it: synthetic feed, scipy fit, one call.

    ``start`` is pinned so that the expiry the provider places its chain on is knowable *before*
    the pipeline builds its own provider from this configuration -- the position has to name that
    same instant, and a feed reading the wall clock twice would place it twice.
    """
    return make_app_config(
        markets=(
            make_market_config(
                market_id=MARKET_ID,
                provider=SYNTHETIC_PROVIDER,
                cadence_seconds=0.02,
                synthetic=SyntheticSettings(config=clean_market(), start=start),
            ),
        ),
        calibration=make_calibration_config(calibrators=(PRODUCER_ID,)),
        risk=make_risk_config(
            positions=(
                Position(
                    underlying=UNDERLYING,
                    expiry=expiry,
                    strike=FORWARD,
                    kind=OptionKindR.CALL,
                    quantity=1.0,
                ),
            )
        ),
    )


def quoted_expiry(config: AppConfig) -> datetime:
    """The instant the feed will place its only expiry on, asked of the feed itself.

    Built from the same conventions, the same settings and the same pinned origin as the provider
    the pipeline builds, so the two agree by construction rather than by a date restated here --
    which would be this test computing the venue's expiry hour on its own.
    """
    market = config.markets[0]
    assert market.synthetic is not None
    provider = SyntheticProvider(
        market.conventions, market.synthetic.config, market.synthetic.start
    )
    return provider.instruments[0].expiry


def with_writer(writer: ReportWriter) -> Adapters:
    """The real registry with only its far end swapped, so the test holds the report itself."""
    registry = default_adapters()
    return Adapters(
        providers=registry.providers,
        calibrators=registry.calibrators,
        writers={"recording": lambda _risk: writer},
    )


async def test_the_report_carries_the_volatility_the_surface_was_generated_from() -> None:
    """One quote in, one fitted volatility out, across the two adapters that do the mathematics.

    The run has no stopping rule and the feed is finite, so it ends when the last cycle has been
    ingested and everything in flight has settled -- and the report asserted is the *last*. The
    first snapshot of a session goes out the moment the first quote lands, so it rests on a single
    instrument, and five parameters through one point is not a recovery of anything; only once the
    whole ladder is in the chain does the fit have a surface to find.
    """
    start = datetime.now(UTC)
    writer = RecordingWriter()
    config = vertical_config(start, quoted_expiry(vertical_config(start, start)))

    pipeline = build_pipeline(
        config,
        with_writer(writer),
        SystemClock(),
        InProcessConflatingBus(NullMetricsSink()),
        NullMetricsSink(),
    )
    await pipeline.run()

    report = writer.reports[-1]
    assert (report.market_id, report.producer_id) == (MARKET_ID, PRODUCER_ID)
    assert report.freshness is FreshnessDecision.NORMAL
    tenor_years = (report.positions[0].position.expiry - start).total_seconds() / (365 * 86_400)
    assert report.positions[0].vol == pytest.approx(true_vol(tenor_years), abs=5e-3)


def test_a_position_off_the_generating_surface_would_have_moved_that_number() -> None:
    """The guard on the assertion above: it is not one any surface would have satisfied.

    ``pytest.approx`` at five thousandths of a volatility point is a real bound only if a
    different slice lands outside it -- so the same closed form, evaluated on the tenor the
    generator does *not* use, has to disagree.
    """
    thirty_days = true_vol(30 / 365)
    ninety_days = true_vol(90 / 365)

    assert abs(thirty_days - ninety_days) > 5e-3


def test_the_command_line_writes_a_csv_a_reader_can_parse(tmp_path: Path) -> None:
    """``volengine report`` on a file naming the synthetic feed, the fit and the CSV writer.

    F2's definition of done, as a third party meets it: a configuration file, one command, and a
    file that ``csv.DictReader`` -- or a spreadsheet, or ``pandas`` -- opens.
    """
    output = tmp_path / "reports.csv"
    text = vertical_toml(output)

    result = CliRunner().invoke(
        app, ["report", "--config", str(write_config(tmp_path, text)), "--count", "2"]
    )

    assert result.exit_code == 0, result.output
    rows = list(csv.DictReader(output.read_text(encoding="utf-8").splitlines()))
    assert [row["market_id"] for row in rows] == [MARKET_ID, MARKET_ID]
    assert all(float(row["vol"]) > 0.0 for row in rows)


def test_the_shipped_example_names_only_adapters_this_build_registers() -> None:
    """The guard on the file a first-time reader runs, without pinning it to a date.

    A renamed adapter, a moved key or a threshold the domain has since tightened all show up here.
    What is deliberately *not* asserted is the position's expiry: it is a fixed instant in a file
    that has no way to say "next year" (``docs/SEAMS.md``), so it is documented as needing a bump
    rather than guarded by a test that would fail on a calendar rather than on a change anybody
    made.
    """
    config = load_config(EXAMPLE)
    registry = default_adapters()

    assert config.markets[0].provider in registry.providers
    assert config.markets[0].synthetic is not None
    assert list(config.calibration.calibrators) == [PRODUCER_ID]
    assert config.calibration.fit is not None
    assert config.risk.writer in registry.writers


def vertical_toml(output: Path) -> str:
    """A file naming the three F2 adapters, with its two instants generated rather than pinned.

    The dates are computed because the feed's expiries are relative to start-up: a literal expiry
    would sit inside the chain today and outside it next month, which is the maintenance seam the
    shipped example carries and no test should inherit.
    """
    start = datetime.now(UTC)
    held = (start + timedelta(days=60)).strftime("%Y-%m-%dT08:00:00Z")
    return f"""
[[market]]
id = "{MARKET_ID}"
underlying = "{UNDERLYING}"
provider = "synthetic"
max_skew_seconds = 30.0

[market.conventions]
day_count = "ACT/365F"
expiry_time_utc = 08:00:00
numeraire = "INVERSE"
forward_method = "PROVIDER_UNDERLYING"

[market.admissibility]
max_spread_rel = 0.5
max_age_seconds = 5.0
moneyness_range = [-1.5, 1.5]
max_iv_divergence_bp = 500.0
convexity_tolerance = 0.0005
min_size = 1.0

[market.snapshot]
cadence_seconds = 0.001
material_move_threshold = 0.0
min_coverage_ratio = 0.0

[market.synthetic]
forward0 = {FORWARD}
strikes_per_expiry = 7
log_moneyness_range = [-0.2, 0.2]
spread_bp = 200.0
vol_noise_bp = 0.0
size = 10.0
junk_quote_rate = 0.0
forward_move_rel = 0.0
latency_seconds = 0.008
jitter_seconds = 0.0
cycles = 3
interval_seconds = 0.01
seed = 1

[[market.synthetic.slice]]
expiry_days = 30.0
a = {TRUE.a}
b = {TRUE.b}
rho = {TRUE.rho}
m = {TRUE.m}
sigma = {TRUE.sigma}

[[market.synthetic.slice]]
expiry_days = 90.0
a = 0.055
b = 0.090
rho = -0.25
m = 0.0
sigma = 0.25

[calibration]
calibrators = ["{PRODUCER_ID}"]

[calibration.grid]
k_min = -0.4
k_max = 0.4
n_nodes = 9

[calibration.weighting]
spread_scale = 0.05
flagged_factor = 0.25
unpaired_itm_factor = 0.1

[calibration.acceptance]
max_rmse_vol_bp = 50.0

[calibration.fit]
huber_scale_bp = 100.0
durrleman_penalty_bp = 10000.0
durrleman_mesh_nodes = 21
durrleman_mesh_margin = 0.5
min_quotes_for_free_shape = 5
ridge_bp = 5.0
max_nfev = 500

[risk]
writer = "csv"
output_path = "{output}"

[risk.freshness]
warn_seconds = 30.0
reject_seconds = 120.0

[risk.report]
discount = 1.0

[risk.report.bumps]
forward_rel = 0.01
vol_abs = 0.01

[[risk.position]]
underlying = "{UNDERLYING}"
expiry = {held}
strike = {FORWARD}
kind = "CALL"
quantity = 1.0
"""
