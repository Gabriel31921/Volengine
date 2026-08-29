"""The proof that the architecture closes: one quote in, one risk report out, on real adapters.

Every other test in this repository exercises one context, or exercises the graph on fakes. This
one runs the engine the way an operator does -- ``default_adapters()``, a configuration file, the
command line -- and asserts that a number produced by ``market_data`` survives ingestion,
admissibility, the snapshot policy, the bus, the inversion, a fit, the acceptance rule, a grid,
the cache, an interpolation and a valuation, and comes out the far end recognisable.

That last property is what makes this more than a smoke test. The constant provider prices its
chain from one volatility; the flat calibrator fits exactly that shape; so the volatility printed
on the report has to be the volatility the feed was built from, to four decimals. Any hop that
quietly dropped, rescaled or reinterpreted it -- a units mix-up in a mid, a tenor read off the
wrong calendar, a moneyness axis inverted -- moves that number, and no single-context test would
see it.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from typer.testing import CliRunner

from tests.entrypoints.builders import (
    CONFIG_TOML,
    RecordingWriter,
    make_app_config,
    make_calibration_config,
    make_market_config,
    make_risk_config,
    replacing,
    write_config,
)
from volengine.entrypoints.cli import app
from volengine.entrypoints.config import AppConfig, load_config
from volengine.entrypoints.pipeline import Adapters, build_pipeline, default_adapters
from volengine.market_data.adapters.constant import VOL
from volengine.parametric_pricing.adapters.flat_vol import PRODUCER_ID
from volengine.platform.bus import InProcessConflatingBus
from volengine.platform.clock import SystemClock
from volengine.platform.metrics import NullMetricsSink
from volengine.risk.domain.freshness_policy import FreshnessDecision
from volengine.risk.domain.portfolio import Position
from volengine.risk.domain.ports import ReportWriter
from volengine.risk.domain.pricing import OptionKindR

pytestmark = pytest.mark.e2e

EXAMPLE = Path(__file__).resolve().parents[2] / "examples" / "walking-skeleton.toml"
"""The configuration shipped for a first run. Its own header holds the two commands."""

PROVIDER_NAME = "constant"
WRITER_NAME = "console"


def held_position() -> Position:
    """One at-the-money call inside the tenors the constant provider quotes.

    Built against the wall clock rather than pinned to a date, because the provider's expiries are
    relative to start-up: a fixed instant here would drift out of the grid and, eventually, into
    the past, where the valuation refuses it outright.
    """
    return Position(
        underlying="BTC",
        expiry=datetime.now(UTC) + timedelta(days=45),
        strike=60_000.0,
        kind=OptionKindR.CALL,
        quantity=1.0,
    )


def skeleton_config() -> AppConfig:
    """The walking skeleton, as configuration: the three registered names and one position."""
    return make_app_config(
        markets=(make_market_config(provider=PROVIDER_NAME),),
        calibration=make_calibration_config(calibrators=(PRODUCER_ID,)),
        risk=make_risk_config(writer=WRITER_NAME, positions=(held_position(),)),
    )


def with_writer(writer: ReportWriter) -> Adapters:
    """The real registry with its writer replaced, so a test can hold the report itself.

    The provider and the calibrator stay real -- they are what is under test. Only the far end is
    swapped, because a console writer's output is text and the assertions below are about a
    ``RiskReport``.
    """
    registry = default_adapters()
    return Adapters(
        providers=registry.providers,
        calibrators=registry.calibrators,
        writers={WRITER_NAME: lambda _risk: writer},
    )


async def test_a_quote_becomes_a_risk_report_carrying_the_volatility_it_was_priced_at() -> None:
    """The whole engine, end to end, on the adapters the composition root actually registers."""
    writer = RecordingWriter()
    pipeline = build_pipeline(
        skeleton_config(),
        with_writer(writer),
        SystemClock(),
        InProcessConflatingBus(NullMetricsSink()),
        NullMetricsSink(),
    )

    await pipeline.run(max_reports=1)

    assert pipeline.reports_written == 1
    report = writer.reports[0]
    assert (report.market_id, report.producer_id) == ("BTC-DERIBIT", PRODUCER_ID)
    assert report.freshness is FreshnessDecision.NORMAL
    assert len(report.positions) == 1
    assert report.positions[0].vol == pytest.approx(VOL, abs=1e-4)


async def test_the_report_reaches_the_console_writer_the_registry_builds(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The same run with nothing swapped: the far end really is ``ConsoleReportWriter``.

    Without this, every assertion above would hold with a writer the shipped configuration never
    names, and ``volengine run`` could still print nothing at all.
    """
    pipeline = build_pipeline(
        skeleton_config(),
        default_adapters(),
        SystemClock(),
        InProcessConflatingBus(NullMetricsSink()),
        NullMetricsSink(),
    )

    await pipeline.run(max_reports=1)

    assert f"[BTC-DERIBIT / {PRODUCER_ID}]" in capsys.readouterr().out


def test_the_command_line_produces_a_report_and_exits_cleanly(tmp_path: Path) -> None:
    """``volengine report`` on a file naming the three adapters. F1's definition of done.

    Two reports rather than one, because the first snapshot of a session goes out the moment the
    first quote lands -- one instrument, one expiry -- and it is the second, one cadence later,
    that rests on the whole chain. Both are real reports; asking for both is what proves the
    engine keeps running rather than merely starting.
    """
    text = replacing(CONFIG_TOML, "calibrators", f'calibrators = ["{PRODUCER_ID}"]')
    text = replacing(text, "material_move_threshold", "material_move_threshold = 0.0")
    expiry = datetime.now(UTC) + timedelta(days=45)
    # "expiry = " and not "expiry": `replacing` matches on a line prefix, and the shorter one
    # also catches `expiry_time_utc` in the conventions table.
    text = replacing(text, "expiry = ", f"expiry = {expiry:%Y-%m-%d}T08:00:00Z")

    result = CliRunner().invoke(
        app, ["report", "--config", str(write_config(tmp_path, text)), "--count", "2"]
    )

    assert result.exit_code == 0
    assert result.output.count(f"[BTC-DERIBIT / {PRODUCER_ID}]") == 2


def test_the_shipped_example_names_only_adapters_this_build_registers() -> None:
    """The guard on the file a first-time reader runs, without pinning it to a date.

    A renamed adapter, a moved key or a threshold the domain has since tightened all show up
    here. What is deliberately *not* asserted is the position's expiry: it is a fixed instant in a
    file that has no way to say "next year", so it is documented as needing a bump rather than
    guarded by a test that would fail on a calendar rather than on a change anybody made.
    """
    config = load_config(EXAMPLE)
    registry = default_adapters()

    assert [market.provider for market in config.markets] == list(registry.providers)
    assert list(config.calibration.calibrators) == list(registry.calibrators)
    assert config.risk.writer in registry.writers
