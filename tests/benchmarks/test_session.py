"""One engine session run as the CLI runs it, with the surfaces and the comparison kept."""

from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

import pytest
from benchmarks.metrics_file import read_metrics
from benchmarks.session import MonotonicClock, last_surfaces, run_session

from volengine.entrypoints.config import AppConfig, load_config
from volengine.parametric_pricing.adapters.flat_vol import PRODUCER_ID as FLAT_ID
from volengine.parametric_pricing.adapters.scipy_calibrator import PRODUCER_ID as SCIPY_ID

EXAMPLES = Path(__file__).resolve().parents[2] / "examples"


def short_session() -> AppConfig:
    """The synthetic vertical with a second producer, its feed sped up to a fraction of a second.

    The interval goes from a second to a twentieth and the cycles from twenty to six: what the
    feed quotes is unchanged, only how long it takes. Two producers so a comparison exists.
    """
    config = load_config(EXAMPLES / "synthetic-svi.toml")
    market = config.markets[0]
    assert market.synthetic is not None
    feed = replace(market.synthetic.config, interval_seconds=0.05, cycles=6)
    market = replace(market, synthetic=replace(market.synthetic, config=feed))
    return replace(
        config,
        markets=(market,),
        calibration=replace(config.calibration, calibrators=(SCIPY_ID, FLAT_ID)),
    )


@pytest.mark.e2e
async def test_the_tap_keeps_every_producers_surfaces_and_the_comparison(tmp_path: Path) -> None:
    result = await run_session(short_session(), tmp_path / "m.csv")

    assert result.producers == (SCIPY_ID, FLAT_ID)
    assert all(result.surfaces[producer] for producer in result.producers)
    assert result.comparison is not None
    assert result.comparison.baseline.producer_id == SCIPY_ID
    assert result.comparison.challenger.producer_id == FLAT_ID
    assert set(last_surfaces(result)) == {SCIPY_ID, FLAT_ID}


@pytest.mark.e2e
async def test_the_metrics_file_is_replaced_not_appended_to(tmp_path: Path) -> None:
    path = tmp_path / "m.csv"
    path.write_text("left over from another session\n", encoding="utf-8")

    await run_session(short_session(), path)

    rows = read_metrics(path)
    assert any(row.name == "pricing.rmse_vol_bp" for row in rows)


@pytest.mark.e2e
async def test_the_reports_go_nowhere_while_the_engine_still_computes_them(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    await run_session(short_session(), tmp_path / "m.csv")

    assert capsys.readouterr().out == ""
    assert any(row.name == "risk.report.freshness" for row in read_metrics(tmp_path / "m.csv"))


async def test_a_session_with_two_markets_is_refused(tmp_path: Path) -> None:
    config = load_config(EXAMPLES / "multi-market.toml")

    with pytest.raises(ValueError, match="one market"):
        await run_session(config, tmp_path / "m.csv")


def test_the_monotonic_clock_starts_at_the_wall_clock_and_never_steps_back() -> None:
    clock = MonotonicClock()
    readings = [clock.now() for _ in range(1000)]

    # Within a few seconds of the wall clock, not to the microsecond: the host this was written
    # on steps its wall clock back by 2.6 s, which is the whole reason the class exists.
    assert abs((readings[0] - datetime.now(UTC)).total_seconds()) < 5.0
    assert readings == sorted(readings)
