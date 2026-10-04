"""The helpers both benchmark reports share, and the neural regime split."""

from __future__ import annotations

from pathlib import Path

import pytest
from benchmarks.metrics_file import read_metrics, summarise
from benchmarks.parametric_vs_neural import regimes
from benchmarks.report import (
    DASH,
    dropped_by_conflation,
    fmt_summary,
    handler_failures,
    table,
    with_cycles,
)
from benchmarks.session import TAP_PREFIX

from tests.benchmarks.builders import write_metrics
from volengine.entrypoints.config import load_config

EXAMPLES = Path(__file__).resolve().parents[2] / "examples"


def test_the_network_rmse_is_split_at_its_first_restart(tmp_path: Path) -> None:
    path = write_metrics(
        tmp_path / "m.csv",
        [
            (1.0, "gauge", "neural.rmse_vol_bp", 680.0, {}),
            (2.0, "gauge", "neural.rmse_vol_bp", 200.0, {}),
            (3.0, "counter", "neural.restart", 1, {}),
            (3.0, "gauge", "neural.rmse_vol_bp", 46.0, {}),
            (4.0, "gauge", "neural.rmse_vol_bp", 48.0, {}),
        ],
    )

    before, after = regimes(read_metrics(path))

    assert (before.count, before.maximum) == (2, 680.0)
    assert (after.count, after.maximum) == (2, 48.0)


def test_without_a_restart_every_neural_point_is_the_first_regime(tmp_path: Path) -> None:
    path = write_metrics(tmp_path / "m.csv", [(1.0, "gauge", "neural.rmse_vol_bp", 5.0, {})])

    before, after = regimes(read_metrics(path))

    assert (before.count, after.count) == (1, 0)


def test_conflation_losses_exclude_the_benchmarks_own_tap(tmp_path: Path) -> None:
    path = write_metrics(
        tmp_path / "m.csv",
        [
            (0.0, "counter", "bus.dropped", 1, {"subscriber": "svi-scipy@X"}),
            (1.0, "counter", "bus.dropped", 1, {"subscriber": f"{TAP_PREFIX}svi-scipy@X"}),
        ],
    )

    assert dropped_by_conflation(read_metrics(path), TAP_PREFIX) == 1


def test_handler_failures_are_counted_per_producer_and_market(tmp_path: Path) -> None:
    path = write_metrics(
        tmp_path / "m.csv",
        [
            (0.0, "counter", "runner.handler_failed", 1, {"subscriber": "svi-scipy@X"}),
            (1.0, "counter", "runner.handler_failed", 1, {"subscriber": "risk-svi-scipy@X"}),
        ],
    )

    assert handler_failures(read_metrics(path), "svi-scipy", "X") == 1


def test_a_summary_with_nothing_in_it_prints_a_dash_never_a_zero() -> None:
    assert fmt_summary(summarise([])) == DASH


def test_a_table_aligns_every_column_but_the_first_to_the_right() -> None:
    assert table(["a", "b", "c"], [["x", "1", "2"]]).splitlines()[1] == "|---|---:|---:|"


def test_the_cycle_override_lengthens_the_synthetic_feed_and_nothing_else() -> None:
    config = load_config(EXAMPLES / "synthetic-svi.toml")

    longer = with_cycles(config, 60)

    assert longer.markets[0].synthetic is not None
    assert config.markets[0].synthetic is not None
    assert longer.markets[0].synthetic.config.cycles == 60
    assert longer.calibration == config.calibration


def test_the_cycle_override_refuses_a_market_that_is_not_synthetic() -> None:
    config = load_config(EXAMPLES / "deribit-live.toml")

    with pytest.raises(ValueError, match="not a synthetic feed"):
        with_cycles(config, 60)


def test_the_cycle_override_refuses_a_count_that_is_not_positive() -> None:
    with pytest.raises(ValueError, match="positive"):
        with_cycles(load_config(EXAMPLES / "synthetic-svi.toml"), 0)
