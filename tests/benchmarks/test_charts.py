"""Charts drawn from a metrics file and nothing else."""

from __future__ import annotations

from pathlib import Path

from benchmarks.charts import STANDARD_CHARTS, draw_all, main
from benchmarks.metrics_file import read_metrics

from tests.benchmarks.builders import write_metrics


def session_file(path: Path) -> Path:
    return write_metrics(
        path,
        [
            (0.0, "gauge", "pricing.rmse_vol_bp", 37.0, {"producer": "svi-scipy"}),
            (0.5, "gauge", "neural.rmse_vol_bp", 680.0, {"market": "BTC-SYNTH"}),
            (1.0, "gauge", "pricing.rmse_vol_bp", 39.0, {"producer": "svi-scipy"}),
            (1.5, "gauge", "neural.rmse_vol_bp", 47.0, {"market": "BTC-SYNTH"}),
            (2.0, "timing", "pricing.cycle_ms", 90.0, {"producer": "svi-scipy"}),
        ],
    )


def test_the_rmse_chart_puts_parametric_and_neural_producers_on_one_axis(tmp_path: Path) -> None:
    charts = draw_all(read_metrics(session_file(tmp_path / "m.csv")))

    assert "svi-scipy" in charts["rmse"]
    assert ">neural<" in charts["rmse"]


def test_a_chart_whose_metric_is_absent_is_skipped_not_drawn_empty(tmp_path: Path) -> None:
    charts = draw_all(read_metrics(session_file(tmp_path / "m.csv")))

    assert set(charts) == {"rmse", "cycle-ms"}


def test_every_standard_chart_has_a_distinct_file_name() -> None:
    slugs = [spec.slug for spec in STANDARD_CHARTS]

    assert len(slugs) == len(set(slugs))


def test_the_command_writes_one_svg_per_drawable_chart(tmp_path: Path) -> None:
    metrics = session_file(tmp_path / "m.csv")
    out = tmp_path / "charts"

    assert main([str(metrics), "--out-dir", str(out), "--prefix", "s-"]) == 0
    assert sorted(path.name for path in out.iterdir()) == ["s-cycle-ms.svg", "s-rmse.svg"]
