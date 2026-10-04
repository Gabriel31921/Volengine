"""Every chart, drawn from a ``CsvMetricsSink`` file and nothing else (Design §8.3, §12).

The metrics are "the direct basis of the charts for write-ups" (Design §8.3), and this module
takes that literally: each chart below is one metric name, grouped by one tag, plotted against
seconds since the first row of the file. Nothing is recomputed from surfaces or reports, so a chart
of a session says exactly what the engine measured during it -- and it can be drawn from any
session's file, including one an operator recorded with ``[metrics] sink = "csv"`` and no
benchmark in sight::

    uv run python -m benchmarks.charts metrics.csv --out-dir charts/

A chart whose metric is absent from the file is skipped, not drawn empty: a session without a
neural producer has no drift to chart.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

from benchmarks.metrics_file import MetricRow, as_points, origin, read_metrics, select, tag_values
from benchmarks.svg import Series, line_chart

SECONDS = "seconds since the session's first measurement"


@dataclass(frozen=True, slots=True)
class ChartSpec:
    """One chart: which metric, split by which tag, and how to label it."""

    slug: str
    """The file name, without the extension."""

    title: str
    y_label: str
    metric: str
    group_by: str | None
    """The tag whose values become the series, or ``None`` for one series named after the
    metric -- the neural context tags by market only, because it has one producer."""

    label: Callable[[str], str] = str
    """How a tag value is shown in the legend."""

    also: tuple[tuple[str, str], ...] = ()
    """Further ``(metric, legend label)`` pairs drawn as one series each on the same axes -- how
    the parametric and the neural RMSE, which carry different names, share one chart."""


STANDARD_CHARTS: tuple[ChartSpec, ...] = (
    ChartSpec(
        "rmse",
        "Fit RMSE of every published surface, all producers",
        "RMSE (vol bp)",
        "pricing.rmse_vol_bp",
        "producer",
        also=(("neural.rmse_vol_bp", "neural"),),
    ),
    ChartSpec(
        "cycle-ms",
        "Calibration cycle time (the fit alone)",
        "milliseconds",
        "pricing.cycle_ms",
        "producer",
    ),
    ChartSpec(
        "snapshot-to-surface",
        "Snapshot to surface, as Risk receives it",
        "milliseconds",
        "risk.surface.snapshot_to_surface_ms",
        "producer",
    ),
    ChartSpec(
        "distance",
        "Distance between producers (RMS over shared nodes)",
        "vol bp",
        "risk.comparison.distance_rms_vol_bp",
        "challenger",
        label=lambda challenger: f"{challenger} vs baseline",
    ),
    ChartSpec(
        "restart-drift",
        "Neural restart drift (before vs after a scheduled retrain)",
        "vol bp",
        "neural.restart.drift_vol_bp",
        None,
    ),
    ChartSpec(
        "neural-gate",
        "Neural butterfly depth on the gate's mesh",
        "depth of g below zero",
        "neural.butterfly_violation",
        None,
    ),
)


def draw(rows: Sequence[MetricRow], spec: ChartSpec) -> str | None:
    """One chart as SVG text, or ``None`` when the file holds no finite value of its metrics."""
    if not rows:
        return None
    start = origin(rows)
    if spec.group_by is None:
        series = [Series(spec.metric, as_points(select(rows, spec.metric), start))]
    else:
        series = [
            Series(
                spec.label(value),
                as_points(select(rows, spec.metric, **{spec.group_by: value}), start),
            )
            for value in tag_values(rows, spec.metric, spec.group_by)
        ]
    series += [
        Series(label, as_points(select(rows, metric), start))
        for metric, label in spec.also
        if any(row.name == metric for row in rows)
    ]
    if not any(one.points for one in series):
        return None
    return line_chart(spec.title, SECONDS, spec.y_label, series)


def draw_all(
    rows: Sequence[MetricRow], specs: Sequence[ChartSpec] = STANDARD_CHARTS
) -> dict[str, str]:
    """Every chart the rows can feed, keyed by slug."""
    drawn = {spec.slug: draw(rows, spec) for spec in specs}
    return {slug: svg for slug, svg in drawn.items() if svg is not None}


def write_charts(charts: dict[str, str], out_dir: Path, prefix: str = "") -> tuple[Path, ...]:
    """Write each chart as ``<prefix><slug>.svg`` and return the paths, in order."""
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for slug, svg in charts.items():
        path = out_dir / f"{prefix}{slug}.svg"
        path.write_text(svg, encoding="utf-8")
        written.append(path)
    return tuple(written)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Draw the standard charts from a metrics CSV.")
    parser.add_argument("metrics", type=Path, help="A file written by CsvMetricsSink.")
    parser.add_argument("--out-dir", type=Path, default=Path("."), help="Where the SVGs go.")
    parser.add_argument("--prefix", default="", help="Prepended to every file name.")
    args = parser.parse_args(argv)
    for path in write_charts(draw_all(read_metrics(args.metrics)), args.out_dir, args.prefix):
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
