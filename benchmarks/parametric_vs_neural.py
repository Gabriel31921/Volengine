"""The SVI fit against the network: RMSE, violations, drift (Design §12).

The two engines the project exists to compare -- five raw-SVI numbers per expiry against one small
network over the whole ``(k, T)`` plane -- on one market, behind one contract, from the shipped
``examples/svi-scipy-vs-mlp-torch.toml``. Both publish on the same ``[calibration.grid]`` nodes and
weigh quotes by the same ``[calibration.weighting]``, so the distance between them is a distance
between models (Design §6.5).

**Which regime is measured, stated before any number** (STATE's warning to F3-G, and
``docs/SEAMS.md``, Neural Surface): the network's first snapshot can rest on a handful of quotes,
its cold start fits those, and ten warm steps a snapshot move it only slowly towards the full chain
until the first scheduled restart retrains over the whole buffer. The network has no RMSE
acceptance, so everything it fits cleanly is published, bad fits included. Its RMSE is therefore
reported in **two regimes** -- before the first restart and after it -- split at the instant of the
first ``neural.restart`` row. One number over the whole session would average a cold-start
artefact into the model's steady state.

What is measured, and where each number comes from:

* **RMSE.** ``pricing.rmse_vol_bp`` and ``neural.rmse_vol_bp``: each producer's fit against the
  quotes of the snapshot it was fitted on, in the same vega-weighted vol units.
* **Violations, with one ruler.** :mod:`benchmarks.surface_checks` on every published grid of
  both producers -- Durrleman's ``g`` and the calendar condition on the nodes a consumer receives.
  Beside it, the network's own gate on its wider mesh (``neural.butterfly_violation``,
  ``neural.calendar_violation``) and every refusal either producer made.
* **Drift.** ``neural.restart.drift_vol_bp``, the RMS distance between the surface a scheduled
  restart replaces and the one it makes (Design §6.4's "honest measure of accumulated drift").
  The SVI producer has no restart -- every snapshot is a fresh fit, warm-started -- so the
  comparable number for both is the **step**: Risk's ``surface_distance`` between each producer's
  consecutive published surfaces. A model that drifts shows a large restart jump against small
  steps; a noisy one shows large steps.

Run with the ``neural`` extra::

    uv run --extra neural python -m benchmarks.parametric_vs_neural
"""

from __future__ import annotations

import argparse
import asyncio
from collections.abc import Sequence
from datetime import datetime
from itertools import pairwise
from pathlib import Path

from benchmarks.charts import draw_all, write_charts
from benchmarks.comparative import render_comparison, render_last_surfaces
from benchmarks.metrics_file import MetricRow, Summary, read_metrics, select, summarise
from benchmarks.report import (
    counter_total,
    display_path,
    dropped_by_conflation,
    fmt_summary,
    handler_failures,
    image,
    metric_summary,
    provenance,
    table,
    with_cycles,
)
from benchmarks.session import TAP_PREFIX, SessionResult, last_surfaces, run_session
from benchmarks.surface_checks import grid_arbitrage
from volengine.contracts.calibrated_surface import CalibratedSurface, SurfaceStatus
from volengine.entrypoints.config import load_config
from volengine.risk.application.acl import to_surface_view
from volengine.risk.domain.comparison import surface_distance

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "examples" / "svi-scipy-vs-mlp-torch.toml"
DEFAULT_OUT = ROOT / "benchmarks" / "results"
SLUG = "parametric-vs-neural"


def fitted(surfaces: Sequence[CalibratedSurface]) -> tuple[CalibratedSurface, ...]:
    """The surfaces that are a fit of their own snapshot -- republished ones are a copy."""
    return tuple(s for s in surfaces if s.status is not SurfaceStatus.STALE_REPUBLISH)


def first_restart(rows: Sequence[MetricRow]) -> datetime | None:
    """When the network first retrained from scratch, or ``None`` if it never did."""
    restarts = select(rows, "neural.restart")
    return restarts[0].ts if restarts else None


def regimes(rows: Sequence[MetricRow]) -> tuple[Summary, Summary]:
    """The network's RMSE before its first restart, and from it on."""
    series = select(rows, "neural.rmse_vol_bp")
    split = first_restart(rows)
    if split is None:
        return summarise(r.value for r in series), summarise(())
    return (
        summarise(r.value for r in series if r.ts < split),
        summarise(r.value for r in series if r.ts >= split),
    )


def steps(surfaces: Sequence[CalibratedSurface]) -> Summary:
    """RMS vol distance between consecutive published fits, through Risk's own measure."""
    distances: list[float] = []
    for before, after in pairwise(fitted(surfaces)):
        distance = surface_distance(to_surface_view(before), to_surface_view(after))
        if distance is not None:
            distances.append(distance.rms_vol_bp)
    return summarise(distances)


def violations(surfaces: Sequence[CalibratedSurface]) -> tuple[Summary, Summary, int]:
    """Butterfly depth and calendar crossing on every published grid, and how many breached."""
    checks = [grid_arbitrage(s.grid) for s in fitted(surfaces)]
    breached = sum(1 for c in checks if c.butterfly > 0.0 or c.calendar > 0.0)
    return (
        summarise(c.butterfly for c in checks),
        summarise(c.calendar for c in checks),
        breached,
    )


def build_report(
    config_path: Path, out_dir: Path, result: SessionResult, cycles: int | None = None
) -> str:
    """The whole Markdown report; the charts are written beside it."""
    rows = read_metrics(out_dir / f"{SLUG}.metrics.csv")
    charts = write_charts(draw_all(rows), out_dir, prefix=f"{SLUG}-")
    parametric, neural = result.producers[0], result.producers[1]
    cold, warm = regimes(rows)
    restart_at = first_restart(rows)
    p_bfly, p_cal, p_breached = violations(result.surfaces[parametric])
    n_bfly, n_cal, n_breached = violations(result.surfaces[neural])
    n_fitted = {p: len(fitted(result.surfaces[p])) for p in (parametric, neural)}

    def digits(summary: Summary) -> str:
        return fmt_summary(summary, 1, scientific=True)

    rmse = table(
        ["RMSE, vol bp (p50 / p95 / max)", f"`{parametric}`", f"`{neural}`"],
        [
            [
                "Whole session",
                fmt_summary(metric_summary(rows, "pricing.rmse_vol_bp", producer=parametric)),
                fmt_summary(metric_summary(rows, "neural.rmse_vol_bp")),
            ],
            ["Before the network's first restart", "", fmt_summary(cold)],
            ["From the first restart on", "", fmt_summary(warm)],
        ],
    )
    arbitrage = table(
        ["Violations", f"`{parametric}`", f"`{neural}`"],
        [
            ["Published fits", str(n_fitted[parametric]), str(n_fitted[neural])],
            [
                "Snapshots lost to a handler failure",
                str(handler_failures(rows, parametric, result.market_id)),
                str(handler_failures(rows, neural, result.market_id)),
            ],
            [
                "Grids with any breach on the published nodes",
                str(p_breached),
                str(n_breached),
            ],
            ["Butterfly depth on the published nodes", digits(p_bfly), digits(n_bfly)],
            ["Calendar crossing on the published nodes", digits(p_cal), digits(n_cal)],
            [
                "Butterfly depth, network's own gate mesh",
                "",
                digits(metric_summary(rows, "neural.butterfly_violation")),
            ],
            [
                "Calendar crossing, network's own gate mesh",
                "",
                digits(metric_summary(rows, "neural.calendar_violation")),
            ],
            [
                "Refused (parametric: calibration · neural: gate / diverged)",
                str(counter_total(rows, "pricing.calibration.refused", producer=parametric)),
                f"{counter_total(rows, 'neural.publication.refused')} / "
                f"{counter_total(rows, 'neural.surface.diverged')}",
            ],
        ],
    )
    drift = table(
        ["Drift, vol bp (p50 / p95 / max)", f"`{parametric}`", f"`{neural}`"],
        [
            [
                "Restart jump (before vs after a retrain)",
                "no restart",
                fmt_summary(metric_summary(rows, "neural.restart.drift_vol_bp")),
            ],
            [
                "Step between consecutive published fits",
                fmt_summary(steps(result.surfaces[parametric])),
                fmt_summary(steps(result.surfaces[neural])),
            ],
        ],
    )
    latency = table(
        ["Time, ms (p50 / p95 / max)", f"`{parametric}`", f"`{neural}`"],
        [
            [
                "Fit (FitMetrics.duration_ms)",
                fmt_summary(
                    summarise(s.fit.duration_ms for s in fitted(result.surfaces[parametric]))
                ),
                fmt_summary(summarise(s.fit.duration_ms for s in fitted(result.surfaces[neural]))),
            ],
            [
                "Snapshot to surface, as Risk receives it",
                fmt_summary(
                    metric_summary(rows, "risk.surface.snapshot_to_surface_ms", producer=parametric)
                ),
                fmt_summary(
                    metric_summary(rows, "risk.surface.snapshot_to_surface_ms", producer=neural)
                ),
            ],
        ],
    )
    distance = metric_summary(rows, "risk.comparison.distance_rms_vol_bp")
    comparison = (
        render_last_surfaces(last_surfaces(result)) + "\n" + render_comparison(result.comparison)
        if result.comparison is not None
        else "No comparison: a producer never published.\n"
    )
    if restart_at is None:
        restart_line = (
            "The network never restarted, so every neural number below is the pre-restart regime."
        )
    else:
        offset = (restart_at - min(r.ts for r in rows)).total_seconds()
        restart_line = (
            f"The network first restarted {offset:.1f} s into the session; "
            f"restarts in all: {counter_total(rows, 'neural.restart')}."
        )
    sections = [
        "# Parametric vs. neural — RMSE, violations, drift",
        "",
        provenance(
            display_path(config_path), result.wall_seconds, ("numpy", "scipy", "torch"), cycles
        ),
        "",
        "Generated by `benchmarks/parametric_vs_neural.py`. Every number below is read back from "
        f"the session's metrics file, `{SLUG}.metrics.csv`, or computed on the surfaces the "
        "session published (violations on the published nodes, consecutive steps, fit time).",
        "",
        restart_line,
        "",
        "## RMSE",
        "",
        rmse,
        "",
        "## Violations",
        "",
        arbitrage,
        "",
        "## Drift",
        "",
        drift,
        "",
        "## Time (side by side, one process)",
        "",
        latency,
        "",
        "## Between the two",
        "",
        f"Distance between the two surfaces, RMS vol bp (p50 / p95 / max): {fmt_summary(distance)}."
        f" Events discarded by conflation on the engine's subscriptions: "
        f"{dropped_by_conflation(rows, TAP_PREFIX)} -- a snapshot that arrived while a producer was"
        " still busy replaced the one waiting for it (ADR-003).",
        "",
        "### Risk's comparative report, at the end of the session",
        "",
        comparison,
        "### Charts",
        "",
    ]
    sections += [image(path, out_dir, path.stem) for path in charts]
    return "\n".join(sections) + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="The SVI fit against the network.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--cycles", type=int, default=None, help="Override the feed's cycles.")
    args = parser.parse_args(argv)
    config_path: Path = args.config.resolve()
    out_dir: Path = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    config = with_cycles(load_config(config_path), args.cycles)
    if len(dict.fromkeys(config.calibration.calibrators)) != 2:
        parser.error("the configuration must run exactly two producers, parametric first")
    result = asyncio.run(run_session(config, out_dir / f"{SLUG}.metrics.csv"))
    report = out_dir / f"{SLUG}.md"
    report.write_text(build_report(config_path, out_dir, result, args.cycles), encoding="utf-8")
    print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
