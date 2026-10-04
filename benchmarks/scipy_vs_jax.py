"""scipy against JAX: the same SVI objective, two optimisers (Design §5.7).

Design §5.7 names the criteria -- **time, convergence, lines of code** -- and the shipped
configuration ``examples/svi-scipy-vs-jax.toml`` is the experiment: one invented market whose true
parameters the file states, both calibrators on it, one loss (the two tables agree on every number
that defines it, which ``tests/entrypoints/test_svi_side_by_side.py`` holds). The difference
between the two surfaces is therefore a difference between two searches, not two problems.

**Three sessions, not one.** Each calibrator runs once *alone* on the file's market, and the time
and convergence columns come from those solo sessions; then both run *side by side*, which is what
the distance between the surfaces, the comparative report and the charts come from. The split is
the GIL: the two producers sit on two thread pools (ADR-005) in one process, and a fit timed while
the other engine holds the interpreter measures the pair, not the engine. The side-by-side warm
cycle is still reported, in its own row, because it is what a deployment running both would see.

What is measured, and where each number comes from:

* **Time.** ``pricing.cycle_ms``, split into the first calibration of the session (a cold start)
  and every later one (warm-started from the previous surface), and
  ``risk.surface.snapshot_to_surface_ms`` -- the fit plus the wait for a free pool. Plus one number
  the metrics cannot hold: how long each calibrator takes to *construct*, which for JAX is the
  jit compilation (ADR-009). It is taken first, in this process, because the compiled functions
  are cached per process and every later construction hits the cache.
* **Convergence.** Slices accepted of attempted, the three refusal reasons, the fit RMSE, and the
  objective evaluations per fit (``FitMetrics.n_iterations``, the same physical quantity for both
  -- ADR-029).
* **Lines of code.** The modules each engine needs beyond the shared domain, counted with and
  without docstrings and comments (:mod:`benchmarks.loc`).

Run with the ``jax`` extra::

    uv run --extra jax python -m benchmarks.scipy_vs_jax
"""

from __future__ import annotations

import argparse
import asyncio
import time
from collections.abc import Sequence
from dataclasses import dataclass, replace
from pathlib import Path

from benchmarks.charts import draw_all, write_charts
from benchmarks.comparative import render_comparison, render_last_surfaces
from benchmarks.loc import LineCount, count_files
from benchmarks.metrics_file import MetricRow, read_metrics, select, summarise
from benchmarks.report import (
    DASH,
    counter_total,
    display_path,
    dropped_by_conflation,
    fmt,
    fmt_summary,
    handler_failures,
    image,
    metric_summary,
    provenance,
    table,
    with_cycles,
)
from benchmarks.session import TAP_PREFIX, SessionResult, last_surfaces, run_session
from volengine.contracts.calibrated_surface import SurfaceStatus
from volengine.entrypoints.config import AppConfig, load_config
from volengine.entrypoints.pipeline import default_adapters

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "examples" / "svi-scipy-vs-jax.toml"
DEFAULT_OUT = ROOT / "benchmarks" / "results"
SLUG = "scipy-vs-jax"

_ADAPTERS = ROOT / "src" / "volengine" / "parametric_pricing" / "adapters"
ENGINE_MODULES: dict[str, tuple[Path, ...]] = {
    "svi-scipy": (_ADAPTERS / "scipy_calibrator.py",),
    # The JAX calibrator also imports the scipy module, for the practical bounds and the reference
    # total variance it deliberately does not restate (ADR-029); those lines are counted once,
    # against scipy, where they live.
    "svi-jax": (
        _ADAPTERS / "jax_calibrator.py",
        _ADAPTERS / "padding.py",
        _ADAPTERS / "jax_black76.py",
    ),
}
"""What each engine needs beyond the domain both share. ``jax_greeks.py`` is excluded: it sits
outside the contract and the calibrator does not import it (Design §5.8)."""


@dataclass(frozen=True, slots=True)
class Run:
    """One session and the metrics it wrote."""

    result: SessionResult
    rows: tuple[MetricRow, ...]


def construction_seconds(config: AppConfig) -> dict[str, float]:
    """How long each configured calibrator takes to build, on a cold process."""
    registry = default_adapters()
    timings: dict[str, float] = {}
    for name in config.calibration.calibrators:
        factory = registry.calibrators.get(name)
        if factory is None:
            continue
        started = time.perf_counter()
        factory(config.calibration)
        timings[name] = time.perf_counter() - started
    return timings


def _received(rows: Sequence[MetricRow], producer: str) -> str:
    counts = [
        counter_total(rows, "risk.surface.received", producer=producer, status=status.value)
        for status in (SurfaceStatus.OK, SurfaceStatus.DEGRADED, SurfaceStatus.STALE_REPUBLISH)
    ]
    return " / ".join(str(count) for count in counts)


def _slices(rows: Sequence[MetricRow], producer: str) -> str:
    accepted = sum(r.value for r in select(rows, "pricing.slices.accepted", producer=producer))
    attempted = sum(r.value for r in select(rows, "pricing.slices.attempted", producer=producer))
    return f"{accepted:.0f} / {attempted:.0f}"


def _refusals(rows: Sequence[MetricRow], producer: str) -> str:
    names = ("pricing.slice.not_converged", "pricing.slice.at_bound", "pricing.slice.rmse_exceeded")
    return " / ".join(str(counter_total(rows, name, producer=producer)) for name in names)


def _cycles(rows: Sequence[MetricRow], producer: str) -> tuple[str, str]:
    """The first cycle of the session, then a summary of every later one."""
    values = [r.value for r in select(rows, "pricing.cycle_ms", producer=producer)]
    if not values:
        return DASH, DASH
    return fmt(values[0]), fmt_summary(summarise(values[1:]))


def _evaluations(result: SessionResult, producer: str) -> str:
    fitted = [
        s.fit.n_iterations
        for s in result.surfaces.get(producer, ())
        if s.status is not SurfaceStatus.STALE_REPUBLISH
    ]
    if not fitted:
        return DASH
    warm = summarise(float(n) for n in fitted[1:])
    return f"{fitted[0]} / {fmt(warm.p50, 0)}"


def producer_table(
    solo: dict[str, Run], joint: Run, built: dict[str, float], lines: dict[str, LineCount]
) -> str:
    """Criteria as rows, producers as columns; solo sessions except where a row says otherwise."""
    producers = tuple(solo)
    body = [
        ["Construction, s (includes jit)", *(fmt(built.get(p), 2) for p in producers)],
        [
            "Surfaces received by Risk (OK / DEGRADED / STALE)",
            *(_received(solo[p].rows, p) for p in producers),
        ],
        [
            "Snapshots lost to a handler failure",
            *(str(handler_failures(solo[p].rows, p, solo[p].result.market_id)) for p in producers),
        ],
        [
            "Calibrations refused",
            *(
                str(counter_total(solo[p].rows, "pricing.calibration.refused", producer=p))
                for p in producers
            ),
        ],
        ["Slices accepted / attempted", *(_slices(solo[p].rows, p) for p in producers)],
        [
            "Slices not converged / at bound / over RMSE",
            *(_refusals(solo[p].rows, p) for p in producers),
        ],
        [
            "Fit RMSE, vol bp (p50 / p95 / max)",
            *(
                fmt_summary(metric_summary(solo[p].rows, "pricing.rmse_vol_bp", producer=p))
                for p in producers
            ),
        ],
        [
            "First cycle (cold start, thin first snapshot), ms",
            *(_cycles(solo[p].rows, p)[0] for p in producers),
        ],
        [
            "Later cycles (warm start), ms (p50 / p95 / max)",
            *(_cycles(solo[p].rows, p)[1] for p in producers),
        ],
        [
            "Later cycles **side by side**, ms (p50 / p95 / max)",
            *(_cycles(joint.rows, p)[1] for p in producers),
        ],
        [
            "Snapshot to surface, ms (p50 / p95 / max)",
            *(
                fmt_summary(
                    metric_summary(solo[p].rows, "risk.surface.snapshot_to_surface_ms", producer=p)
                )
                for p in producers
            ),
        ],
        [
            "Objective evaluations, first / warm p50",
            *(_evaluations(solo[p].result, p) for p in producers),
        ],
        [
            "Lines: code / physical",
            *(f"{lines[p].code} / {lines[p].physical}" if p in lines else DASH for p in producers),
        ],
    ]
    return table(["Criterion", *(f"`{p}`" for p in producers)], body)


def build_report(
    config_path: Path,
    out_dir: Path,
    solo: dict[str, Run],
    joint: Run,
    built: dict[str, float],
    cycles: int | None = None,
) -> str:
    """The whole Markdown report; the side-by-side session's charts are written beside it."""
    charts = write_charts(draw_all(joint.rows), out_dir, prefix=f"{SLUG}-")
    lines = {name: count_files(paths) for name, paths in ENGINE_MODULES.items()}
    wall = joint.result.wall_seconds + sum(run.result.wall_seconds for run in solo.values())
    distance = metric_summary(joint.rows, "risk.comparison.distance_rms_vol_bp")
    comparison = (
        render_last_surfaces(last_surfaces(joint.result))
        + "\n"
        + render_comparison(joint.result.comparison)
        if joint.result.comparison is not None
        else "No comparison: a producer never published.\n"
    )
    sections = [
        "# scipy vs. JAX — the same SVI objective, two optimisers",
        "",
        provenance(display_path(config_path), wall, ("numpy", "scipy", "jax", "optax"), cycles),
        "",
        "Generated by `benchmarks/scipy_vs_jax.py`. Every number below except construction time "
        "and lines of code is read back from a session's metrics file: "
        + ", ".join(f"`{SLUG}.{p}.metrics.csv`" for p in solo)
        + f" for the solo sessions and `{SLUG}.metrics.csv` for the side-by-side one.",
        "",
        "## Time, convergence, lines of code",
        "",
        producer_table(solo, joint, built, lines),
        "",
        "## Side by side",
        "",
        f"Distance between the two surfaces, RMS vol bp (p50 / p95 / max): {fmt_summary(distance)}."
        f" Events discarded by conflation on the engine's subscriptions: "
        f"{dropped_by_conflation(joint.rows, TAP_PREFIX)}.",
        "",
        "### Risk's comparative report, at the end of the session",
        "",
        comparison,
        "### Charts",
        "",
    ]
    sections += [image(path, out_dir, path.stem) for path in charts]
    return "\n".join(sections) + "\n"


async def _session(config: AppConfig, metrics_path: Path) -> Run:
    result = await run_session(config, metrics_path)
    return Run(result=result, rows=read_metrics(metrics_path))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="scipy vs. JAX on one synthetic market.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--cycles", type=int, default=None, help="Override the feed's cycles.")
    args = parser.parse_args(argv)
    config_path: Path = args.config.resolve()
    out_dir: Path = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    config = with_cycles(load_config(config_path), args.cycles)
    built = construction_seconds(config)
    solo = {
        producer: asyncio.run(
            _session(
                replace(config, calibration=replace(config.calibration, calibrators=(producer,))),
                out_dir / f"{SLUG}.{producer}.metrics.csv",
            )
        )
        for producer in dict.fromkeys(config.calibration.calibrators)
    }
    joint = asyncio.run(_session(config, out_dir / f"{SLUG}.metrics.csv"))
    report = out_dir / f"{SLUG}.md"
    text = build_report(config_path, out_dir, solo, joint, built, args.cycles)
    report.write_text(text, encoding="utf-8")
    print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
