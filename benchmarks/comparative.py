"""Risk's comparative report (Design §7.3) as a Markdown section.

The engine computes a ``ComparativeReport`` every time either producer on a market publishes, and
writes none: the ``ReportWriter`` port takes a ``RiskReport``, and ``docs/SEAMS.md`` (Risk) records
that a second shape needs a writer of its own. This module is the rendering a benchmark needs and
nothing more -- it is not that writer, because nothing in the running engine calls it. A
``ComparativeReportWriter`` port and its wiring would be a change to Risk and to the composition
root, and F3-G names neither.

Same book, two surfaces, and per position: the volatility each surface gives it, the difference in
value and in each greek. Then the one number neither report can produce alone, the distance between
the two surfaces themselves.
"""

from __future__ import annotations

from collections.abc import Mapping

from volengine.contracts.calibrated_surface import CalibratedSurface
from volengine.risk.domain.comparison import ComparativeReport

BASIS_POINTS = 10_000.0


def render_comparison(report: ComparativeReport) -> str:
    """One Markdown section: the verdicts, the paired lines, the distance.

    A comparison against a refusal has no lines (``ComparativeReport`` says why); the section then
    prints both verdicts and the refusing report's message rather than an empty table, because
    "no comparison was possible, and here is why" is the result in that case.
    """
    baseline, challenger = report.baseline, report.challenger
    out = [
        f"Market `{report.market_id}` · baseline `{baseline.producer_id}` "
        f"({baseline.freshness.value}) · challenger `{challenger.producer_id}` "
        f"({challenger.freshness.value})",
        "",
    ]
    if not report.comparable:
        for side in (baseline, challenger):
            if side.message is not None:
                out.append(f"- `{side.producer_id}`: {side.message}")
        out.append("")
        out.append("No comparison: at least one side has no surface it may value off.")
        return "\n".join(out) + "\n"

    out += [
        "| Position | Vol baseline | Vol challenger | Vol diff (bp) | Value diff | Delta diff "
        "| Gamma diff | Vega diff |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for line in report.lines:
        position = line.baseline.position
        name = (
            f"{position.quantity:g} x {position.underlying} {position.kind.value} "
            f"{position.strike:g} {position.expiry:%Y-%m-%d}"
        )
        out.append(
            f"| {name} | {line.baseline.vol:.4f} | {line.challenger.vol:.4f} "
            f"| {line.vol_diff * BASIS_POINTS:+.1f} | {line.value_diff:+.4f} "
            f"| {line.delta_diff:+.6f} | {line.gamma_diff:+.3e} | {line.vega_diff:+.4f} |"
        )
    out.append("")
    out.append(f"Total value difference over paired lines: {report.total_value_diff:+.4f}.")
    if report.distance is None:
        out.append("The two grids share no region, so no surface distance is defined.")
    else:
        out.append(
            f"Surface distance over {report.distance.n_points} nodes: "
            f"RMS {report.distance.rms_vol_bp:.1f} bp, max {report.distance.max_vol_bp:.1f} bp."
        )
    return "\n".join(out) + "\n"


def render_last_surfaces(last: Mapping[str, CalibratedSurface]) -> str:
    """What each side of the comparison was valued off: status, tenors, fit.

    The comparative report cannot say this itself -- ``RiskReport`` carries the snapshot instant
    and not the surface's status, on purpose (``SurfaceView`` drops it; ``docs/SEAMS.md``, Risk,
    "A ``DEGRADED`` surface is valued into a ``NORMAL`` report"). A comparison between a full
    surface and a degraded one that lost a slice is a comparison of coverage, not of models, and
    the reader needs to see that before reading the differences.
    """
    lines = []
    for producer, surface in last.items():
        tenors = ", ".join(f"{tenor:.3f}" for tenor in surface.grid.tenors)
        lines.append(
            f"- `{producer}` valued off a `{surface.status.value}` surface, snapshot "
            f"`{surface.source_snapshot_id}`, tenors (years) {tenors}, "
            f"fit RMSE {surface.fit.rmse_vol_bp:.1f} bp."
        )
    return "\n".join(lines) + "\n"
