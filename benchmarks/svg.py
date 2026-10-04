"""Line charts as SVG text, in the standard library.

**No plotting library, on purpose.** matplotlib would be the first dependency in this repository
that exists only to draw pictures, and it would have to be added to ``pyproject.toml`` and
``uv.lock`` for every installation of an engine that never draws one. A line chart of a handful of
series is a few dozen lines of SVG; GitHub renders SVG inline in a README; and the output is text,
so two runs over the same metrics file produce the same bytes and a regenerated chart diffs like
any other file.

The charts are deliberately plain: one polyline per series, a dot per point, linear axes with
rounded ticks, and a legend. Anything fancier belongs in a notebook reading the same CSV.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from xml.sax.saxutils import escape

PALETTE: tuple[str, ...] = ("#1f77b4", "#d62728", "#2ca02c", "#ff7f0e", "#9467bd", "#8c564b")
"""One colour per series, in order. Six is more than any chart here draws."""

WIDTH = 760
HEIGHT = 380
_MARGIN_LEFT = 72
_MARGIN_RIGHT = 180
_MARGIN_TOP = 40
_MARGIN_BOTTOM = 52


@dataclass(frozen=True, slots=True)
class Series:
    """One named line: ``(x, y)`` points, drawn in the order given."""

    label: str
    points: tuple[tuple[float, float], ...]

    def __post_init__(self) -> None:
        if not self.label:
            raise ValueError("A series needs a label for the legend")
        for x, y in self.points:
            # Finiteness first: a NaN coordinate would print as "nan" inside the path and the
            # browser would drop the whole line without a word.
            if not math.isfinite(x) or not math.isfinite(y):
                raise ValueError(f"Series {self.label!r} holds a non-finite point ({x}, {y})")


def nice_ticks(low: float, high: float, target: int = 5) -> tuple[float, ...]:
    """Round tick positions covering ``[low, high]``: steps of 1, 2 or 5 times a power of ten.

    Raises:
        ValueError: If either bound is not finite or ``high < low``.
    """
    if not math.isfinite(low) or not math.isfinite(high) or high < low:
        raise ValueError(f"Cannot place ticks on [{low}, {high}]")
    if high == low:
        # A flat series still needs an axis; widen symmetrically around the value.
        pad = abs(low) * 0.1 if low != 0.0 else 1.0
        low, high = low - pad, high + pad
    raw = (high - low) / max(target, 1)
    magnitude = 10.0 ** math.floor(math.log10(raw))
    step = next(m * magnitude for m in (1.0, 2.0, 5.0, 10.0) if m * magnitude >= raw)
    first = math.floor(low / step) * step
    # Enough steps that the last tick reaches `high`: the axis must contain every point, or a
    # series would be drawn outside the frame. The epsilon absorbs a quotient that lands a
    # rounding error above an integer.
    count = max(1, math.ceil((high - first) / step - 1e-9))
    return tuple(round(first + index * step, 12) for index in range(count + 1))


def _label(value: float) -> str:
    """A tick label short enough to fit the margin."""
    if value == 0.0:
        return "0"
    magnitude = abs(value)
    if magnitude >= 1e5 or magnitude < 1e-3:
        return f"{value:.0e}"
    return f"{value:g}"


def line_chart(
    title: str,
    x_label: str,
    y_label: str,
    series: Sequence[Series],
    *,
    y_floor: float | None = 0.0,
) -> str:
    """One SVG document: every series on shared axes, with a legend.

    Args:
        title: Printed above the plot.
        x_label: Under the horizontal axis.
        y_label: Beside the vertical axis.
        series: The lines. Empty series are listed in the legend as "(no data)" rather than
            dropped, so a producer that never published is visibly absent instead of silently so.
        y_floor: The lowest the vertical axis may start, ``0.0`` by default because every quantity
            charted here (RMSE, milliseconds, distances) is non-negative and an axis starting at
            the minimum exaggerates small differences. ``None`` fits the axis to the data.

    Raises:
        ValueError: If no series has a single point -- an empty chart is a benchmark that measured
            nothing, and an image of empty axes would hide that.
    """
    points = [point for one in series for point in one.points]
    if not points:
        raise ValueError(f"Nothing to draw for {title!r}: every series is empty")

    xs = [x for x, _ in points]
    ys = [y for _, y in points]
    y_low = min(ys) if y_floor is None else min(y_floor, min(ys))
    x_ticks = nice_ticks(min(xs), max(xs))
    y_ticks = nice_ticks(y_low, max(ys))
    x0, x1 = x_ticks[0], x_ticks[-1]
    y0, y1 = y_ticks[0], y_ticks[-1]

    plot_w = WIDTH - _MARGIN_LEFT - _MARGIN_RIGHT
    plot_h = HEIGHT - _MARGIN_TOP - _MARGIN_BOTTOM

    def sx(x: float) -> float:
        return _MARGIN_LEFT + (x - x0) / (x1 - x0) * plot_w

    def sy(y: float) -> float:
        return _MARGIN_TOP + plot_h - (y - y0) / (y1 - y0) * plot_h

    out: list[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{HEIGHT}" '
        f'viewBox="0 0 {WIDTH} {HEIGHT}" font-family="sans-serif" font-size="12">',
        f'<rect width="{WIDTH}" height="{HEIGHT}" fill="white"/>',
        f'<text x="{WIDTH / 2:.1f}" y="22" text-anchor="middle" font-size="14" '
        f'font-weight="bold">{escape(title)}</text>',
    ]
    for tick in y_ticks:
        y = sy(tick)
        out.append(
            f'<line x1="{_MARGIN_LEFT}" y1="{y:.1f}" x2="{_MARGIN_LEFT + plot_w}" y2="{y:.1f}" '
            'stroke="#e5e5e5"/>'
        )
        out.append(
            f'<text x="{_MARGIN_LEFT - 6}" y="{y + 4:.1f}" text-anchor="end">{_label(tick)}</text>'
        )
    for tick in x_ticks:
        x = sx(tick)
        out.append(
            f'<line x1="{x:.1f}" y1="{_MARGIN_TOP}" x2="{x:.1f}" y2="{_MARGIN_TOP + plot_h}" '
            'stroke="#f0f0f0"/>'
        )
        out.append(
            f'<text x="{x:.1f}" y="{_MARGIN_TOP + plot_h + 16}" text-anchor="middle">'
            f"{_label(tick)}</text>"
        )
    out.append(
        f'<rect x="{_MARGIN_LEFT}" y="{_MARGIN_TOP}" width="{plot_w}" height="{plot_h}" '
        'fill="none" stroke="#333"/>'
    )
    out.append(
        f'<text x="{_MARGIN_LEFT + plot_w / 2:.1f}" y="{HEIGHT - 12}" text-anchor="middle">'
        f"{escape(x_label)}</text>"
    )
    mid_y = _MARGIN_TOP + plot_h / 2
    out.append(
        f'<text x="16" y="{mid_y:.1f}" text-anchor="middle" '
        f'transform="rotate(-90 16 {mid_y:.1f})">{escape(y_label)}</text>'
    )

    for index, one in enumerate(series):
        colour = PALETTE[index % len(PALETTE)]
        if one.points:
            path = " ".join(f"{sx(x):.1f},{sy(y):.1f}" for x, y in one.points)
            out.append(
                f'<polyline points="{path}" fill="none" stroke="{colour}" stroke-width="1.5"/>'
            )
            out.extend(
                f'<circle cx="{sx(x):.1f}" cy="{sy(y):.1f}" r="2.5" fill="{colour}"/>'
                for x, y in one.points
            )
        legend_y = _MARGIN_TOP + 14 + index * 18
        legend_x = _MARGIN_LEFT + plot_w + 14
        name = one.label if one.points else f"{one.label} (no data)"
        out.append(
            f'<line x1="{legend_x}" y1="{legend_y - 4}" x2="{legend_x + 18}" y2="{legend_y - 4}" '
            f'stroke="{colour}" stroke-width="2"/>'
        )
        out.append(f'<text x="{legend_x + 24}" y="{legend_y}">{escape(name)}</text>')

    out.append("</svg>")
    return "\n".join(out) + "\n"
