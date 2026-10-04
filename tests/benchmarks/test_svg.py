"""Line charts as SVG text: every point inside the frame, every series in the legend."""

from __future__ import annotations

import math
import re

import pytest
from benchmarks.svg import HEIGHT, WIDTH, Series, line_chart, nice_ticks

FRAME = re.compile(r'<rect x="(\d+)" y="(\d+)" width="(\d+)" height="(\d+)" fill="none"')
CIRCLE = re.compile(r'<circle cx="([-\d.]+)" cy="([-\d.]+)"')


def two_series() -> list[Series]:
    return [
        Series("svi-scipy", ((0.0, 37.0), (1.0, 41.5), (2.0, 39.0))),
        Series("svi-jax", ((0.5, 38.0), (1.5, 88.0))),
    ]


def test_one_polyline_per_series_and_one_dot_per_point() -> None:
    svg = line_chart("t", "x", "y", two_series())

    assert svg.count("<polyline") == 2
    assert svg.count("<circle") == 5


def test_every_point_is_drawn_inside_the_plot_frame() -> None:
    # The axis must reach the largest value: a tick range that stopped short would draw the
    # 88 bp point above the frame, which is how a chart quietly crops an outlier.
    svg = line_chart("t", "x", "y", two_series())
    match = FRAME.search(svg)
    assert match is not None
    left, top, width, height = (float(group) for group in match.groups())

    for cx, cy in CIRCLE.findall(svg):
        assert left <= float(cx) <= left + width
        assert top <= float(cy) <= top + height


def test_legend_labels_are_escaped() -> None:
    svg = line_chart("a < b", "x", "y", [Series("p&q", ((0.0, 1.0),))])

    assert "a &lt; b" in svg
    assert "p&amp;q" in svg


def test_an_empty_series_is_listed_as_having_no_data_rather_than_dropped() -> None:
    svg = line_chart("t", "x", "y", [*two_series(), Series("mlp-torch", ())])

    assert "mlp-torch (no data)" in svg


def test_a_chart_with_nothing_to_draw_is_refused() -> None:
    with pytest.raises(ValueError, match="Nothing to draw"):
        line_chart("t", "x", "y", [Series("a", ())])


def test_a_non_finite_point_is_refused() -> None:
    with pytest.raises(ValueError, match="non-finite"):
        Series("a", ((0.0, math.nan),))


def test_the_same_input_draws_the_same_bytes() -> None:
    assert line_chart("t", "x", "y", two_series()) == line_chart("t", "x", "y", two_series())


def test_the_document_declares_its_size() -> None:
    svg = line_chart("t", "x", "y", two_series())

    assert f'width="{WIDTH}" height="{HEIGHT}"' in svg


@pytest.mark.parametrize(
    ("low", "high"),
    [(0.0, 88.0), (0.0, 1.0), (37.0, 41.5), (0.0, 0.00012), (-3.0, 7.0), (0.0, 12_345.0)],
)
def test_ticks_cover_the_whole_range(low: float, high: float) -> None:
    ticks = nice_ticks(low, high)

    assert ticks[0] <= low
    assert ticks[-1] >= high
    assert len(ticks) >= 2


def test_a_flat_range_still_gets_an_axis_around_its_value() -> None:
    ticks = nice_ticks(5.0, 5.0)

    assert ticks[0] < 5.0 < ticks[-1]


def test_an_inverted_range_is_refused() -> None:
    with pytest.raises(ValueError, match="Cannot place ticks"):
        nice_ticks(2.0, 1.0)
