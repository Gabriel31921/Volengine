"""Reading a ``CsvMetricsSink`` file back: the reader every chart and table rests on."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest
from benchmarks.metrics_file import (
    MetricRow,
    Summary,
    as_points,
    origin,
    quantile,
    read_metrics,
    select,
    summarise,
    tag_values,
    total,
)

from tests.benchmarks.builders import START, write_metrics


def test_rows_written_by_the_sink_read_back_with_their_name_value_tags_and_instant(
    tmp_path: Path,
) -> None:
    path = write_metrics(
        tmp_path / "m.csv",
        [(1.5, "gauge", "pricing.rmse_vol_bp", 37.25, {"producer": "svi-scipy", "market": "X"})],
    )

    (row,) = read_metrics(path)

    assert row.name == "pricing.rmse_vol_bp"
    assert row.kind == "gauge"
    assert row.value == 37.25
    assert row.tags == {"market": "X", "producer": "svi-scipy"}
    assert (row.ts - START).total_seconds() == 1.5


def test_a_file_without_the_sink_header_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "other.csv"
    path.write_text("a,b,c\n1,2,3\n", encoding="utf-8")

    with pytest.raises(ValueError, match="metrics header"):
        read_metrics(path)


def test_tags_that_are_not_an_object_are_refused(tmp_path: Path) -> None:
    path = write_metrics(tmp_path / "m.csv", [(0.0, "counter", "bus.published", 1, {})])
    text = path.read_text(encoding="utf-8").replace("{}", "[]")
    path.write_text(text, encoding="utf-8")

    with pytest.raises(ValueError, match="not a JSON object"):
        read_metrics(path)


def test_a_naive_instant_is_refused() -> None:
    with pytest.raises(ValueError, match="ts"):
        MetricRow(ts=START.replace(tzinfo=None), kind="gauge", name="x", value=1.0, tags={})


def test_select_keeps_only_rows_carrying_every_requested_tag_sorted_by_instant(
    tmp_path: Path,
) -> None:
    path = write_metrics(
        tmp_path / "m.csv",
        [
            (2.0, "gauge", "pricing.rmse_vol_bp", 2.0, {"producer": "a"}),
            (1.0, "gauge", "pricing.rmse_vol_bp", 1.0, {"producer": "b"}),
            (3.0, "gauge", "pricing.rmse_vol_bp", 3.0, {"producer": "a"}),
            (4.0, "gauge", "pricing.cycle_ms", 9.0, {"producer": "a"}),
        ],
    )
    rows = read_metrics(path)
    # Written out of order on purpose: two threads emit into one file (ADR-005).
    shuffled = (rows[2], rows[0], rows[1], rows[3])

    chosen = select(shuffled, "pricing.rmse_vol_bp", producer="a")

    assert [row.value for row in chosen] == [2.0, 3.0]


def test_tag_values_lists_each_value_once_in_order_of_first_appearance(tmp_path: Path) -> None:
    path = write_metrics(
        tmp_path / "m.csv",
        [
            (0.0, "gauge", "m", 1.0, {"producer": "b"}),
            (1.0, "gauge", "m", 1.0, {"producer": "a"}),
            (2.0, "gauge", "m", 1.0, {"producer": "b"}),
            (3.0, "gauge", "other", 1.0, {"producer": "c"}),
        ],
    )

    assert tag_values(read_metrics(path), "m", "producer") == ("b", "a")


def test_points_are_seconds_since_the_origin_and_skip_non_finite_values(tmp_path: Path) -> None:
    path = write_metrics(
        tmp_path / "m.csv",
        [
            (1.0, "gauge", "m", 5.0, {}),
            (2.0, "gauge", "m", math.nan, {}),
            (4.0, "gauge", "m", 7.0, {}),
        ],
    )
    rows = read_metrics(path)

    assert as_points(rows, origin(rows)) == ((0.0, 5.0), (3.0, 7.0))


def test_a_counter_totals_its_increments(tmp_path: Path) -> None:
    path = write_metrics(
        tmp_path / "m.csv",
        [(0.0, "counter", "c", 2, {}), (1.0, "counter", "c", 3, {})],
    )

    assert total(read_metrics(path)) == 5.0


def test_an_empty_file_has_no_origin() -> None:
    with pytest.raises(ValueError, match="no origin"):
        origin(())


@pytest.mark.parametrize("q", [0.0, 0.25, 0.5, 0.95, 1.0])
def test_quantile_agrees_with_numpys_linear_default(q: float) -> None:
    values = [3.0, 1.0, 4.0, 1.5, 9.0, 2.6]

    assert quantile(values, q) == pytest.approx(float(np.quantile(values, q)), abs=1e-12)


@pytest.mark.parametrize("q", [-0.1, 1.1, math.nan])
def test_quantile_outside_the_unit_interval_is_refused(q: float) -> None:
    with pytest.raises(ValueError, match="quantile"):
        quantile([1.0], q)


def test_a_summary_keeps_non_finite_values_apart_from_the_statistics() -> None:
    summary = summarise([10.0, math.nan, 20.0, math.inf, 30.0])

    assert summary.count == 3
    assert summary.non_finite == 2
    assert summary.mean == 20.0
    assert summary.p50 == 20.0
    assert summary.maximum == 30.0


def test_the_guard_above_is_not_vacuous_a_nan_folded_into_a_mean_poisons_it() -> None:
    assert math.isnan(math.fsum([10.0, math.nan, 20.0]) / 3)


def test_a_series_with_no_finite_value_has_no_statistics_rather_than_zeros() -> None:
    summary = summarise([math.nan])

    assert (summary.count, summary.non_finite) == (0, 1)
    assert summary.mean is None
    assert summary.maximum is None


def test_a_summary_with_statistics_but_no_count_is_refused() -> None:
    with pytest.raises(ValueError, match="exactly when"):
        Summary(count=0, non_finite=0, mean=1.0, p50=None, p95=None, maximum=None)
