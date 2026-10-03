"""The metrics file the write-up charts are drawn from (Design 8.3).

What is pinned: the header and the row shape, the clock the rows are stamped with, that tags are
unambiguous, that a non-finite value is written rather than refused, that appending keeps an
earlier run, and that two threads cannot tear a row.
"""

from __future__ import annotations

import csv
import json
import math
import threading
from datetime import UTC, datetime
from pathlib import Path

import pytest

from volengine.market_data.domain.ports import MetricsSink as MarketDataMetrics
from volengine.neural_surface.domain.ports import MetricsSink as NeuralMetrics
from volengine.parametric_pricing.domain.ports import MetricsSink as PricingMetrics
from volengine.platform.adapters.csv_metrics_sink import COLUMNS, CsvMetricsSink
from volengine.platform.clock import ManualClock
from volengine.platform.metrics import MetricsSink as PlatformMetrics
from volengine.risk.domain.ports import MetricsSink as RiskMetrics

TS = datetime(2026, 7, 27, 8, 0, tzinfo=UTC)


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def test_the_sink_satisfies_every_contexts_metrics_port(tmp_path: Path) -> None:
    """Structural conformance is checked where it is assigned, which is here and in the root."""
    with CsvMetricsSink(tmp_path / "m.csv", ManualClock(TS)) as sink:
        ports: tuple[
            PlatformMetrics, MarketDataMetrics, PricingMetrics, NeuralMetrics, RiskMetrics
        ] = (sink, sink, sink, sink, sink)

    assert len(ports) == 5


def test_a_new_file_starts_with_the_header(tmp_path: Path) -> None:
    path = tmp_path / "m.csv"
    CsvMetricsSink(path, ManualClock(TS)).close()

    assert path.read_text(encoding="utf-8") == ",".join(COLUMNS) + "\n"


def test_each_kind_of_measurement_is_one_row(tmp_path: Path) -> None:
    path = tmp_path / "m.csv"
    with CsvMetricsSink(path, ManualClock(TS)) as sink:
        sink.gauge("pricing.rmse_vol_bp", 12.5, market="BTC-DERIBIT", producer="svi-scipy")
        sink.counter("bus.dropped", topic="snapshots.BTC-DERIBIT")
        sink.timing("pricing.cycle_ms", 8.25, producer="svi-scipy")

    rows = read_rows(path)

    assert [(row["kind"], row["name"], row["value"]) for row in rows] == [
        ("gauge", "pricing.rmse_vol_bp", "12.5"),
        ("counter", "bus.dropped", "1"),
        ("timing", "pricing.cycle_ms", "8.25"),
    ]


def test_rows_are_stamped_by_the_injected_clock_not_the_wall(tmp_path: Path) -> None:
    """A replay must plot recorded time, not the seconds it took to replay it (ADR-004)."""
    path = tmp_path / "m.csv"
    clock = ManualClock(TS)
    with CsvMetricsSink(path, clock) as sink:
        sink.counter("a")
        clock.advance(90.0)
        sink.counter("b")

    assert [row["ts"] for row in read_rows(path)] == [
        "2026-07-27T08:00:00+00:00",
        "2026-07-27T08:01:30+00:00",
    ]


def test_tags_are_a_json_object_with_sorted_keys(tmp_path: Path) -> None:
    """Sorted so that one measurement always renders identically, whatever the keyword order."""
    path = tmp_path / "m.csv"
    with CsvMetricsSink(path, ManualClock(TS)) as sink:
        sink.counter("x", subscriber="risk-svi@BTC", topic="surfaces.BTC")
        sink.counter("x", topic="surfaces.BTC", subscriber="risk-svi@BTC")

    first, second = read_rows(path)

    assert first["tags"] == second["tags"] == '{"subscriber":"risk-svi@BTC","topic":"surfaces.BTC"}'


def test_a_tag_containing_delimiters_survives_the_round_trip(tmp_path: Path) -> None:
    """The reason the tags are JSON and not ``key=value;key=value``."""
    awkward = 'a,b;c=d "quoted"'
    path = tmp_path / "m.csv"
    with CsvMetricsSink(path, ManualClock(TS)) as sink:
        sink.counter("x", label=awkward)

    [row] = read_rows(path)

    assert json.loads(row["tags"]) == {"label": awkward}


def test_a_measurement_with_no_tags_writes_an_empty_object(tmp_path: Path) -> None:
    path = tmp_path / "m.csv"
    with CsvMetricsSink(path, ManualClock(TS)) as sink:
        sink.counter("x")

    assert read_rows(path)[0]["tags"] == "{}"


def test_a_non_finite_gauge_is_written_not_refused(tmp_path: Path) -> None:
    """A NaN gauge is a finding about the emitter; a sink that raised would make it an outage."""
    path = tmp_path / "m.csv"
    with CsvMetricsSink(path, ManualClock(TS)) as sink:
        sink.gauge("broken", math.nan)
        sink.gauge("unbounded", math.inf)

    assert [row["value"] for row in read_rows(path)] == ["nan", "inf"]


def test_an_integer_gauge_is_written_as_a_float(tmp_path: Path) -> None:
    """So that the value column of a gauge parses as one type."""
    path = tmp_path / "m.csv"
    with CsvMetricsSink(path, ManualClock(TS)) as sink:
        sink.gauge("size", 3)

    assert read_rows(path)[0]["value"] == "3.0"


def test_values_are_written_at_full_precision(tmp_path: Path) -> None:
    path = tmp_path / "m.csv"
    with CsvMetricsSink(path, ManualClock(TS)) as sink:
        sink.gauge("tiny", 1.234567890123e-19)

    assert float(read_rows(path)[0]["value"]) == 1.234567890123e-19


def test_a_second_run_appends_without_a_second_header(tmp_path: Path) -> None:
    path = tmp_path / "m.csv"
    with CsvMetricsSink(path, ManualClock(TS)) as sink:
        sink.counter("first")
    with CsvMetricsSink(path, ManualClock(TS)) as sink:
        sink.counter("second")

    assert [row["name"] for row in read_rows(path)] == ["first", "second"]
    assert path.read_text(encoding="utf-8").count("ts,kind,name,value,tags") == 1


def test_flush_makes_rows_visible_while_the_sink_is_open(tmp_path: Path) -> None:
    path = tmp_path / "m.csv"
    sink = CsvMetricsSink(path, ManualClock(TS))
    try:
        sink.counter("x")
        sink.flush()

        assert [row["name"] for row in read_rows(path)] == ["x"]
    finally:
        sink.close()


def test_close_is_idempotent(tmp_path: Path) -> None:
    sink = CsvMetricsSink(tmp_path / "m.csv", ManualClock(TS))
    sink.close()
    sink.close()


def test_a_metric_after_close_is_an_error_not_silence(tmp_path: Path) -> None:
    """A measurement after the run ended is a shutdown-ordering bug; silence would hide it."""
    sink = CsvMetricsSink(tmp_path / "m.csv", ManualClock(TS))
    sink.close()

    with pytest.raises(ValueError, match="closed file"):
        sink.counter("late")


def test_a_path_that_cannot_be_opened_fails_at_construction(tmp_path: Path) -> None:
    with pytest.raises(OSError):
        CsvMetricsSink(tmp_path / "missing" / "m.csv", ManualClock(TS))


def test_two_threads_never_tear_a_row(tmp_path: Path) -> None:
    """The calibrations emit from their pools (ADR-005) while the loop emits everything else."""
    path = tmp_path / "m.csv"
    per_thread = 2_000
    with CsvMetricsSink(path, ManualClock(TS)) as sink:

        def emit(producer: str) -> None:
            for index in range(per_thread):
                sink.gauge("pricing.rmse_vol_bp", float(index), producer=producer)

        threads = [threading.Thread(target=emit, args=(name,)) for name in ("svi", "mlp")]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

    rows = read_rows(path)

    assert len(rows) == 2 * per_thread
    assert all(set(row) == set(COLUMNS) and None not in row.values() for row in rows)
    assert all(json.loads(row["tags"])["producer"] in {"svi", "mlp"} for row in rows)
