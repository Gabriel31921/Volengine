"""Builders for the benchmark harness's tests: grids by total variance, and metrics files."""

from __future__ import annotations

import math
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from pathlib import Path

from volengine.contracts.calibrated_surface import VolGrid
from volengine.platform.adapters.csv_metrics_sink import CsvMetricsSink
from volengine.platform.clock import ManualClock

START = datetime(2026, 10, 4, 12, 0, tzinfo=UTC)

K_AXIS = tuple(-0.4 + 0.05 * step for step in range(17))
"""The shipped examples' published mesh: 17 uniform nodes over [-0.4, 0.4]."""

TENORS = (30 / 365.0, 90 / 365.0, 1.0)


def make_grid(
    total_variance: Callable[[float, float], float] | None = None,
    tenors: tuple[float, ...] = TENORS,
    log_moneyness: tuple[float, ...] = K_AXIS,
) -> VolGrid:
    """A published grid whose total variance at ``(tenor, k)`` is the function given.

    The default is a mild, arbitrage-free smile whose level grows with tenor: convex in ``k``,
    increasing in ``T`` at every node. Stated in total variance because both no-arbitrage
    conditions are statements about it.
    """

    def healthy(tenor: float, k: float) -> float:
        return tenor * (0.25 + 0.2 * k * k)

    w = healthy if total_variance is None else total_variance
    return VolGrid(
        log_moneyness=log_moneyness,
        tenors=tenors,
        expiries=tuple(START + timedelta(days=365.0 * tenor) for tenor in tenors),
        forwards=tuple(60_000.0 for _ in tenors),
        vols=tuple(
            tuple(math.sqrt(w(tenor, k) / tenor) for k in log_moneyness) for tenor in tenors
        ),
    )


def write_metrics(path: Path, rows: list[tuple[float, str, str, float, dict[str, str]]]) -> Path:
    """A metrics file written by the real ``CsvMetricsSink``, one row per entry.

    Each entry is ``(seconds after START, kind, name, value, tags)``; the clock is advanced to each
    instant before the row is emitted, so the file is stamped exactly as a session would be.
    """
    clock = ManualClock(START)
    elapsed = 0.0
    with CsvMetricsSink(path, clock) as sink:
        for seconds, kind, name, value, tags in rows:
            clock.advance(seconds - elapsed)
            elapsed = seconds
            if kind == "gauge":
                sink.gauge(name, value, **tags)
            elif kind == "counter":
                sink.counter(name, int(value), **tags)
            elif kind == "timing":
                sink.timing(name, value, **tags)
            else:
                raise ValueError(f"unknown metric kind {kind!r}")
    return path
