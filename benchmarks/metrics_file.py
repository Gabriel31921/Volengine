"""Reading a ``CsvMetricsSink`` file back: rows, filters, series and summaries.

The sink's own docstring names its reader -- something "grouping by ``name`` and plotting ``value``
against ``ts``" -- and this is that reader, in the standard library, so that drawing a chart does
not add pandas to an engine that does not otherwise need it.

**Non-finite values are kept apart, never averaged.** The sink refuses no value on purpose
(``csv_metrics_sink._emit``): a ``nan`` RMSE is a finding about the code that emitted it. A summary
that folded it into a mean would print ``nan`` and hide every other number; one that dropped it
silently would report a broken series as healthy. :class:`Summary` therefore counts the finite
values it summarises and, separately, the ones it could not.
"""

from __future__ import annotations

import csv
import json
import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from volengine.platform.adapters.csv_metrics_sink import COLUMNS
from volengine.shared_kernel.domain.instants import require_aware


@dataclass(frozen=True, slots=True)
class MetricRow:
    """One measurement, as the sink wrote it."""

    ts: datetime
    kind: str
    name: str
    value: float
    tags: Mapping[str, str]

    def __post_init__(self) -> None:
        require_aware(self.ts, "ts")
        if not self.name:
            raise ValueError("A metric row must carry a name")


def read_metrics(path: Path) -> tuple[MetricRow, ...]:
    """Every row of one metrics file, in file order.

    Args:
        path: A file written by ``CsvMetricsSink``.

    Returns:
        The rows, in the order they were written -- which is not strictly the order of ``ts``,
        because two threads emit into one file (ADR-005) and a row is stamped before it takes the
        lock. Sort by ``ts`` if order matters.

    Raises:
        ValueError: If the header is not the sink's, or a row does not parse. Refused rather than
            skipped: a malformed metrics file means the benchmark is reading something the engine
            did not write.
    """
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader, None)
        if header is None or tuple(header) != COLUMNS:
            raise ValueError(f"{path} does not start with the metrics header {COLUMNS}")
        rows: list[MetricRow] = []
        for number, cells in enumerate(reader, start=2):
            if len(cells) != len(COLUMNS):
                raise ValueError(f"{path}:{number} has {len(cells)} cells, expected {len(COLUMNS)}")
            ts, kind, name, value, tags = cells
            decoded = json.loads(tags)
            if not isinstance(decoded, dict):
                raise ValueError(f"{path}:{number} carries tags that are not a JSON object")
            rows.append(
                MetricRow(
                    ts=datetime.fromisoformat(ts),
                    kind=kind,
                    name=name,
                    value=float(value),
                    tags={str(key): str(tag) for key, tag in decoded.items()},
                )
            )
    return tuple(rows)


def select(rows: Iterable[MetricRow], name: str, **tags: str) -> tuple[MetricRow, ...]:
    """The rows of one metric whose tags include every ``key=value`` given, sorted by ``ts``."""
    chosen = (
        row
        for row in rows
        if row.name == name and all(row.tags.get(key) == value for key, value in tags.items())
    )
    return tuple(sorted(chosen, key=lambda row: row.ts))


def tag_values(rows: Iterable[MetricRow], name: str, tag: str) -> tuple[str, ...]:
    """Every distinct value ``tag`` takes on metric ``name``, in order of first appearance."""
    found = (row.tags[tag] for row in rows if row.name == name and tag in row.tags)
    return tuple(dict.fromkeys(found))


def origin(rows: Sequence[MetricRow]) -> datetime:
    """The earliest instant in a file: where every series' time axis starts.

    Raises:
        ValueError: If there are no rows.
    """
    if not rows:
        raise ValueError("An empty metrics file has no origin")
    return min(row.ts for row in rows)


def as_points(rows: Iterable[MetricRow], start: datetime) -> tuple[tuple[float, float], ...]:
    """``(seconds since start, value)`` for every finite row -- the shape a line chart takes."""
    return tuple(
        ((row.ts - start).total_seconds(), row.value) for row in rows if math.isfinite(row.value)
    )


def total(rows: Iterable[MetricRow]) -> float:
    """The sum of a counter's increments."""
    return math.fsum(row.value for row in rows)


@dataclass(frozen=True, slots=True)
class Summary:
    """What a series came to: its finite values summarised, its non-finite ones counted.

    The statistics are ``None`` when there is no finite value to summarise, rather than ``nan`` or
    zero -- a zero would read as a perfect fit and a ``nan`` would propagate into every table cell
    that arithmetic touched.
    """

    count: int
    non_finite: int
    mean: float | None
    p50: float | None
    p95: float | None
    maximum: float | None

    def __post_init__(self) -> None:
        if self.count < 0 or self.non_finite < 0:
            raise ValueError("Counts cannot be negative")
        stats = (self.mean, self.p50, self.p95, self.maximum)
        if (self.count == 0) != all(stat is None for stat in stats):
            raise ValueError("The statistics are present exactly when there is a finite value")


def quantile(values: Sequence[float], q: float) -> float:
    """The ``q``-quantile by linear interpolation between order statistics (numpy's default).

    Raises:
        ValueError: If ``values`` is empty, holds a non-finite value, or ``q`` is outside [0, 1].
    """
    if not math.isfinite(q) or q < 0.0 or q > 1.0:
        raise ValueError(f"A quantile must lie in [0, 1], got {q}")
    if not values:
        raise ValueError("The quantile of no values is undefined")
    if not all(math.isfinite(value) for value in values):
        raise ValueError("Quantiles are taken over finite values only")
    ordered = sorted(values)
    position = q * (len(ordered) - 1)
    lower = math.floor(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def summarise(values: Iterable[float]) -> Summary:
    """Count, mean, median, 95th percentile and maximum of the finite values given."""
    everything = list(values)
    finite = [value for value in everything if math.isfinite(value)]
    non_finite = len(everything) - len(finite)
    if not finite:
        return Summary(0, non_finite, None, None, None, None)
    return Summary(
        count=len(finite),
        non_finite=non_finite,
        mean=math.fsum(finite) / len(finite),
        p50=quantile(finite, 0.5),
        p95=quantile(finite, 0.95),
        maximum=max(finite),
    )
