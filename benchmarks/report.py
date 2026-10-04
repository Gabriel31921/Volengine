"""What both benchmark reports share: the provenance header, number formatting, table rows."""

from __future__ import annotations

import platform
import sys
from collections.abc import Iterable, Sequence
from dataclasses import replace
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from benchmarks.metrics_file import MetricRow, Summary, select, summarise, total
from volengine.entrypoints.config import AppConfig

DASH = "—"
"""Printed where there is no number. Never ``0``, which would read as a measurement."""


def library_version(name: str) -> str:
    """The installed version of a distribution, or ``"absent"``."""
    try:
        return version(name)
    except PackageNotFoundError:
        return "absent"


def provenance(
    config: Path, wall_seconds: float, libraries: Sequence[str], cycles: int | None = None
) -> str:
    """Where, when and on what the numbers below were taken.

    Benchmark numbers without their machine are folklore. The header is written into the report
    file itself so the numbers cannot travel without it.
    """
    stamp = datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")
    libs = ", ".join(f"{name} {library_version(name)}" for name in libraries)
    return (
        f"Measured {stamp} on {platform.system()} {platform.machine()} "
        f"({platform.processor() or 'unknown CPU'}), Python {sys.version.split()[0]}; {libs}. "
        f"Configuration `{config.as_posix()}`"
        + ("" if cycles is None else f" with the feed's `cycles` overridden to {cycles}")
        + f", session wall time {wall_seconds:.1f} s."
    )


def with_cycles(config: AppConfig, cycles: int | None) -> AppConfig:
    """The configuration with its synthetic feed's cycle count replaced, or unchanged.

    The one override the benchmarks offer: a longer session of the same market, so a regime that
    only starts after a restart has more than a couple of points. Everything else is the file.

    Raises:
        ValueError: If ``cycles`` is given and a market is not synthetic, or is not positive.
    """
    if cycles is None:
        return config
    if cycles <= 0:
        raise ValueError(f"The cycle count must be positive, got {cycles}")
    markets = []
    for market in config.markets:
        if market.synthetic is None:
            raise ValueError(f"{market.market_id} is not a synthetic feed; it has no cycle count")
        feed = replace(market.synthetic.config, cycles=cycles)
        markets.append(replace(market, synthetic=replace(market.synthetic, config=feed)))
    return replace(config, markets=tuple(markets))


def display_path(path: Path) -> Path:
    """``path`` relative to the working directory when it is inside it, so a committed report
    does not carry the home directory of whoever generated it."""
    try:
        return path.resolve().relative_to(Path.cwd())
    except ValueError:
        return path


def fmt(value: float | None, digits: int = 1, *, scientific: bool = False) -> str:
    """A number at fixed decimals (or in scientific notation), or the dash."""
    if value is None:
        return DASH
    return f"{value:.{digits}e}" if scientific else f"{value:.{digits}f}"


def fmt_summary(summary: Summary, digits: int = 1, *, scientific: bool = False) -> str:
    """``p50 / p95 / max (n)``, plus the non-finite count when there is one."""
    if summary.count == 0:
        text = DASH
    else:
        cells = (summary.p50, summary.p95, summary.maximum)
        text = " / ".join(fmt(cell, digits, scientific=scientific) for cell in cells)
        text += f" (n={summary.count})"
    if summary.non_finite:
        text += f", {summary.non_finite} non-finite"
    return text


def metric_summary(rows: Iterable[MetricRow], name: str, **tags: str) -> Summary:
    """The summary of one metric's values under the given tags."""
    return summarise(row.value for row in select(rows, name, **tags))


def counter_total(rows: Iterable[MetricRow], name: str, **tags: str) -> int:
    """The sum of a counter's increments under the given tags."""
    return int(total(select(rows, name, **tags)))


def table(header: Sequence[str], body: Iterable[Sequence[str]]) -> str:
    """A Markdown table, first column left-aligned and the rest right-aligned."""
    lines = [
        "| " + " | ".join(header) + " |",
        "|---|" + "---:|" * (len(header) - 1),
    ]
    lines += ["| " + " | ".join(row) + " |" for row in body]
    return "\n".join(lines)


def handler_failures(rows: Iterable[MetricRow], producer: str, market_id: str) -> int:
    """Snapshots a producer's handler raised on -- lost, counted by ``BusRunner`` and survived.

    Reported beside every benchmark because a loss is invisible everywhere else: the surface is
    simply never published, and a producer that loses its fast fits looks slower than it is.
    """
    return counter_total(rows, "runner.handler_failed", subscriber=f"{producer}@{market_id}")


def dropped_by_conflation(rows: Iterable[MetricRow], tap_prefix: str) -> int:
    """Events conflation discarded on the engine's own subscriptions, a tap's excluded."""
    return int(
        sum(
            row.value
            for row in select(rows, "bus.dropped")
            if not row.tags.get("subscriber", "").startswith(tap_prefix)
        )
    )


def image(path: Path, relative_to: Path, alt: str) -> str:
    """A Markdown image reference relative to the report's own directory."""
    return f"![{alt}]({path.relative_to(relative_to).as_posix()})"
