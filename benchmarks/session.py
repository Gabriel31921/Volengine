"""One engine session, run as ``volengine run`` runs it, with the evidence kept.

A benchmark needs three things the CLI does not hand back: the metrics file (the CLI writes one,
but this module opens it so the path is known and fresh), every surface each producer published,
and the comparative report at the end of the session. The first is the ``CsvMetricsSink``; the
other two come from **a tap on the bus** -- one extra subscription per producer's surface topic,
opened before :func:`build_pipeline` so it cannot miss the first surface.

**The tap is a subscriber, not a change to the engine.** ``build_pipeline`` takes the bus as an
argument precisely so the composition is observable from outside (ADR-022); subscribing here is
what any other consumer of ``CalibratedSurface`` would do. Its cost is visible rather than hidden:
its mailbox appears in ``bus.dropped`` under a ``benchmark-tap-`` subscriber name, which the
reports filter out of the conflation numbers.

**The comparison is recomputed once, after the run**, from the last surface each producer
published, with the configuration's own book, freshness policy and bumps, through Risk's own
``CompareProducersUseCase`` -- the use case the pipeline runs on every arrival and discards the
result of (``docs/SEAMS.md``, Risk: "measured, not written"). Same code, same inputs, kept.
"""

from __future__ import annotations

import asyncio
import contextlib
import time
from collections.abc import Mapping
from dataclasses import dataclass, replace
from datetime import UTC, datetime, timedelta
from pathlib import Path

from volengine.contracts.calibrated_surface import CalibratedSurface
from volengine.contracts.events import SurfaceCalibrated
from volengine.entrypoints.config import AppConfig
from volengine.entrypoints.pipeline import (
    Adapters,
    build_pipeline,
    default_adapters,
    surface_topic,
)
from volengine.platform.adapters.csv_metrics_sink import CsvMetricsSink
from volengine.platform.bus import InProcessConflatingBus, Subscription
from volengine.platform.clock import Clock
from volengine.platform.metrics import NullMetricsSink
from volengine.risk.application.compare_producers import CompareProducersUseCase
from volengine.risk.application.surface_cache import LastValueSurfaceProvider
from volengine.risk.domain.comparison import ComparativeReport
from volengine.risk.domain.portfolio import Portfolio
from volengine.risk.domain.risk_report import RiskReport

TAP_PREFIX = "benchmark-tap-"
"""The subscriber names the tap's mailboxes carry in ``bus.dropped``."""

SILENT_WRITER = "benchmark-silent"
"""The registry name the session's report writer is swapped to."""


class MonotonicClock:
    """Real elapsed time that never steps backwards: the wall clock once, then the monotonic one.

    **Why a benchmark does not run on ``SystemClock``.** The host these numbers were taken on
    steps its wall clock back by about 2.6 s every thirty seconds (WSL2 time sync; STATE's open
    decision on the backwards wall clock). Under ``SystemClock`` a step lands a snapshot stamped
    *after* the instant its fit finishes, ``CalibratedSurface`` refuses ``ts_calibrated <
    ts_snapshot``, and ``BusRunner`` counts the loss as ``runner.handler_failed`` and moves on.
    Measured on the scipy-alone session before this clock was used: ten of seventeen snapshots
    lost, and the faster engine loses more of them, because a slow fit outlives the step. A
    benchmark on that clock would rank the host's time sync, not the calibrators.

    ``now()`` is the wall clock read at construction plus the monotonic time elapsed since, so it
    agrees with ``datetime.now(UTC)`` at the start -- which is what the synthetic feed's own
    timeline is anchored on -- and then advances at the same rate, steps excluded. The same
    construction as ``tests/entrypoints/builders.SteadyClock``, which exists for the same reason;
    it is restated rather than imported because nothing outside ``tests/`` imports from it.
    """

    def __init__(self) -> None:
        self._start = datetime.now(UTC)
        self._origin = time.monotonic()

    def now(self) -> datetime:
        return self._start + timedelta(seconds=time.monotonic() - self._origin)

    async def sleep(self, seconds: float) -> None:
        await asyncio.sleep(seconds)


class _DiscardingWriter:
    """A ``ReportWriter`` that writes nothing.

    The per-producer reports would otherwise go to the console the example names, interleaved
    with the benchmark's own output; what a benchmark keeps of Risk is the comparative report,
    which carries both producers' reports whole.
    """

    def write(self, report: RiskReport) -> None:
        return None


@dataclass(frozen=True, slots=True)
class SessionResult:
    """What one session left behind, beside its metrics file."""

    market_id: str
    producers: tuple[str, ...]
    """In configuration order: the first is the comparison's baseline."""

    surfaces: Mapping[str, tuple[CalibratedSurface, ...]]
    """Every surface the tap took from each producer's topic, in arrival order, republished
    (``STALE_REPUBLISH``) ones included. Conflation can drop one the tap was too slow for, exactly
    as it can for Risk; ``bus.dropped`` counts it."""

    comparison: ComparativeReport | None
    """Baseline against the second producer, from each one's last surface. ``None`` when the
    configuration runs fewer than two producers or one of them never published."""

    wall_seconds: float
    """How long the run took on the wall, including the producers' construction."""


async def run_session(
    config: AppConfig,
    metrics_path: Path,
    *,
    adapters: Adapters | None = None,
    clock: Clock | None = None,
) -> SessionResult:
    """Run one single-market configuration to completion and keep the evidence.

    Args:
        config: A loaded configuration with exactly one market. Its ``[metrics]`` table is
            ignored: the session always writes to ``metrics_path``.
        metrics_path: Where the metrics go. Replaced if it exists -- the sink appends, and a
            benchmark mixing two sessions' rows in one file would summarise both as one.
        adapters: The registry; :func:`default_adapters` when omitted. Its writers are replaced
            by one that discards.
        clock: The engine's clock; a fresh :class:`MonotonicClock` when omitted.

    Raises:
        ValueError: If the configuration runs more than one market. A benchmark summarises one
            market's producers; two markets would put two chains' numbers in one table.
    """
    if len(config.markets) != 1:
        raise ValueError(f"A benchmark session runs one market, got {len(config.markets)}")
    market_id = config.markets[0].market_id
    producers = tuple(dict.fromkeys(config.calibration.calibrators))
    registry = adapters if adapters is not None else default_adapters()
    registry = replace(registry, writers={SILENT_WRITER: lambda _risk: _DiscardingWriter()})
    config = replace(config, risk=replace(config.risk, writer=SILENT_WRITER))
    clock = clock if clock is not None else MonotonicClock()

    metrics_path.unlink(missing_ok=True)
    started = time.perf_counter()
    with CsvMetricsSink(metrics_path, clock) as sink:
        bus = InProcessConflatingBus(sink)
        taps = {
            producer: bus.subscribe(
                surface_topic(market_id, producer), f"{TAP_PREFIX}{producer}@{market_id}"
            )
            for producer in producers
        }
        pipeline = build_pipeline(config, registry, clock, bus, sink)
        collected: dict[str, list[CalibratedSurface]] = {producer: [] for producer in producers}
        drains = [
            asyncio.create_task(_drain(subscription, collected[producer]))
            for producer, subscription in taps.items()
        ]
        try:
            await pipeline.run()
            # The last publication may still sit in a tap's mailbox; one turn of the loop per
            # drain lets each take it before they are cancelled.
            for _ in drains:
                await asyncio.sleep(0)
        finally:
            for drain in drains:
                drain.cancel()
            for drain in drains:
                with contextlib.suppress(asyncio.CancelledError):
                    await drain
    wall_seconds = time.perf_counter() - started

    surfaces = {producer: tuple(found) for producer, found in collected.items()}
    return SessionResult(
        market_id=market_id,
        producers=producers,
        surfaces=surfaces,
        comparison=_compare(config, market_id, producers, surfaces, clock),
        wall_seconds=wall_seconds,
    )


async def _drain(subscription: Subscription, into: list[CalibratedSurface]) -> None:
    while True:
        event = await subscription.receive()
        if isinstance(event, SurfaceCalibrated):
            into.append(event.surface)


def _compare(
    config: AppConfig,
    market_id: str,
    producers: tuple[str, ...],
    surfaces: Mapping[str, tuple[CalibratedSurface, ...]],
    clock: Clock,
) -> ComparativeReport | None:
    """Risk's comparison of the first two producers, off each one's last surface."""
    if len(producers) < 2:
        return None
    baseline, challenger = producers[0], producers[1]
    if not surfaces[baseline] or not surfaces[challenger]:
        return None
    cache = LastValueSurfaceProvider(clock=clock, metrics=NullMetricsSink())
    for producer in (baseline, challenger):
        cache.accept(surfaces[producer][-1])
    underlying = config.markets[0].conventions.underlying
    book = Portfolio(
        positions=tuple(
            position
            for position in config.risk.portfolio.positions
            if position.underlying == underlying
        )
    )
    return CompareProducersUseCase(
        cache=cache,
        portfolio=book,
        policy=config.risk.freshness,
        clock=clock,
        metrics=NullMetricsSink(),
        settings=config.risk.settings,
        baseline_producer_id=baseline,
        challenger_producer_id=challenger,
    ).compute(market_id)


def last_surfaces(result: SessionResult) -> dict[str, CalibratedSurface]:
    """The last surface each of the first two producers published -- what the comparison used."""
    return {
        producer: result.surfaces[producer][-1]
        for producer in result.producers[:2]
        if result.surfaces[producer]
    }
