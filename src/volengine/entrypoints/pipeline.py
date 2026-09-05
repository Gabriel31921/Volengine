"""The composition root: the one module that knows every context exists at the same time.

Four bounded contexts that may not import each other (rule 6), five use cases that never touch a
bus (ADR-016), and a set of ports nobody in the domain layer can satisfy. Something has to build
the concrete objects, decide which topic each event belongs on, and publish what the handlers hand
back. That something is this file, and keeping it to one file is what lets every other module in
the engine be ignorant of the wiring.

**What it owns, and what it deliberately does not.**

* It owns **routing**. Topic names exist nowhere else -- :func:`topic_of` derives one from the
  event itself, so a producer never names a destination and a consumer never guesses a spelling.
* It owns **publication**. Every handler is a synchronous function returning
  ``tuple[Event, ...]``; the closures below call them and publish the result (ADR-016).
* It owns **the threads**. ADR-005's per-producer pool is applied here, so a calibration can be
  pushed off the event loop without the use case ever learning that threads exist.
* It owns **the heartbeat**, and that closes the seam ``IngestStreamUseCase`` left open.
  ``max_quiet_seconds`` was written so that a calm market is still heard from, but the ingestion
  loop only evaluates the snapshot policy when an update *arrives* -- so a feed that stops
  produces silence, which is the one symptom downstream cannot tell from a dead process. The timer
  in :meth:`Pipeline._heartbeat` races the stream and asks the same use case the same question on
  a clock instead of on a tick.
* It does **not** own the mathematics, the policies or the language. Nothing here decides whether
  a fit is good, whether a snapshot is worth publishing or how old is too old.

**Adapters arrive by name.** :func:`default_adapters` is the registry that maps a configured
string -- ``provider = "constant"`` -- onto the callable that builds the object, and it is the
only place in the engine that *constructs* one. It holds six: the walking skeleton's three and
F2's three beside them, so a file may name a constant feed or a synthetic one, a mean or a fit, a
console or a file. A name it does not hold fails here with a message that says which one was not
registered. Passing the registry into :func:`build_pipeline` rather than reaching for it keeps the
graph testable with fakes.

**Recording and replay are modes of a run, not entries in that registry** (ADR-004).
:func:`with_recording` decorates every provider with a tap onto a file; :func:`with_replay`
replaces every provider with the file itself and hands it the engine's clock. Both take a registry
and return a registry, so the graph below is built once and does not know which mode it is in --
which is the point, because a replay that took a different path through this module would be
reproducing that path rather than the session.

``entrypoints/config.py`` also imports from two ``*/adapters/`` modules since F2-07 (ADR-028,
superseding one sentence of ADR-022), for the types ``SyntheticConfig`` and ``FitSettings`` and
nothing else: their fields are thresholds a
deployment retunes (ADR-012) and mirroring them in the loader would be a second copy of their
guards. So the claim above is about construction, not about imports -- what neither module allows
is a *configuration file* naming a class.

**Three of the four contexts are wired here, and Neural Surface is not.** The graph is Market
Data to Parametric Pricing to Risk; ``TrainOnSnapshot`` has no place in it and ``AppConfig`` has
no section that would feed one. That is a gap rather than an omission, and it is in
``docs/SEAMS.md``: the learner behind that use case is torch, an optional extra that arrives in
F3-C, and wiring it needs a replay buffer, an arbitrage mesh, gate thresholds, a restart schedule
and a seeded generator -- five configuration sections whose values nothing in this build could
exercise or check. Adding them now would be five thresholds chosen by guesswork, which is the
opposite of what ADR-012 asks configuration to be. The shape it will take is settled: a second
producer kind alongside :data:`CalibratorFactory`, publishing the same ``CalibratedSurface`` onto
the same topic, which is the arrangement Design 6.5 and 7.3 exist to compare.

**Everything except the calibrations runs on the event loop**, single-threaded, and that is a
choice rather than an oversight. ``QuoteChain``, ``CalibrationState`` and
``LastValueSurfaceProvider`` are all documented as not thread-safe; the calibrations are the only
work long enough to be worth an executor, they are one per producer, and the pool has one worker,
so each producer's state is still touched by one thread at a time. The results are published back
on the loop -- never from the worker -- because ``InProcessConflatingBus`` drives an
``asyncio.Event`` and is not thread-safe either.
"""

from __future__ import annotations

import asyncio
import math
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import Executor
from dataclasses import dataclass, replace
from pathlib import Path

from volengine.contracts.events import (
    CalibrationFailed,
    ChainCompositionChanged,
    Event,
    SnapshotReady,
    SurfaceCalibrated,
)
from volengine.entrypoints.config import (
    SYNTHETIC_PROVIDER,
    AppConfig,
    CalibrationConfig,
    ConfigError,
    MarketConfig,
    RiskConfig,
)
from volengine.market_data.adapters.constant import ConstantProvider
from volengine.market_data.adapters.recorded import RecordedProvider, Recording, ReplayClock
from volengine.market_data.adapters.recorder import RecordingProvider, RecordingSink
from volengine.market_data.adapters.synthetic import SyntheticProvider
from volengine.market_data.application.acl import to_snapshot_ready
from volengine.market_data.application.build_snapshot import BuildSnapshotUseCase
from volengine.market_data.application.ingest_stream import IngestStreamUseCase
from volengine.market_data.domain.ports import MarketDataProvider
from volengine.market_data.domain.quote_chain import QuoteChain
from volengine.market_data.domain.snapshot_policy import SnapshotPolicy
from volengine.parametric_pricing.adapters.flat_vol import PRODUCER_ID as FLAT_VOL_ID
from volengine.parametric_pricing.adapters.flat_vol import FlatVolCalibrator
from volengine.parametric_pricing.adapters.scipy_calibrator import PRODUCER_ID as SCIPY_ID
from volengine.parametric_pricing.adapters.scipy_calibrator import ScipyCalibrator
from volengine.parametric_pricing.application.calibrate_on_snapshot import CalibrateOnSnapshot
from volengine.parametric_pricing.application.calibration_state import CalibrationState
from volengine.parametric_pricing.domain.ports import Calibrator
from volengine.platform.bus import EventBus, Subscription
from volengine.platform.clock import Clock
from volengine.platform.executors import NamedExecutors
from volengine.platform.metrics import MetricsSink
from volengine.platform.runner import BusRunner, EventHandler
from volengine.risk.adapters.console_report_writer import ConsoleReportWriter
from volengine.risk.adapters.csv_report_writer import CsvReportWriter
from volengine.risk.application.compute_report import ComputeReportUseCase
from volengine.risk.application.surface_cache import LastValueSurfaceProvider
from volengine.risk.domain.portfolio import Portfolio
from volengine.risk.domain.ports import ReportWriter
from volengine.risk.domain.surface_view import SurfaceView

type ProviderFactory = Callable[[MarketConfig], MarketDataProvider]
"""Builds one market's quote source from its configuration."""

type CalibratorFactory = Callable[[CalibrationConfig], Calibrator]
"""Builds one producer. It names itself through ``Calibrator.producer_id``, which is why the
factory is keyed by the configured string and the *identity* still comes from the object."""

type WriterFactory = Callable[[RiskConfig], ReportWriter]
"""Builds the destination a finished report goes to."""

SETTLE_HOPS = 3
"""Loop iterations a settling pass yields before it decides nothing more is pending.

An allowance for scheduling, not a timeout: ``_Mailbox.put`` raises an ``asyncio.Event`` and the
runner waiting on it resumes on the next iteration of the loop, so one hop is what the current bus
actually needs and three is the margin for the callback hops an ``Event`` may add. It is safe to
count hops rather than to ask the bus whether it is empty *because this class owns both ends*:
every publisher and every subscriber in the process was created by :func:`build_pipeline`, so
there is no third party whose work could arrive after the pass that found none. A settling pass
that waited on wall-clock time instead would make a ``ManualClock`` run non-deterministic, which
is the one thing ADR-004 will not have.
"""


@dataclass(frozen=True, slots=True)
class Adapters:
    """The registry: configured names on the left, the things they build on the right.

    Three mappings rather than three instances, because a name in a file is the only thing the
    configuration is allowed to say about infrastructure. Wiring by name keeps ``config.py`` free
    of imports it has no business holding, and keeps the failure -- *nobody registered that* -- at
    start-up rather than at the first snapshot.
    """

    providers: Mapping[str, ProviderFactory]
    calibrators: Mapping[str, CalibratorFactory]
    writers: Mapping[str, WriterFactory]


def default_adapters() -> Adapters:
    """Every concrete adapter this build knows how to make: two feeds, two producers, two writers.

    **The only function in the engine that constructs an adapter**, which is what keeps every
    other module -- and every configuration file -- ignorant of infrastructure. A name it does not
    hold is refused at start-up by :func:`_lookup`, so a typo says so on the first line of output
    rather than after a session that publishes nothing.

    Each factory reads what its adapter needs out of the configuration section it is handed, and
    nothing else: a provider takes the market it belongs to, a calibrator the calibration section,
    a writer the risk section. There is deliberately no place for an adapter to be handed a
    threshold that ADR-012 already put somewhere else, and no factory reaches past its own section
    to find one.

    Three of the six take no settings at all, and that is a property of those adapters rather than
    an oversight: a constant feed, a weighted mean and a console have nothing a deployment could
    retune. The other three read the sections F2-07 gave them.
    """
    return Adapters(
        providers={
            "constant": lambda market: ConstantProvider(market.conventions),
            SYNTHETIC_PROVIDER: _synthetic_provider,
        },
        calibrators={
            FLAT_VOL_ID: lambda _calibration: FlatVolCalibrator(),
            # `fit` is `None` unless the file states a `[calibration.fit]` table, and the
            # calibrator answers that with the settings it ships with -- which are argued for in
            # its own docstring. Passing it through rather than substituting a default here keeps
            # this module from holding an opinion about a number it does not own.
            SCIPY_ID: lambda calibration: ScipyCalibrator(calibration.fit),
        },
        writers={
            "console": lambda _risk: ConsoleReportWriter(),
            "csv": _csv_report_writer,
        },
    )


def _synthetic_provider(market: MarketConfig) -> MarketDataProvider:
    """The synthetic feed, on the market it belongs to and the settings the file gave it.

    Named rather than a lambda because of the ``except``. ``SyntheticProvider`` refuses a
    configuration it cannot generate a chain from -- a session that would outlive its own nearest
    expiry is the one that actually happens -- with a plain ``ValueError``, which is right for the
    adapter and wrong for an operator: it is a file that has to change, so it leaves here as a
    ``ConfigError`` naming the market, like every other rejected number.
    """
    settings = market.synthetic
    try:
        if settings is None:
            return SyntheticProvider(market.conventions)
        return SyntheticProvider(market.conventions, settings.config, settings.start)
    except ValueError as failure:
        raise ConfigError(f"market[{market.market_id}].synthetic: {failure}") from failure


def _csv_report_writer(risk: RiskConfig) -> ReportWriter:
    """The file writer, on the path the risk section names.

    Two failures belong to the file rather than to the adapter and are translated here. A missing
    ``output_path`` is a writer with nowhere to write, refused at start-up rather than at the
    first report; an ``OSError`` is a directory that does not exist or cannot be written to, which
    ``CsvReportWriter`` provokes deliberately in its constructor so that it happens now and not
    thirty seconds into a session whose quotes are already gone.
    """
    if risk.output_path is None:
        raise ConfigError("risk: the 'csv' writer needs an 'output_path' naming the file to write")
    try:
        return CsvReportWriter(risk.output_path)
    except OSError as failure:
        message = f"risk: cannot write reports to {risk.output_path} ({failure})"
        raise ConfigError(message) from failure


# --- the two modes of ADR-004


def with_recording(adapters: Adapters, path: Path) -> Adapters:
    """The same registry, with every provider wrapped so the session is written to disk.

    **Recording is a mode of a run, not a provider a file can name**, and this is where that is
    decided. The alternative -- a ``provider = "recording"`` entry pointing at a wrapper, with the
    real feed named in a sub-table -- would let a configuration file describe a market in two
    places and would make "record the Deribit feed" a different market from "run the Deribit feed".
    Here the file describes one market, and the *command* decides whether a tap is fitted to it.

    Every name in the registry is wrapped, including the ones a given file does not use: which
    provider the tap ends up on is decided by the same lookup as always, so nothing here has to
    know which one the configuration named.

    Args:
        adapters: The registry to decorate, normally :func:`default_adapters`.
        path: The file the session is written to. Opened when the provider is built, which is
            during :func:`build_pipeline` and therefore before a single quote arrives.

    Returns:
        A registry whose providers record. The calibrators and writers are untouched.
    """
    return replace(
        adapters,
        providers={
            name: _recording_factory(factory, path) for name, factory in adapters.providers.items()
        },
    )


def _recording_factory(inner: ProviderFactory, path: Path) -> ProviderFactory:
    """One provider factory, wrapped in the tap, with the file's failure translated.

    An ``OSError`` here is a directory that does not exist or cannot be written to -- the operator's
    mistake, not the engine's -- so it leaves as a ``ConfigError`` naming the path, exactly like
    the CSV writer's does.
    """

    def build(market: MarketConfig) -> MarketDataProvider:
        try:
            sink = RecordingSink(path)
        except OSError as failure:
            raise ConfigError(f"cannot record to {path} ({failure})") from failure
        return RecordingProvider(
            inner=inner(market),
            sink=sink,
            market_id=market.market_id,
            underlying=market.underlying,
        )

    return build


def with_replay(adapters: Adapters, recording: Recording, clock: ReplayClock) -> Adapters:
    """The same registry, with every provider replaced by the recording.

    Under a replay the configured provider is not built at all: there is no venue to connect to and
    no synthetic market to invent, only a file. The registry is still consulted by name, so a
    configuration naming an adapter nobody registered is refused exactly as ``run`` would refuse it
    -- a replay is run against a file that must describe a runnable engine, and quietly accepting a
    name that does not exist would hide the day the recording and the configuration parted company.

    The clock is where determinism comes from. It is the engine's own, and this is the one place it
    is handed to an adapter: ``ProviderFactory`` takes a ``MarketConfig`` and nothing else
    (ADR-022), so the clock arrives through the closure rather than through the signature, and only
    for the one adapter whose whole job is to move it.

    Args:
        adapters: The registry to replace the providers of.
        recording: The opened session, from ``adapters.recorded.open_recording``.
        clock: The clock the recorded instants are written into, normally a ``SimulatedClock``
            already placed at ``recording.started_at``.
    """
    return replace(
        adapters,
        providers={
            name: (lambda _market: RecordedProvider(recording, clock))
            for name in adapters.providers
        },
    )


# --- routing


def snapshot_topic(market_id: str) -> str:
    """Where a market's snapshots go. Both calibrators subscribe to it."""
    return f"snapshot.{market_id}"


def composition_topic(market_id: str) -> str:
    """Where a market's live-instrument set goes (ADR-013)."""
    return f"composition.{market_id}"


def surface_topic(market_id: str, producer_id: str) -> str:
    """Where one producer's surfaces go, in the spelling ``Calibrator.producer_id`` documents."""
    return f"surface.{market_id}.{producer_id}"


def failure_topic(market_id: str, producer_id: str) -> str:
    """Where one producer's refusals go.

    **A topic of its own rather than sharing the surface topic**, and the bus is the reason. A
    refused calibration publishes ``CalibrationFailed`` *and* republishes the last good surface
    (ADR-006); on one topic the second would overwrite the first in a mailbox that holds one event
    (ADR-003), so the pair that says "this producer is failing, and here is what it is still
    standing on" would arrive as half of itself.
    """
    return f"calibration_failed.{market_id}.{producer_id}"


def topic_of(event: Event) -> str:
    """The topic an event belongs on, derived from the event itself.

    Routing by identity rather than by a caller's opinion: a use case returns a tuple of events
    that may be of two different kinds, and asking the publisher to remember which goes where is
    how a failure ends up on a surface topic. ``Event`` is a closed union (``contracts/events``),
    so the type checker proves this is exhaustive and a fifth event cannot be added without
    landing here.
    """
    match event:
        case SnapshotReady():
            return snapshot_topic(event.snapshot.market_id)
        case ChainCompositionChanged():
            return composition_topic(event.market_id)
        case SurfaceCalibrated():
            return surface_topic(event.surface.market_id, event.surface.producer_id)
        case CalibrationFailed():
            return failure_topic(event.market_id, event.producer_id)


# --- the graph


@dataclass(frozen=True, slots=True)
class _MarketLoop:
    """One market's ingestion, and the timer that keeps it audible when it goes quiet."""

    market_id: str
    ingest: IngestStreamUseCase
    build: BuildSnapshotUseCase
    heartbeat_seconds: float | None
    """How often the timer asks the snapshot policy again, or ``None`` when the market has no
    heartbeat configured.

    Derived from ``cadence_seconds`` rather than configured separately: the cadence is already the
    ceiling on how often anything may be published, so polling faster could not emit sooner, and a
    second knob could only contradict the first. The heartbeat therefore fires within one cadence
    of ``max_quiet_seconds``, which is the resolution the cadence itself defines.
    """


@dataclass(frozen=True, slots=True)
class _Producer:
    """One calibrator on one market, and the risk report computed off what it publishes."""

    market_id: str
    producer_id: str
    calibrate: CalibrateOnSnapshot
    report: ComputeReportUseCase
    snapshots: Subscription
    surfaces: Subscription
    executor: Executor


class _ProducerSurfaces:
    """One named producer's slot in the shared cache, seen as a ``SurfaceProvider``.

    The composition root's answer to Design 7.3. ``LastValueSurfaceProvider.latest`` picks the
    newest surface across every producer, which is the right answer to the question the port asks
    and the wrong one for a comparative report: valuing one book under two producers means asking
    for a *named* one twice. That is ``for_producer``, which is deliberately not on the port -- so
    the binding from "the surface for this market" to "this producer's surface for this market"
    happens here, in the only layer allowed to know that two producers exist at once.
    """

    def __init__(self, cache: LastValueSurfaceProvider, producer_id: str) -> None:
        self._cache = cache
        self._producer_id = producer_id

    def latest(self, market_id: str) -> SurfaceView | None:
        return self._cache.for_producer(market_id, self._producer_id)


def build_pipeline(
    config: AppConfig,
    adapters: Adapters,
    clock: Clock,
    bus: EventBus,
    metrics: MetricsSink,
) -> Pipeline:
    """Build every context instance, wire the subscriptions, and hand back something runnable.

    Args:
        config: The whole configuration, already validated by ``load_config``.
        adapters: The registry of concrete infrastructure, and a fifth argument
            ``Implementation.md`` does not list (ADR-022). Passed in rather than taken from
            :func:`default_adapters` so that a test can run the entire graph on fakes -- which is
            what makes this module testable before F1-08 supplies any real adapter.
        clock: One clock for every context (ADR-004). ``SystemClock`` in production, ``ManualClock``
            in a test, ``SimulatedClock`` under a replay; each context's own ``Clock`` protocol is
            satisfied structurally here, which is the single point where that connection is made.
        bus: Where events go. Subscriptions are opened during this call, before anything can
            publish, so no consumer can miss the first snapshot of a session.
        metrics: One sink for every context, for the same reason as the clock.

    Returns:
        A :class:`Pipeline` whose subscriptions are already live.

    Raises:
        ConfigError: If the configuration names a provider, calibrator or writer that no adapter
            is registered for. A start-up failure by design: the alternative is an engine that
            runs with a context missing and publishes nothing, which looks exactly like a market
            that is not moving. Also if the book and the markets do not line up -- see
            :func:`_book_for`.
    """
    executors = NamedExecutors()
    cache = LastValueSurfaceProvider()
    writer = _lookup(adapters.writers, config.risk.writer, "writer")(config.risk)
    _require_every_position_is_quoted(config)

    markets: list[_MarketLoop] = []
    producers: list[_Producer] = []
    for market in config.markets:
        book = _book_for(market, config.risk.portfolio)
        chain = QuoteChain(market.conventions, market.admissibility)
        build = BuildSnapshotUseCase(
            chain=chain,
            policy=SnapshotPolicy(market.snapshot),
            clock=clock,
            metrics=metrics,
            max_skew_seconds=market.max_skew_seconds,
        )
        provider = _lookup(adapters.providers, market.provider, "provider")(market)
        markets.append(
            _MarketLoop(
                market_id=market.market_id,
                ingest=IngestStreamUseCase(
                    provider=provider,
                    chain=chain,
                    build_snapshot=build,
                    clock=clock,
                    metrics=metrics,
                    market_id=market.market_id,
                ),
                build=build,
                heartbeat_seconds=(
                    market.snapshot.cadence_seconds
                    if market.snapshot.max_quiet_seconds is not None
                    else None
                ),
            )
        )

        for name in config.calibration.calibrators:
            calibrator = _lookup(adapters.calibrators, name, "calibrator")(config.calibration)
            producer_id = calibrator.producer_id
            producers.append(
                _Producer(
                    market_id=market.market_id,
                    producer_id=producer_id,
                    calibrate=CalibrateOnSnapshot(
                        calibrator=calibrator,
                        state=CalibrationState(),
                        clock=clock,
                        metrics=metrics,
                        weighting=config.calibration.weighting,
                        grid=config.calibration.grid,
                        acceptance=config.calibration.acceptance,
                    ),
                    report=ComputeReportUseCase(
                        provider=_ProducerSurfaces(cache, producer_id),
                        portfolio=book,
                        policy=config.risk.freshness,
                        clock=clock,
                        metrics=metrics,
                        settings=config.risk.settings,
                        expected_producer_id=producer_id,
                    ),
                    snapshots=bus.subscribe(
                        snapshot_topic(market.market_id), f"{producer_id}@{market.market_id}"
                    ),
                    surfaces=bus.subscribe(
                        surface_topic(market.market_id, producer_id),
                        f"risk-{producer_id}@{market.market_id}",
                    ),
                    executor=executors.for_producer(producer_id),
                )
            )

    return Pipeline(
        bus=bus,
        clock=clock,
        metrics=metrics,
        markets=tuple(markets),
        producers=tuple(producers),
        cache=cache,
        writer=writer,
        executors=executors,
    )


def _book_for(market: MarketConfig, portfolio: Portfolio) -> Portfolio:
    """The part of the book this market can value: the positions written on its underlying.

    **The composition root's obligation, and it is written down as one** (ADR-022).
    ``position_risk`` never
    compares a position's underlying against the surface it is valued on, because a
    ``CalibratedSurface`` publishes a ``market_id`` and no underlying at all (``docs/SEAMS.md``);
    handing every market the whole book would therefore value BTC options off the ETH smile and
    print a number rather than an error. Splitting the book here is the only place that can be
    done, because this is the only place that sees both.

    Raises:
        ConfigError: If no position is written on this market's underlying. Refused rather than
            wired with an empty book: ``Portfolio`` requires at least one position, and a market
            configured with nothing to value is either a typo in ``underlying`` or a market
            nobody meant to run -- and both are cheaper to hear about now than to notice in an
            end-of-day report that quietly covered one market fewer than it was asked to.
    """
    held = tuple(
        position
        for position in portfolio.positions
        if position.underlying == market.conventions.underlying
    )
    if not held:
        raise ConfigError(
            f"the book holds no position on {market.conventions.underlying!r}, "
            f"which is what {market.market_id} quotes"
        )
    return Portfolio(positions=held)


def _require_every_position_is_quoted(config: AppConfig) -> None:
    """Refuse a book naming an underlying no configured market supplies quotes for.

    The other half of :func:`_book_for`, and it catches what that one cannot: with two markets
    configured, a position on a third underlying leaves every market with a non-empty book of its
    own and is simply never valued by anybody. Silently unvalued risk is the one outcome this
    whole context exists to prevent.
    """
    quoted = {market.conventions.underlying for market in config.markets}
    orphans = sorted(
        {
            position.underlying
            for position in config.risk.portfolio.positions
            if position.underlying not in quoted
        }
    )
    if orphans:
        raise ConfigError(
            f"the book holds positions on {', '.join(orphans)}, which no configured market "
            f"quotes; the markets quote {', '.join(sorted(quoted))}"
        )


def _lookup[F](registry: Mapping[str, F], name: str, kind: str) -> F:
    factory = registry.get(name)
    if factory is None:
        known = ", ".join(sorted(registry)) or "nothing"
        raise ConfigError(f"no {kind} adapter is registered under {name!r}; known: {known}")
    return factory


class Pipeline:
    """Every task the engine runs, and the rule for stopping.

    Built by :func:`build_pipeline` and not directly: the constructor takes objects that are
    already wired to each other, so a caller assembling one by hand would be writing the
    composition twice.
    """

    def __init__(
        self,
        bus: EventBus,
        clock: Clock,
        metrics: MetricsSink,
        markets: tuple[_MarketLoop, ...],
        producers: tuple[_Producer, ...],
        cache: LastValueSurfaceProvider,
        writer: ReportWriter,
        executors: NamedExecutors,
    ) -> None:
        self._bus = bus
        self._clock = clock
        self._metrics = metrics
        self._markets = markets
        self._producers = producers
        self._cache = cache
        self._writer = writer
        self._executors = executors
        self._reports_written = 0
        self._max_reports: int | None = None
        self._enough = asyncio.Event()
        self._published = 0
        self._handled = 0
        self._busy = 0
        self._idle = asyncio.Event()
        self._idle.set()
        self._runners = tuple(
            runner
            for producer in producers
            for runner in (
                BusRunner(
                    subscription=producer.snapshots,
                    handler=self._tracked(self._calibration_handler(producer)),
                    metrics=metrics,
                    name=f"{producer.producer_id}@{producer.market_id}",
                ),
                BusRunner(
                    subscription=producer.surfaces,
                    handler=self._tracked(self._report_handler(producer)),
                    metrics=metrics,
                    name=f"risk-{producer.producer_id}@{producer.market_id}",
                ),
            )
        )

    @property
    def reports_written(self) -> int:
        """How many reports reached the writer during the last run. Read-only, for the CLI."""
        return self._reports_written

    async def run(
        self, max_reports: int | None = None, duration_seconds: float | None = None
    ) -> None:
        """Run every task until ingestion ends, a stopping rule fires, or the caller cancels.

        Four ways a run finishes, and each of them is somebody's normal:

        * **Every provider's stream ended.** A recorded replay and the walking skeleton's fixed
          chain are finite, and when the last one runs out there is nothing left to *start*. What
          is already under way still finishes: the sources are cancelled, then :meth:`_settle`
          lets the events in flight reach the end of the chain, and only then are the consumers
          stopped. Without that, a fit still running on its pool when the last quote arrived would
          be thrown away, and a run over a finite chain would return successfully having written
          nothing -- indistinguishable, from outside, from a market that produced no report.
        * **The report goal was met**, which is what ``volengine report`` asks for: value the book
          once against a live surface and stop. Nothing is settled in that case, deliberately --
          the caller has what it asked for, and draining further work would only produce reports
          the stopping rule has already refused to write.
        * **The session ran its time**, which is what ``volengine run --duration`` asks for: a
          bounded observation of an unbounded feed. Nothing is settled there either -- the caller
          asked to stop at a wall-clock instant, not to stop and then keep working -- so it ends
          like the goal rather than like an exhausted stream.
        * **Cancellation**, which is how a long-running session is stopped from outside. Nothing
          is settled there either: an interrupt means stop now, and awaiting a calibration inside
          a cancelled ``finally`` is how a shutdown hangs.

        Args:
            max_reports: Stop once this many reports have been written, or ``None`` to run until
                ingestion ends. Counting *reports* rather than snapshots or surfaces is
                deliberate: it is the only event at the end of the whole chain, so a run that
                reaches it has exercised every hop.
            duration_seconds: Stop after this long, or ``None`` for no time limit. Measured on the
                injected ``Clock`` and not on ``asyncio.sleep``, like the heartbeat and for the
                same reason (ADR-004): under a ``ManualClock`` a run bounded by time has to be
                bounded by *simulated* time, or a replay would end when the machine felt like it.
                The consequence is the heartbeat's: this task advances a manual clock on its own.

        Raises:
            ValueError: If ``max_reports`` is not positive, or ``duration_seconds`` is not
                positive and finite. Zero would mean "stop before starting", which is a caller
                mistake dressed as a configuration.
            Exception: Whatever a task raised. A provider that fails to connect, a chain fed
                another market's quotes -- both are wiring or infrastructure failures that must
                surface rather than be counted and survived. Handler exceptions never reach here:
                ``BusRunner`` counts and swallows those on purpose.
        """
        if max_reports is not None and max_reports <= 0:
            raise ValueError(f"The report goal must be positive, got {max_reports}")
        if duration_seconds is not None and (
            not math.isfinite(duration_seconds) or duration_seconds <= 0
        ):
            raise ValueError(
                f"The run duration must be positive and finite, got {duration_seconds}"
            )

        self._max_reports = max_reports
        self._reports_written = 0
        self._published = 0
        self._handled = 0
        self._enough.clear()

        # Sources and consumers are kept apart because shutdown treats them differently: the
        # sources are silenced first and the consumers are given the chance to finish what those
        # sources already produced.
        sources = [asyncio.create_task(self._ingest_all(), name="volengine-ingestion")]
        sources += [
            asyncio.create_task(self._heartbeat(market), name=f"heartbeat-{market.market_id}")
            for market in self._markets
            if market.heartbeat_seconds is not None
        ]
        consumers = [asyncio.create_task(runner.run(), name="runner") for runner in self._runners]
        # The stopping rules, kept together: whichever of them completes, the run is over and
        # nothing is drained. Everything else in the wait set finishing means either the feed ran
        # out -- which does drain -- or a task failed, which must be re-raised.
        stoppers = {asyncio.create_task(self._wait_for_goal(), name="report-goal")}
        if duration_seconds is not None:
            stoppers.add(asyncio.create_task(self._clock.sleep(duration_seconds), name="duration"))

        try:
            finished, _ = await asyncio.wait(
                {*stoppers, *sources, *consumers}, return_when=asyncio.FIRST_COMPLETED
            )
            for task in finished:
                if task not in stoppers:
                    # Re-raises whatever ended it, and does nothing for a clean return. The
                    # consumers and the heartbeat never return on their own, so anything here
                    # other than ingestion finishing is a failure that must not be swallowed.
                    task.result()
            if not finished & stoppers:
                await self._settle(sources)
        finally:
            # The order matters. Cancel and await the tasks first so a provider's `finally` gets
            # to close its connection while the loop is still running, and only then take the
            # thread pools down -- shutting them first would leave a cancelled handler awaiting a
            # future on a pool that is going away.
            for task in (*stoppers, *sources, *consumers):
                task.cancel()
            await asyncio.gather(*stoppers, *sources, *consumers, return_exceptions=True)
            self._executors.shutdown()

    async def _settle(self, sources: Sequence[asyncio.Task[None]]) -> None:
        """Silence the sources, then let everything already published reach the end of the chain.

        The engine is a chain of one-slot mailboxes (ADR-003) with a thread pool in the middle of
        it (ADR-005), so "ingestion has finished" says nothing about whether a snapshot is still
        being fitted, or a surface is sitting unread in a mailbox, or a report is one hop away
        from the writer. Cancelling the consumers at that moment throws all three away.

        The sources go first and are awaited, which is also what closes the providers. After that
        nothing new can enter the chain, so waiting for a fixpoint terminates: each pass yields
        the loop long enough for a pending event to reach a runner, waits for every handler
        currently running -- including one blocked on an executor -- and stops on the first pass
        in which nothing was published and no handler ran. The two counters are what make it a
        fixpoint rather than a guess: a report handler publishes nothing, so counting only
        publications would call the pass quiet while a report was being written.
        """
        for task in sources:
            task.cancel()
        await asyncio.gather(*sources, return_exceptions=True)

        while True:
            before = (self._published, self._handled)
            for _ in range(SETTLE_HOPS):
                await asyncio.sleep(0)
            await self._idle.wait()
            if (self._published, self._handled) == before:
                return

    # --- the tasks

    async def _wait_for_goal(self) -> None:
        """Complete once enough reports have been written.

        A coroutine of its own rather than ``asyncio.Event.wait`` directly, so that every task in
        the run is a ``Task[None]`` and the set handed to ``asyncio.wait`` has one element type.
        """
        await self._enough.wait()

    async def _ingest_all(self) -> None:
        """Run every market's ingestion loop, and finish when the last of them does."""
        await asyncio.gather(*(self._ingest(market) for market in self._markets))

    async def _ingest(self, market: _MarketLoop) -> None:
        """Publish what one market's ingestion yields.

        The one use case in the engine that is a loop rather than a handler (ADR-016), because it
        owns an ``AsyncIterator`` of domain objects nobody outside its context may touch. It still
        does not publish: it yields, and this is where a topic is attached.
        """
        async for event in market.ingest.run():
            self._publish(event)

    async def _heartbeat(self, market: _MarketLoop) -> None:
        """Ask the snapshot policy on a timer, so silence is not mistaken for calm.

        The seam ``IngestStreamUseCase`` documents, closed. It calls exactly the same
        ``build()`` the stream calls, so the decision stays entirely in ``SnapshotPolicy`` --
        this task supplies an occasion, never a verdict, and on a market that is ticking normally
        every one of these calls returns ``None`` against the cadence.

        Safe beside the stream despite sharing one use case and one chain: ``build()`` contains no
        ``await``, so the two callers cannot interleave inside it on a single event loop.

        Note what this does to a ``ManualClock``, whose ``sleep`` advances time instead of waiting:
        a heartbeat task drives that clock forward on its own. That is the intended behaviour --
        it is how a test reaches the heartbeat without waiting -- but a test that wants time to
        stand still configures ``max_quiet_seconds`` as absent, which is what leaves this task
        uncreated.
        """
        poll = market.heartbeat_seconds
        if poll is None:  # pragma: no cover - the task is not created in that case
            return
        while True:
            await self._clock.sleep(poll)
            snapshot = market.build.build()
            if snapshot is not None:
                self._metrics.counter("pipeline.heartbeat.emitted", market=market.market_id)
                self._publish(to_snapshot_ready(snapshot))

    # --- the handlers

    def _calibration_handler(self, producer: _Producer) -> EventHandler:
        """Fit a snapshot on this producer's thread pool, then publish what came back.

        ADR-005 and ADR-016 meeting in six lines. The use case is synchronous and knows nothing
        about threads or topics; the executor keeps a fit that holds the GIL for milliseconds to
        seconds from stalling every other market's ingestion; and the publication happens after
        the ``await``, back on the event loop, because the bus is not thread-safe.

        Events other than ``SnapshotReady`` are ignored rather than refused: a topic is a
        contract about *where*, not about *what*, and a handler that raised on an unexpected event
        would trade a harmless no-op for a counted failure.
        """

        async def handle(event: Event) -> None:
            if not isinstance(event, SnapshotReady):
                return
            loop = asyncio.get_running_loop()
            produced = await loop.run_in_executor(
                producer.executor, producer.calibrate.handle, event.snapshot
            )
            for outcome in produced:
                self._publish(outcome)

        return handle

    def _report_handler(self, producer: _Producer) -> EventHandler:
        """Cache the surface, value the book against it, and write the result.

        On the event loop rather than on a pool, unlike the calibration above. The cache is shared
        by every producer and documents itself as single-threaded; a report is one interpolation
        and a handful of bumps per position, which is nothing beside a fit; and the alternative --
        two risk handlers on two producer pools writing into one dictionary -- would be a data
        race bought for no measurable latency.

        A rejected report is written like any other (Design 7.2): "there is no valid surface" is
        the loudest thing this context says, and swallowing it here would turn it into silence.

        **Once the goal is met, the surface is still cached and no further report is written.**
        Raising the flag does not stop the engine on the spot: the run only ends when
        :meth:`run` is scheduled again, and every task already queued behind this one gets to
        finish first. Without the guard, ``--count 1`` would write one report *and then a few
        more* depending on how the loop happened to interleave, which is a stopping rule nobody
        could assert on -- and the count is what the operator asked for, not a lower bound. The
        cache is deliberately still fed, so what it holds does not depend on when we stop.
        """

        async def handle(event: Event) -> None:
            if not isinstance(event, SurfaceCalibrated):
                return
            self._cache.accept(event.surface)
            if self._goal_met():
                return
            self._writer.write(producer.report.compute(producer.market_id))
            self._count_report()

        return handle

    # --- plumbing

    def _tracked(self, handler: EventHandler) -> EventHandler:
        """Wrap a handler so that :meth:`_settle` can tell whether anything is still happening.

        Two counters and a flag, kept here rather than asked of the bus or of ``BusRunner``.
        Neither of those can answer the question: a mailbox holds one event and does not say
        whether the last one it handed out has been dealt with, and a runner counts what it
        finished without knowing what it is in the middle of. The composition root, which created
        every handler in the process, can simply observe them.

        ``_busy`` is incremented before the ``await``, so a handler that is blocked on a thread
        pool is visible as work in flight rather than as an idle loop.
        """

        async def handle(event: Event) -> None:
            self._busy += 1
            self._idle.clear()
            try:
                await handler(event)
            finally:
                self._handled += 1
                self._busy -= 1
                # `== 0` rather than `not self._busy`: the truthiness form is right here by
                # accident and wrong everywhere else a count is tested in this repository.
                if self._busy == 0:
                    self._idle.set()

        return handle

    def _publish(self, event: Event) -> None:
        self._published += 1
        self._bus.publish(topic_of(event), event)

    def _goal_met(self) -> bool:
        return self._max_reports is not None and self._reports_written >= self._max_reports

    def _count_report(self) -> None:
        self._reports_written += 1
        if self._goal_met():
            self._enough.set()
