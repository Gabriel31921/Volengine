"""The command line: four verbs, and the only place an event loop is started.

Deliberately thin. Everything below it is already composed by ``pipeline.build_pipeline``, so what
is left here is the decisions a person makes when they type the command -- which file to read,
which market to run, which producers to run it with, and how long or how many reports to run for
-- plus turning a ``ConfigError`` into an exit code instead of a traceback.

**``record`` and ``replay`` are operative since F3-B**, and they are the same graph as ``run``
seen twice: once with a tap on the feed, once with the file in the feed's place. Neither is a
second code path through the engine -- that is what ``pipeline.with_recording`` and
``pipeline.with_replay`` buy, and it is what makes the replay reproduce the session rather than a
route through the composition root nothing else takes.

**A recording holds one market.** Both verbs therefore narrow the configuration to one, ``record``
from ``--market`` and ``replay`` from the market the file says it holds, and both refuse rather
than guess when the choice is ambiguous. One file per market is what keeps the format free of a
routing field nothing would read, and it is the shape a golden fixture takes anyway: an hour of one
venue's chain.

**The metrics sink is opened and closed here** (F3-W1). ``[metrics]`` names it, ``--metrics`` and
``--no-metrics`` override the name, and :func:`_metrics_sink` turns the choice into an object. The
CSV sink holds a file open and the ``MetricsSink`` port has no ``close()`` -- deliberately, since a
use case has no business ending a run -- so the code that opens the file is the code that closes
it, in a context manager wrapped around the whole of :func:`_drive`: a clean end, a refused
configuration, an interrupt and an exception out of the run all leave a closed, complete file.
"""

from __future__ import annotations

import asyncio
import logging
import math
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from typing import Annotated, NoReturn

import typer

from volengine.entrypoints.config import (
    AppConfig,
    ConfigError,
    MetricsConfig,
    MetricsSinkKind,
    load_config,
)
from volengine.entrypoints.pipeline import (
    Adapters,
    build_pipeline,
    default_adapters,
    with_recording,
    with_replay,
)
from volengine.market_data.adapters.recorded import Recording, open_recording
from volengine.platform.adapters.csv_metrics_sink import CsvMetricsSink
from volengine.platform.bus import InProcessConflatingBus
from volengine.platform.clock import Clock, SimulatedClock, SystemClock
from volengine.platform.metrics import LoggingMetricsSink, MetricsSink, NullMetricsSink

CONFIG_EXIT_CODE = 2
"""What a bad configuration exits with.

Distinct from 1 so a script can tell "you typed something wrong" from "the engine failed". It
covers every refusal this layer makes before the first quote arrives: an unreadable file, a
threshold the domain rejects, a market or a producer the file does not name, an adapter nobody
registered, and -- since F3-B -- a recording that cannot be opened or does not match the
configuration it was handed.
"""

app = typer.Typer(
    add_completion=False,
    help="Real-time implied volatility surface calibration engine.",
)

ConfigOption = Annotated[
    Path,
    typer.Option("--config", "-c", help="TOML file holding every threshold (ADR-012)."),
]
MarketOption = Annotated[
    str | None,
    typer.Option("--market", help="Run only this market, out of those the file configures."),
]
CalibratorsOption = Annotated[
    str | None,
    # The example was `svi,neural` in Design 8.1 until the neural producer turned out not to be
    # wired into the pipeline (docs/SEAMS.md): a help line promising a name the engine cannot
    # build is worse than one that names what the shipped configurations actually list.
    typer.Option("--calibrators", help="Comma-separated producers to run, e.g. svi-scipy,svi-jax."),
]
MetricsOption = Annotated[
    bool | None,
    typer.Option(
        "--metrics/--no-metrics",
        help="Log every metric, or none, whatever the file's metrics table chooses.",
    ),
]
"""Three states, not two: absent leaves the choice to the file's ``[metrics]`` table, and either
spelling overrides it -- the way ``--market`` and ``--calibrators`` narrow theirs."""
DurationOption = Annotated[
    float | None,
    typer.Option("--duration", help="Stop after this many seconds of session."),
]
RecordingOption = Annotated[
    Path,
    typer.Option("--recording", "-r", help="JSON Lines file holding one market's session."),
]


@app.command()
def run(
    config: ConfigOption,
    market: MarketOption = None,
    calibrators: CalibratorsOption = None,
    metrics: MetricsOption = None,
    duration: DurationOption = None,
) -> None:
    """Ingest, calibrate and report until the streams end, the duration elapses or you interrupt.

    ``--duration`` is the stopping rule a live feed needs and a finite one does not: a synthetic
    session ends when it runs out of cycles and a websocket never does, so a bounded observation
    of the second is otherwise only possible with an interrupt. It is on ``run`` alone --
    ``report`` already stops at ``--count`` -- and it is counted on the engine's clock, so a
    replay under a simulated clock is bounded by the session's own time rather than by ours
    (ADR-028).
    """
    # Finiteness first, and joined with `or`: `nan <= 0` is `False`, so the ordering test alone
    # would pass `--duration nan` down to `Pipeline.run`, whose own guard raises a `ValueError`
    # this module does not catch -- a traceback and exit 1 where the contract is a message and
    # exit 2. An `inf` slips through the same hole and sleeps forever.
    if duration is not None and (not math.isfinite(duration) or duration <= 0):
        _fail(f"--duration must be positive and finite, got {duration}", CONFIG_EXIT_CODE)
    _drive(
        _configure(config, market, calibrators),
        metrics=metrics,
        max_reports=None,
        duration=duration,
    )


@app.command()
def report(
    config: ConfigOption,
    market: MarketOption = None,
    calibrators: CalibratorsOption = None,
    metrics: MetricsOption = None,
    count: Annotated[int, typer.Option(help="Stop after this many reports.")] = 1,
) -> None:
    """Run the pipeline only until it has valued the book, then stop.

    The same graph as ``run`` with a stopping rule, rather than a second code path that reads a
    surface from somewhere: nothing in this engine persists a surface between processes, so a
    report is always the end of a live pipeline -- and saying so in one line of wiring is more
    honest than a command that would quietly always refuse for want of a surface.
    """
    if count <= 0:
        _fail(f"--count must be positive, got {count}", CONFIG_EXIT_CODE)
    _drive(_configure(config, market, calibrators), metrics=metrics, max_reports=count)


@app.command()
def record(
    config: ConfigOption,
    recording: RecordingOption,
    market: MarketOption = None,
    calibrators: CalibratorsOption = None,
    metrics: MetricsOption = None,
    duration: DurationOption = None,
) -> None:
    """Run a session normally and write its normalised quote stream to a file (ADR-004).

    An ordinary run with a tap on it: the same ingestion, the same fits, the same reports. That is
    deliberate rather than incidental -- a recorder that ran a reduced pipeline would be recording
    a session nobody will ever have, and the first thing a replay is asked to reproduce is a run
    that actually happened.

    The file holds one market, so ``--market`` is what selects it when the configuration names
    more than one.
    """
    if duration is not None and (not math.isfinite(duration) or duration <= 0):
        _fail(f"--duration must be positive and finite, got {duration}", CONFIG_EXIT_CODE)
    selected = _one_market(_configure(config, market, calibrators), "record")
    _drive(
        selected,
        metrics=metrics,
        max_reports=None,
        duration=duration,
        adapters=with_recording(default_adapters(), recording),
    )


@app.command()
def replay(
    config: ConfigOption,
    recording: RecordingOption,
    calibrators: CalibratorsOption = None,
    metrics: MetricsOption = None,
    count: Annotated[int | None, typer.Option(help="Stop after this many reports.")] = None,
) -> None:
    """Replay a recorded session through the whole engine, deterministically (ADR-004).

    Every number a replayed report carries comes out of the file: the quotes, and the instants the
    engine read them at, because the clock is a ``SimulatedClock`` the recording moves quote by
    quote rather than one that ticks on its own. Two replays of one recording write the same
    report, byte for byte.

    **What that does not reproduce is the live interleaving** (ADR-003). ``RecordedProvider``
    awaits nothing between quotes, so the ingestion task holds the event loop until the file ends:
    every snapshot is published, each overwrites the last in the calibrators' one-slot mailboxes,
    and the fit sees only the final one. Measured on the golden fixture in F3-F: 25 snapshots, one
    surface, one report. That makes a replay's *report* deterministic for a reason that is not the
    one this docstring used to give, and it is in ``docs/SEAMS.md`` with the choice it leaves.

    Which market runs is the recording's decision, not the flag's: the file names the market it
    holds and the configuration is narrowed to it. A ``--market`` option here could only agree or
    disagree with the file, and the disagreement would be a session replayed against another
    market's conventions.

    **No ``--duration``.** A recording is finite and ends when it ends; a time limit would be
    counted on the recorded clock, where "ten seconds" is a property of the file rather than of
    the person waiting. ``--count`` stops early on a number the engine controls.

    **No timers, and the heartbeat's rule kept.** The heartbeat task and the rediscovery poll are
    not started: under a ``SimulatedClock`` only the recording moves time, so either would fire on
    the loop's schedule rather than the session's. ``max_quiet_seconds`` itself stays in the
    policy, which asks it on every recorded tick at that tick's recorded instant -- so a quiet
    market is still heard from every ``max_quiet_seconds``, deterministically. Until F3-F the
    replay dropped the setting outright, and a recording whose first snapshot rested on one quote
    replayed as that snapshot and nothing after it.
    """
    if count is not None and count <= 0:
        _fail(f"--count must be positive, got {count}", CONFIG_EXIT_CODE)
    session = _open(recording)
    selected = _configure(config, session.market_id, calibrators)
    _require_same_underlying(selected, session)
    clock = SimulatedClock(session.started_at)
    _drive(
        selected,
        metrics=metrics,
        max_reports=count,
        adapters=with_replay(default_adapters(), session, clock),
        clock=clock,
        timers=False,
    )


# --- plumbing


def _configure(path: Path, market: str | None, calibrators: str | None) -> AppConfig:
    """Read the file, then narrow it to what the flags asked for.

    The flags override rather than extend: a market or a producer that is not in the file cannot
    be conjured from the command line, because everything either of them needs -- conventions,
    thresholds, a mesh -- lives in the file. What the command line chooses is a *subset*, which is
    why an unknown name is an error here rather than a silently empty run.
    """
    try:
        config = load_config(path)
        if market is not None:
            config = _only_market(config, market)
        if calibrators is not None:
            config = _only_calibrators(config, calibrators)
    except ConfigError as failure:
        _fail(str(failure), CONFIG_EXIT_CODE)
    return config


def _only_market(config: AppConfig, market_id: str) -> AppConfig:
    selected = tuple(one for one in config.markets if one.market_id == market_id)
    if not selected:
        known = ", ".join(one.market_id for one in config.markets)
        raise ConfigError(f"no market named {market_id!r} is configured; known: {known}")
    # `replace` rather than a fresh `AppConfig(...)`: a constructor call names every field it keeps,
    # and the one it forgets -- `metrics`, the day it arrived -- would silently fall back to its
    # default.
    return replace(config, markets=selected)


def _only_calibrators(config: AppConfig, names: str) -> AppConfig:
    wanted = tuple(name.strip() for name in names.split(",") if name.strip())
    if not wanted:
        raise ConfigError(f"--calibrators names nothing, got {names!r}")
    unknown = [name for name in wanted if name not in config.calibration.calibrators]
    if unknown:
        known = ", ".join(config.calibration.calibrators)
        raise ConfigError(f"no calibrator named {unknown[0]!r} is configured; known: {known}")
    return replace(config, calibration=replace(config.calibration, calibrators=wanted))


def _one_market(config: AppConfig, verb: str) -> AppConfig:
    """Refuse to guess which market a one-market file is about.

    A recording holds one market, so ``record`` has to be told which one when the configuration
    describes several. Refusing is the only honest answer: picking the first would write a file
    named after a market chosen by the order of the tables in a TOML file, and recording all of
    them would need one path per market and a flag that takes a directory.
    """
    if len(config.markets) > 1:
        known = ", ".join(one.market_id for one in config.markets)
        _fail(
            f"{verb} writes one market to one file, and the configuration names several; "
            f"choose one with --market: {known}",
            CONFIG_EXIT_CODE,
        )
    return config


def _open(path: Path) -> Recording:
    """Open the recording, translating a missing or malformed file into an exit code.

    Both failures belong to the person holding the file rather than to the engine, which is why
    neither reaches ``asyncio.run`` as a traceback: an ``OSError`` is a path that is not there and
    a ``ValueError`` is a file that is not a recording of a schema this build reads.
    """
    try:
        return open_recording(path)
    except OSError as failure:
        _fail(f"cannot read the recording at {path} ({failure})", CONFIG_EXIT_CODE)
    except ValueError as failure:
        _fail(str(failure), CONFIG_EXIT_CODE)


def _require_same_underlying(config: AppConfig, recording: Recording) -> None:
    """Refuse to feed one market's quotes to another market's chain.

    ``QuoteChain.apply`` raises on an update for a different underlying, and it is right to: every
    slice-level rule in this context would be measuring two markets at once. But it raises on the
    *first quote*, inside a task, halfway through a run -- so the same mistake is caught here,
    where it is one comparison and a message naming both sides.
    """
    configured = config.markets[0]
    if configured.underlying != recording.underlying:
        _fail(
            f"the recording at {recording.path} holds {recording.underlying} quotes and "
            f"{configured.market_id} is configured on {configured.underlying}",
            CONFIG_EXIT_CODE,
        )


def _drive(
    config: AppConfig,
    metrics: bool | None,
    max_reports: int | None,
    duration: float | None = None,
    adapters: Adapters | None = None,
    clock: Clock | None = None,
    timers: bool = True,
) -> None:
    """Build the pipeline and run it, translating a wiring failure into an exit code.

    ``timers=False`` is the replay's: no heartbeat task and no rediscovery poll, with the snapshot
    policy left exactly as the file states it (``pipeline.build_pipeline`` says why).

    ``metrics`` is the command line's override of ``[metrics]``, or ``None`` to take the file's
    choice. The sink is opened before the pipeline is built and closed after the run whatever
    ended it (:func:`_metrics_sink`). The clock is resolved first because a CSV row is stamped by
    the engine's own clock, so a replay's metrics carry recorded time (ADR-004).
    """
    engine_clock = clock if clock is not None else SystemClock()
    with _metrics_sink(_metrics_choice(config.metrics, metrics), engine_clock) as sink:
        bus = InProcessConflatingBus(sink)
        try:
            pipeline = build_pipeline(
                config,
                adapters if adapters is not None else default_adapters(),
                engine_clock,
                bus,
                sink,
                timers=timers,
            )
        except ConfigError as failure:
            _fail(str(failure), CONFIG_EXIT_CODE)
        try:
            asyncio.run(pipeline.run(max_reports=max_reports, duration_seconds=duration))
        except KeyboardInterrupt:  # pragma: no cover - requires a real signal
            # An interrupt is how a session is meant to end, so it is not a traceback.
            # `asyncio.run` has already cancelled the tasks, which is what closes the providers.
            typer.echo("interrupted", err=True)


def _metrics_choice(configured: MetricsConfig, flag: bool | None) -> MetricsConfig:
    """The file's ``[metrics]``, unless ``--metrics`` or ``--no-metrics`` says otherwise.

    The flag overrides rather than combines: ``--metrics`` beside ``sink = "csv"`` logs instead of
    writing, because one sink per run is what the port carries and a person who typed the flag
    wants to *see* the numbers now. A file's CSV is therefore one flag away from being suppressed,
    and that is the precedence every other flag here has over the file.
    """
    if flag is None:
        return configured
    return MetricsConfig(sink=MetricsSinkKind.LOGGING if flag else MetricsSinkKind.NULL)


@contextmanager
def _metrics_sink(choice: MetricsConfig, clock: Clock) -> Iterator[MetricsSink]:
    """The sink a run reports through, closed when the run ends -- however it ends.

    The one place in the engine that configures ``logging``, and it does so only for the logging
    sink. ``LoggingMetricsSink`` emits at INFO and an unconfigured root logger drops everything
    below WARNING, so without this line the choice would run the whole session and print nothing --
    a switch that reports success by staying silent. Nothing below this layer may touch logging at
    all (the domain has ``MetricsSink`` instead), which is what makes one call here sufficient.

    The CSV sink is opened here and closed in the ``finally``, which a ``typer.Exit`` from a
    refused configuration, an exception out of the run and a clean return all pass through: the
    file is flushed and released on every path, rather than left to the garbage collector with
    its buffered tail. A path that cannot be opened is the operator's mistake and exits with the
    configuration code, naming the path.
    """
    if choice.sink is MetricsSinkKind.LOGGING:
        logging.basicConfig(level=logging.INFO, format="%(message)s")
        yield LoggingMetricsSink()
        return
    if choice.sink is MetricsSinkKind.NULL or choice.path is None:
        # `path is None` cannot happen for a CSV choice -- `MetricsConfig` refuses it -- and is
        # spelled here only so the type checker knows the branch below has a path.
        yield NullMetricsSink()
        return
    try:
        sink = CsvMetricsSink(choice.path, clock)
    except OSError as failure:
        _fail(f"metrics: cannot write to {choice.path} ({failure})", CONFIG_EXIT_CODE)
    try:
        yield sink
    finally:
        sink.close()


def _fail(message: str, code: int) -> NoReturn:
    """Say what is wrong on stderr and stop with a code a script can branch on."""
    typer.echo(message, err=True)
    raise typer.Exit(code=code)
