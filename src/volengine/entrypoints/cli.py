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
"""

from __future__ import annotations

import asyncio
import logging
import math
from dataclasses import replace
from pathlib import Path
from typing import Annotated, NoReturn

import typer

from volengine.entrypoints.config import AppConfig, ConfigError, load_config
from volengine.entrypoints.pipeline import (
    Adapters,
    build_pipeline,
    default_adapters,
    with_recording,
    with_replay,
)
from volengine.market_data.adapters.recorded import Recording, open_recording
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
    # build is worse than one that names what the shipped configuration actually lists.
    typer.Option("--calibrators", help="Comma-separated producers to run, e.g. svi-scipy."),
]
MetricsOption = Annotated[
    bool,
    typer.Option("--metrics/--no-metrics", help="Log every metric the contexts emit."),
]
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
    metrics: MetricsOption = False,
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
    metrics: MetricsOption = False,
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
    metrics: MetricsOption = False,
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
    metrics: MetricsOption = False,
    count: Annotated[int | None, typer.Option(help="Stop after this many reports.")] = None,
) -> None:
    """Replay a recorded session through the whole engine, deterministically (ADR-004).

    Every number a replayed report carries comes out of the file: the quotes, and the instants the
    engine read them at, because the clock is a ``SimulatedClock`` the recording moves quote by
    quote rather than one that ticks on its own. Two replays of one recording write the same
    report, byte for byte.

    **What that does not remove is conflation** (ADR-003). A replay hands its quotes over as fast
    as the consumer takes them, so a snapshot published while the previous fit is still on its pool
    is overwritten in a one-slot mailbox -- and *how many* reports a long recording produces is
    therefore still a function of how fast this machine fits. That is the engine behaving as it
    does live, reproduced rather than removed; ``docs/SEAMS.md`` says where it bites.

    Which market runs is the recording's decision, not the flag's: the file names the market it
    holds and the configuration is narrowed to it. A ``--market`` option here could only agree or
    disagree with the file, and the disagreement would be a session replayed against another
    market's conventions.

    **No ``--duration``.** A recording is finite and ends when it ends; a time limit would be
    counted on the recorded clock, where "ten seconds" is a property of the file rather than of
    the person waiting. ``--count`` stops early on a number the engine controls.
    """
    if count is not None and count <= 0:
        _fail(f"--count must be positive, got {count}", CONFIG_EXIT_CODE)
    session = _open(recording)
    selected = _without_heartbeat(_configure(config, session.market_id, calibrators))
    _require_same_underlying(selected, session)
    clock = SimulatedClock(session.started_at)
    _drive(
        selected,
        metrics=metrics,
        max_reports=count,
        adapters=with_replay(default_adapters(), session, clock),
        clock=clock,
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
    return AppConfig(markets=selected, calibration=config.calibration, risk=config.risk)


def _only_calibrators(config: AppConfig, names: str) -> AppConfig:
    wanted = tuple(name.strip() for name in names.split(",") if name.strip())
    if not wanted:
        raise ConfigError(f"--calibrators names nothing, got {names!r}")
    unknown = [name for name in wanted if name not in config.calibration.calibrators]
    if unknown:
        known = ", ".join(config.calibration.calibrators)
        raise ConfigError(f"no calibrator named {unknown[0]!r} is configured; known: {known}")
    return AppConfig(
        markets=config.markets,
        calibration=replace(config.calibration, calibrators=wanted),
        risk=config.risk,
    )


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


def _without_heartbeat(config: AppConfig) -> AppConfig:
    """Drop ``max_quiet_seconds`` for a replay, which is the one run where it cannot mean anything.

    The heartbeat is a timer racing the stream so that a *stopped feed* is heard about rather than
    mistaken for a calm market (``Pipeline._heartbeat``). Under a replay there is no such thing as
    a stopped feed: time only moves when the recording moves it, so a timer either never fires --
    the clock never reaches its deadline on its own -- or spins the loop asking the same question
    of a chain nothing has changed. Worse, whether it fires at all would depend on how the event
    loop interleaved two tasks, which is precisely the non-determinism this command exists to
    remove.

    Left as a transformation of the configuration rather than a flag on ``Pipeline``: the graph is
    built from the file, and the honest way to say "this run has no heartbeat" is a file that does
    not configure one. ``docs/SEAMS.md`` records what stays open -- a recorded session's *quiet*
    periods replay as quiet, and nothing in a replay reproduces the snapshots the heartbeat emitted
    during the original run.
    """
    return AppConfig(
        markets=tuple(
            replace(market, snapshot=replace(market.snapshot, max_quiet_seconds=None))
            for market in config.markets
        ),
        calibration=config.calibration,
        risk=config.risk,
    )


def _drive(
    config: AppConfig,
    metrics: bool,
    max_reports: int | None,
    duration: float | None = None,
    adapters: Adapters | None = None,
    clock: Clock | None = None,
) -> None:
    """Build the pipeline and run it, translating a wiring failure into an exit code.

    The one place in the engine that configures ``logging``, and it does so only for ``--metrics``.
    ``LoggingMetricsSink`` emits at INFO and an unconfigured root logger drops everything below
    WARNING, so without this line the flag would run the whole session and print nothing --
    a switch that reports success by staying silent. Nothing below this layer may touch logging at
    all (the domain has ``MetricsSink`` instead), which is what makes one call here sufficient.
    """
    sink: MetricsSink = NullMetricsSink()
    if metrics:
        logging.basicConfig(level=logging.INFO, format="%(message)s")
        sink = LoggingMetricsSink()
    bus = InProcessConflatingBus(sink)
    try:
        pipeline = build_pipeline(
            config,
            adapters if adapters is not None else default_adapters(),
            clock if clock is not None else SystemClock(),
            bus,
            sink,
        )
    except ConfigError as failure:
        _fail(str(failure), CONFIG_EXIT_CODE)
    try:
        asyncio.run(pipeline.run(max_reports=max_reports, duration_seconds=duration))
    except KeyboardInterrupt:  # pragma: no cover - requires a real signal
        # An interrupt is how a session is meant to end, so it is not a traceback. `asyncio.run`
        # has already cancelled the tasks, which is what closes the providers.
        typer.echo("interrupted", err=True)


def _fail(message: str, code: int) -> NoReturn:
    """Say what is wrong on stderr and stop with a code a script can branch on."""
    typer.echo(message, err=True)
    raise typer.Exit(code=code)
