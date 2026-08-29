"""The command line: four verbs, and the only place an event loop is started.

Deliberately thin. Everything below it is already composed by ``pipeline.build_pipeline``, so what
is left here is the three decisions a person makes when they type the command -- which file to
read, which market to run, which producers to run it with -- plus turning a ``ConfigError`` into
an exit code instead of a traceback.

**``record`` and ``replay`` are declared and refuse.** They belong to F3-B, which needs a
``RecordedProvider`` and a recorder adapter that do not exist yet. Declaring them now rather than
adding them later is what makes ``volengine --help`` an honest description of the engine's shape:
the replay of ADR-004 is not an afterthought, it is a mode this system was designed around, and a
CLI that did not mention it would read as though determinism had been forgotten.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import replace
from pathlib import Path
from typing import Annotated, NoReturn

import typer

from volengine.entrypoints.config import AppConfig, ConfigError, load_config
from volengine.entrypoints.pipeline import build_pipeline, default_adapters
from volengine.platform.bus import InProcessConflatingBus
from volengine.platform.clock import SystemClock
from volengine.platform.metrics import LoggingMetricsSink, MetricsSink, NullMetricsSink

CONFIG_EXIT_CODE = 2
"""What a bad configuration exits with. Distinct from 1 so a script can tell "you typed something
wrong" from "the command is not implemented yet"."""

NOT_IMPLEMENTED_EXIT_CODE = 1

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


@app.command()
def run(
    config: ConfigOption,
    market: MarketOption = None,
    calibrators: CalibratorsOption = None,
    metrics: MetricsOption = False,
) -> None:
    """Ingest, calibrate and report continuously until the streams end or you interrupt."""
    _drive(_configure(config, market, calibrators), metrics=metrics, max_reports=None)


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
def record(config: ConfigOption) -> None:
    """Record a session to disk for later replay. **Not implemented until F3-B.**"""
    _fail(
        "record: not implemented until F3-B, which adds the recorder adapter (ADR-004)",
        NOT_IMPLEMENTED_EXIT_CODE,
    )


@app.command()
def replay(config: ConfigOption) -> None:
    """Replay a recorded session deterministically. **Not implemented until F3-B.**"""
    _fail(
        "replay: not implemented until F3-B, which adds the recorded provider (ADR-004)",
        NOT_IMPLEMENTED_EXIT_CODE,
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


def _drive(config: AppConfig, metrics: bool, max_reports: int | None) -> None:
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
        pipeline = build_pipeline(config, default_adapters(), SystemClock(), bus, sink)
    except ConfigError as failure:
        _fail(str(failure), CONFIG_EXIT_CODE)
    try:
        asyncio.run(pipeline.run(max_reports=max_reports))
    except KeyboardInterrupt:  # pragma: no cover - requires a real signal
        # An interrupt is how a session is meant to end, so it is not a traceback. `asyncio.run`
        # has already cancelled the tasks, which is what closes the providers.
        typer.echo("interrupted", err=True)


def _fail(message: str, code: int) -> NoReturn:
    """Say what is wrong on stderr and stop with a code a script can branch on."""
    typer.echo(message, err=True)
    raise typer.Exit(code=code)
