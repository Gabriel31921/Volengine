"""Every threshold this engine draws a line at, read from one TOML file at start-up (ADR-012).

Cadence, admissibility, acceptance, freshness, bump sizes and the published mesh are all
*operational* decisions: the right number for Deribit's wings is not the right number for SPX, and
a deployment that has to edit Python to retune one is a deployment that never retunes it. So they
arrive as data, and this module is the one place that turns text into them.

**It parses; it does not restate.** ``AdmissibilityThresholds``, ``SnapshotPolicyConfig``,
``Acceptance``, ``FreshnessPolicy``, ``BumpSpec``, ``GridSpec``, ``Weighting`` and
``MarketConventions`` already exist, already carry their invariants in ``__post_init__``, and are
already the types the use cases are built from. A parallel set of ``…Config`` dataclasses here
would be a second declaration of the same numbers with a second copy of the same guards, free to
drift from the first -- so what this module adds is a *reader*, and the objects it hands back are
the ones the contexts own. That is why the TOML keys are spelled exactly like the fields they
fill: one vocabulary, no translation table to keep honest.

**Every failure is a ``ConfigError`` naming the table it came from.** A domain constructor raises
``ValueError("The reject_seconds must be above warn_seconds")``, which is the right message and
the wrong context -- a traceback out of ``risk/domain/`` for a typo in a file tells an operator to
read the wrong code. Each construction is therefore wrapped, and the original is chained rather
than swallowed, so ``__cause__`` still holds the rule that was broken.
"""

from __future__ import annotations

import math
import tomllib
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import datetime, time
from enum import StrEnum
from pathlib import Path
from typing import Any

from volengine.market_data.domain.admissibility import AdmissibilityThresholds
from volengine.market_data.domain.market_conventions import (
    DayCount,
    ForwardMethod,
    MarketConventions,
    Numeraire,
)
from volengine.market_data.domain.snapshot_policy import SnapshotPolicyConfig
from volengine.parametric_pricing.application.acl import Weighting
from volengine.parametric_pricing.application.calibrate_on_snapshot import Acceptance
from volengine.parametric_pricing.application.grid_spec import GridSpec
from volengine.risk.application.compute_report import ReportSettings
from volengine.risk.domain.freshness_policy import FreshnessPolicy
from volengine.risk.domain.portfolio import Portfolio, Position
from volengine.risk.domain.pricing import OptionKindR
from volengine.risk.domain.valuation import BumpSpec


class ConfigError(Exception):
    """A configuration file that cannot be turned into a running engine.

    One exception type for the whole of ``entrypoints/``, deliberately: an unreadable file, a
    missing key, a value of the wrong shape, a threshold the domain refuses and an adapter name
    nobody registered are all the same event to whoever is holding the file -- *fix line N* -- and
    a hierarchy here would only ask the CLI to translate five classes back into one message.

    It is not a ``MarketDataError`` or a ``RiskError`` and must never become one: those hierarchies
    are for conditions a market can produce, and a typo in a TOML file is not one of them.
    """


@dataclass(frozen=True, slots=True)
class MarketConfig:
    """One market: how it is read, how it is judged, and who supplies its quotes."""

    conventions: MarketConventions
    """Daycount, expiry time, numeraire and forward method (ADR-002).

    It carries ``market_id`` and ``underlying`` already, which is why neither is a field of its
    own here even though ``Implementation.md`` lists them separately: two spellings of one
    identifier can disagree, and the copy that lost would be the one every published event is
    stamped with.
    """

    admissibility: AdmissibilityThresholds
    """Where each quote-quality rule draws its line for this market."""

    snapshot: SnapshotPolicyConfig
    """Cadence, material move, coverage floor and the heartbeat."""

    provider: str
    """Name of the adapter that supplies this market's quotes, e.g. ``"constant"``.

    A name rather than an import path: the mapping from name to class is made once, in
    ``pipeline.default_adapters()``, so a configuration file cannot reach an arbitrary object.
    """

    max_skew_seconds: float
    """How far the venue's clock may sit from ours before its stamps are disbelieved (ADR-021)."""

    @property
    def market_id(self) -> str:
        """Identity of the market, as every published event will spell it."""
        return self.conventions.market_id

    @property
    def underlying(self) -> str:
        """Symbol this market's chain is written on."""
        return self.conventions.underlying

    def __post_init__(self) -> None:
        if not self.provider.strip():
            raise ValueError(f"The provider name must not be empty, got {self.provider!r}")
        if not math.isfinite(self.max_skew_seconds) or self.max_skew_seconds <= 0:
            raise ValueError(
                f"The max_skew_seconds must be positive and finite, got {self.max_skew_seconds}"
            )


@dataclass(frozen=True, slots=True)
class CalibrationConfig:
    """Which calibrators run, on what mesh, weighted how, and when a fit may be published."""

    calibrators: tuple[str, ...]
    """Adapter names of the producers to run, e.g. ``("svi-scipy",)``. Non-empty, no duplicates.

    Two entries is the arrangement the project exists to compare, and the same name twice would
    be two producers with one identity publishing onto one topic -- a wiring mistake that would
    read downstream as a producer contradicting itself.
    """

    grid: GridSpec
    """The moneyness mesh every fitted surface is published on (ADR-001)."""

    weighting: Weighting
    """How much each quote counts in the loss (ADR-018)."""

    acceptance: Acceptance
    """The RMSE a slice must beat to go out rather than be republished stale (ADR-006)."""

    def __post_init__(self) -> None:
        if not self.calibrators:
            raise ValueError("At least one calibrator must be configured")
        if len(set(self.calibrators)) != len(self.calibrators):
            raise ValueError(f"The calibrators must be distinct, got {self.calibrators}")


@dataclass(frozen=True, slots=True)
class RiskConfig:
    """The book, the freshness policy, the bump sizes and where a finished report goes."""

    portfolio: Portfolio
    """The positions every report is computed over. Non-empty, by its own constructor."""

    freshness: FreshnessPolicy
    """The two ages at which a report calls itself degraded, then refuses (Design 7.2)."""

    settings: ReportSettings
    """Bump sizes and the discount factor."""

    writer: str
    """Name of the adapter a finished report is handed to, e.g. ``"console"``."""

    def __post_init__(self) -> None:
        if not self.writer.strip():
            raise ValueError(f"The writer name must not be empty, got {self.writer!r}")


@dataclass(frozen=True, slots=True)
class AppConfig:
    """One file, one engine: every market it runs and the two consumers behind them."""

    markets: tuple[MarketConfig, ...]
    """The markets to ingest. Non-empty, with distinct identifiers.

    Distinct because ``market_id`` is what every topic, every metric tag and every surface is
    keyed by: two markets sharing one would interleave two chains on one topic and average two
    latencies into one series.
    """

    calibration: CalibrationConfig
    """Shared by every market. One mesh and one acceptance rule keep the published surfaces
    comparable, which is the measurement this engine is for; a per-market mesh is a change this
    type can grow the day a market needs one."""

    risk: RiskConfig
    """Shared by every market, for the same reason: one book valued the same way everywhere."""

    def __post_init__(self) -> None:
        if not self.markets:
            raise ValueError("At least one market must be configured")
        identifiers = [market.market_id for market in self.markets]
        if len(set(identifiers)) != len(identifiers):
            raise ValueError(f"The market identifiers must be distinct, got {identifiers}")


def load_config(path: Path) -> AppConfig:
    """Read one TOML file and build the objects the composition root wires.

    Args:
        path: The file to read.

    Returns:
        The whole configuration, validated: every threshold has already passed the invariants of
        the type that owns it, so nothing downstream of here can fail on a number.

    Raises:
        ConfigError: If the file cannot be read, is not valid TOML, is missing a key, holds a
            value of the wrong shape, or holds one the domain refuses. The message names the
            table and the key; the original exception is chained.
    """
    try:
        with path.open("rb") as handle:
            raw: Mapping[str, Any] = tomllib.load(handle)
    except OSError as failure:
        raise ConfigError(f"{path}: cannot be read ({failure})") from failure
    except tomllib.TOMLDecodeError as failure:
        raise ConfigError(f"{path}: is not valid TOML ({failure})") from failure

    return _built(
        str(path),
        lambda: AppConfig(
            markets=tuple(
                _market(table, f"market[{index}]")
                for index, table in enumerate(_tables(raw, "market", "the file"))
            ),
            calibration=_calibration(_table(raw, "calibration", "the file")),
            risk=_risk(_table(raw, "risk", "the file")),
        ),
    )


# --- section readers


def _market(raw: Mapping[str, Any], where: str) -> MarketConfig:
    conventions = _table(raw, "conventions", where)
    return _built(
        where,
        lambda: MarketConfig(
            conventions=_built(
                f"{where}.conventions",
                lambda: MarketConventions(
                    market_id=_text(raw, "id", where),
                    underlying=_text(raw, "underlying", where),
                    day_count=_member(conventions, "day_count", f"{where}.conventions", DayCount),
                    expiry_time_utc=_clock_time(
                        conventions, "expiry_time_utc", f"{where}.conventions"
                    ),
                    numeraire=_member(conventions, "numeraire", f"{where}.conventions", Numeraire),
                    forward_method=_member(
                        conventions, "forward_method", f"{where}.conventions", ForwardMethod
                    ),
                ),
            ),
            admissibility=_admissibility(
                _table(raw, "admissibility", where), f"{where}.admissibility"
            ),
            snapshot=_snapshot(_table(raw, "snapshot", where), f"{where}.snapshot"),
            provider=_text(raw, "provider", where),
            max_skew_seconds=_number(raw, "max_skew_seconds", where),
        ),
    )


def _admissibility(raw: Mapping[str, Any], where: str) -> AdmissibilityThresholds:
    return _built(
        where,
        lambda: AdmissibilityThresholds(
            max_spread_rel=_number(raw, "max_spread_rel", where),
            max_age_seconds=_number(raw, "max_age_seconds", where),
            moneyness_range=_pair(raw, "moneyness_range", where),
            max_iv_divergence_bp=_number(raw, "max_iv_divergence_bp", where),
            convexity_tolerance=_number(raw, "convexity_tolerance", where),
            min_size=_number(raw, "min_size", where),
        ),
    )


def _snapshot(raw: Mapping[str, Any], where: str) -> SnapshotPolicyConfig:
    return _built(
        where,
        lambda: SnapshotPolicyConfig(
            cadence_seconds=_number(raw, "cadence_seconds", where),
            material_move_threshold=_number(raw, "material_move_threshold", where),
            min_coverage_ratio=_number(raw, "min_coverage_ratio", where),
            # Absent means "no heartbeat", which is a real deployment choice and not a default
            # anyone should have to spell. Present and null is refused by `_optional_number`,
            # because TOML has no null and the intent would be unreadable.
            max_quiet_seconds=_optional_number(raw, "max_quiet_seconds", where),
        ),
    )


def _calibration(raw: Mapping[str, Any]) -> CalibrationConfig:
    where = "calibration"
    grid = _table(raw, "grid", where)
    weighting = _table(raw, "weighting", where)
    acceptance = _table(raw, "acceptance", where)
    return _built(
        where,
        lambda: CalibrationConfig(
            calibrators=_texts(raw, "calibrators", where),
            grid=_built(
                f"{where}.grid",
                lambda: GridSpec(
                    k_min=_number(grid, "k_min", f"{where}.grid"),
                    k_max=_number(grid, "k_max", f"{where}.grid"),
                    n_nodes=_integer(grid, "n_nodes", f"{where}.grid"),
                ),
            ),
            weighting=_built(
                f"{where}.weighting",
                lambda: Weighting(
                    spread_scale=_number(weighting, "spread_scale", f"{where}.weighting"),
                    flagged_factor=_number(weighting, "flagged_factor", f"{where}.weighting"),
                    unpaired_itm_factor=_number(
                        weighting, "unpaired_itm_factor", f"{where}.weighting"
                    ),
                ),
            ),
            acceptance=_built(
                f"{where}.acceptance",
                lambda: Acceptance(
                    max_rmse_vol_bp=_number(acceptance, "max_rmse_vol_bp", f"{where}.acceptance")
                ),
            ),
        ),
    )


def _risk(raw: Mapping[str, Any]) -> RiskConfig:
    where = "risk"
    freshness = _table(raw, "freshness", where)
    report = _table(raw, "report", where)
    bumps = _table(report, "bumps", f"{where}.report")
    return _built(
        where,
        lambda: RiskConfig(
            portfolio=_built(
                f"{where}.position",
                lambda: Portfolio(
                    positions=tuple(
                        _position(table, f"{where}.position[{index}]")
                        for index, table in enumerate(_tables(raw, "position", where))
                    )
                ),
            ),
            freshness=_built(
                f"{where}.freshness",
                lambda: FreshnessPolicy(
                    warn_seconds=_number(freshness, "warn_seconds", f"{where}.freshness"),
                    reject_seconds=_number(freshness, "reject_seconds", f"{where}.freshness"),
                ),
            ),
            settings=_built(
                f"{where}.report",
                lambda: ReportSettings(
                    bumps=_built(
                        f"{where}.report.bumps",
                        lambda: BumpSpec(
                            forward_rel=_number(bumps, "forward_rel", f"{where}.report.bumps"),
                            vol_abs=_number(bumps, "vol_abs", f"{where}.report.bumps"),
                        ),
                    ),
                    discount=_number(report, "discount", f"{where}.report"),
                ),
            ),
            writer=_text(raw, "writer", where),
        ),
    )


def _position(raw: Mapping[str, Any], where: str) -> Position:
    return _built(
        where,
        lambda: Position(
            underlying=_text(raw, "underlying", where),
            expiry=_instant(raw, "expiry", where),
            strike=_number(raw, "strike", where),
            kind=_member(raw, "kind", where, OptionKindR),
            quantity=_number(raw, "quantity", where),
        ),
    )


# --- primitives
#
# Every reader takes the table, the key and where the table came from, and every failure names all
# three. A `KeyError` out of a nested comprehension says `'k_min'` and nothing else, which is the
# one thing an operator holding a two-hundred-line file already knows.


def _built[T](where: str, build: Callable[[], T]) -> T:
    """Run a constructor, and blame the configuration rather than the domain if it refuses.

    The domain's message is kept verbatim -- it is the one that names the rule -- and the table is
    prefixed onto it. Chained rather than replaced, so ``__cause__`` still holds the original for
    anyone reading a traceback.
    """
    try:
        return build()
    except ValueError as failure:
        raise ConfigError(f"{where}: {failure}") from failure


def _require(raw: Mapping[str, Any], key: str, where: str) -> Any:
    if key not in raw:
        raise ConfigError(f"{where}: the key {key!r} is missing")
    return raw[key]


def _table(raw: Mapping[str, Any], key: str, where: str) -> Mapping[str, Any]:
    value = _require(raw, key, where)
    if not isinstance(value, dict):
        raise ConfigError(f"{where}: {key!r} must be a table, got {type(value).__name__}")
    return value


def _tables(raw: Mapping[str, Any], key: str, where: str) -> tuple[Mapping[str, Any], ...]:
    """An array of tables -- ``[[market]]`` -- which is how repetition is spelled in TOML."""
    value = _require(raw, key, where)
    if not isinstance(value, list) or not all(isinstance(entry, dict) for entry in value):
        raise ConfigError(f"{where}: {key!r} must be an array of tables, e.g. [[{key}]]")
    return tuple(value)


def _number(raw: Mapping[str, Any], key: str, where: str) -> float:
    return _as_number(_require(raw, key, where), key, where)


def _optional_number(raw: Mapping[str, Any], key: str, where: str) -> float | None:
    """A number, or ``None`` when the key is simply absent. TOML has no null to mean it with."""
    if key not in raw:
        return None
    return _as_number(raw[key], key, where)


def _as_number(value: Any, key: str, where: str) -> float:
    # `isinstance(True, int)` is True, so a bare `isinstance(value, int | float)` accepts `true`
    # and turns it into 1.0 -- a threshold nobody chose, passing every guard downstream.
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ConfigError(f"{where}: {key!r} must be a number, got {value!r}")
    return float(value)


def _integer(raw: Mapping[str, Any], key: str, where: str) -> int:
    value = _require(raw, key, where)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConfigError(f"{where}: {key!r} must be an integer, got {value!r}")
    return value


def _text(raw: Mapping[str, Any], key: str, where: str) -> str:
    value = _require(raw, key, where)
    if not isinstance(value, str):
        raise ConfigError(f"{where}: {key!r} must be a string, got {value!r}")
    return value


def _texts(raw: Mapping[str, Any], key: str, where: str) -> tuple[str, ...]:
    value = _require(raw, key, where)
    if not isinstance(value, list) or not all(isinstance(entry, str) for entry in value):
        raise ConfigError(f"{where}: {key!r} must be an array of strings, got {value!r}")
    return tuple(value)


def _pair(raw: Mapping[str, Any], key: str, where: str) -> tuple[float, float]:
    value = _require(raw, key, where)
    if not isinstance(value, list) or len(value) != 2:
        raise ConfigError(f"{where}: {key!r} must be an array of two numbers, got {value!r}")
    return _as_number(value[0], key, where), _as_number(value[1], key, where)


def _member[E: StrEnum](raw: Mapping[str, Any], key: str, where: str, kind: type[E]) -> E:
    """One member of a ``StrEnum``, by its wire value.

    By value rather than by name because the value is what the enum documents as its wire format
    and what a human recognises -- ``"ACT/365F"`` is a daycount, ``ACT_365F`` is a Python
    identifier -- and the error lists every alternative, which is the whole of what an operator
    needs to fix it.
    """
    value = _text(raw, key, where)
    try:
        return kind(value)
    except ValueError as failure:
        allowed = ", ".join(member.value for member in kind)
        raise ConfigError(f"{where}: {key!r} must be one of {allowed}, got {value!r}") from failure


def _clock_time(raw: Mapping[str, Any], key: str, where: str) -> time:
    """A TOML local time -- ``08:00:00`` -- which ``tomllib`` already hands back as a ``time``."""
    value = _require(raw, key, where)
    if not isinstance(value, time):
        raise ConfigError(f"{where}: {key!r} must be a time of day, e.g. 08:00:00, got {value!r}")
    return value


def _instant(raw: Mapping[str, Any], key: str, where: str) -> datetime:
    """A TOML offset date-time -- ``2026-09-25T08:00:00Z``.

    The offset is not optional and the check is here rather than left to ``require_aware``: a
    local date-time is valid TOML and parses to a *naive* ``datetime``, which every subtraction in
    this engine would then raise ``TypeError`` on, a long way from the line that wrote it.
    """
    value = _require(raw, key, where)
    if not isinstance(value, datetime):
        raise ConfigError(
            f"{where}: {key!r} must be a date-time with an offset, "
            f"e.g. 2026-09-25T08:00:00Z, got {value!r}"
        )
    if value.tzinfo is None:
        raise ConfigError(f"{where}: {key!r} must carry a UTC offset, got the local time {value}")
    return value
