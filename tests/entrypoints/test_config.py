"""Reading a TOML file: what it builds, and what it refuses to build.

Every rejection test starts from one valid file and breaks a single line, so what the test asserts
is exactly what it changed. The point of most of them is not that a bad number is refused --
the domain types already do that, and their own tests already say so -- but that the refusal
arrives as a ``ConfigError`` naming the table, rather than as a ``ValueError`` out of a module the
operator has no reason to be reading.
"""

from __future__ import annotations

from datetime import UTC, datetime, time, timedelta
from importlib.util import find_spec
from pathlib import Path

import pytest

from tests.entrypoints.builders import CONFIG_TOML, replacing, without, write_config
from volengine.entrypoints import config as config_module
from volengine.entrypoints.config import (
    JAX_EXTRA_LIBRARIES,
    AppConfig,
    ConfigError,
    MetricsConfig,
    MetricsSinkKind,
    load_config,
)
from volengine.market_data.adapters.deribit_ws import DeribitSettings
from volengine.market_data.adapters.synthetic import SVIParamsSpec
from volengine.market_data.domain.market_conventions import DayCount, ForwardMethod, Numeraire
from volengine.neural_surface.application.train_on_snapshot import GateThresholds, TrainingSchedule
from volengine.neural_surface.domain.replay_buffer import StratificationSpec
from volengine.risk.domain.pricing import OptionKindR


def load(tmp_path: Path, text: str = CONFIG_TOML) -> AppConfig:
    return load_config(write_config(tmp_path, text))


# --- what a valid file builds


def test_a_market_carries_its_conventions(tmp_path: Path) -> None:
    market = load_config(write_config(tmp_path)).markets[0]

    assert market.conventions.day_count is DayCount.ACT_365F
    assert market.conventions.numeraire is Numeraire.INVERSE
    assert market.conventions.forward_method is ForwardMethod.PROVIDER_UNDERLYING


def test_the_expiry_time_is_read_as_a_time_of_day(tmp_path: Path) -> None:
    market = load_config(write_config(tmp_path)).markets[0]

    assert market.conventions.expiry_time_utc == time(8, 0)


def test_the_market_id_is_not_stored_twice(tmp_path: Path) -> None:
    """One spelling of the identity: the property reads the conventions, never a second field."""
    market = load_config(write_config(tmp_path)).markets[0]

    assert market.market_id == market.conventions.market_id == "BTC-DERIBIT"


def test_the_thresholds_arrive_as_the_types_the_contexts_own(tmp_path: Path) -> None:
    config = load_config(write_config(tmp_path))

    assert config.markets[0].admissibility.max_spread_rel == 0.5
    assert config.markets[0].snapshot.cadence_seconds == 1.0
    assert config.calibration.acceptance.max_rmse_vol_bp == 50.0
    assert config.risk.freshness.reject_seconds == 120.0
    assert config.risk.settings.bumps.vol_abs == 0.01


def test_a_position_is_read_with_its_offset(tmp_path: Path) -> None:
    position = load_config(write_config(tmp_path)).risk.portfolio.positions[0]

    assert position.expiry == datetime(2026, 8, 27, 8, 0, tzinfo=UTC)
    assert position.kind is OptionKindR.CALL


def test_an_absent_heartbeat_is_none_rather_than_a_default(tmp_path: Path) -> None:
    """TOML has no null, so absence is the only way to say "no heartbeat" -- and it must mean it."""
    config = load_config(write_config(tmp_path, without(CONFIG_TOML, "max_quiet_seconds")))

    assert config.markets[0].snapshot.max_quiet_seconds is None


# --- what it refuses


def test_a_missing_file_names_the_path(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="cannot be read"):
        load_config(tmp_path / "absent.toml")


def test_a_malformed_file_is_not_valid_toml(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="not valid TOML"):
        load(tmp_path, "[market\n")


def test_a_missing_key_names_the_key_and_its_table(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match=r"market\[0\].*'provider' is missing"):
        load(tmp_path, without(CONFIG_TOML, "provider ="))


def test_a_missing_section_names_the_section(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="the key 'risk' is missing"):
        load(tmp_path, CONFIG_TOML.split("[risk]")[0])


def test_an_unknown_enum_member_lists_the_alternatives(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="ACT/365F"):
        load(tmp_path, replacing(CONFIG_TOML, "day_count", 'day_count = "ACT/360"'))


def test_a_threshold_the_domain_refuses_is_blamed_on_its_table(tmp_path: Path) -> None:
    """The domain's message survives; the table it came from is prefixed onto it."""
    with pytest.raises(ConfigError, match=r"risk\.freshness: .*above"):
        load(tmp_path, replacing(CONFIG_TOML, "reject_seconds", "reject_seconds = 5.0"))


def test_the_rule_that_was_broken_is_kept_as_the_cause(tmp_path: Path) -> None:
    """Chained rather than swallowed: a traceback still reaches the invariant that fired."""
    with pytest.raises(ConfigError) as failure:
        load(tmp_path, replacing(CONFIG_TOML, "reject_seconds", "reject_seconds = 5.0"))

    assert isinstance(failure.value.__cause__, ValueError)


def test_a_boolean_is_not_a_number(tmp_path: Path) -> None:
    """``isinstance(True, int)`` is True, so a bare number check would take ``true`` as 1.0."""
    with pytest.raises(ConfigError, match="must be a number"):
        load(tmp_path, replacing(CONFIG_TOML, "vol_abs", "vol_abs = true"))


def test_a_number_where_a_string_belongs_is_refused(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="must be a string"):
        load(tmp_path, replacing(CONFIG_TOML, "writer =", "writer = 3"))


def test_a_moneyness_range_of_the_wrong_length_is_refused(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="two numbers"):
        load(tmp_path, replacing(CONFIG_TOML, "moneyness_range", "moneyness_range = [-1.5]"))


def test_a_fractional_node_count_is_refused(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="must be an integer"):
        load(tmp_path, replacing(CONFIG_TOML, "n_nodes", "n_nodes = 9.5"))


def test_a_naive_expiry_is_refused_by_name(tmp_path: Path) -> None:
    """A local date-time is valid TOML and parses naive, which every subtraction later hates."""
    with pytest.raises(ConfigError, match="UTC offset"):
        load(tmp_path, replacing(CONFIG_TOML, "expiry =", "expiry = 2026-08-27T08:00:00"))


def test_a_file_with_no_market_is_refused(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="array of tables"):
        load(tmp_path, replacing(CONFIG_TOML, "[[market]]", "[market]"))


def test_two_markets_may_not_share_an_identifier(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="distinct"):
        load(tmp_path, CONFIG_TOML + CONFIG_TOML.split("[calibration]")[0])


def test_the_same_calibrator_may_not_be_listed_twice(tmp_path: Path) -> None:
    """Two producers with one identity would publish onto one topic and read as a contradiction."""
    with pytest.raises(ConfigError, match="distinct"):
        load(
            tmp_path,
            replacing(CONFIG_TOML, "calibrators =", 'calibrators = ["svi-scipy", "svi-scipy"]'),
        )


def test_an_empty_calibrator_list_is_refused(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="At least one calibrator"):
        load(tmp_path, replacing(CONFIG_TOML, "calibrators =", "calibrators = []"))


# --- the two adapter sections (F2-07)


SYNTHETIC_TOML = """
[market.synthetic]
forward0 = 50000.0
strikes_per_expiry = 7
log_moneyness_range = [-0.2, 0.2]
spread_bp = 150.0
vol_noise_bp = 25.0
size = 5.0
junk_quote_rate = 0.01
forward_move_rel = 0.001
latency_seconds = 0.01
jitter_seconds = 0.02
cycles = 4
interval_seconds = 0.5
seed = 7

[[market.synthetic.slice]]
expiry_days = 30.0
a = 0.020
b = 0.050
rho = -0.30
m = 0.0
sigma = 0.20

[[market.synthetic.slice]]
expiry_days = 90.0
a = 0.055
b = 0.090
rho = -0.25
m = 0.0
sigma = 0.25
"""
"""The invented market, appended to the valid file. Every value differs from the adapter's own
default, so a test asserting one of them cannot pass on a section that was never read."""


def synthetic(text: str = CONFIG_TOML) -> str:
    """The valid file, quoted by the synthetic feed and carrying its settings.

    The provider has to change with the section: a ``[market.synthetic]`` beside any other name
    is refused, which is its own test below.
    """
    return replacing(text, "provider =", 'provider = "synthetic"') + SYNTHETIC_TOML


def with_start(text: str, start: str) -> str:
    """The same file with an origin pinned *inside* the synthetic table.

    Appended after the slice array it would belong to the last slice instead, which TOML would
    accept and the loader would then blame on the wrong table.
    """
    return replacing(text, "seed =", f"seed = 7\nstart = {start}")


FIT_TOML = """
[calibration.fit]
huber_scale_bp = 80.0
durrleman_penalty_bp = 5000.0
durrleman_mesh_nodes = 21
durrleman_mesh_margin = 0.25
min_quotes_for_free_shape = 4
ridge_bp = 1.0
max_nfev = 300
"""


def test_the_synthetic_table_fills_the_generator_own_type(tmp_path: Path) -> None:
    """No mirror type: the file builds the ``SyntheticConfig`` the adapter itself declares."""
    settings = load(tmp_path, synthetic()).markets[0].synthetic

    assert settings is not None
    assert settings.config.forward0 == 50_000.0
    assert settings.config.strikes_per_expiry == 7
    assert settings.config.log_moneyness_range == (-0.2, 0.2)
    assert settings.config.cycles == 4
    assert settings.config.seed == 7


def test_the_slices_become_the_expiries_and_the_parameters_at_once(tmp_path: Path) -> None:
    """One array of tables fills two fields, which is what keeps them naming the same set.

    ``SyntheticConfig`` refuses an expiry with no parameters and a parameter set with no expiry;
    reading them from one array means that invariant cannot be broken by a file at all.
    """
    settings = load(tmp_path, synthetic()).markets[0].synthetic

    assert settings is not None
    assert settings.config.expiries == (timedelta(days=30), timedelta(days=90))
    assert set(settings.config.true_params) == set(settings.config.expiries)
    # `true_params` is typed as the generator protocol since F3-B, so that a Heston market can sit
    # in the same field. What a *file* builds is still an SVI slice, and the `isinstance` says so
    # before reading a parameter only that spelling has.
    generated = settings.config.true_params[timedelta(days=90)]
    assert isinstance(generated, SVIParamsSpec)
    assert generated.sigma == 0.25


def test_a_market_with_no_synthetic_table_carries_none(tmp_path: Path) -> None:
    """Absent is not a defaulted table: the adapter, not the loader, owns what absence means."""
    assert load(tmp_path).markets[0].synthetic is None


def test_an_absent_start_leaves_the_feed_on_the_wall_clock(tmp_path: Path) -> None:
    settings = load(tmp_path, synthetic()).markets[0].synthetic

    assert settings is not None
    assert settings.start is None


def test_a_pinned_start_is_read_with_its_offset(tmp_path: Path) -> None:
    """The other half of reproducibility: the stream is a function of the seed *and* this."""
    settings = load(tmp_path, with_start(synthetic(), "2026-09-01T00:00:00Z")).markets[0].synthetic

    assert settings is not None
    assert settings.start == datetime(2026, 9, 1, tzinfo=UTC)


def test_a_naive_start_is_refused_by_name(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="UTC offset"):
        load(tmp_path, with_start(synthetic(), "2026-09-01T00:00:00"))


def test_a_missing_key_in_the_synthetic_table_names_the_table(tmp_path: Path) -> None:
    """Complete when present: there is no partial override to fall back through."""
    text = without(synthetic(), "seed")

    with pytest.raises(ConfigError, match=r"market\[0\].synthetic: the key 'seed'"):
        load(tmp_path, text)


def test_a_number_of_days_the_calendar_cannot_hold_names_its_slice(tmp_path: Path) -> None:
    """``timedelta`` answers a NaN with a ``ValueError`` about floats, which names nothing."""
    text = replacing(synthetic(), "expiry_days = 30.0", "expiry_days = nan")

    with pytest.raises(ConfigError, match=r"slice\[0\]: 'expiry_days'"):
        load(tmp_path, text)


def test_a_slice_the_generator_refuses_is_blamed_on_its_own_table(tmp_path: Path) -> None:
    text = replacing(synthetic(), "sigma = 0.20", "sigma = 0.0")

    with pytest.raises(ConfigError, match=r"slice\[0\]: The SVI parameter sigma"):
        load(tmp_path, text)


def test_two_slices_on_one_expiry_are_refused(tmp_path: Path) -> None:
    text = replacing(synthetic(), "expiry_days = 90.0", "expiry_days = 30.0")

    with pytest.raises(ConfigError, match="distinct"):
        load(tmp_path, text)


def test_settings_given_to_a_provider_that_would_never_read_them_are_refused(
    tmp_path: Path,
) -> None:
    """The failure mode a silently ignored section has: a block of numbers nobody reads.

    The table is named after the provider it configures, which is what makes the pairing
    checkable without the loader knowing anything else about the registry.
    """
    with pytest.raises(ConfigError, match="which would never read them"):
        load(tmp_path, CONFIG_TOML + SYNTHETIC_TOML)


def test_the_fit_table_fills_the_calibrator_own_type(tmp_path: Path) -> None:
    fit = load(tmp_path, CONFIG_TOML + FIT_TOML).calibration.fit

    assert fit is not None
    assert fit.huber_scale_bp == 80.0
    assert fit.durrleman_mesh_nodes == 21
    assert fit.ridge_bp == 1.0
    assert fit.max_nfev == 300


def test_a_file_with_no_fit_table_leaves_the_tuning_absent(tmp_path: Path) -> None:
    """A file running ``flat-vol`` alone states nothing here, and must not have to."""
    assert load(tmp_path).calibration.fit is None


def test_a_fit_value_the_calibrator_refuses_is_blamed_on_its_table(tmp_path: Path) -> None:
    text = replacing(CONFIG_TOML + FIT_TOML, "max_nfev", "max_nfev = 0")

    with pytest.raises(ConfigError, match=r"calibration\.fit: The evaluation budget"):
        load(tmp_path, text)


def test_the_output_path_is_read_as_written(tmp_path: Path) -> None:
    """Relative and unresolved: against the working directory, like every other tool."""
    text = replacing(CONFIG_TOML, 'writer = "console"', 'writer = "csv"\noutput_path = "out.csv"')

    assert load(tmp_path, text).risk.output_path == Path("out.csv")


def test_a_writer_with_no_output_path_carries_none(tmp_path: Path) -> None:
    """Which writers need one is the registry's question, not this module's."""
    assert load(tmp_path).risk.output_path is None


def test_a_blank_output_path_is_refused_by_the_reader_rather_than_by_the_writer(
    tmp_path: Path,
) -> None:
    """``Path("")`` is ``PosixPath('.')``, so ``RiskConfig``'s own guard never sees this one.

    The check has to happen on the raw string, and it has to happen here: left to the adapter it
    arrives as an ``OSError`` about ``'.'``, naming a directory the operator never typed instead
    of the line they did.
    """
    text = replacing(CONFIG_TOML, 'writer = "console"', 'writer = "csv"\noutput_path = "   "')

    with pytest.raises(ConfigError, match=r"risk: 'output_path' must not be empty"):
        load(tmp_path, text)


def test_an_output_path_of_one_dot_is_still_the_working_directory(tmp_path: Path) -> None:
    """The guard above rejects blankness, not the path the blank used to collapse into: ``"."``
    is a thing an operator can legitimately write, and the writer is what refuses it."""
    text = replacing(CONFIG_TOML, 'writer = "console"', 'writer = "csv"\noutput_path = "."')

    assert load(tmp_path, text).risk.output_path == Path(".")


# --- the venue's transport (F3-C)


DERIBIT_TOML = """
[market.deribit]
ws_url = "wss://test.deribit.com/ws/api/v2"
rest_url = "https://test.deribit.com/api/v2"
heartbeat_seconds = 20.0
silence_timeout_seconds = 50.0
reconnect_initial_seconds = 2.0
reconnect_max_seconds = 30.0
request_timeout_seconds = 5.0
subscribe_batch_size = 100
"""
"""The transport, appended to the valid file. Every value differs from the adapter's own default,
so a test asserting one of them cannot pass on a section that was never read."""


def deribit(text: str = CONFIG_TOML) -> str:
    """The valid file, quoted by the live venue and carrying its transport settings."""
    return replacing(text, "provider =", 'provider = "deribit"') + DERIBIT_TOML


def test_the_deribit_table_fills_the_adapter_own_type(tmp_path: Path) -> None:
    settings = load(tmp_path, deribit()).markets[0].deribit

    assert settings == DeribitSettings(
        ws_url="wss://test.deribit.com/ws/api/v2",
        rest_url="https://test.deribit.com/api/v2",
        heartbeat_seconds=20.0,
        silence_timeout_seconds=50.0,
        reconnect_initial_seconds=2.0,
        reconnect_max_seconds=30.0,
        request_timeout_seconds=5.0,
        subscribe_batch_size=100,
    )


def test_a_market_with_no_deribit_table_carries_none(tmp_path: Path) -> None:
    """Absent is the adapter's defaults, decided by the adapter and not restated here."""
    assert (
        load(tmp_path, replacing(CONFIG_TOML, "provider =", 'provider = "deribit"'))
        .markets[0]
        .deribit
        is None
    )


def test_a_missing_key_in_the_deribit_table_names_the_table(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match=r"market\[0\].deribit: the key 'ws_url'"):
        load(tmp_path, without(deribit(), "ws_url"))


def test_a_transport_value_the_adapter_refuses_is_blamed_on_its_table(tmp_path: Path) -> None:
    text = replacing(deribit(), "heartbeat_seconds =", "heartbeat_seconds = 5.0")

    with pytest.raises(ConfigError, match=r"market\[0\].deribit: The heartbeat_seconds"):
        load(tmp_path, text)


def test_a_fractional_batch_size_is_refused(tmp_path: Path) -> None:
    text = replacing(deribit(), "subscribe_batch_size =", "subscribe_batch_size = 2.5")

    with pytest.raises(ConfigError, match="must be an integer"):
        load(tmp_path, text)


def test_transport_settings_given_to_another_provider_are_refused(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="'deribit' were given to the provider 'constant'"):
        load(tmp_path, CONFIG_TOML + DERIBIT_TOML)


# --- rediscovery (F3-C)


def test_the_rediscovery_cadence_is_read(tmp_path: Path) -> None:
    text = replacing(
        CONFIG_TOML, "max_skew_seconds =", "max_skew_seconds = 30.0\nrediscovery_seconds = 300.0"
    )

    assert load(tmp_path, text).markets[0].rediscovery_seconds == 300.0


def test_an_absent_rediscovery_cadence_is_none_rather_than_a_default(tmp_path: Path) -> None:
    """Absent means "discover once", the way an absent heartbeat means "no heartbeat"."""
    assert load(tmp_path).markets[0].rediscovery_seconds is None


@pytest.mark.parametrize("value", ["0.0", "-5.0", "nan", "inf"])
def test_a_rediscovery_cadence_that_could_not_be_polled_on_is_refused(
    tmp_path: Path, value: str
) -> None:
    text = replacing(
        CONFIG_TOML, "max_skew_seconds =", f"max_skew_seconds = 30.0\nrediscovery_seconds = {value}"
    )

    with pytest.raises(ConfigError, match="rediscovery_seconds must be positive and finite"):
        load(tmp_path, text)


# --- the JAX calibrator's table (F3-W1)


JAX_TOML = """
[calibration.jax]
huber_scale_bp = 80.0
durrleman_penalty_bp = 5000.0
durrleman_mesh_margin = 0.25
ridge_bp = 1.0
min_quotes_for_free_shape = 4
learning_rate = 0.02
hot_steps = 150
cold_steps = 120
tolerance_bp = 0.01
linesearch_steps = 9
"""
"""Every value differs from ``JaxFitSettings``'s own default, so an assertion on one of them cannot
pass on a table that was never read."""

HAS_JAX_EXTRA = all(find_spec(name) is not None for name in JAX_EXTRA_LIBRARIES)


def hide(monkeypatch: pytest.MonkeyPatch, library: str) -> None:
    """Make the loader's extra check see an installation without ``library``.

    Patched on the loader's own name for the lookup, so the refusal is observable on every
    installation rather than only on the CI leg that happens to lack the extra.
    """
    monkeypatch.setattr(
        config_module, "find_spec", lambda name: None if name == library else find_spec(name)
    )


@pytest.mark.skipif(not HAS_JAX_EXTRA, reason="needs the jax extra")
def test_the_jax_table_fills_the_calibrator_own_type(tmp_path: Path) -> None:
    """No mirror type: the file builds the ``JaxFitSettings`` the adapter itself declares."""
    jax_calibrator = pytest.importorskip("volengine.parametric_pricing.adapters.jax_calibrator")

    settings = load(tmp_path, CONFIG_TOML + JAX_TOML).calibration.jax

    assert settings == jax_calibrator.JaxFitSettings(
        huber_scale_bp=80.0,
        durrleman_penalty_bp=5000.0,
        durrleman_mesh_margin=0.25,
        ridge_bp=1.0,
        min_quotes_for_free_shape=4,
        learning_rate=0.02,
        hot_steps=150,
        cold_steps=120,
        tolerance_bp=0.01,
        linesearch_steps=9,
    )


def test_a_file_with_no_jax_table_leaves_the_tuning_absent(tmp_path: Path) -> None:
    assert load(tmp_path).calibration.jax is None


def test_a_file_with_no_jax_table_is_read_without_the_extra(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The guard on the test below: the extra is asked for by the table, not by every file read."""
    hide(monkeypatch, "jax")

    assert load(tmp_path).calibration.jax is None


def test_a_jax_table_without_the_extra_is_refused_with_the_remedy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A ``ConfigError`` naming the table and the extra, never an ``ImportError``."""
    hide(monkeypatch, "jax")

    with pytest.raises(ConfigError, match=r"calibration\.jax: .*uv sync --extra jax"):
        load(tmp_path, CONFIG_TOML + JAX_TOML)


@pytest.mark.skipif(not HAS_JAX_EXTRA, reason="needs the jax extra")
def test_a_missing_key_in_the_jax_table_names_the_table(tmp_path: Path) -> None:
    """Complete when present, like ``[calibration.fit]``."""
    with pytest.raises(ConfigError, match=r"calibration\.jax: the key 'hot_steps'"):
        load(tmp_path, without(CONFIG_TOML + JAX_TOML, "hot_steps"))


@pytest.mark.skipif(not HAS_JAX_EXTRA, reason="needs the jax extra")
def test_a_jax_value_the_calibrator_refuses_is_blamed_on_its_table(tmp_path: Path) -> None:
    text = replacing(CONFIG_TOML + JAX_TOML, "learning_rate", "learning_rate = 0.0")

    with pytest.raises(ConfigError, match=r"calibration\.jax: The learning rate"):
        load(tmp_path, text)


# --- the metrics sink (F3-W1)


def with_metrics(table: str) -> str:
    return CONFIG_TOML + "\n[metrics]\n" + table + "\n"


def test_a_file_with_no_metrics_table_gets_the_null_sink(tmp_path: Path) -> None:
    """Absent is what every run did before the table existed: nothing recorded."""
    assert load(tmp_path).metrics == MetricsConfig(sink=MetricsSinkKind.NULL)


def test_the_csv_sink_is_read_with_its_path(tmp_path: Path) -> None:
    metrics = load(tmp_path, with_metrics('sink = "csv"\npath = "out/metrics.csv"')).metrics

    assert metrics.sink is MetricsSinkKind.CSV
    assert metrics.path == Path("out/metrics.csv")


def test_the_logging_sink_takes_no_path(tmp_path: Path) -> None:
    assert load(tmp_path, with_metrics('sink = "logging"')).metrics.sink is MetricsSinkKind.LOGGING


def test_a_csv_sink_with_nowhere_to_write_is_refused(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="metrics: The 'csv' metrics sink needs a 'path'"):
        load(tmp_path, with_metrics('sink = "csv"'))


@pytest.mark.parametrize("sink", ["logging", "null"])
def test_a_path_given_to_a_sink_that_writes_no_file_is_refused(tmp_path: Path, sink: str) -> None:
    """Refused rather than ignored: a path is a file the operator expects to find afterwards."""
    with pytest.raises(ConfigError, match="writes no file"):
        load(tmp_path, with_metrics(f'sink = "{sink}"\npath = "metrics.csv"'))


def test_an_unknown_sink_lists_the_three_there_are(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="logging, null, csv"):
        load(tmp_path, with_metrics('sink = "prometheus"'))


def test_a_blank_metrics_path_is_refused_by_name(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="metrics: 'path' must not be empty"):
        load(tmp_path, with_metrics('sink = "csv"\npath = " "'))


# --- the neural producer (F3-W2)

NEURAL_TOML = """
[neural]
seed = 77

[neural.buffer]
moneyness_edges = [-0.4, -0.1, 0.1, 0.4]
tenor_edges = [0.02, 0.1, 0.5]
capacity_per_cell = 16
max_age_seconds = 900.0

[neural.mesh]
k_min = -0.5
k_max = 0.5
n_nodes = 11
tenors = [0.08, 0.25]

[neural.gate]
butterfly = 0.001
calendar = 0.0002

[neural.schedule]
replay_size = 32
restart_seconds = 1800.0
"""
"""A complete ``[neural]`` with no optional sub-table, so it is read on every installation."""

NETWORK_TOML = """
[neural.network]
hidden = [8, 8, 8]
activation = "softplus"
k_scale = 0.25
tenor_scale = 2.0
"""
"""Every value differs from ``NetworkSpec``'s own default."""

TORCH_FIT_TOML = """
[neural.fit]
cold_steps = 123
warm_steps = 7
cold_learning_rate = 0.02
warm_learning_rate = 0.002
butterfly_penalty = 50.0
calendar_penalty = 60.0
init_seed = 9
"""
"""Every value differs from ``TorchFitSettings``'s own default."""

HAS_TORCH_EXTRA = find_spec("torch") is not None

needs_torch = pytest.mark.skipif(not HAS_TORCH_EXTRA, reason="needs the neural extra")


def test_a_file_with_no_neural_table_leaves_the_producer_unconfigured(tmp_path: Path) -> None:
    assert load(tmp_path).neural is None


def test_the_neural_table_fills_the_types_neural_surface_owns(tmp_path: Path) -> None:
    """No mirror type: the buffer, the gate and the schedule are the context's own objects."""
    neural = load(tmp_path, CONFIG_TOML + NEURAL_TOML).neural

    assert neural is not None
    assert neural.buffer == StratificationSpec(
        moneyness_edges=(-0.4, -0.1, 0.1, 0.4),
        tenor_edges=(0.02, 0.1, 0.5),
        capacity_per_cell=16,
        max_age_seconds=900.0,
    )
    assert neural.gate == GateThresholds(butterfly=0.001, calendar=0.0002)
    assert neural.schedule == TrainingSchedule(replay_size=32, restart_seconds=1800.0)
    assert neural.seed == 77


def test_the_mesh_is_built_uniform_from_a_range_and_a_count(tmp_path: Path) -> None:
    """The one table spelled differently from its type: the axis lands exactly on its ends."""
    neural = load(tmp_path, CONFIG_TOML + NEURAL_TOML).neural

    assert neural is not None
    assert neural.mesh.log_moneyness[0] == -0.5
    assert neural.mesh.log_moneyness[-1] == 0.5
    assert len(neural.mesh.log_moneyness) == 11
    assert neural.mesh.log_moneyness[5] == pytest.approx(0.0, abs=1e-15)
    assert neural.mesh.tenors == (0.08, 0.25)


def test_an_absent_restart_interval_means_never(tmp_path: Path) -> None:
    neural = load(tmp_path, without(CONFIG_TOML + NEURAL_TOML, "restart_seconds")).neural

    assert neural is not None
    assert neural.schedule.restart_seconds is None


def test_a_neural_table_without_its_optional_tables_leaves_the_learner_defaults(
    tmp_path: Path,
) -> None:
    neural = load(tmp_path, CONFIG_TOML + NEURAL_TOML).neural

    assert neural is not None
    assert neural.network is None
    assert neural.fit is None


def test_a_neural_table_without_optional_tables_is_read_without_the_extra(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The extra is asked for by the two tables whose types live beside torch, not by ``[neural]``.

    Which is what lets a file carry the neural producer's configuration and still run its
    parametric producers under ``--calibrators`` on an installation without torch.
    """
    hide(monkeypatch, "torch")

    assert load(tmp_path, CONFIG_TOML + NEURAL_TOML).neural is not None


@pytest.mark.parametrize(
    ("table", "name"), [(NETWORK_TOML, r"neural\.network"), (TORCH_FIT_TOML, r"neural\.fit")]
)
def test_a_torch_table_without_the_extra_is_refused_with_the_remedy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, table: str, name: str
) -> None:
    hide(monkeypatch, "torch")

    with pytest.raises(ConfigError, match=rf"{name}: .*uv sync --extra neural"):
        load(tmp_path, CONFIG_TOML + NEURAL_TOML + table)


@needs_torch
def test_the_network_table_fills_the_learner_own_type(tmp_path: Path) -> None:
    torch_learner = pytest.importorskip("volengine.neural_surface.adapters.torch_learner")

    neural = load(tmp_path, CONFIG_TOML + NEURAL_TOML + NETWORK_TOML).neural

    assert neural is not None
    assert neural.network is not None
    assert neural.network == torch_learner.NetworkSpec(
        hidden=(8, 8, 8),
        activation=torch_learner.Activation.SOFTPLUS,
        k_scale=0.25,
        tenor_scale=2.0,
    )
    assert isinstance(neural.network.activation, torch_learner.Activation)


@needs_torch
def test_the_fit_table_fills_the_learner_own_type(tmp_path: Path) -> None:
    torch_learner = pytest.importorskip("volengine.neural_surface.adapters.torch_learner")

    neural = load(tmp_path, CONFIG_TOML + NEURAL_TOML + TORCH_FIT_TOML).neural

    assert neural is not None
    assert neural.fit == torch_learner.TorchFitSettings(
        cold_steps=123,
        warm_steps=7,
        cold_learning_rate=0.02,
        warm_learning_rate=0.002,
        butterfly_penalty=50.0,
        calendar_penalty=60.0,
        init_seed=9,
    )


@needs_torch
def test_a_missing_key_in_the_fit_table_names_the_table(tmp_path: Path) -> None:
    """Complete when present, like every adapter table."""
    with pytest.raises(ConfigError, match=r"neural\.fit: the key 'warm_steps'"):
        load(tmp_path, without(CONFIG_TOML + NEURAL_TOML + TORCH_FIT_TOML, "warm_steps"))


@needs_torch
def test_an_activation_the_learner_does_not_offer_lists_the_ones_it_does(tmp_path: Path) -> None:
    text = replacing(CONFIG_TOML + NEURAL_TOML + NETWORK_TOML, "activation", 'activation = "relu"')

    with pytest.raises(ConfigError, match="tanh, softplus"):
        load(tmp_path, text)


def test_a_missing_neural_sub_table_names_it(tmp_path: Path) -> None:
    text = (CONFIG_TOML + NEURAL_TOML).replace("[neural.gate]\nbutterfly = 0.001\n", "")
    text = text.replace("calendar = 0.0002\n", "")

    with pytest.raises(ConfigError, match="neural: the key 'gate' is missing"):
        load(tmp_path, text)


def test_a_missing_seed_is_refused(tmp_path: Path) -> None:
    """No default: the seed is what makes a replay of the draw reproducible (ADR-004)."""
    with pytest.raises(ConfigError, match="neural: the key 'seed'"):
        load(tmp_path, without(CONFIG_TOML + NEURAL_TOML, "seed"))


def test_a_gate_tolerance_the_use_case_refuses_is_blamed_on_its_table(tmp_path: Path) -> None:
    text = replacing(CONFIG_TOML + NEURAL_TOML, "butterfly", "butterfly = -1.0")

    with pytest.raises(ConfigError, match=r"neural\.gate: The butterfly tolerance"):
        load(tmp_path, text)


def test_a_buffer_the_domain_refuses_is_blamed_on_its_table(tmp_path: Path) -> None:
    text = replacing(CONFIG_TOML + NEURAL_TOML, "capacity_per_cell", "capacity_per_cell = 0")

    with pytest.raises(ConfigError, match=r"neural\.buffer: The capacity per cell"):
        load(tmp_path, text)


def test_a_mesh_too_coarse_to_judge_is_blamed_on_its_table(tmp_path: Path) -> None:
    """Two nodes span no interior point; the refusal is ``ArbitrageMesh``'s, named by the table.

    ``n_nodes`` appears in ``[calibration.grid]`` too, so the replacement is spelled on the mesh's
    own neighbourhood rather than by key.
    """
    text = (CONFIG_TOML + NEURAL_TOML).replace(
        "k_max = 0.5\nn_nodes = 11", "k_max = 0.5\nn_nodes = 2"
    )

    with pytest.raises(ConfigError, match=r"neural\.mesh: The moneyness mesh"):
        load(tmp_path, text)


def test_a_single_node_mesh_is_refused_before_it_divides_by_zero(tmp_path: Path) -> None:
    text = (CONFIG_TOML + NEURAL_TOML).replace(
        "k_max = 0.5\nn_nodes = 11", "k_max = 0.5\nn_nodes = 1"
    )

    with pytest.raises(ConfigError, match=r"neural\.mesh: 'n_nodes' must be at least 2"):
        load(tmp_path, text)


def test_a_boolean_among_the_mesh_tenors_is_refused(tmp_path: Path) -> None:
    """``isinstance(True, int)`` is true; a ``true`` in an array must not become a tenor of 1.0."""
    text = replacing(CONFIG_TOML + NEURAL_TOML, "tenors", "tenors = [0.08, true]")

    with pytest.raises(ConfigError, match=r"neural\.mesh: 'tenors' must be a number"):
        load(tmp_path, text)


@needs_torch
def test_a_boolean_among_the_hidden_widths_is_refused(tmp_path: Path) -> None:
    text = replacing(CONFIG_TOML + NEURAL_TOML + NETWORK_TOML, "hidden", "hidden = [8, true]")

    with pytest.raises(ConfigError, match=r"neural\.network: 'hidden' must be an array of int"):
        load(tmp_path, text)
