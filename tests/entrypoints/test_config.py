"""Reading a TOML file: what it builds, and what it refuses to build.

Every rejection test starts from one valid file and breaks a single line, so what the test asserts
is exactly what it changed. The point of most of them is not that a bad number is refused --
the domain types already do that, and their own tests already say so -- but that the refusal
arrives as a ``ConfigError`` naming the table, rather than as a ``ValueError`` out of a module the
operator has no reason to be reading.
"""

from __future__ import annotations

from datetime import UTC, datetime, time, timedelta
from pathlib import Path

import pytest

from tests.entrypoints.builders import CONFIG_TOML, replacing, without, write_config
from volengine.entrypoints.config import AppConfig, ConfigError, load_config
from volengine.market_data.domain.market_conventions import DayCount, ForwardMethod, Numeraire
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
    assert settings.config.true_params[timedelta(days=90)].sigma == 0.25


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
