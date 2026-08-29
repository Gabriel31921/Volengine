"""Reading a TOML file: what it builds, and what it refuses to build.

Every rejection test starts from one valid file and breaks a single line, so what the test asserts
is exactly what it changed. The point of most of them is not that a bad number is refused --
the domain types already do that, and their own tests already say so -- but that the refusal
arrives as a ``ConfigError`` naming the table, rather than as a ``ValueError`` out of a module the
operator has no reason to be reading.
"""

from __future__ import annotations

from datetime import UTC, datetime, time
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
