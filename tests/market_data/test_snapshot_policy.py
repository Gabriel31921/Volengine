from __future__ import annotations

from dataclasses import fields, replace
from datetime import UTC, datetime, timedelta

import pytest

from volengine.market_data.domain.quote_chain import ChainStats
from volengine.market_data.domain.snapshot_policy import (
    DegradationReason,
    QualityAssessment,
    SnapshotPolicy,
    SnapshotPolicyConfig,
)

NOW = datetime(2026, 7, 27, 8, 0, tzinfo=UTC)
NAIVE = datetime(2026, 7, 27, 8, 0)


# --- builders


def make_config() -> SnapshotPolicyConfig:
    return SnapshotPolicyConfig(
        cadence_seconds=1.0,
        material_move_threshold=0.002,
        min_coverage_ratio=0.80,
        max_quiet_seconds=30.0,
    )


def make_stats(
    n_instruments_known: int = 10,
    n_quotes_total: int = 10,
    n_quotes_admissible: int = 10,
    max_age_seconds: float = 0.5,
    coverage_ratio: float = 1.0,
) -> ChainStats:
    """A fully quoted, fully admissible chain. One knob per test."""
    return ChainStats(
        n_instruments_known=n_instruments_known,
        n_quotes_total=n_quotes_total,
        n_quotes_admissible=n_quotes_admissible,
        max_age_seconds=max_age_seconds,
        coverage_ratio=coverage_ratio,
    )


def decide(
    config: SnapshotPolicyConfig | None = None,
    stats: ChainStats | None = None,
    max_relative_move: float = 0.05,
    last_emit: datetime | None = NOW - timedelta(seconds=2),
    now: datetime = NOW,
) -> bool:
    """should_emit with every knob at a value that emits, so a test moves exactly one."""
    policy = SnapshotPolicy(config if config is not None else make_config())
    return policy.should_emit(
        stats=stats if stats is not None else make_stats(),
        max_relative_move=max_relative_move,
        last_emit=last_emit,
        now=now,
    )


def assess(
    config: SnapshotPolicyConfig | None = None,
    stats: ChainStats | None = None,
) -> QualityAssessment:
    policy = SnapshotPolicy(config if config is not None else make_config())
    return policy.assess_quality(stats if stats is not None else make_stats())


# --- SnapshotPolicyConfig


@pytest.mark.parametrize("bad", [0.0, -1.0, float("nan"), float("inf")])
def test_config_rejects_a_non_positive_cadence(bad: float) -> None:
    with pytest.raises(ValueError, match="cadence_seconds"):
        replace(make_config(), cadence_seconds=bad)


@pytest.mark.parametrize("bad", [-0.001, float("nan"), float("inf")])
def test_config_rejects_a_negative_move_threshold(bad: float) -> None:
    with pytest.raises(ValueError, match="material_move_threshold"):
        replace(make_config(), material_move_threshold=bad)


def test_config_accepts_a_zero_move_threshold() -> None:
    """Zero is how a deployment switches the movement filter off, not an invalid setting."""
    assert replace(make_config(), material_move_threshold=0.0).material_move_threshold == 0.0


@pytest.mark.parametrize("bad", [-0.1, 1.1, float("nan")])
def test_config_rejects_a_coverage_ratio_outside_the_unit_interval(bad: float) -> None:
    with pytest.raises(ValueError, match="min_coverage_ratio"):
        replace(make_config(), min_coverage_ratio=bad)


@pytest.mark.parametrize("edge", [0.0, 1.0])
def test_config_accepts_the_coverage_bounds(edge: float) -> None:
    assert replace(make_config(), min_coverage_ratio=edge).min_coverage_ratio == edge


@pytest.mark.parametrize("bad", [0.0, -1.0, float("nan"), float("inf")])
def test_config_rejects_a_non_positive_heartbeat(bad: float) -> None:
    with pytest.raises(ValueError, match="max_quiet_seconds"):
        replace(make_config(), max_quiet_seconds=bad)


def test_config_rejects_a_heartbeat_shorter_than_the_cadence() -> None:
    """It would be a second cadence contradicting the first: the heartbeat can never fire
    before the cadence permits an emission at all.
    """
    with pytest.raises(ValueError, match="below the cadence"):
        replace(make_config(), max_quiet_seconds=0.5)


def test_config_accepts_a_heartbeat_equal_to_the_cadence() -> None:
    config = replace(make_config(), cadence_seconds=30.0, max_quiet_seconds=30.0)
    assert config.max_quiet_seconds == config.cadence_seconds


def test_config_accepts_no_heartbeat_at_all() -> None:
    """None is how a deployment opts out of the heartbeat."""
    assert replace(make_config(), max_quiet_seconds=None).max_quiet_seconds is None


# --- should_emit: preconditions


def test_should_emit_rejects_a_naive_now() -> None:
    with pytest.raises(ValueError, match="timezone-aware"):
        decide(now=NAIVE)


def test_should_emit_rejects_a_naive_last_emit() -> None:
    """Both stamps get subtracted from each other; mixing the two kinds raises TypeError from
    inside the arithmetic, with nothing in the message naming the field.
    """
    with pytest.raises(ValueError, match="last_emit"):
        decide(last_emit=NAIVE)


# --- should_emit: one branch at a time


def test_an_empty_chain_is_not_worth_publishing() -> None:
    """A snapshot with no quotes in it informs nobody, however much time has passed."""
    assert decide(stats=make_stats(coverage_ratio=0.0)) is False


def test_the_first_snapshot_always_goes_out() -> None:
    """There is no baseline to measure a move against, so the movement test cannot apply."""
    assert decide(last_emit=None, max_relative_move=0.0) is True


def test_nothing_is_published_before_the_cadence_elapses() -> None:
    """The cadence is a ceiling on frequency and outranks everything below it."""
    assert decide(last_emit=NOW - timedelta(milliseconds=200)) is False


def test_a_material_move_after_the_cadence_publishes() -> None:
    assert decide(max_relative_move=0.05) is True


def test_a_move_below_the_threshold_does_not_publish() -> None:
    """Republishing a motionless chain burns a whole calibration to reproduce the same number."""
    assert decide(max_relative_move=0.001) is False


def test_a_move_exactly_at_the_threshold_publishes() -> None:
    """The comparison is `>=`, which is what makes a threshold of 0.0 mean 'filter off'."""
    assert decide(max_relative_move=0.002) is True


def test_a_zero_threshold_disables_the_movement_filter() -> None:
    """With the filter off the policy degenerates to a pure cadence timer, which is a
    legitimate deployment choice.
    """
    config = replace(make_config(), material_move_threshold=0.0)
    assert decide(config=config, max_relative_move=0.0) is True


# --- should_emit: the heartbeat


def test_a_motionless_market_still_publishes_once_the_heartbeat_expires() -> None:
    """Without this, a calm market emits nothing at all and downstream cannot tell that from a
    dead process: Risk watches staleness grow, goes DEGRADED then REJECT, and reports 'no valid
    surface' because the market was quiet.
    """
    assert decide(max_relative_move=0.0, last_emit=NOW - timedelta(seconds=31)) is True


def test_without_a_heartbeat_a_motionless_market_stays_silent() -> None:
    """None really disables it -- that is the whole difference the field makes."""
    config = replace(make_config(), max_quiet_seconds=None)
    assert decide(config=config, max_relative_move=0.0, last_emit=NOW - timedelta(hours=3)) is False


def test_the_heartbeat_never_overrides_the_cadence() -> None:
    """An expired heartbeat inside the cadence window must still not publish; the config
    forbids that combination, and the ordering of the checks enforces it.
    """
    config = replace(make_config(), cadence_seconds=60.0, max_quiet_seconds=60.0)
    inside_the_window = NOW - timedelta(seconds=5)
    assert decide(config=config, max_relative_move=0.0, last_emit=inside_the_window) is False


def test_a_heartbeat_expiring_exactly_now_publishes() -> None:
    assert decide(max_relative_move=0.0, last_emit=NOW - timedelta(seconds=30)) is True


# --- should_emit: the clock going backwards


def test_a_clock_running_backwards_does_not_publish() -> None:
    """A rewound ManualClock in a replay gives a negative elapsed, which is below any positive
    cadence. Time has not passed, so nothing is due -- and the heartbeat must not fire either.
    """
    assert decide(max_relative_move=0.5, last_emit=NOW + timedelta(seconds=90)) is False


# --- should_emit: quality does not gate emission


def test_a_degraded_chain_is_still_published() -> None:
    """Architectural, not behavioural. A degraded snapshot goes out *marked* -- that is the
    entire reason `degraded` exists. Wiring min_coverage_ratio into should_emit would turn an
    observable degradation into a silence, and silence is the one symptom downstream cannot
    distinguish from a dead process.
    """
    barely_usable = make_stats(n_quotes_admissible=1, coverage_ratio=1.0)
    assert assess(stats=barely_usable).degraded is True
    assert decide(stats=barely_usable) is True


# --- assess_quality


def test_a_fully_admissible_chain_is_not_degraded() -> None:
    assert assess().reasons == ()


def test_a_thinly_admissible_chain_is_degraded() -> None:
    assert assess(stats=make_stats(n_quotes_admissible=3)).reasons == (
        DegradationReason.LOW_COVERAGE,
    )


def test_coverage_exactly_at_the_minimum_is_not_degraded() -> None:
    """The comparison is `ratio < minimum`: the configured floor is acceptable, not forbidden."""
    assert assess(stats=make_stats(n_quotes_admissible=8)).degraded is False


def test_quality_is_judged_on_the_admissible_ratio_not_on_coverage() -> None:
    """Pins Design 4.2: degradation is 'fresh quotes with a reasonable spread', not 'a mid
    arrived'. Every instrument here is quoted -- coverage_ratio is a perfect 1.0 -- and only two
    of the ten quotes survive their flags. Judged by coverage this chain looks healthy; judged
    by what a calibrator can actually use, it is nearly empty.
    """
    every_quote_flagged = make_stats(n_quotes_admissible=2, coverage_ratio=1.0)
    assert every_quote_flagged.coverage_ratio >= make_config().min_coverage_ratio
    assert assess(stats=every_quote_flagged).degraded is True


def test_an_unknown_chain_is_degraded_rather_than_dividing_by_zero() -> None:
    """No known instruments means an undefined ratio, and an unconfigured market that reported
    itself healthy would be the most misleading answer available.
    """
    empty = make_stats(
        n_instruments_known=0, n_quotes_total=0, n_quotes_admissible=0, coverage_ratio=0.0
    )
    assert assess(stats=empty).reasons == (DegradationReason.LOW_COVERAGE,)


def test_an_empty_chain_is_degraded_even_with_the_rule_relaxed_to_zero() -> None:
    """A ratio of nothing over nothing is not 'above any threshold', it is unanswerable."""
    config = replace(make_config(), min_coverage_ratio=0.0)
    empty = make_stats(
        n_instruments_known=0, n_quotes_total=0, n_quotes_admissible=0, coverage_ratio=0.0
    )
    assert assess(config=config, stats=empty).degraded is True


# --- QualityAssessment


def test_degraded_is_derived_from_the_reasons_and_cannot_contradict_them() -> None:
    """Architectural. As a stored field, `degraded=False` beside two reasons is an object that
    constructs cleanly and lies to every consumer that trusts the flag.
    """
    assert "degraded" not in {field.name for field in fields(QualityAssessment)}
    assert QualityAssessment(reasons=()).degraded is False
    assert QualityAssessment(reasons=(DegradationReason.LOW_COVERAGE,)).degraded is True


def test_a_degradation_reason_carries_its_wire_value() -> None:
    """The string is what the ACL maps onto the published QualityBlock; a rename must not
    silently change it.
    """
    assert DegradationReason.LOW_COVERAGE == "LOW_COVERAGE"
