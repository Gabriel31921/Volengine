from __future__ import annotations

from dataclasses import FrozenInstanceError, replace
from datetime import datetime, timedelta

import pytest

from tests.risk.builders import NAIVE, NOW, make_policy
from volengine.risk.domain.freshness_policy import FreshnessDecision, FreshnessPolicy

# make_policy warns at five seconds and rejects past thirty. Every expected verdict below is read
# off those two numbers, so they are named once here rather than repeated as literals.
WARN = 5.0
REJECT = 30.0

# A millisecond, which is a thousand microseconds and therefore survives timedelta's resolution
# intact. That is not incidental: an epsilon below a microsecond would round to zero, "just below
# the boundary" would silently become "exactly on the boundary", and half the tests below would be
# asserting something other than what they say. See the guard test at the end of this section.
EPS = 0.001


def at_age(seconds: float) -> datetime:
    """The instant at which a surface stamped ``NOW`` is ``seconds`` old."""
    return NOW + timedelta(seconds=seconds)


# --- the three bands, both boundaries pinned from both sides


def test_a_fresh_surface_is_normal() -> None:
    assert make_policy().evaluate(NOW, at_age(1.0)) is FreshnessDecision.NORMAL


def test_an_age_just_below_the_warning_threshold_is_still_normal() -> None:
    assert make_policy().evaluate(NOW, at_age(WARN - EPS)) is FreshnessDecision.NORMAL


def test_an_age_exactly_at_the_warning_threshold_is_degraded() -> None:
    """The bands are half-open at the bottom: reaching the warning is already worth the caveat."""
    assert make_policy().evaluate(NOW, at_age(WARN)) is FreshnessDecision.DEGRADED


def test_an_age_between_the_thresholds_is_degraded() -> None:
    assert make_policy().evaluate(NOW, at_age(15.0)) is FreshnessDecision.DEGRADED


def test_an_age_exactly_at_the_rejection_threshold_is_still_degraded() -> None:
    """Closed at the top: "we stop at thirty" means the stop has not happened at thirty."""
    assert make_policy().evaluate(NOW, at_age(REJECT)) is FreshnessDecision.DEGRADED


def test_an_age_just_above_the_rejection_threshold_is_rejected() -> None:
    assert make_policy().evaluate(NOW, at_age(REJECT + EPS)) is FreshnessDecision.REJECT


def test_a_long_dead_feed_is_rejected() -> None:
    assert make_policy().evaluate(NOW, at_age(3_600.0)) is FreshnessDecision.REJECT


def test_the_boundary_epsilon_actually_moves_the_instant() -> None:
    """Vacuity guard for every boundary test above.

    ``timedelta`` stores microseconds, so an epsilon finer than one would round away and the
    "just below" and "just above" instants would collapse onto the boundary itself -- at which
    point those tests would be re-asserting the boundary case under a misleading name. This
    pins that the step is real and lands on the side it claims.
    """
    assert at_age(WARN - EPS) != at_age(WARN)
    assert at_age(REJECT + EPS) != at_age(REJECT)
    assert (at_age(WARN - EPS) - NOW).total_seconds() < WARN
    assert (at_age(REJECT + EPS) - NOW).total_seconds() > REJECT


# --- a clock running ahead of us


def test_a_surface_stamped_in_the_future_is_normal() -> None:
    """A venue clock a second fast is routine, and ts_exchange is unclamped upstream by design.

    Refusing it would take the whole risk report down for as long as the skew lasted, which is
    the same trade ``ReplayBuffer.prune`` refuses to make in ``neural_surface/``.
    """
    assert make_policy().evaluate(NOW, at_age(-1.0)) is FreshnessDecision.NORMAL


def test_an_age_of_exactly_zero_is_normal() -> None:
    assert make_policy().evaluate(NOW, NOW) is FreshnessDecision.NORMAL


def test_the_same_policy_still_degrades_and_rejects() -> None:
    """Vacuity guard for the future-stamp test: NORMAL is a verdict, not this policy's only one.

    A policy that returned NORMAL unconditionally -- an inverted comparison, a NaN threshold that
    slipped past the constructor -- would pass every test above that asserts NORMAL and nothing
    would say which of them was proving anything. One policy object producing all three verdicts
    is what makes the future-stamped case a decision rather than a default.
    """
    policy = make_policy()
    verdicts = {policy.evaluate(NOW, at_age(age)) for age in (-1.0, 15.0, 100.0)}

    assert verdicts == {
        FreshnessDecision.NORMAL,
        FreshnessDecision.DEGRADED,
        FreshnessDecision.REJECT,
    }


# --- naive instants


def test_evaluate_rejects_a_naive_snapshot() -> None:
    with pytest.raises(ValueError, match="ts_snapshot must be timezone-aware"):
        make_policy().evaluate(NAIVE, at_age(1.0))


def test_evaluate_rejects_a_naive_now() -> None:
    with pytest.raises(ValueError, match="now must be timezone-aware"):
        make_policy().evaluate(NOW, NAIVE)


# --- the thresholds themselves


def test_thresholds_that_are_equal_are_rejected() -> None:
    """Equal thresholds would make DEGRADED one instant wide, so the band would never be seen."""
    with pytest.raises(ValueError, match="strictly above the warning threshold"):
        replace(make_policy(), reject_seconds=WARN)


def test_a_rejection_threshold_below_the_warning_one_is_rejected() -> None:
    with pytest.raises(ValueError, match="strictly above the warning threshold"):
        replace(make_policy(), reject_seconds=1.0)


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_a_non_finite_warning_threshold_is_rejected(bad: float) -> None:
    """NaN first of all: `nan <= 0` is False, so a bare ordering guard would let it through."""
    with pytest.raises(ValueError, match="warn_seconds must be positive and finite"):
        replace(make_policy(), warn_seconds=bad)


@pytest.mark.parametrize("bad", [0.0, -1.0])
def test_a_non_positive_warning_threshold_is_rejected(bad: float) -> None:
    """Zero would degrade every report ever produced, which teaches readers to ignore the label."""
    with pytest.raises(ValueError, match="warn_seconds must be positive and finite"):
        replace(make_policy(), warn_seconds=bad)


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_a_non_finite_rejection_threshold_is_rejected(bad: float) -> None:
    """An infinite rejection threshold is a policy that never rejects -- the one it must not be."""
    with pytest.raises(ValueError, match="reject_seconds must be finite"):
        replace(make_policy(), reject_seconds=bad)


def test_a_nan_warning_threshold_would_otherwise_degrade_a_fresh_surface() -> None:
    """Vacuity guard for the NaN rejections: prove the poisoned value really would have hidden.

    Every comparison against NaN is False, so a NaN ``warn_seconds`` makes the NORMAL band
    unreachable: a one-second-old surface falls through to DEGRADED and stays there for the life
    of the process, with nothing raised anywhere to say why the healthy verdict disappeared. That
    is what the constructor's finiteness check buys, and without this test the rejections above
    assert that a guard fires while nothing asserts that its absence would cost anything.

    The policy is built around ``__post_init__`` on purpose -- the guard being tested is exactly
    the one that makes this object otherwise unconstructible.
    """
    unchecked = FreshnessPolicy.__new__(FreshnessPolicy)
    object.__setattr__(unchecked, "warn_seconds", float("nan"))
    object.__setattr__(unchecked, "reject_seconds", REJECT)

    assert unchecked.evaluate(NOW, at_age(1.0)) is FreshnessDecision.DEGRADED
    assert make_policy().evaluate(NOW, at_age(1.0)) is FreshnessDecision.NORMAL


def test_the_policy_is_frozen() -> None:
    """Configuration shared across markets and reports; a mutable one would drift between them."""
    with pytest.raises(FrozenInstanceError):
        make_policy().warn_seconds = 60.0  # type: ignore[misc]


# --- the enum is a wire format


def test_the_decision_is_a_string() -> None:
    """A StrEnum member is a str, which is what lets it reach a CSV column and a metrics tag."""
    assert isinstance(FreshnessDecision.REJECT, str)


def test_the_decision_values_are_pinned() -> None:
    """Explicit values, never auto(): renaming a member must not change what a consumer parses."""
    assert FreshnessDecision.NORMAL.value == "NORMAL"
    assert FreshnessDecision.DEGRADED.value == "DEGRADED"
    assert FreshnessDecision.REJECT.value == "REJECT"


def test_there_are_exactly_three_decisions() -> None:
    """No NO_SURFACE, no REPUBLISHED: the first is REJECT with a message, the second is a cause."""
    assert len(FreshnessDecision) == 3
