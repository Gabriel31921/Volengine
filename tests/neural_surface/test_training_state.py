"""The memory a pure ``SurfaceLearner`` refuses to keep.

The distinction that carries this file is between the surface *trained* and the surface
*published*. They are equal in the ordinary case and differ in exactly the situation the hard gate
exists for, so a test suite that only ever exercised the happy path would never notice them being
merged.
"""

from __future__ import annotations

from datetime import timedelta

from tests.neural_surface.builders import NOW, FlatVolSurface
from tests.risk.builders import make_calibrated_surface
from volengine.neural_surface.application.training_state import TrainingState

MINUTE = timedelta(minutes=1)


# --- the surface being trained


def test_a_fresh_state_has_no_surface() -> None:
    """The cold start the port documents an implementation must handle."""
    assert TrainingState().surface is None


def test_the_trained_surface_is_what_the_next_step_continues_from() -> None:
    state = TrainingState()
    surface = FlatVolSurface(version=3)

    state.trained(surface, NOW)

    assert state.surface is surface


def test_a_surface_that_failed_the_gate_is_still_kept() -> None:
    """Throwing it away would answer a refusal with a cold restart, the most expensive move."""
    state = TrainingState()
    refused = FlatVolSurface(version=9)

    state.trained(refused, NOW)

    assert state.surface is refused
    assert state.last_published is None


# --- the surface that was published


def test_a_fresh_state_has_nothing_to_republish() -> None:
    assert TrainingState().last_published is None


def test_the_published_surface_is_the_one_republished_after_a_refusal() -> None:
    state = TrainingState()
    published = make_calibrated_surface()
    state.remember(published)

    state.trained(FlatVolSurface(version=2), NOW)

    assert state.last_published is published


def test_training_past_a_publication_does_not_change_what_would_be_republished() -> None:
    """The two diverge after a refusal, and it is the validated DTO that has to go out again."""
    state = TrainingState()
    state.remember(make_calibrated_surface(surface_id="published"))

    state.trained(FlatVolSurface(version=4), NOW)

    assert state.last_published is not None
    assert state.last_published.surface_id == "published"


# --- the restart schedule


def test_the_first_cycle_never_restarts() -> None:
    """Nothing has been trained yet, so there is nothing to restart from."""
    assert TrainingState().is_restart_due(NOW, 600.0) is False


def test_a_restart_is_not_due_before_the_interval_has_passed() -> None:
    state = TrainingState()
    state.trained(FlatVolSurface(), NOW)

    assert state.is_restart_due(NOW + MINUTE, 600.0) is False


def test_a_restart_is_due_once_the_interval_has_passed() -> None:
    state = TrainingState()
    state.trained(FlatVolSurface(), NOW)

    assert state.is_restart_due(NOW + 11 * MINUTE, 600.0) is True


def test_the_interval_runs_from_the_first_step_rather_than_from_an_absent_history() -> None:
    """Otherwise the first snapshot of every session retrains over one snapshot's worth of data."""
    state = TrainingState()
    state.trained(FlatVolSurface(), NOW + 10 * MINUTE)

    assert state.is_restart_due(NOW + 11 * MINUTE, 600.0) is False


def test_a_configured_restart_of_none_never_fires() -> None:
    state = TrainingState()
    state.trained(FlatVolSurface(), NOW)

    assert state.is_restart_due(NOW + 10_000 * MINUTE, None) is False


def test_asking_whether_a_restart_is_due_changes_nothing() -> None:
    """A predicate that moved the clock would make a second look disagree with the first."""
    state = TrainingState()
    state.trained(FlatVolSurface(), NOW)

    first = state.is_restart_due(NOW + 11 * MINUTE, 600.0)
    second = state.is_restart_due(NOW + 11 * MINUTE, 600.0)

    assert first is second is True


def test_a_restart_resets_the_interval() -> None:
    state = TrainingState()
    state.trained(FlatVolSurface(), NOW)
    state.restarted(NOW + 11 * MINUTE)

    assert state.is_restart_due(NOW + 12 * MINUTE, 600.0) is False


def test_a_restart_that_was_decided_on_but_never_happened_leaves_the_clock_alone() -> None:
    """Asking and doing are two events, and only the second one may move the reference."""
    state = TrainingState()
    state.trained(FlatVolSurface(), NOW)

    assert state.is_restart_due(NOW + 11 * MINUTE, 600.0) is True
    assert state.is_restart_due(NOW + 12 * MINUTE, 600.0) is True


def test_a_later_step_does_not_push_the_restart_clock_forward() -> None:
    """Only the first step sets it; training every second would otherwise defer restarts forever."""
    state = TrainingState()
    state.trained(FlatVolSurface(), NOW)
    state.trained(FlatVolSurface(version=2), NOW + 5 * MINUTE)

    assert state.is_restart_due(NOW + 11 * MINUTE, 600.0) is True
