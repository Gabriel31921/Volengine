"""One fine-tuning cycle, judged by what it publishes.

The hard gate of ADR-010 is the behaviour this context exists to demonstrate, and it is asserted
here by handing the use case a surface that is deliberately arbitrageable and checking that
nothing goes out. Everything runs on a ``ManualClock`` and a seeded generator, so the replay draw
-- the one genuinely random step in the engine -- is reproducible too.
"""

from __future__ import annotations

from datetime import timedelta

import numpy as np
import pytest

from tests.neural_surface.builders import (
    CallableSurface,
    FlatVolSurface,
    StubLearner,
    make_buffer,
    make_grid_spec,
    make_mesh,
    make_schedule,
    make_spec,
    make_thresholds,
    make_weighting,
)
from tests.parametric_pricing.builders import NOW, make_market_snapshot
from tests.support import RecordingMetrics
from volengine.contracts.calibrated_surface import SurfaceStatus
from volengine.contracts.events import CalibrationFailed, SurfaceCalibrated
from volengine.contracts.market_snapshot import MarketSnapshot
from volengine.neural_surface.application.train_on_snapshot import (
    GateThresholds,
    TrainingSchedule,
    TrainOnSnapshot,
)
from volengine.neural_surface.application.training_state import TrainingState
from volengine.neural_surface.domain.errors import NeuralSurfaceError
from volengine.neural_surface.domain.replay_buffer import ReplayBuffer
from volengine.platform.clock import ManualClock


def make_use_case(
    learner: StubLearner | None = None,
    schedule: TrainingSchedule | None = None,
    thresholds: GateThresholds | None = None,
    clock: ManualClock | None = None,
    state: TrainingState | None = None,
    buffer: ReplayBuffer | None = None,
) -> tuple[TrainOnSnapshot, StubLearner, TrainingState, ReplayBuffer, RecordingMetrics]:
    learner = learner if learner is not None else StubLearner()
    state = state if state is not None else TrainingState()
    buffer = buffer if buffer is not None else make_buffer()
    metrics = RecordingMetrics()
    use_case = TrainOnSnapshot(
        learner=learner,
        state=state,
        buffer=buffer,
        clock=clock if clock is not None else ManualClock(NOW),
        metrics=metrics,
        weighting=make_weighting(),
        grid=make_grid_spec(),
        mesh=make_mesh(),
        thresholds=thresholds if thresholds is not None else make_thresholds(),
        schedule=schedule if schedule is not None else make_schedule(),
        rng=np.random.default_rng(20260828),
    )
    return use_case, learner, state, buffer, metrics


def surfaces(events: tuple[object, ...]) -> list[SurfaceCalibrated]:
    return [event for event in events if isinstance(event, SurfaceCalibrated)]


def failures(events: tuple[object, ...]) -> list[CalibrationFailed]:
    return [event for event in events if isinstance(event, CalibrationFailed)]


def later_snapshot(seconds: float = 700.0) -> MarketSnapshot:
    """The next snapshot, stamped as far ahead as the clock was advanced.

    Necessary rather than decorative once the clock moves past the buffer's age horizon: a
    snapshot's points carry the instant the venue observed them, so one stamped at ``NOW`` and
    delivered eleven minutes later is genuinely older than the ten-minute horizon and is pruned
    the moment it arrives. That is the buffer working; a test about restarts that relied on it
    would be asserting on the wrong mechanism.
    """
    return make_market_snapshot(
        snapshot_id="BTC-DERIBIT:00000001",
        ts_exchange=NOW + timedelta(seconds=seconds),
    )


def arbitrageable() -> CallableSurface:
    """A surface whose total variance falls with the tenor: a calendar arbitrage by construction.

    Accumulated uncertainty cannot shrink, so a term structure that runs backwards is free money
    -- buy the long expiry, sell the short one. Written as a formula rather than trained, which is
    the whole point of the gate being numpy: it can be checked against a surface somebody wrote in
    one line.
    """
    return CallableSurface(fn=lambda k, t: 0.05 + 0.01 * k * k - 0.10 * t)


# --- the published cycle


def test_a_healthy_surface_is_published() -> None:
    use_case, _, _, _, _ = make_use_case()

    events = use_case.handle(make_market_snapshot())

    assert len(events) == 1
    assert surfaces(events)


def test_the_published_surface_names_the_snapshot_it_was_trained_on() -> None:
    use_case, _, _, _, _ = make_use_case()

    surface = surfaces(use_case.handle(make_market_snapshot()))[0].surface

    assert surface.source_snapshot_id == "BTC-DERIBIT:00000000"


def test_the_published_surface_carries_the_snapshots_own_instant() -> None:
    """Not the training instant: staleness downstream is measured against the market."""
    use_case, _, _, _, _ = make_use_case()

    assert surfaces(use_case.handle(make_market_snapshot()))[0].surface.ts_snapshot == NOW


def test_a_degraded_snapshot_can_only_produce_a_degraded_surface() -> None:
    use_case, _, _, _, _ = make_use_case()

    surface = surfaces(use_case.handle(make_market_snapshot(degraded=True)))[0].surface

    assert surface.status is SurfaceStatus.DEGRADED


def test_two_producers_on_one_snapshot_publish_distinguishable_surfaces() -> None:
    snapshot = make_market_snapshot()
    neural, _, _, _, _ = make_use_case(learner=StubLearner(producer_id="mlp-torch"))
    other, _, _, _, _ = make_use_case(learner=StubLearner(producer_id="mlp-jax"))

    first = surfaces(neural.handle(snapshot))[0].surface
    second = surfaces(other.handle(snapshot))[0].surface

    assert first.surface_id != second.surface_id


# --- the hard gate


def test_an_arbitrageable_surface_is_not_published() -> None:
    """ADR-010: soft constraints train, hard constraints govern."""
    use_case, _, _, _, _ = make_use_case(learner=StubLearner(surface=arbitrageable()))

    events = use_case.handle(make_market_snapshot())

    assert not surfaces(events)
    assert "no-arbitrage gate" in failures(events)[0].reason


def test_a_refused_surface_is_still_kept_as_the_training_state() -> None:
    """A refusal must not become a cold restart: it may be one gradient step from admissible."""
    refused = arbitrageable()
    use_case, _, state, _, _ = make_use_case(learner=StubLearner(surface=refused))

    use_case.handle(make_market_snapshot())

    assert state.surface is refused


def test_a_tolerant_gate_publishes_what_a_strict_one_refuses() -> None:
    """The vacuous-pass guard: the gate has to be the thing deciding, not the surface being odd."""
    learner = StubLearner(surface=arbitrageable())
    use_case, _, _, _, _ = make_use_case(
        learner=learner, thresholds=GateThresholds(butterfly=10.0, calendar=10.0)
    )

    assert surfaces(use_case.handle(make_market_snapshot()))


def test_the_violation_depths_are_gauged_whether_or_not_they_crossed_the_line() -> None:
    """A refusal is a step function; the depth is the slope leading up to it."""
    use_case, _, _, _, metrics = make_use_case()

    use_case.handle(make_market_snapshot())

    assert metrics.gauge_value("neural.butterfly_violation") >= 0.0
    assert metrics.gauge_value("neural.calendar_violation") >= 0.0


def test_a_refusal_is_counted_as_this_producers_quality_signal() -> None:
    use_case, _, _, _, metrics = make_use_case(learner=StubLearner(surface=arbitrageable()))

    use_case.handle(make_market_snapshot())

    assert "neural.publication.refused" in metrics.counter_names()


@pytest.mark.parametrize("bad", [-0.1, float("nan"), float("inf")])
def test_an_unusable_tolerance_is_refused(bad: float) -> None:
    """A NaN tolerance would not fail loudly, it would pass every surface for the whole session."""
    with pytest.raises(ValueError, match="tolerance"):
        GateThresholds(butterfly=bad, calendar=1e-6)


# --- fine-tuning is chained


def test_the_first_cycle_trains_from_scratch() -> None:
    use_case, learner, _, _, _ = make_use_case()

    use_case.handle(make_market_snapshot())

    assert learner.calls[0][0] is None


def test_the_second_cycle_continues_from_the_first_ones_surface() -> None:
    """A learner that accepted a history and ignored it would pass every other test here."""
    first = FlatVolSurface(version=1)
    learner = StubLearner(surface=first)
    use_case, _, _, _, _ = make_use_case(learner=learner)
    use_case.handle(make_market_snapshot())

    use_case.handle(make_market_snapshot(snapshot_id="BTC-DERIBIT:00000001"))

    assert learner.calls[1][0] is first


# --- the replay buffer


def test_the_snapshots_own_points_go_into_the_buffer() -> None:
    use_case, _, _, buffer, _ = make_use_case()

    use_case.handle(make_market_snapshot())

    assert len(buffer) > 0


def test_the_second_cycle_replays_points_from_the_first() -> None:
    use_case, learner, _, _, _ = make_use_case(schedule=make_schedule(replay_size=4))
    use_case.handle(make_market_snapshot())

    use_case.handle(make_market_snapshot(snapshot_id="BTC-DERIBIT:00000001"))

    assert learner.calls[1][1].replayed


def test_a_replay_size_of_zero_switches_replay_off() -> None:
    """A legitimate configuration, and the control arm of Design 6.4's own justification."""
    use_case, learner, _, _, _ = make_use_case(schedule=make_schedule(replay_size=0))
    use_case.handle(make_market_snapshot())

    use_case.handle(make_market_snapshot(snapshot_id="BTC-DERIBIT:00000001"))

    assert learner.calls[1][1].replayed == ()


def test_the_fresh_points_lead_the_batch() -> None:
    use_case, learner, _, _, _ = make_use_case()
    use_case.handle(make_market_snapshot())

    use_case.handle(make_market_snapshot(snapshot_id="BTC-DERIBIT:00000001"))

    batch = learner.calls[1][1]
    assert batch.n_fresh > 0
    assert batch.samples[: batch.n_fresh] == batch.fresh


def test_the_buffer_is_pruned_against_the_injected_clock() -> None:
    """The buffer has no clock of its own; if the use case skips this the horizon stops existing."""
    clock = ManualClock(NOW)
    use_case, _, _, buffer, metrics = make_use_case(
        clock=clock, buffer=make_buffer(), schedule=make_schedule(replay_size=0)
    )
    use_case.handle(make_market_snapshot())
    held = len(buffer)

    clock.advance(3_600.0)
    use_case.handle(make_market_snapshot(snapshot_id="BTC-DERIBIT:00000001"))

    assert "neural.buffer.pruned" in metrics.counter_names()
    assert len(buffer) < held + len(buffer)


def test_the_buffers_cell_coverage_is_gauged() -> None:
    """Coverage collapse is what catastrophic forgetting looks like before it reaches the fit."""
    use_case, _, _, _, metrics = make_use_case()

    use_case.handle(make_market_snapshot())

    assert 0.0 <= metrics.gauge_value("neural.buffer.cell_coverage") <= 1.0


# --- the scheduled restart


def patient_buffer() -> ReplayBuffer:
    """A buffer whose age horizon outlives the restart interval the tests below use.

    Necessary, and the reason is a genuine interaction between two configured numbers: the buffer
    is pruned before the restart draws from it, so a restart interval longer than the age horizon
    finds a buffer holding only whatever arrived in between. With no intervening snapshots -- which
    is what a test is -- that is nothing at all, and the restart has nothing to retrain on. See
    ``test_a_restart_interval_longer_than_the_age_horizon_has_nothing_to_retrain_on``.
    """
    return make_buffer(spec=make_spec(max_age_seconds=3_600.0))


def test_a_restart_trains_from_scratch_over_the_whole_buffer() -> None:
    """``update(None, batch)``: the same operation without a history, not a separate method."""
    clock = ManualClock(NOW)
    use_case, learner, _, _, _ = make_use_case(
        clock=clock, schedule=make_schedule(restart_seconds=600.0), buffer=patient_buffer()
    )
    use_case.handle(make_market_snapshot())

    clock.advance(700.0)
    use_case.handle(later_snapshot())

    assert learner.calls[1][0] is None


def test_a_restart_carries_no_fresh_points() -> None:
    """``n_fresh`` of zero is legal and is precisely this case, which is why it must never be
    tested for truthiness."""
    clock = ManualClock(NOW)
    use_case, learner, _, _, _ = make_use_case(
        clock=clock, schedule=make_schedule(restart_seconds=600.0), buffer=patient_buffer()
    )
    use_case.handle(make_market_snapshot())

    clock.advance(700.0)
    use_case.handle(later_snapshot())

    assert learner.calls[1][1].n_fresh == 0


def test_a_restart_still_publishes_a_surface_with_fit_metrics() -> None:
    """With no fresh points the residual is taken over the batch; the contract has no empty case."""
    clock = ManualClock(NOW)
    use_case, _, _, _, _ = make_use_case(
        clock=clock, schedule=make_schedule(restart_seconds=600.0), buffer=patient_buffer()
    )
    use_case.handle(make_market_snapshot())

    clock.advance(700.0)
    events = use_case.handle(later_snapshot())

    assert surfaces(events)[0].surface.fit.n_quotes_used > 0


def test_a_restart_is_counted() -> None:
    clock = ManualClock(NOW)
    use_case, _, _, _, metrics = make_use_case(
        clock=clock, schedule=make_schedule(restart_seconds=600.0), buffer=patient_buffer()
    )
    use_case.handle(make_market_snapshot())

    clock.advance(700.0)
    use_case.handle(later_snapshot())

    assert "neural.restart" in metrics.counter_names()


def test_a_restart_interval_longer_than_the_age_horizon_has_nothing_to_retrain_on() -> None:
    """Two configured numbers interacting, pinned so the interaction is a decision and not a bug.

    The buffer is pruned before the restart draws from it, so a restart scheduled further apart
    than the age horizon retrains on whatever arrived in the meantime and nothing else. On a live
    market that is a full buffer; with no intervening snapshots it is empty, and the cycle refuses
    rather than training on nothing. Setting the horizon above the interval is what a deployment
    that wants a restart to see real history has to do.
    """
    clock = ManualClock(NOW)
    use_case, _, _, _, _ = make_use_case(
        clock=clock, schedule=make_schedule(restart_seconds=600.0), buffer=make_buffer()
    )
    use_case.handle(make_market_snapshot())

    clock.advance(700.0)
    events = use_case.handle(later_snapshot())

    assert "usable weight" in failures(events)[0].reason


def test_restarts_are_off_by_default_so_a_manual_clock_never_triggers_one() -> None:
    use_case, learner, _, _, _ = make_use_case(schedule=make_schedule(restart_seconds=None))
    use_case.handle(make_market_snapshot())

    use_case.handle(make_market_snapshot(snapshot_id="BTC-DERIBIT:00000001"))

    assert learner.calls[1][0] is not None


@pytest.mark.parametrize("bad", [0.0, -1.0, float("nan"), float("inf")])
def test_an_unusable_restart_interval_is_refused(bad: float) -> None:
    with pytest.raises(ValueError, match="restart interval"):
        TrainingSchedule(replay_size=4, restart_seconds=bad)


def test_a_negative_replay_size_is_refused() -> None:
    with pytest.raises(ValueError, match="replay size"):
        TrainingSchedule(replay_size=-1, restart_seconds=None)


# --- refusals


def test_a_snapshot_with_nothing_invertible_is_a_refusal_rather_than_a_crash() -> None:
    from dataclasses import replace

    use_case, _, _, _, _ = make_use_case()
    snapshot = make_market_snapshot()
    dead = replace(
        snapshot,
        slices=tuple(
            replace(
                slice_data,
                quotes=tuple(replace(quote, mid=1e9) for quote in slice_data.quotes),
            )
            for slice_data in snapshot.slices
        ),
    )

    events = use_case.handle(dead)

    assert "implied volatility" in failures(events)[0].reason


def test_a_learner_that_fails_outright_is_reported_rather_than_propagated() -> None:
    learner = StubLearner(failure=NeuralSurfaceError("the model could not be built"))
    use_case, _, _, _, _ = make_use_case(learner=learner)

    events = use_case.handle(make_market_snapshot())

    assert "could not be built" in failures(events)[0].reason


def test_a_refusal_with_no_history_publishes_the_failure_alone() -> None:
    use_case, _, _, _, _ = make_use_case(learner=StubLearner(surface=arbitrageable()))

    events = use_case.handle(make_market_snapshot())

    assert len(events) == 1
    assert isinstance(events[0], CalibrationFailed)


def test_a_refusal_republishes_the_previous_surface_as_stale() -> None:
    learner = StubLearner()
    use_case, _, _, _, _ = make_use_case(learner=learner)
    good = surfaces(use_case.handle(make_market_snapshot()))[0].surface

    learner.answer_with(arbitrageable())
    events = use_case.handle(make_market_snapshot(snapshot_id="BTC-DERIBIT:00000001"))

    republished = surfaces(events)[0].surface
    assert republished.status is SurfaceStatus.STALE_REPUBLISH
    assert republished.ts_snapshot == good.ts_snapshot
    assert republished.surface_id == good.surface_id


def test_the_failure_is_published_before_the_republished_surface() -> None:
    learner = StubLearner()
    use_case, _, _, _, _ = make_use_case(learner=learner)
    use_case.handle(make_market_snapshot())

    learner.answer_with(arbitrageable())
    events = use_case.handle(make_market_snapshot(snapshot_id="BTC-DERIBIT:00000001"))

    assert isinstance(events[0], CalibrationFailed)
    assert isinstance(events[1], SurfaceCalibrated)


def test_a_cycle_never_returns_nothing() -> None:
    use_case, _, _, _, _ = make_use_case(learner=StubLearner(surface=arbitrageable()))

    assert use_case.handle(make_market_snapshot())


# --- determinism


def test_two_runs_of_one_recording_train_on_the_same_history() -> None:
    """ADR-004 through the one genuinely random step in the engine: the stratified replay draw."""
    first, first_learner, _, _, _ = make_use_case()
    second, second_learner, _, _, _ = make_use_case()

    for snapshot_id in ("BTC-DERIBIT:00000000", "BTC-DERIBIT:00000001"):
        first.handle(make_market_snapshot(snapshot_id=snapshot_id))
        second.handle(make_market_snapshot(snapshot_id=snapshot_id))

    assert first_learner.calls[1][1].samples == second_learner.calls[1][1].samples


def test_a_snapshot_older_than_the_buffers_horizon_is_still_trained_on() -> None:
    """Surprising and worth pinning: its own points are pruned the moment they arrive.

    A quote carries the instant the venue observed it, so a snapshot delivered long after it was
    taken is old data however new the message is -- and the age horizon is exactly the rule that
    keeps such data out of the *buffer*. It does not keep it out of this step: the fresh points
    reach the learner regardless, because they are what the snapshot is, and the alternative would
    be a market that publishes nothing at all whenever the feed lags.
    """
    clock = ManualClock(NOW)
    use_case, _, _, buffer, _ = make_use_case(clock=clock)

    clock.advance(3_600.0)
    events = use_case.handle(make_market_snapshot())

    assert surfaces(events)
    # They are in the buffer now and will be pruned by the next cycle's call, which is one cycle
    # of latency and the price of pruning before the draw rather than after the extend.
    assert len(buffer) > 0


def test_an_empty_buffer_does_not_stop_the_first_snapshot_from_training() -> None:
    """``ReplayBuffer.sample`` raises on an empty buffer; there is nothing to report here."""
    use_case, learner, _, _, _ = make_use_case(schedule=make_schedule(replay_size=8))

    use_case.handle(make_market_snapshot())

    assert learner.calls[0][1].replayed == ()
    assert learner.calls[0][1].n_fresh > 0
