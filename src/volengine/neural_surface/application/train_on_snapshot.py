"""One fine-tuning cycle: a snapshot in, the events it caused out.

The neural twin of ``parametric_pricing``'s ``CalibrateOnSnapshot``, and the resemblance is the
design rather than a coincidence. Both are synchronous handlers, both return a tuple of events,
both refuse a result they cannot vouch for and republish the last good surface behind the refusal.
Design 5.7 and 6.5 compare the two producers, and a comparison in which one of them reacted to
failure differently would be measuring the plumbing.

**Everything stateful about continuous learning lives here**, which is what lets
``SurfaceLearner.update`` be a pure function: the surface trained so far, the replay buffer, and
the instant of the last scheduled restart. The port's docstring names this module as their home,
and the placement is the trick -- the statefulness is real, it is simply kept outside the thing
being compared and tested.

**The hard gate of ADR-010 is applied here and nowhere else.** ``update`` hands back surfaces it
suspects of arbitrage on purpose, because a learner that refused would take the decision away from
the layer that owns it and leave the gate with nothing to measure. Soft constraints train; hard
constraints govern; this file is where the second kind lives.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from volengine.contracts.calibrated_surface import SurfaceStatus
from volengine.contracts.events import Event
from volengine.contracts.market_snapshot import MarketSnapshot
from volengine.neural_surface.application.acl import (
    Weighting,
    as_stale_republish,
    fit_metrics,
    to_calibrated_surface,
    to_calibration_failed,
    to_surface_calibrated,
    to_training_batch,
    to_training_samples,
)
from volengine.neural_surface.application.grid_spec import GridSpec
from volengine.neural_surface.application.training_state import TrainingState
from volengine.neural_surface.domain.errors import EmptyBufferError, NeuralSurfaceError
from volengine.neural_surface.domain.invariants import ArbitrageMesh, check_surface
from volengine.neural_surface.domain.ports import Clock, MetricsSink, SurfaceLearner
from volengine.neural_surface.domain.replay_buffer import ReplayBuffer
from volengine.neural_surface.domain.training_batch import TrainingBatch, TrainingSample

MILLISECONDS_PER_SECOND = 1000.0


@dataclass(frozen=True, slots=True)
class GateThresholds:
    """How much arbitrage a surface may show and still be published. ADR-010, as configuration.

    Two numbers, and neither of them is the rule: ``ArbitrageReport.exceeds`` is the rule and it
    lives in the domain, where a gradient cannot reach it and a config edit cannot switch it off.
    What is configurable is where the line sits, because that is a policy about how much the engine
    is willing to be blind to on a particular market.
    """

    butterfly: float
    """Largest butterfly depth still publishable. Non-negative and finite.

    ``0.0`` is meaningful and is the strictest setting: no breach at all tolerated. It works
    because the comparison is strict -- a clean report holds exactly ``0.0`` and passes.
    """

    calendar: float
    """Largest calendar depth still publishable, in total-variance units. Same terms."""

    def __post_init__(self) -> None:
        # Finiteness first, and the NaN case is the one that matters: every `>` against a NaN is
        # `False`, so a misconfigured gate would not fail, it would silently publish every surface
        # for the rest of the session -- which is the exact outcome the gate exists to prevent.
        for name, value in (("butterfly", self.butterfly), ("calendar", self.calendar)):
            if not math.isfinite(value) or value < 0:
                raise ValueError(
                    f"The {name} tolerance must be non-negative and finite, got {value}"
                )


@dataclass(frozen=True, slots=True)
class TrainingSchedule:
    """How much history each step replays, and how often the model is retrained from scratch.

    Both are configuration (ADR-012) and both are statements about a market rather than about this
    code: how fast a surface drifts decides how much history helps, and how badly continuous
    fine-tuning accumulates error decides how often it is worth throwing away.
    """

    replay_size: int
    """How many buffered points join each step, drawn stratified across the cells. Non-negative.

    Zero is legal and turns off replay entirely, which is worth being able to configure precisely
    because it is the experiment: Design 6.4 justifies replay by catastrophic forgetting, and a
    run with it switched off is the measurement that shows whether the justification holds on a
    real chain. It is also the honest setting for a session so short the buffer never fills.
    """

    restart_seconds: float | None
    """How long between scheduled retrains from scratch, or ``None`` to never restart.

    The restart of Design 6.4: ``update(None, batch)`` over the whole buffer, with no history at
    all. It is not a different operation and it has no method of its own -- which is exactly why
    it is a number here rather than a branch in the port.

    ``None`` disables it, and that is not the same as a very large value: a session run under a
    ``ManualClock`` that never advances would otherwise be one restart away from firing on an
    arbitrary tick, and a test asserting on continuous fine-tuning should be able to say "no
    restarts" outright.
    """

    def __post_init__(self) -> None:
        if self.replay_size < 0:
            raise ValueError(f"The replay size cannot be negative, got {self.replay_size}")
        if self.restart_seconds is not None and (
            not math.isfinite(self.restart_seconds) or self.restart_seconds <= 0
        ):
            raise ValueError(
                f"The restart interval must be positive and finite, got {self.restart_seconds}"
            )


class TrainOnSnapshot:
    """Fine-tune on one snapshot and say what happened, in the published language.

    One instance per market. Not thread-safe: the buffer and the held surface are mutable state
    owned by one asyncio task, and the training itself runs on a named thread pool (ADR-005) that
    is handed an immutable batch and hands back an immutable surface.
    """

    def __init__(
        self,
        learner: SurfaceLearner,
        state: TrainingState,
        buffer: ReplayBuffer,
        clock: Clock,
        metrics: MetricsSink,
        weighting: Weighting,
        grid: GridSpec,
        mesh: ArbitrageMesh,
        thresholds: GateThresholds,
        schedule: TrainingSchedule,
        rng: np.random.Generator,
    ) -> None:
        """Wire the cycle to its collaborators.

        Args:
            learner: The training step. Pure, so everything it needs arrives in its arguments.
            state: The surface trained so far, the last good published one, and the restart clock.
            buffer: The stratified history. Owned here in the sense that nothing else prunes it.
            clock: Read for the buffer's age horizon, the restart schedule and the stamps.
            metrics: Where the observations go. The refusal rate is this producer's honest quality
                signal (ADR-010), and it exists only through here.
            weighting: How much each observed point counts.
            grid: Where the published moneyness nodes sit.
            mesh: Where the gate looks -- a third axis, neither the training data nor the published
                grid, chosen for the judgement rather than for the fit.
            thresholds: How much arbitrage is tolerable.
            schedule: Replay size and restart interval.
            rng: The generator the stratified draw uses.

                Injected rather than created here, and that is ADR-004 rather than fastidiousness:
                the replay draw is the one genuinely random step in this engine, so a generator
                seeded at the composition root is the difference between a recorded session
                replaying exactly and replaying approximately. A module-level ``default_rng()``
                would make every run of one recording train on a different history.
        """
        self._learner = learner
        self._state = state
        self._buffer = buffer
        self._clock = clock
        self._metrics = metrics
        self._weighting = weighting
        self._grid = grid
        self._mesh = mesh
        self._thresholds = thresholds
        self._schedule = schedule
        self._rng = rng

    def handle(self, snapshot: MarketSnapshot) -> tuple[Event, ...]:
        """Train on the snapshot and return every event the cycle produced.

        The sequence, and why it is this one:

        1. **Invert and weight the quotes.** One point per strike, from the out-of-the-money leg.
        2. **Prune**, once, against the injected clock. The buffer has no clock of its own by
           design, so if this call is skipped the age horizon silently stops being enforced.
        3. **Draw the replayed points, then add the fresh ones.** In that order: a snapshot's own
           quotes must not be drawn back out of the buffer in the same cycle they entered it, or
           they train the model twice at double weight -- guaranteed on the first cycle, when the
           buffer holds nothing else.
        4. **Decide whether this is a scheduled restart.** If it is, the batch is the *whole*
           buffer and ``previous`` is ``None``: the same operation without a history.
        5. **Train**, and time it on the injected clock.
        6. **Judge**, on the mesh, against the tolerances. A surface over either one is refused.
        7. **Publish**, or refuse and republish the last good surface (ADR-006).

        Returns:
            The events, in the order they should be published: one ``SurfaceCalibrated``, or a
            ``CalibrationFailed`` followed by the republished surface when there is one to fall
            back on, or the failure alone. Never empty -- silence downstream is indistinguishable
            from a process that died.
        """
        now = self._clock.now()
        fresh = to_training_samples(snapshot, self._weighting)
        if not fresh:
            return self._refuse(snapshot, "no quote in the snapshot admits an implied volatility")

        dropped = self._buffer.prune(now)
        restarting = self._state.is_restart_due(now, self._schedule.restart_seconds)
        batch = self._build_batch(snapshot, fresh, restarting)
        # Added *after* the draw, and the order is the whole meaning of "replay". Extending first
        # would let this snapshot's own points come back out of the buffer in the same breath, so
        # every one of them would enter the step twice with double the weight -- always, on the
        # first cycle of a session, when the buffer holds nothing else. They are perfectly good
        # replay candidates for the *next* snapshot, which is where they will be drawn from.
        self._buffer.extend(fresh)
        self._metrics.gauge("neural.buffer.size", len(self._buffer), market=snapshot.market_id)
        self._metrics.gauge(
            "neural.buffer.cell_coverage", self._buffer.cell_coverage, market=snapshot.market_id
        )
        if dropped:
            self._metrics.counter("neural.buffer.pruned", dropped, market=snapshot.market_id)

        if batch is None:
            return self._refuse(snapshot, "the batch carried no usable weight")

        started = self._clock.now()
        try:
            surface = self._learner.update(None if restarting else self._state.surface, batch)
        except NeuralSurfaceError as failure:
            return self._refuse(snapshot, str(failure))
        finished = self._clock.now()

        if restarting:
            self._state.restarted(now)
            self._metrics.counter("neural.restart", market=snapshot.market_id)

        report = check_surface(surface, self._mesh)
        self._metrics.gauge(
            "neural.butterfly_violation", report.butterfly_violation, market=snapshot.market_id
        )
        self._metrics.gauge(
            "neural.calendar_violation", report.calendar_violation, market=snapshot.market_id
        )

        # The surface is kept even when it is refused. It is the state fine-tuning continues from,
        # and throwing it away on a failed gate would turn every refusal into a cold restart --
        # which is the most expensive possible response to a surface that may be one gradient step
        # from admissible.
        self._state.trained(surface, now)

        if report.exceeds(self._thresholds.butterfly, self._thresholds.calendar):
            return self._refuse(snapshot, "the surface failed the no-arbitrage gate")

        published = to_calibrated_surface(
            surface=surface,
            snapshot=snapshot,
            grid=self._grid,
            producer_id=self._learner.producer_id,
            surface_id=self._surface_id(snapshot),
            ts_calibrated=finished,
            status=(SurfaceStatus.DEGRADED if snapshot.quality.degraded else SurfaceStatus.OK),
            fit=fit_metrics(
                surface=surface,
                # On a restart there are no fresh points to measure against, so the residual is
                # taken over the whole batch instead. It is a different question -- "does the model
                # fit its own history" rather than "does it fit the market now" -- and reporting it
                # under the same field is the honest option only because the alternative is
                # publishing a surface with no fit metrics at all, which the contract forbids.
                fresh=batch.fresh if batch.n_fresh else batch.samples,
                n_iterations=1,
                duration_ms=(finished - started).total_seconds() * MILLISECONDS_PER_SECOND,
            ),
        )
        if published is None:
            return self._refuse(snapshot, "the snapshot carried no tenor to publish on")

        self._state.remember(published)
        self._metrics.gauge(
            "neural.rmse_vol_bp",
            published.fit.rmse_vol_bp,
            market=snapshot.market_id,
            producer=self._learner.producer_id,
        )
        return (to_surface_calibrated(published),)

    def _build_batch(
        self,
        snapshot: MarketSnapshot,
        fresh: Sequence[TrainingSample],
        restarting: bool,
    ) -> TrainingBatch | None:
        """This snapshot's points plus a draw from the buffer, or the whole buffer on a restart.

        A restart carries **no fresh points at all** -- ``n_fresh`` is zero, which
        ``TrainingBatch`` documents as legal and meaningful precisely for this case: the model is
        retrained from scratch on the accumulated history, and the before-and-after comparison of
        that surface against the one it replaces is the honest drift measurement of Design 6.4.

        This snapshot's own quotes therefore sit out the restart, and join the buffer immediately
        after it for the next cycle to train on. Losing one snapshot's worth of fresh data every
        few hours costs nothing measurable, and the alternative -- extending the buffer first, only
        on this branch -- would make the restart the one cycle in the session whose batch was
        assembled by a different rule.
        """
        if restarting:
            return to_training_batch(snapshot, fresh=(), replayed=self._buffer.snapshot())
        return to_training_batch(snapshot, fresh=fresh, replayed=self._replay())

    def _replay(self) -> tuple[TrainingSample, ...]:
        """A stratified draw, or nothing when there is nothing to draw from.

        ``ReplayBuffer.sample`` raises ``EmptyBufferError`` on an empty buffer, and that is right
        for a caller who asked for history and needs to know there is none. Here it is not a
        failure at all: the fresh quotes are a perfectly good batch on their own, and the buffer is
        legitimately empty on the first snapshot of a session and again after a prune that outlived
        everything held. Refusing to train would mean a cold-started market publishing nothing
        until its second snapshot, for no reason a consumer could act on.
        """
        if not self._schedule.replay_size:
            return ()
        try:
            return self._buffer.sample(self._schedule.replay_size, self._rng)
        except EmptyBufferError:
            return ()

    def _surface_id(self, snapshot: MarketSnapshot) -> str:
        """``"BTC-DERIBIT:00000042/mlp-torch"`` -- the snapshot and who trained on it.

        Deterministic, and built the same way the parametric producer builds its own: two
        producers see the same snapshots by design, so an id derived from the snapshot alone would
        give their surfaces one identity.
        """
        return f"{snapshot.snapshot_id}/{self._learner.producer_id}"

    def _refuse(self, snapshot: MarketSnapshot, reason: str) -> tuple[Event, ...]:
        """Publish the failure, and the previous surface behind it if there is one (ADR-006)."""
        producer = self._learner.producer_id
        self._metrics.counter(
            "neural.publication.refused", market=snapshot.market_id, producer=producer
        )
        failed = to_calibration_failed(
            market_id=snapshot.market_id,
            source_snapshot_id=snapshot.snapshot_id,
            producer_id=producer,
            reason=reason,
            ts=self._clock.now(),
        )
        previous = self._state.last_published
        if previous is None:
            return (failed,)
        self._metrics.counter(
            "neural.surface.republished", market=snapshot.market_id, producer=producer
        )
        return (failed, to_surface_calibrated(as_stale_republish(previous)))
