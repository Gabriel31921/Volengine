"""What the Neural Surface domain needs from the outside world, said in its own words.

These are the *driven* ports of this context: the domain declares the shape of the collaborators
it depends on, and the composition root supplies something that fits. The torch MLP of F3-D is
one implementation of ``SurfaceLearner``; it is not imported here, and it never will be. That is
the direction of the dependency arrow in a hexagon -- the outside knows about the inside, not the
reverse -- and in this context it carries more weight than usual, because the thing on the outside
is a machine-learning framework and the thing on the inside is the invariant that governs it.
``domain/invariants.py`` refuses to publish a surface that admits arbitrage (ADR-010), and it can
only be trusted to do so while it is written in ordinary arrays, checkable against a surface
somebody wrote by hand in four lines. ``SurfaceLearner`` is the seam that keeps torch on the far
side of that line.

Everything below is a ``typing.Protocol``, so conformance is **structural**: an adapter is never
asked to inherit from anything, it simply has the right methods. Nothing here is
``@runtime_checkable``, deliberately. ``isinstance`` against a runtime-checkable Protocol only
compares *member names*, so an ``update`` that returned ``None``, or that took its two arguments
in the other order, would pass it happily. The failure that matters most here would pass it too: a
learner that accepted ``previous`` and then ignored it has the right member with the right name,
and would silently turn continuous fine-tuning into a cold restart on every snapshot without
anything ever raising. A name check that blesses that is worse than no check at all, because it
reads like one. The real guard is ``mypy --strict`` at the composition root, where the concrete
object is assigned to the port, plus a call made through the port in the tests.

``Clock`` and ``MetricsSink`` are duplicated on purpose across contexts rather than imported from
``platform/``: a port describes a need, and a need belongs to whoever has it. This context's
``Clock`` is visibly not Market Data's and not Parametric Pricing's either -- see its docstring --
which is the divergence that duplication was chosen to allow.
"""

from __future__ import annotations

from datetime import datetime
from typing import Protocol

from volengine.neural_surface.domain.learned_surface import LearnedSurface
from volengine.neural_surface.domain.training_batch import TrainingBatch


class SurfaceLearner(Protocol):
    """A training step: weights in, weights out. **A pure function wearing an object's clothes.**

    This is the central port of the context, and it is the file that contradicts the sketch in
    ``Implementation.md``. Two departures, both deliberate, both explained where they bite.

    **The sketch's ``evaluate(surface, k, t)`` is gone.** ``LearnedSurface`` is itself an evaluable
    abstraction of sigma(k, T) (Design 6.1), so a learner evaluating on its surface's behalf would
    be a second way to do one thing, and the two would drift. Worse, it would make evaluation a
    capability of the *learner*: the hard gate of ADR-010 would then need a live learner in hand
    before it could judge anything, which is precisely the coupling that would drag torch back
    within reach of the domain. Deleting it is what lets ``invariants.check_surface`` be
    self-contained and lets a surface written in four lines of numpy stand in for a trained model
    in every test of the gate.

    **The sketch's ``update(batch)`` gained a ``previous``.** Continuous fine-tuning (Design 6.4)
    is stateful by nature -- K gradient steps per snapshot, applied on top of whatever the weights
    already were -- and the obvious shape for it is a learner that holds those weights and mutates
    them. That shape is rejected here. ``update`` performs no I/O, reads no clock, and **keeps no
    state between calls**: everything it needs arrives in its arguments and everything it produces
    leaves in its return value. An implementation that stashed the last surface, or that kept an
    optimiser's moment estimates on itself between invocations, would satisfy the type checker and
    break every property below.

    Three things follow, and all three are the reason the port is shaped this way:

    * **Replay is reproducible for free.** Replaying a recorded session means replaying the same
      sequence of ``(previous, batch)`` pairs, and the surfaces come out identical -- no reset
      ritual between runs, no optimiser momentum quietly carried over from the previous experiment.
      ADR-004 asks that of the whole engine; here it costs nothing, because there is nothing to
      reset.
    * **The neural producer is comparable with the parametric one on identical terms.** Design 6.5
      measures the distance between the two surfaces, and Design 5.7 compares the producers. Those
      numbers only mean something if the two ports have the same shape and the same purity:
      ``Calibrator.calibrate(previous, task)`` and ``SurfaceLearner.update(previous, batch)`` are
      the same function twice, differing in the mathematics and in nothing else. A stateful learner
      would carry its own history into the comparison and there would be no way to tell a better
      model from a luckier one.
    * **Tests need no mocks, and no GPU.** A training step is checked by calling a function and
      looking at what came back.

    The state that fine-tuning obviously *is* still exists -- it lives in
    ``application/train_stream.py``, which chains one snapshot's surface into the next snapshot's
    ``previous``, owns the replay buffer, and keeps the instant of the last scheduled restart. That
    placement is the whole trick: the statefulness is real, it is simply kept outside the thing
    being compared and tested, in a module whose only job is to hold it.

    Note what this port does *not* constrain: how much a concrete ``LearnedSurface`` carries. The
    domain sees only ``version`` and ``total_variance``; an implementation is free to hide network
    weights, optimiser moments and input normalisation statistics inside its own concrete type,
    because the domain never constructs one and never looks. Purity is a statement about the
    learner, not a demand that a surface be a bare array.
    """

    @property
    def producer_id(self) -> str:
        """Stable identity of this implementation, such as ``"mlp-torch"``.

        What makes this producer distinguishable everywhere downstream: it names the topic the
        surface is published on (``surface.{market_id}.{producer_id}``), it travels inside
        ``CalibratedSurface`` so a risk report can say which producer it trusted, and it is the
        key the comparative report of F3-E groups by. This context exists to be run **beside**
        the parametric one on the same market, publishing the same contract -- if a consumer could
        not tell the two apart, the comparison of Design 6.5 would have nothing to name its two
        sides, and the two producers' metric series would average into one that describes neither.

        A read-only property rather than a plain attribute so that the protocol constrains neither
        how it is stored nor whether it is computed -- a frozen dataclass field, a class constant
        and a derived string all satisfy it.
        """
        ...

    def update(self, previous: LearnedSurface | None, batch: TrainingBatch) -> LearnedSurface:
        """Train from one batch, optionally continuing from the surface trained so far.

        Args:
            previous: The surface to fine-tune, or ``None`` to train from scratch.

                ``None`` is the cold start -- the first snapshot of a market, and the recovery
                path after a failure -- and an implementation must handle it rather than assume it
                never happens. It is *also* the **scheduled restart** of Design 6.4: retraining
                from zero over the whole buffer every M hours is exactly ``update(None, batch)``
                with a batch drawn from ``ReplayBuffer.snapshot()``. That is why there is no
                ``restart`` method here and no flag saying which mode we are in. The restart is not
                a different operation, it is the same operation without a history, and giving it
                its own method would create a second code path that only runs once every few hours
                -- the rarest path in the system and therefore the least exercised.

                Passed as an argument rather than held on the learner because the drift measurement
                of Design 6.4 needs both surfaces to exist **at the same time**: the restart is
                honest only if the surface it replaces can still be evaluated on the same mesh
                afterwards and the two compared. A learner that retrained itself in place would
                destroy the very thing the measurement is about.

            batch: The quotes to train on: this snapshot's, followed by a stratified draw from the
                replay buffer, with ``n_fresh`` marking the split. See ``training_batch.py``: no
                venue, no day count, no strike in currency, and no DTO -- the domain does not know
                the published language (rule 3), so a learner cannot depend on it either. The
                batch is a *set of scattered points* in ``(k, T)`` rather than a stack of slices,
                which is the whole modelling difference between this context and SVI's
                slice-by-slice fit, and it reaches the learner already shuffled across tenors.

        Returns:
            A **new** ``LearnedSurface``, with a ``version`` distinguishable from ``previous``'s.
            Never ``previous`` mutated and handed back: the caller still holds the old surface and
            still needs it, both for the before-and-after comparison above and for the far more
            ordinary case in which the new one is refused.

            **Including a surface that will turn out to be unpublishable.** Deciding whether a
            surface may be published is not this port's job: the hard gate of ADR-010 --
            ``invariants.check_surface`` against a mesh, judged with tolerances that are TOML
            configuration (ADR-012) -- belongs to the use case, which republishes the last good
            surface when the new one fails. A learner that refused to return a surface it suspected
            of arbitrage would take that decision away from the layer that owns it, and would leave
            the gate with nothing to measure. Soft constraints train, hard constraints govern; this
            method is where the soft ones end.

        Raises:
            Nothing, in the ordinary course of business. An arbitrageable surface is a return
            value, not an exception; see ``domain/errors.py``, which explains that boundary and is
            the module this one must not contradict. A model that has diverged into NaN is not
            caught here either -- it surfaces as ``SurfaceEvaluationError`` when the surface is
            evaluated, which is where the numbers are actually looked at, and which is a
            consequence of the deleted ``evaluate`` rather than an oversight. An implementation may
            still fail outright -- a model that cannot be constructed at all -- and such a failure
            should surface as a ``NeuralSurfaceError``.

        Synchronous, not ``async``, and not by omission. K gradient steps are compute-bound, so the
        use case hands the call to the named thread pool of ADR-005 and awaits the executor rather
        than the training. Typing it ``async`` would invite an implementation to reach for I/O
        inside -- the one thing the purity above forbids -- and would put a coroutine in a function
        that never waits for anything.
        """
        ...


class Clock(Protocol):
    """Time, as this context needs it: read the current instant. Nothing else.

    Declared here rather than imported from ``platform/``, and the duplication is the design.
    ``platform.clock.SystemClock`` happens to meet this shape; neither module imports the other,
    in either direction, and the connection is made exactly once, in the composition root.

    **This context reads the clock for two things no other context has**, and that is the argument
    for a port per context made concrete:

    * **The replay buffer's max-age policy.** ``ReplayBuffer.prune(now)`` takes the instant as an
      argument and never reads a clock itself -- the buffer is domain state, not a scheduler -- so
      somebody upstream has to supply that instant, and this is the port they get it from. Nothing
      in Market Data or Parametric Pricing holds data that expires by age in this way.
    * **The scheduled restart of Design 6.4.** Retraining from scratch every M hours is a decision
      about elapsed time, and elapsed time is ``now()`` minus the instant of the last restart, held
      in ``application/train_stream.py``.

    Note what does *not* follow from the second one: this ``Clock`` still has **no ``sleep``**,
    where Market Data's does. A restart every M hours sounds like a timer, and a timer would need
    one -- but the restart is evaluated on the arrival of a snapshot, by comparing instants, not by
    a task waking itself up. That keeps this context purely reactive, like its parametric sibling,
    and it also keeps the restart from firing into an empty room: the point of retraining is to
    train, and there is nothing to train on until quotes arrive. If this context ever needs a
    method the others do not, it is added *here*, and only the implementations wired into *this*
    context have to grow it. A single shared ``Clock`` would force every context's needs onto every
    context at once, which is how a shared kernel quietly becomes a god object.

    Note also who does **not** get this port: the learner, the buffer and the gate. ``update`` takes
    no clock and is the better for it, ``prune`` is handed an instant, and an arbitrage check has
    no notion of when it ran. Time enters this context in one layer only, which is what ADR-004
    requires for a recorded session to replay exactly: a module calling ``datetime.now()`` directly
    is a module whose output cannot be reproduced.
    """

    def now(self) -> datetime:
        """Current instant, always timezone-aware UTC.

        Aware, never naive: this value is subtracted from ``TrainingSample.ts_observed`` to age
        out the buffer and compared against ``TrainingBatch.ts_snapshot`` to stamp a cycle, and
        subtracting a naive datetime from an aware one raises ``TypeError``. ``datetime.utcnow()``
        returns a naive value despite its name and is banned everywhere in this repo.
        """
        ...


class MetricsSink(Protocol):
    """Where this context's observations go, without it knowing where that is.

    ``logging`` is banned in the domain, and this is the replacement: this layer states *what* it
    observed and stops, leaving format, level and destination to the composition root. Declared by
    this context for the same reason as ``Clock``, and satisfied structurally by
    ``platform.metrics.LoggingMetricsSink`` and ``NullMetricsSink`` with no import in either
    direction.

    The series that matter here are not the ones a machine-learning project usually watches. A
    training loss is a statement about the model's opinion of itself; the numbers below are
    statements about whether the model may be trusted downstream:

    * **How often publication was refused.** ADR-010 says a rejected publication is observable as a
      metric and that the failure rate is the honest quality signal for this producer -- and it is
      honest precisely because the soft constraints in the loss are a *preference*, not a
      guarantee. The refusal rate is the only measurement that says whether they actually held.
    * **How deep the violations ran.** ``ArbitrageReport``'s butterfly and calendar depths, gauged
      whether or not they crossed the tolerance, so a surface drifting towards the gate is visible
      before it hits it. A refusal is a step function; the depth is the slope leading up to it.
    * **Buffer cell coverage.** The fraction of stratification cells holding anything at all
      (Design 6.5). A coverage collapse is what catastrophic forgetting looks like *before* it
      reaches the fit -- the wings go quiet first and the RMSE, dominated by at-the-money ticks,
      says nothing about it for a while.
    * **The neural-to-parametric distance.** Both producers emit the same contract, so the distance
      between their surfaces is trivially computable (Design 6.5), and where on the surface they
      diverge is a signal in itself.

    Tags are the dimensions a measurement is filtered by later -- market, producer, tenor band,
    reason -- and they are ``str`` so that any backend can carry them. ``producer_id`` belongs on
    every one of them: this producer runs beside the parametric one by design, and two producers
    publishing the same metric name under the same tags would silently average into a series that
    describes neither.
    """

    def gauge(self, name: str, value: float, **tags: str) -> None:
        """A value that goes up and down: the depth of the worst butterfly violation on the mesh."""
        ...

    def counter(self, name: str, value: int = 1, **tags: str) -> None:
        """A value that only grows: how many surfaces the hard gate refused to publish."""
        ...

    def timing(self, name: str, ms: float, **tags: str) -> None:
        """A duration in milliseconds: snapshot received to surface published, or refused."""
        ...
