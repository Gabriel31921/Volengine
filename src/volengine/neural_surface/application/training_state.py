"""The state a pure ``SurfaceLearner`` refuses to keep.

``SurfaceLearner.update`` takes the previous surface as an argument and returns a new one, so that
replaying a session means replaying the same ``(previous, batch)`` pairs and the weights come out
identical -- no optimiser momentum quietly carried between experiments, no reset ritual. The
statefulness that continuous fine-tuning obviously *is* has to live somewhere, and this is it.

Three pieces, and they are kept apart because they answer different questions:

* **The surface trained so far** is what the next step continues from. It survives a refused
  publication, because a surface the gate turned down is still the state training has reached and
  discarding it would make every refusal a cold restart.
* **The last published surface** is what gets republished when a cycle is refused (ADR-006). Only
  ever a surface that actually went out.
* **The instant of the last restart** is what the scheduled retrain of Design 6.4 is measured
  from.

The first two are the distinction that matters most, and merging them would be the easy mistake:
they are equal only in the ordinary case, and they differ in exactly the situation the gate exists
for.
"""

from __future__ import annotations

from datetime import datetime

from volengine.contracts.calibrated_surface import CalibratedSurface
from volengine.neural_surface.domain.learned_surface import LearnedSurface


class TrainingState:
    """One market's memory across fine-tuning cycles.

    Not thread-safe, and it does not need to be: one instance belongs to one use case, which
    belongs to one asyncio task. The training itself runs on a thread pool (ADR-005) and is handed
    an immutable batch.
    """

    def __init__(self) -> None:
        self._surface: LearnedSurface | None = None
        self._last_published: CalibratedSurface | None = None
        self._last_restart: datetime | None = None

    @property
    def surface(self) -> LearnedSurface | None:
        """The surface trained so far, or ``None`` before the first step.

        ``None`` is the cold start, which the port documents as a case an implementation must
        handle rather than assume away -- the first snapshot of a market and the recovery path
        after a failure both land here.
        """
        return self._surface

    @property
    def last_published(self) -> CalibratedSurface | None:
        """The last surface that actually went out, or ``None`` if none ever has.

        What ADR-006 republishes when a cycle is refused. Deliberately **not** the same as
        :attr:`surface`: the model keeps training through a refusal, so after one the two differ,
        and it is the published one -- already validated, already a DTO -- that has to go out
        again. Reassembling a contract from the newer weights would mean publishing precisely the
        surface the gate just turned down.
        """
        return self._last_published

    def trained(self, surface: LearnedSurface, now: datetime) -> None:
        """Record where training has reached, published or not, and when.

        Called on **every** successful ``update``, including one whose result is about to be
        refused. That is the whole reason this is separate from :meth:`remember`: a surface that
        failed the gate is one gradient step's worth of progress, possibly one step from
        admissible, and throwing it away would answer a refusal with the most expensive move
        available -- a cold restart on the next snapshot.

        The instant is taken here so that the restart interval starts running from the first step
        rather than from an absent history. A cold start *is* a training-from-scratch, so measuring
        the next scheduled restart from it is the reading that stops the first snapshot of every
        session from immediately retraining over a buffer holding one snapshot's worth of quotes.
        """
        self._surface = surface
        if self._last_restart is None:
            self._last_restart = now

    def remember(self, published: CalibratedSurface) -> None:
        """Keep this surface as the one to republish if a later cycle is refused.

        Called only with a surface that was genuinely published. A republished stale surface is
        not passed back through here: it is already the last good one, and storing it again under
        its ``STALE_REPUBLISH`` status would send the next republication out labelled
        stale-of-a-stale, a distinction the contract does not make.
        """
        self._last_published = published

    def is_restart_due(self, now: datetime, restart_seconds: float | None) -> bool:
        """Whether the scheduled retrain of Design 6.4 should fire on this cycle.

        Evaluated on the arrival of a snapshot rather than by a timer, which is why this context's
        ``Clock`` has no ``sleep``: the point of retraining is to train, and there is nothing to
        train on until quotes arrive. A timer would fire the most expensive operation in the
        system into an empty room.

        **The first cycle never restarts**, because no step has run yet and there is nothing to
        restart from: the interval starts running at the first :meth:`trained`. This method is a
        pure predicate and changes nothing, which is what lets a caller ask it before deciding
        anything.

        Args:
            now: The instant being decided about. Timezone-aware.
            restart_seconds: The configured interval, or ``None`` to never restart.

        Returns:
            ``True`` when a restart is configured and at least that long has passed.
        """
        if restart_seconds is None or self._last_restart is None:
            return False
        return (now - self._last_restart).total_seconds() >= restart_seconds

    def restarted(self, now: datetime) -> None:
        """Record that a restart happened, so the next one is measured from here.

        Separate from :meth:`is_restart_due` rather than folded into it, because a restart that
        was decided on and then failed -- an empty buffer, a learner that could not be built --
        must not reset the interval. Asking and doing are two events and only the second one
        moves the clock.
        """
        self._last_restart = now
