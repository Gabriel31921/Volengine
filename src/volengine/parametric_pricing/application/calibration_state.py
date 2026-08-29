"""The state a calibration cycle needs and ``Calibrator`` refuses to hold.

``Calibrator.calibrate`` is a pure function on purpose -- no memory between calls, so a replay
reproduces a session bit for bit and two calibrators handed identical inputs differ only in their
mathematics. The statefulness a warm start obviously *is* has to live somewhere, and this is the
somewhere: one object per market and producer, whose entire job is to chain one cycle's output
into the next cycle's input.

Two pieces of state, and they are kept apart because they answer different questions. The **warm
start** is where the next optimisation begins, and it is per expiry. The **last good surface** is
what gets republished when a cycle is refused (ADR-006), and it is one whole published object.
Merging them would be tempting -- both are "the last thing that worked" -- and wrong: a warm start
survives a rejected cycle, because the parameters it holds were accepted on an earlier one and are
still the best starting point available, while the surface republished has to be a complete,
already-validated DTO and cannot be reassembled from parameters after the fact.
"""

from __future__ import annotations

from collections.abc import Collection, Mapping, Sequence
from datetime import datetime

from volengine.contracts.calibrated_surface import CalibratedSurface
from volengine.parametric_pricing.domain.calibration import SliceResult
from volengine.parametric_pricing.domain.svi_slice import SVIParams


class CalibrationState:
    """One market's memory across calibration cycles.

    Not thread-safe, and it does not need to be: one instance belongs to one use case, which
    belongs to one asyncio task. The fit itself runs on a thread pool (ADR-005), but it is handed
    an immutable task and hands back an immutable result -- nothing in this object crosses that
    boundary.
    """

    def __init__(self) -> None:
        self._warm_start: dict[datetime, SVIParams] = {}
        self._last_good: CalibratedSurface | None = None

    @property
    def warm_start(self) -> Mapping[datetime, SVIParams] | None:
        """Where the next fit should start, or ``None`` for a cold start.

        ``None`` rather than an empty mapping, matching what ``Calibrator.calibrate`` documents:
        the port's ``previous`` is ``Mapping | None`` and ``None`` means "no history", which is
        the first cycle of a market and the recovery path after a failure. An empty dict would be
        a third state meaning the same thing, and an implementation checking ``if previous:``
        would treat the two identically anyway -- so the ambiguity would exist only to be
        resolved by accident.

        Keyed by expiry, never by position: strikes and whole expiries are born and die between
        snapshots (ADR-013), and a positional structure would pair a warm start with whichever
        tenor happened to occupy that index this time. That is not a wrong number anyone would
        notice; the fit would simply converge somewhere slightly odd.
        """
        return dict(self._warm_start) if self._warm_start else None

    @property
    def last_good(self) -> CalibratedSurface | None:
        """The last surface that was actually published, or ``None`` if none ever was.

        What ADR-006 republishes when a cycle is refused. ``None`` at start-up is the one case
        where a refusal has nothing to fall back on, and the use case then publishes the failure
        alone -- which is honest: there is no old surface to be stale about.
        """
        return self._last_good

    def accept(self, accepted: Sequence[SliceResult], live_expiries: Collection[datetime]) -> None:
        """Record the accepted parameters as the next cycle's starting point.

        Args:
            accepted: The slices that passed the acceptance rule. **Only the accepted ones**: a
                slice that finished pinned against a bound is a projection onto the boundary
                rather than a fit, and starting the next optimisation from it would walk the
                search straight back to the same edge -- a bad cycle would then be self-sustaining
                rather than self-correcting.
            live_expiries: Every expiry present in the task just fitted. Anything held from an
                earlier cycle and absent from this one is dropped.

                The pruning is what bounds the memory of a process that runs for months. Without
                it, every expiry the venue has ever listed stays here forever: a warm start keyed
                by ``datetime`` never collides, so nothing would ever overwrite an entry for a
                contract that settled in March. It also keeps the mapping *meaningful* -- an
                expiry that is no longer quoted has no fit to warm-start, and holding stale
                parameters for one would be a plausible answer to a question nobody may ask again.
        """
        live = set(live_expiries)
        self._warm_start = {
            expiry: params for expiry, params in self._warm_start.items() if expiry in live
        }
        for fitted in accepted:
            self._warm_start[fitted.expiry] = fitted.params

    def remember(self, surface: CalibratedSurface) -> None:
        """Keep this surface as the one to republish if a later cycle is refused.

        Called only with a surface that was genuinely published. A republished stale surface is
        deliberately *not* passed back through here: it is already the last good one, and storing
        it again under its ``STALE_REPUBLISH`` status would mean the next republication went out
        labelled stale-of-a-stale, which is a distinction the contract does not make and a
        consumer cannot use.
        """
        self._last_good = surface
