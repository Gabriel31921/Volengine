"""One calibration cycle: a snapshot in, the events it caused out.

**A handler, not a loop, and the name is the departure from ``Implementation.md``.** The sketch
calls this ``CalibrateStreamUseCase`` and pictures an object that subscribes and consumes; what
is here is a synchronous function of one snapshot, with the subscription living in
``entrypoints/pipeline.py``. Three reasons, in increasing order of weight:

* **It is testable by calling it.** Every rule below -- the acceptance threshold, the stale
  republish, the status downgrade -- is asserted by handing this object a snapshot and looking at
  what came back. A loop would need an event loop, a fake bus and a way to wait for "it has
  processed now", and that last one has no non-racy spelling.
* **Replay is a ``for`` statement.** ADR-004 wants a recorded session to reproduce exactly;
  with a handler, replaying is feeding the recorded snapshots back in order.
* **ADR-005 stays outside.** The fit is CPU-bound and belongs on a named thread pool. Whoever
  owns the loop is whoever calls ``run_in_executor``, so a handler can be pushed onto a pool by
  the composition root without ever learning that threads exist -- while a loop would have to
  mix waiting for I/O and doing arithmetic in one object.

**Every decision the ports refused is made here.** ``Calibrator`` returns badly fitted slices
rather than raising, because deciding what may be published is not its job; ADR-006's acceptance
rule, the choice to republish the previous surface, and the ``DEGRADED`` label are all in this
file, judged against thresholds that are TOML configuration (ADR-012).
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

from volengine.contracts.calibrated_surface import SurfaceStatus
from volengine.contracts.events import Event
from volengine.contracts.market_snapshot import MarketSnapshot
from volengine.parametric_pricing.application.acl import (
    Weighting,
    as_stale_republish,
    to_calibrated_surface,
    to_calibration_failed,
    to_calibration_task,
    to_surface_calibrated,
)
from volengine.parametric_pricing.application.calibration_state import CalibrationState
from volengine.parametric_pricing.application.grid_spec import GridSpec
from volengine.parametric_pricing.domain.calibration import SliceResult
from volengine.parametric_pricing.domain.errors import CalibrationError
from volengine.parametric_pricing.domain.ports import Calibrator, Clock, MetricsSink

MILLISECONDS_PER_SECOND = 1000.0


@dataclass(frozen=True, slots=True)
class Acceptance:
    """When a fitted slice is good enough to publish. ADR-006, as configuration.

    One threshold and two flags, and only the threshold is a number anyone chooses -- the other
    two conditions are not configurable and the omission is deliberate. See :meth:`accepts`.
    """

    max_rmse_vol_bp: float
    """Largest weighted RMSE, in basis points of volatility, a slice may carry and still go out.

    Positive and finite. In vol rather than in price so that one number means the same thing at
    every strike, and directly comparable to the spread the fit is trying to beat: a wide crypto
    wing trades a couple of vol points wide, so a fit inside 20 bp is inside the noise.

    Configuration because it is a statement about a market's own noise level, not about this code.
    """

    def __post_init__(self) -> None:
        # Finiteness first: `float("nan") <= 0` is `False`, and a NaN threshold would make every
        # comparison in `accepts` False -- silently rejecting every slice ever fitted, for a
        # market that is behaving perfectly.
        if not math.isfinite(self.max_rmse_vol_bp) or self.max_rmse_vol_bp <= 0:
            raise ValueError(
                f"The RMSE threshold must be positive and finite, got {self.max_rmse_vol_bp}"
            )

    def accepts(self, fitted: SliceResult) -> bool:
        """Whether this slice may be published: converged, off its bounds, and inside the RMSE.

        The three conditions are ``and``-ed and none of them is optional, which is ADR-006 read
        literally -- "RMSE **and** no parameter at a bound" -- with convergence added because
        ``SliceResult`` documents it as the other half of the same context.

        **Why neither flag is a knob.** A fit that exhausted its iteration budget is a snapshot of
        a search in progress, and its RMSE may be under the threshold by luck of where the walk
        happened to stop. A pinned parameter is worse: what came back is a projection onto the
        edge of the admissible region, and the residual it reports is the residual of the
        projection, so such a slice can show a respectable RMSE while describing a smile whose
        wing is held up by the constraint rather than by the market. Publishing either as though
        it were healthy is exactly what "an honest old surface beats a broken new one" refuses,
        and a deployment that could switch the check off would be a deployment that eventually
        does.
        """
        return (
            fitted.converged and not fitted.at_bound and fitted.rmse_vol_bp <= self.max_rmse_vol_bp
        )


class CalibrateOnSnapshot:
    """Fit one snapshot and say what happened, in the published language.

    One instance per market and per calibrator: two producers on one market are two of these,
    each with its own state, publishing the same contract onto different topics. That is the
    arrangement the whole project exists to compare, and it works because nothing in this class
    branches on which calibrator it holds.
    """

    def __init__(
        self,
        calibrator: Calibrator,
        state: CalibrationState,
        clock: Clock,
        metrics: MetricsSink,
        weighting: Weighting,
        grid: GridSpec,
        acceptance: Acceptance,
    ) -> None:
        self._calibrator = calibrator
        self._state = state
        self._clock = clock
        self._metrics = metrics
        self._weighting = weighting
        self._grid = grid
        self._acceptance = acceptance

    def handle(self, snapshot: MarketSnapshot) -> tuple[Event, ...]:
        """Fit the snapshot and return every event the cycle produced.

        **A tuple, because one cycle can be two facts at once.** A refused calibration publishes
        ``CalibrationFailed`` *and* republishes the previous surface as ``STALE_REPUBLISH``
        (ADR-006), and those are genuinely two things a consumer needs: one says this producer
        could not fit this market, the other keeps the risk report alive with numbers that say
        out loud how old they are. Returning a single event would force one of them to be
        dropped, and either choice loses information the pipeline was built to carry.

        The possible answers, exhaustively:

        * one ``SurfaceCalibrated`` -- the fit was accepted;
        * ``CalibrationFailed`` then ``SurfaceCalibrated`` -- refused, with a previous surface to
          fall back on. In that order, so a consumer reading a recording sees the cause before
          the consequence;
        * ``CalibrationFailed`` alone -- refused with nothing to republish, which is the ordinary
          state at start-up.

        Never an empty tuple. A cycle that ran and said nothing would be indistinguishable
        downstream from a cycle that never ran, which is the one thing ADR-006 is written against.

        Args:
            snapshot: The market to fit. Its ``quality`` block is read for exactly one purpose --
                a degraded input can only produce a degraded surface -- and never to decide
                whether to fit at all. Publishing a snapshot that fails its own acceptance rules
                was Market Data's decision, and second-guessing it here would mean two layers
                deciding one thing.

        Returns:
            The events, in the order they should be published.

        Raises:
            ValueError: If the resulting surface cannot be expressed in the contract -- notably a
                snapshot whose ``ts_exchange`` sits after our own clock, which
                ``CalibratedSurface`` refuses. Deliberately not caught and not converted into a
                ``CalibrationFailed``: reconciling a venue clock is the producing context's job
                and it does it (``market_data.application.acl.reconcile_exchange_instant``), so
                one arriving here unreconciled is a wiring bug, and a wiring bug reported as a
                market condition is a wiring bug nobody fixes.
        """
        started = self._clock.now()
        task = to_calibration_task(snapshot, self._weighting)
        if task is None:
            return self._refuse(snapshot, "no quote in the snapshot admits an implied volatility")

        try:
            result = self._calibrator.calibrate(self._state.warm_start, task)
        except CalibrationError as failure:
            return self._refuse(snapshot, str(failure))

        finished = self._clock.now()
        accepted = [fitted for fitted in result.slices if self._acceptance.accepts(fitted)]
        self._observe_fit(snapshot, result.slices, accepted)

        if not accepted:
            return self._refuse(snapshot, "no slice met the acceptance rule")

        surface = to_calibrated_surface(
            task=task,
            accepted=accepted,
            n_iterations=result.n_iterations,
            duration_ms=(finished - started).total_seconds() * MILLISECONDS_PER_SECOND,
            grid=self._grid,
            producer_id=self._calibrator.producer_id,
            surface_id=self._surface_id(snapshot),
            ts_calibrated=finished,
            status=self._status(snapshot, accepted, result.slices),
        )
        if surface is None:
            return self._refuse(snapshot, "no accepted slice could be evaluated on the grid")

        self._state.accept(accepted, [one.expiry for one in task.slices])
        self._state.remember(surface)
        self._metrics.gauge(
            "pricing.rmse_vol_bp",
            surface.fit.rmse_vol_bp,
            market=snapshot.market_id,
            producer=self._calibrator.producer_id,
        )
        self._metrics.timing(
            "pricing.cycle_ms",
            surface.fit.duration_ms,
            market=snapshot.market_id,
            producer=self._calibrator.producer_id,
        )
        return (to_surface_calibrated(surface),)

    def _status(
        self,
        snapshot: MarketSnapshot,
        accepted: Sequence[SliceResult],
        attempted: Sequence[SliceResult],
    ) -> SurfaceStatus:
        """``OK`` only when the input was clean and every slice was published.

        Two independent ways to be degraded, both of them information a consumer cannot recover
        on its own. A surface fitted to a degraded snapshot is at best as trustworthy as its
        input, and the input's quality block does not travel with it. A surface missing an expiry
        is a surface with a hole in its term structure, and the grid alone cannot say whether a
        tenor is absent because the market does not quote it or because its fit was refused.
        """
        if snapshot.quality.degraded or len(accepted) != len(attempted):
            return SurfaceStatus.DEGRADED
        return SurfaceStatus.OK

    def _surface_id(self, snapshot: MarketSnapshot) -> str:
        """``"BTC-DERIBIT:00000042/svi-scipy"`` -- the snapshot and who fitted it.

        Deterministic for the same reason the snapshot id is: a random one would make two runs of
        one recording incomparable line by line while proving nothing a derivation does not. The
        producer has to be part of it because two calibrators fit the *same* snapshot by design,
        and an id built from the snapshot alone would give their two surfaces one identity.
        """
        return f"{snapshot.snapshot_id}/{self._calibrator.producer_id}"

    def _refuse(self, snapshot: MarketSnapshot, reason: str) -> tuple[Event, ...]:
        """Publish the failure, and the previous surface behind it if there is one (ADR-006)."""
        producer = self._calibrator.producer_id
        self._metrics.counter(
            "pricing.calibration.refused", market=snapshot.market_id, producer=producer
        )
        failed = to_calibration_failed(
            market_id=snapshot.market_id,
            source_snapshot_id=snapshot.snapshot_id,
            producer_id=producer,
            reason=reason,
            ts=self._clock.now(),
        )
        previous = self._state.last_good
        if previous is None:
            return (failed,)
        self._metrics.counter(
            "pricing.surface.republished", market=snapshot.market_id, producer=producer
        )
        return (failed, to_surface_calibrated(as_stale_republish(previous)))

    def _observe_fit(
        self,
        snapshot: MarketSnapshot,
        attempted: Sequence[SliceResult],
        accepted: Sequence[SliceResult],
    ) -> None:
        """Count what the acceptance rule did, split by which half of it fired.

        The two rejection reasons are counted separately because they mean different things about
        the calibrator. A slice failing on RMSE is a fit that ran and was not good enough; a slice
        at a bound is a fit that was stopped at the edge of what the parameterisation allows, and
        a market where that keeps happening is a market whose smile the model cannot represent.
        Pooling them into one "rejected" counter would hide the second inside the first.
        """
        market = snapshot.market_id
        producer = self._calibrator.producer_id
        self._metrics.gauge(
            "pricing.slices.attempted", len(attempted), market=market, producer=producer
        )
        self._metrics.gauge(
            "pricing.slices.accepted", len(accepted), market=market, producer=producer
        )
        for fitted in attempted:
            # Tags spelled out at each call rather than splatted from a dict: `counter` takes an
            # optional `value` before its keywords, so `**tags` is a type error waiting to bind a
            # string to it -- which mypy catches here and would not catch at a call site that
            # happened to pass a positional count as well.
            if fitted.at_bound:
                self._metrics.counter("pricing.slice.at_bound", market=market, producer=producer)
            elif not fitted.converged:
                self._metrics.counter(
                    "pricing.slice.not_converged", market=market, producer=producer
                )
            elif fitted.rmse_vol_bp > self._acceptance.max_rmse_vol_bp:
                self._metrics.counter(
                    "pricing.slice.rmse_exceeded", market=market, producer=producer
                )
