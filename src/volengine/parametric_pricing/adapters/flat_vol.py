"""One flat volatility per expiry: the walking skeleton's calibrator, and no optimiser at all.

The least implementation of ``Calibrator`` that is still a *fit*. Each slice is answered with the
weighted mean of the volatilities it was handed, expressed as the degenerate SVI slice
``w(k) = a`` -- ``b = 0``, so the two wings are flat and the smile has no shape. Everything the
port promises still holds: no I/O, no clock, no state between calls, so the same task fed twice
gives the same answer bit for bit and a recorded session replays without a reset ritual.

**It reports the error it actually makes.** The mean is not a fit anybody would publish, and
against a real smile its RMSE is the dispersion of the quotes -- tens to hundreds of basis points
of vol -- which is precisely what the acceptance rule of ADR-006 exists to refuse. That is the
right behaviour and it is why the metrics are computed rather than stubbed at zero: a calibrator
that claimed a perfect fit would take the acceptance decision away from the use case that owns it,
and the walking skeleton would prove that the pipeline publishes anything at all rather than that
it publishes what it should. Fed the constant chain of ``market_data.adapters.constant``, whose
quotes really are one volatility, the RMSE is floating-point noise and the surface is accepted.

**A departure from the plan's wording, not from its intent.** ``Plan.md`` describes a
``FlatVolCalibrator`` "returning a constant vol grid": the port returns ``CalibrationResult``, and
the grid is built one layer up by ``application/acl.to_calibrated_surface``, which evaluates these
parameters on the configured mesh (ADR-011). Flat parameters are how a calibrator asks for a
constant grid, and the alternative -- a port that returned a grid -- would put the mesh, and
therefore configuration, inside every calibrator.

numpy, and not the standard library, because the loss of a real calibrator is written over arrays
and this one should be shaped like the thing it stands in for. The domain layer is allowed numpy
(rule 3); an adapter certainly is.
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime

import numpy as np

from volengine.parametric_pricing.domain.calibration import (
    CalibrationResult,
    CalibrationTask,
    SliceResult,
    SliceTask,
)
from volengine.parametric_pricing.domain.svi_slice import SVIParams

PRODUCER_ID = "flat-vol"
"""How this producer names itself everywhere downstream: the topic its surfaces are published on,
the tag on its metrics, and the ``producer_id`` a risk report says it trusted."""

BASIS_POINTS_PER_VOL = 10_000.0
"""One basis point of volatility is ``0.0001`` in decimal terms, which is the unit every fit
metric in this context is stated in."""

FLAT_SIGMA = 1.0
"""``SVIParams`` refuses ``sigma <= 0`` even though ``b = 0`` makes the curvature term vanish
entirely. Any positive number is therefore equivalent here, and one is the least surprising."""


class FlatVolCalibrator:
    """A ``Calibrator`` that averages instead of optimising.

    Structural conformance, like every adapter in this engine: a ``producer_id`` property and a
    ``calibrate`` method, no base class, and nothing here the domain could import back.
    """

    def __init__(self, producer_id: str = PRODUCER_ID) -> None:
        """Args:
        producer_id: What this instance calls itself. A parameter rather than a constant so
            that two of them can run side by side under different names -- which is the
            arrangement Design 5.7 exists to compare, and which the composition root keys its
            topics and its thread pools by.
        """
        if not producer_id.strip():
            raise ValueError(f"The producer id must not be empty, got {producer_id!r}")
        self._producer_id = producer_id

    @property
    def producer_id(self) -> str:
        return self._producer_id

    def calibrate(
        self,
        previous: Mapping[datetime, SVIParams] | None,
        task: CalibrationTask,
    ) -> CalibrationResult:
        """Fit every slice of the task, ignoring the warm start.

        Args:
            previous: The parameters each expiry was last fitted to. **Read by nothing here**, and
                that is not an oversight: a weighted mean has no search to start, so there is no
                iterate for a previous answer to move. The argument stays in the signature
                because it belongs to the port, and the state that owns it goes on chaining one
                cycle into the next for the calibrators that do use it.
            task: The slices to fit, already inverted, weighted and ordered by tenor.

        Returns:
            One ``SliceResult`` per slice, in the task's own order -- which is what keeps the
            result ordered by tenor, as its constructor requires.

            ``n_iterations`` is zero, which is legal and meaningful: nothing was searched.
            ``duration_ms`` is zero as well, and deliberately not measured -- reading a clock is
            the one thing this port forbids outright, and it would make two runs over the same
            recording differ. The number a consumer sees is the use case's own measurement of the
            call, taken from the injected ``Clock``, so nothing is lost by refusing to guess here.

        Raises:
            Nothing. Every arithmetic guard this fit could need is already discharged by
            ``SliceTask``: strictly positive finite volatilities and non-negative weights summing
            to a positive number make the weighted mean positive and finite, and the constructors
            of ``SVIParams`` and ``SliceResult`` check that again on the way out.
        """
        return CalibrationResult(
            slices=tuple(self._fit(one) for one in task.slices),
            n_iterations=0,
            duration_ms=0.0,
        )

    def _fit(self, task: SliceTask) -> SliceResult:
        """One expiry, answered with the volatility that minimises the weighted squared error.

        The mean *is* the least-squares answer for a constant model, so this is the honest fit of
        a flat slice rather than a shortcut past one -- which is what makes the reported RMSE the
        residual of an optimum instead of the residual of a guess.
        """
        vols = np.asarray(task.implied_vol, dtype=np.float64)
        weights = np.asarray(task.weights, dtype=np.float64)
        flat = float(np.average(vols, weights=weights))

        errors_bp = np.abs(vols - flat) * BASIS_POINTS_PER_VOL
        rmse_vol_bp = float(np.sqrt(np.average(np.square(errors_bp), weights=weights)))
        # Unweighted, unlike the RMSE: the maximum is there to expose the one quote the mean
        # absorbed, and weighting it would let a low weight hide exactly that quote. The
        # ordering invariant `max >= rmse` survives it, because a weighted mean of squares
        # cannot exceed the largest square.
        max_err_vol_bp = float(errors_bp.max())

        return SliceResult(
            expiry=task.expiry,
            tenor_years=task.tenor_years,
            # Total variance rather than volatility, which is the space SVI is written in: a flat
            # `w = vol^2 * T` re-annualises to exactly `vol` at every k, for this tenor and no
            # other.
            params=SVIParams(
                a=flat * flat * task.tenor_years,
                b=0.0,
                rho=0.0,
                m=0.0,
                sigma=FLAT_SIGMA,
            ),
            rmse_vol_bp=rmse_vol_bp,
            max_err_vol_bp=max_err_vol_bp,
            n_quotes_used=len(task.implied_vol),
            # There was no search, so there is no budget to have exhausted and no bound to have
            # been stopped at. Reporting `converged=False` would make the acceptance rule refuse
            # a fit that is exactly as good as this model can be, which is a different claim.
            converged=True,
            at_bound=False,
        )
