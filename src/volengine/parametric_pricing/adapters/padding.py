"""One calibration task, laid out on a grid whose shape never changes (ADR-009).

``jax.jit`` compiles one specialisation per input **shape**, and a recompilation costs hundreds of
milliseconds to seconds. The composition of an option chain changes constantly -- strikes are born
and die as the underlying moves, expiries roll off -- so arrays shaped to today's chain would
recompile on nearly every snapshot. That is the failure ADR-009 exists to prevent, and this module
is the whole of the mechanism: every task, whatever it holds, becomes the same rectangle of
``max_slices`` by ``max_quotes`` plus a boolean mask saying which cells are real.

**No JAX here, deliberately.** Padding is bookkeeping over numpy arrays; it has no derivative and
nothing to compile. Keeping it in a module the optional extra cannot reach means the layout can be
tested -- overflow, ordering, the inertness of a padded cell -- on any machine, including the CI
leg that installs no JAX at all.

**What is a shape and what is a threshold.** Everything that changes the compiled signature lives
in :class:`PadShape`: the two grid dimensions and the number of nodes on the butterfly mesh. The
mesh *margin* does not -- it moves where the nodes sit, not how many there are -- so it stays with
the fit settings that own every other empirical number. Getting that split wrong is how a knob in a
TOML file quietly becomes a recompilation on start-up.

**Padded cells carry benign numbers, never NaN**, and that is not tidiness. The loss is written
``where(mask, err**2, 0)``, whose *value* ignores the padding -- but reverse-mode AD does not:
the gradient of ``where`` multiplies the untaken branch by zero, and ``0 * nan`` is ``nan``, so a
single NaN parked under the mask poisons every parameter of every slice. The fill values below are
chosen to be arithmetically harmless (a zero log-moneyness, a unit volatility, a unit tenor, a
zero weight), which is what makes the inert-mask property test in this stage assert something
about the loss rather than about the fill.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from volengine.parametric_pricing.domain.calibration import CalibrationTask
from volengine.parametric_pricing.domain.errors import CalibrationError

FILL_LOG_MONEYNESS: float = 0.0
"""Log-moneyness written into a padded cell: at the money forward, the most ordinary value there
is. Any finite number would do for the value of the loss; what matters is that it is finite, so
that the model evaluated there produces a real gradient contribution which the mask then zeroes."""

FILL_IMPLIED_VOL: float = 1.0
"""Volatility written into a padded cell. Positive, because the fit divides by a total variance
and takes its square root: a zero would put the model on the one point where ``d sqrt(w) / dw`` is
infinite, and an infinity under the mask is the same poison a NaN is."""

FILL_TENOR_YEARS: float = 1.0
"""Tenor of a padded *slice*. One year rather than zero for the same reason: the annualisation
divides by it."""

FILL_WEIGHT: float = 0.0
"""Weight of a padded cell. The zero that makes the padding inert in the value of the loss, in
addition to -- not instead of -- the mask."""


@dataclass(frozen=True, slots=True)
class PadShape:
    """The rectangle every task is fitted on. **This is the compiled signature.**

    Sized with margin against the real chain, in ADR-009's own words: a Deribit BTC book quotes
    on the order of forty strikes at a liquid expiry and carries a dozen expiries at once, so
    sixty-four by sixteen leaves room for a busy session without ever being reached. The cost of
    the reserve is arithmetic on cells the mask discards, which at this size is nothing.

    Frozen and hashable so that the compiled functions can be cached against it: two calibrators
    built with the same shape and the same settings share one compilation, which is what keeps a
    test suite that constructs a dozen of them from paying for a dozen compilations.
    """

    max_slices: int = 16
    """Expiries the grid reserves. A task with more is refused rather than truncated -- see
    :func:`pad`."""

    max_quotes: int = 64
    """Quotes per expiry the grid reserves."""

    mesh_nodes: int = 51
    """Points on the butterfly penalty mesh of each slice. At least two.

    A shape, not a threshold, which is why it lives here and the margin does not. Fifty-one across
    a band a little over two units wide puts a node every four percent of log-moneyness, finer than
    any feature a five-parameter form can produce.
    """

    def __post_init__(self) -> None:
        if self.max_slices < 1:
            raise ValueError(f"The grid must reserve at least one slice, got {self.max_slices}")
        if self.max_quotes < 1:
            raise ValueError(f"The grid must reserve at least one quote, got {self.max_quotes}")
        if self.mesh_nodes < 2:
            raise ValueError(f"The penalty mesh needs at least two nodes, got {self.mesh_nodes}")


@dataclass(frozen=True, slots=True)
class PaddedTask:
    """A task as rectangles: ``(max_slices, max_quotes)`` for the quotes, ``(max_slices,)`` for the
    expiries, and one boolean mask for each.

    Parallel arrays rather than objects, and index ``[s, q]`` of every quote-shaped array describes
    one quote of one expiry. That is the same parallelism ``SliceTask`` already imposes, carried one
    step further: what arrives here is what a batched loss consumes directly, with no gather and no
    reshape between the boundary and the arithmetic.

    Everything is ``float64`` on the way in. The fit itself runs in the precision the calibrator
    chooses; this type is the layout, not the arithmetic.
    """

    log_moneyness: NDArray[np.float64]
    """``k`` per quote, real cells first, :data:`FILL_LOG_MONEYNESS` after. Shape ``(S, Q)``."""

    implied_vol: NDArray[np.float64]
    """The market volatility the fit is trying to reproduce. Shape ``(S, Q)``."""

    weights: NDArray[np.float64]
    """Influence per quote, **renormalised to sum to one over each active slice**. Shape ``(S, Q)``.

    Renormalised here rather than assumed: ADR-018's weights arrive normalised, and a loss that
    depended on that holding would make the Huber threshold -- "an error of this many basis points
    stops being ordinary noise" -- mean something different the day a task was built by hand with
    unnormalised weights. One division at the boundary buys the guarantee outright.

    A padded cell weighs :data:`FILL_WEIGHT`, and so does a real quote the ACL flagged: zero weight
    is how a suspect quote stays visible in the residuals without steering the fit.
    """

    quote_mask: NDArray[np.bool_]
    """``True`` where the cell is a real quote. Shape ``(S, Q)``."""

    tenor_years: NDArray[np.float64]
    """Year fraction per expiry, :data:`FILL_TENOR_YEARS` on a padded row. Shape ``(S,)``."""

    slice_mask: NDArray[np.bool_]
    """``True`` where the row is a real expiry. Shape ``(S,)``."""

    mesh: NDArray[np.float64]
    """Log-moneyness nodes the butterfly penalty is evaluated on, per slice. Shape ``(S, M)``.

    Built here rather than inside the fit because it depends only on the quoted band and a
    configured margin, so it is data the compiled function receives rather than arithmetic it
    repeats on every iteration. A padded row gets the same nodes as a real one would over a unit
    band: finite, harmless, and discarded by the slice mask.
    """

    n_slices: int
    """How many rows of the rectangle are real. Between one and ``max_slices``."""

    @property
    def shape(self) -> tuple[int, int]:
        """``(max_slices, max_quotes)``, the pair that decides which compilation is used."""
        rows, columns = self.log_moneyness.shape
        return int(rows), int(columns)


def pad(task: CalibrationTask, shape: PadShape, mesh_margin: float) -> PaddedTask:
    """Lay one task out on the fixed grid.

    Args:
        task: The slices to fit, already ordered by tenor and already inverted into ``(k, vol)``
            space by the ACL.
        shape: The rectangle to fill. The same one for every snapshot of a session, which is the
            entire point.
        mesh_margin: How far past the outermost quote of a slice its butterfly mesh reaches, in
            log-moneyness. Non-negative and finite. Not zero in practice: the published grid is
            wider than the quoted band (ADR-001), so the wings a consumer prices against are
            extrapolation -- and extrapolated wings are exactly where raw SVI produces a negative
            density.

    Returns:
        The padded layout, with every real value in the leading cells of its row.

    Raises:
        CalibrationError: If the task does not fit the reserved rectangle -- more expiries than
            ``max_slices``, or more quotes than ``max_quotes`` at some expiry.

            Refused rather than truncated, and that is the decision. Truncating would drop quotes
            the venue really published, silently, and the surface would still be released as
            healthy; refusing makes the use case republish the last good surface as
            ``STALE_REPUBLISH`` (ADR-006) with a message naming the number the grid would need.
            Growing the reservation is a restart, because it is a new compilation -- which is what
            ADR-009 says handling ``ChainCompositionChanged`` will have to mean, and that handling
            is a seam this stage leaves open (``docs/SEAMS.md``).

        ValueError: If ``mesh_margin`` is not a non-negative finite number.
    """
    if not np.isfinite(mesh_margin) or mesh_margin < 0:
        raise ValueError(f"The mesh margin must be non-negative and finite, got {mesh_margin}")

    n_slices = len(task.slices)
    if n_slices > shape.max_slices:
        raise CalibrationError(
            f"The padded grid reserves {shape.max_slices} expiries and the task carries "
            f"{n_slices}; widen the reservation and restart"
        )
    for one in task.slices:
        if len(one.log_moneyness) > shape.max_quotes:
            raise CalibrationError(
                f"The padded grid reserves {shape.max_quotes} quotes per expiry and expiry "
                f"{one.expiry.isoformat()} carries {len(one.log_moneyness)}; widen the "
                "reservation and restart"
            )

    rows, columns = shape.max_slices, shape.max_quotes
    log_moneyness = np.full((rows, columns), FILL_LOG_MONEYNESS, dtype=np.float64)
    implied_vol = np.full((rows, columns), FILL_IMPLIED_VOL, dtype=np.float64)
    weights = np.full((rows, columns), FILL_WEIGHT, dtype=np.float64)
    quote_mask = np.zeros((rows, columns), dtype=np.bool_)
    tenor_years = np.full(rows, FILL_TENOR_YEARS, dtype=np.float64)
    slice_mask = np.zeros(rows, dtype=np.bool_)
    mesh = np.tile(np.linspace(-1.0, 1.0, shape.mesh_nodes), (rows, 1))

    for index, one in enumerate(task.slices):
        width = len(one.log_moneyness)
        k = np.asarray(one.log_moneyness, dtype=np.float64)
        raw = np.asarray(one.weights, dtype=np.float64)

        log_moneyness[index, :width] = k
        implied_vol[index, :width] = np.asarray(one.implied_vol, dtype=np.float64)
        # `SliceTask` already refuses a tuple that sums to zero, so this division is safe --
        # and it is a division rather than an assumption, which is what the docstring on
        # `PaddedTask.weights` argues for.
        weights[index, :width] = raw / raw.sum()
        quote_mask[index, :width] = True
        tenor_years[index] = one.tenor_years
        slice_mask[index] = True
        mesh[index] = np.linspace(
            float(k.min()) - mesh_margin, float(k.max()) + mesh_margin, shape.mesh_nodes
        )

    return PaddedTask(
        log_moneyness=log_moneyness,
        implied_vol=implied_vol,
        weights=weights,
        quote_mask=quote_mask,
        tenor_years=tenor_years,
        slice_mask=slice_mask,
        mesh=mesh,
        n_slices=n_slices,
    )
