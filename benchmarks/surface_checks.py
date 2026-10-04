"""No-arbitrage measured on a *published* grid, with one ruler for every producer.

The benchmark's "violations" column needs a number that means the same thing for the SVI fit and
for the network, and neither producer's own measurement qualifies. The parametric context checks
Durrleman's condition analytically on each fitted slice (``parametric_pricing/domain/durrleman``)
and publishes no metric of it; the neural context checks it by finite differences on its own
``[neural.mesh]`` (``neural.butterfly_violation``), which is wider than the published grid on
purpose. Comparing those two numbers would compare two rulers.

What both producers share is the contract: a ``VolGrid`` on the same nodes (both take
``[calibration.grid]``). So the check here runs on that -- total variance ``w = vol^2 * T`` per
node, Durrleman's ``g`` by central differences along each tenor row, and the calendar condition
``w(T_near) <= w(T_far)`` node by node between consecutive rows -- using the neural context's
finite-difference ``durrleman_g``, because it is the one that takes sampled ``w`` rather than SVI
parameters. It is what a consumer holding only the contract could verify, which is the point.

**Coarser than either producer's own check, and that is stated rather than hidden.** The published
mesh is 17 nodes over ``[-0.4, 0.4]`` in the shipped examples; the outermost node of each row is
consumed by the stencil and never judged, and a breach narrower than the node spacing is invisible.
A zero here is "no breach a grid consumer can see", not a proof.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import pairwise

import numpy as np

from volengine.contracts.calibrated_surface import VolGrid
from volengine.neural_surface.domain.invariants import durrleman_g


@dataclass(frozen=True, slots=True)
class GridArbitrage:
    """How far one published grid is from arbitrage-free, in the units the gate reports in.

    Both are non-negative, and zero means no breach on the grid's nodes.
    """

    butterfly: float
    """The deepest value of Durrleman's ``g`` below zero, over every judged node of every row."""

    calendar: float
    """The largest excess of a nearer row's total variance over the next row's, at any node."""

    def __post_init__(self) -> None:
        for name, value in (("butterfly", self.butterfly), ("calendar", self.calendar)):
            if not np.isfinite(value) or value < 0.0:
                raise ValueError(f"A {name} violation is finite and non-negative, got {value}")


def grid_arbitrage(grid: VolGrid) -> GridArbitrage:
    """Measure both no-arbitrage conditions on one published grid.

    Raises:
        ValueError: If the moneyness axis is not uniform with at least three nodes (the stencil
            needs both), or a total variance comes out non-finite or non-positive -- which a
            contract that admitted the grid already rules out, so it would be a bug here.
    """
    k = np.asarray(grid.log_moneyness, dtype=np.float64)
    vols = np.asarray(grid.vols, dtype=np.float64)
    tenors = np.asarray(grid.tenors, dtype=np.float64)
    w = vols * vols * tenors[:, np.newaxis]
    # Finiteness before sign, joined with `or`: NaN passes every ordering guard (CLAUDE.md).
    if not np.all(np.isfinite(w)) or np.any(w <= 0.0):
        raise ValueError("Total variance must be finite and positive at every node")

    depth = 0.0
    for row in w:
        g = durrleman_g(row, k)
        depth = max(depth, float(-np.min(g)))
    crossing = 0.0
    for near, far in pairwise(w):
        crossing = max(crossing, float(np.max(near - far)))
    return GridArbitrage(butterfly=max(depth, 0.0), calendar=max(crossing, 0.0))
