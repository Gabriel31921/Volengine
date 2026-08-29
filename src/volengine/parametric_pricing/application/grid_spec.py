"""The mesh a fitted surface is published on. Configuration, not mathematics.

ADR-001 publishes a grid of already-evaluated volatilities rather than an evaluable object, so
somebody has to choose where the nodes go. That choice is not the calibrator's: the same fit
published on a coarse mesh and on a fine one is the same fit, and the mesh is a trade between
message size and the interpolation error a consumer inherits between nodes. It is therefore
TOML configuration (ADR-012) and it lives here, in the application layer, beside the ACL that
uses it -- the domain never sees it, because a mesh is a property of the published language.

**Only the moneyness axis is configured.** The tenor axis is whatever the market quoted: one
node per fitted slice, with the expiry and the forward that slice was fitted with. Choosing
tenors independently would mean evaluating an SVI slice at a tenor it was not fitted at, which
raw SVI has no way to do -- each slice is its own parameterisation (ADR-008) and there is nothing
to interpolate *between* them that would not be a modelling decision the consumer is better
placed to make with the grid in hand.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class GridSpec:
    """Where the published moneyness nodes sit.

    A uniform mesh in log-forward-moneyness, which is the axis the whole engine is measured in:
    uniform in ``k`` means uniform in *relative* strike, so the nodes fall in the same places on a
    60,000 coin and a 5,000 index and the published grids are directly comparable.
    """

    k_min: float
    """Lowest published log-moneyness, ``ln(K / F)``. Finite, strictly below ``k_max``.

    Negative in every sensible configuration: it is the low-strike wing, where the puts are.
    """

    k_max: float
    """Highest published log-moneyness. Finite, strictly above ``k_min``."""

    n_nodes: int
    """How many nodes span ``[k_min, k_max]``, endpoints included. At least two.

    Two is the degenerate minimum that still spans an interval, and it is legal so that a test or
    a walking skeleton can publish something tiny. One node would be a grid with no width at all,
    on which every consumer's interpolation would return the same volatility for every strike --
    a surface that constructs cleanly and describes nothing.

    The interpolation error a consumer inherits falls as the square of the spacing, and Design's
    requirement is that it stay well below the market's bid-ask noise. That is a judgement about
    a market rather than about this code, which is exactly why it is configuration.
    """

    def __post_init__(self) -> None:
        # Finiteness first, and the bad cases joined with `or`: `float("nan") < x` is `False`, so
        # a NaN bound written the other way round would pass the ordering test and then produce a
        # step of NaN, an axis of NaN, and a grid the contract accepts as "finite" nowhere.
        if not math.isfinite(self.k_min) or not math.isfinite(self.k_max):
            raise ValueError(
                f"The moneyness bounds must be finite, got ({self.k_min}, {self.k_max})"
            )
        if self.k_min >= self.k_max:
            raise ValueError(
                f"The moneyness range must be non-empty, got k_min={self.k_min} "
                f"not below k_max={self.k_max}"
            )
        if self.n_nodes < 2:
            raise ValueError(f"The grid needs at least two nodes, got {self.n_nodes}")

    def nodes(self) -> tuple[float, ...]:
        """The moneyness axis: ``n_nodes`` points from ``k_min`` to ``k_max`` inclusive.

        Written as ``k_min + step * i`` rather than by accumulating ``step``, because accumulation
        drifts: adding a step two hundred times lands somewhere near ``k_max`` rather than on it,
        and the published axis has to be strictly increasing and to end where the configuration
        says it ends. The last node is assigned exactly rather than computed, so the endpoint is
        the configured number bit for bit and no rounding can make it fall short of its neighbour.

        Plain Python arithmetic and no numpy: ``VolGrid`` carries nested tuples of floats
        (ADR-011), so an array here would only be converted straight back.
        """
        step = (self.k_max - self.k_min) / (self.n_nodes - 1)
        axis = [self.k_min + step * index for index in range(self.n_nodes - 1)]
        axis.append(self.k_max)
        return tuple(axis)
