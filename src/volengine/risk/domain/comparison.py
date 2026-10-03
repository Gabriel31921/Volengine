"""One book, two producers: what changes when the surface does (Design 7.3).

The engine runs two calibrators behind one contract, and this is where that arrangement is cashed
in. Each producer's surface values the same portfolio through the same interpolation, the same
pricer and the same bumps, so whatever differs between the two reports is the surfaces and
nothing else. That is the measurement, and the types below are its shape.

Three types and one function:

* ``PositionComparison`` -- one position valued twice, and the differences.
* ``SurfaceDistance`` -- how far apart the two surfaces are as volatilities, with no book at all.
* ``ComparativeReport`` -- both reports whole, the paired lines, and the distance.
* :func:`surface_distance` -- the computation behind the second.

**Every difference is challenger minus baseline.** One sign convention, stated once: "baseline"
is the producer the comparison is measured *from* -- ordinarily the parametric one, the engine's
reference -- and a positive ``value_diff`` means the challenger values the line higher. Nothing
else in the context distinguishes the two roles, and nothing here branches on which producer
fills either.

**The two reports are kept whole rather than reduced to their differences.** A difference of
``0.0`` is the answer to "do they agree?" and also the answer a comparison of two refusals would
give, so the verdicts, the instants and the messages travel with the numbers. A comparison is as
honest as the two reports under it, and it cannot be more honest than that.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from itertools import product

from volengine.risk.domain.freshness_policy import FreshnessDecision
from volengine.risk.domain.interpolation import implied_vol
from volengine.risk.domain.risk_report import PositionRisk, RiskReport
from volengine.risk.domain.surface_view import SurfaceView

BASIS_POINTS_PER_UNIT = 10_000.0
"""A volatility of ``0.0001`` is one basis point. The unit every RMSE in the engine is quoted in,
so the distance between two producers reads on the same scale as each producer's own fit."""


@dataclass(frozen=True, slots=True)
class PositionComparison:
    """One position, valued off both surfaces.

    The differences are properties rather than fields: they are derived, and storing them would
    create a second copy of a subtraction with nothing keeping it in agreement with the two lines
    it came from.
    """

    baseline: PositionRisk
    """The line as the baseline producer's surface values it."""

    challenger: PositionRisk
    """The same position, off the challenger's surface."""

    def __post_init__(self) -> None:
        """Refuse to pair two different positions.

        The whole object is a statement that these two lines describe one exposure; a pairing
        bug upstream would otherwise publish a "difference" between a call and a put as though it
        were a disagreement between two models.
        """
        if self.baseline.position != self.challenger.position:
            raise ValueError(
                "A comparison pairs one position with itself, got "
                f"{self.baseline.position} against {self.challenger.position}"
            )

    @property
    def vol_diff(self) -> float:
        """Challenger minus baseline implied volatility at the position's own point."""
        return self.challenger.vol - self.baseline.vol

    @property
    def value_diff(self) -> float:
        """Challenger minus baseline position value, quantity included."""
        return self.challenger.value - self.baseline.value

    @property
    def delta_diff(self) -> float:
        """Challenger minus baseline delta, quantity included."""
        return self.challenger.delta - self.baseline.delta

    @property
    def gamma_diff(self) -> float:
        """Challenger minus baseline gamma, quantity included."""
        return self.challenger.gamma - self.baseline.gamma

    @property
    def vega_diff(self) -> float:
        """Challenger minus baseline vega, quantity included."""
        return self.challenger.vega - self.baseline.vega


@dataclass(frozen=True, slots=True)
class SurfaceDistance:
    """How far apart two surfaces are, in basis points of volatility, over the region both cover.

    The neural <-> parametric distance of Design 6.5, measured here because this is the only
    context that holds both surfaces (rule 6 keeps either producer from seeing the other's).
    """

    rms_vol_bp: float
    """Root mean square of the pointwise volatility differences. Non-negative and finite."""

    max_vol_bp: float
    """The largest pointwise difference, in absolute value. At least :attr:`rms_vol_bp`."""

    n_points: int
    """How many points the two were compared at. At least one -- no overlap is ``None``, not a
    distance of zero, because "we could not look" is not "we looked and they agree"."""

    def __post_init__(self) -> None:
        if not math.isfinite(self.rms_vol_bp) or self.rms_vol_bp < 0:
            raise ValueError(
                f"The RMS distance must be finite and non-negative, got {self.rms_vol_bp}"
            )
        if not math.isfinite(self.max_vol_bp) or self.max_vol_bp < self.rms_vol_bp:
            raise ValueError(
                f"The maximum distance must be finite and at least the RMS, got {self.max_vol_bp} "
                f"against an RMS of {self.rms_vol_bp}"
            )
        if self.n_points < 1:
            raise ValueError(f"A distance needs at least one point, got {self.n_points}")


def surface_distance(baseline: SurfaceView, challenger: SurfaceView) -> SurfaceDistance | None:
    """Compare two surfaces at every grid node of either that lies inside the other.

    **The points are the union of both grids' nodes, restricted to the box both cover.** Each
    surface is exact at its own nodes and interpolated elsewhere, so comparing only at the
    baseline's nodes would measure the challenger's interpolation error and never the
    baseline's; the union is symmetric, so swapping the roles gives the same number. Points
    outside the other surface's box are left out rather than compared against its flat
    extrapolation: past the last tenor that is a volatility decaying as ``1 / sqrt(T)``
    (``interpolation``'s module docstring), and a distance dominated by the consumer's own
    clamp would say nothing about either producer. Two grids built from one ``GridSpec`` share
    their nodes and the union collapses to one copy of each.

    Both surfaces are read through this context's own interpolation, :func:`implied_vol`, and at
    the same ``(k, T)`` -- tenor in years from each surface's own snapshot instant. When the two
    rest on different snapshots that is a comparison at equal time-to-expiry rather than at equal
    expiry, which is the right reading for "do these models agree about the shape of the smile"
    and is why the report keeps both instants beside the number.

    Returns:
        The distance, or ``None`` when the two boxes do not overlap at any node.
    """
    points = sorted(
        {
            (k, tenor)
            for view, other in ((baseline, challenger), (challenger, baseline))
            for tenor, k in product(view.tenors, view.log_moneyness)
            if _covers(other, k, tenor)
        }
    )
    if not points:
        return None

    gaps = [
        abs(implied_vol(challenger, k, tenor) - implied_vol(baseline, k, tenor))
        * BASIS_POINTS_PER_UNIT
        for k, tenor in points
    ]
    return SurfaceDistance(
        rms_vol_bp=math.sqrt(math.fsum(gap * gap for gap in gaps) / len(gaps)),
        max_vol_bp=max(gaps),
        n_points=len(gaps),
    )


def _covers(view: SurfaceView, k: float, tenor: float) -> bool:
    """Whether ``(k, tenor)`` lies inside the box of this surface's nodes, edges included."""
    return (
        view.log_moneyness[0] <= k <= view.log_moneyness[-1]
        and view.tenors[0] <= tenor <= view.tenors[-1]
    )


@dataclass(frozen=True, slots=True)
class ComparativeReport:
    """The same book under two producers, line by line, with the distance between the surfaces.

    **Lines exist only when both reports have numbers.** A comparison against a refusal is not a
    difference -- it is one report -- so if either side is ``REJECT`` the lines are empty and the
    distance is absent, and the two reports say why. When both have numbers, a position one
    report valued and the other skipped (it expired between the two snapshots) has no line
    either: the pairing is of exposures valued twice.
    """

    market_id: str
    """The market both reports price."""

    baseline: RiskReport
    """The reference producer's report, whole."""

    challenger: RiskReport
    """The other producer's report, whole."""

    lines: tuple[PositionComparison, ...]
    """One per position both reports valued, in the portfolio's order."""

    distance: SurfaceDistance | None
    """How far apart the surfaces are, or ``None`` when there is nothing to compare."""

    def __post_init__(self) -> None:
        if not self.market_id:
            raise ValueError("A comparative report must name its market")
        if self.baseline.market_id != self.market_id or self.challenger.market_id != self.market_id:
            raise ValueError(
                f"Both reports must price {self.market_id}, got {self.baseline.market_id} and "
                f"{self.challenger.market_id}"
            )
        if self.baseline.producer_id == self.challenger.producer_id:
            raise ValueError(
                f"A comparison needs two producers, got {self.baseline.producer_id} twice"
            )
        if not self.comparable and (self.lines or self.distance is not None):
            raise ValueError("A comparison against a rejected report carries no lines or distance")

    @property
    def comparable(self) -> bool:
        """Whether both reports carry numbers, which is when a difference means anything."""
        return (
            self.baseline.freshness is not FreshnessDecision.REJECT
            and self.challenger.freshness is not FreshnessDecision.REJECT
        )

    @property
    def total_value_diff(self) -> float:
        """Challenger minus baseline, summed over the paired lines. ``0.0`` when there are none.

        Over the *paired* lines, not the two reports' totals: a position only one side valued
        would otherwise enter the difference at its full value and read as model disagreement.
        """
        return math.fsum(line.value_diff for line in self.lines)


def pair_lines(baseline: RiskReport, challenger: RiskReport) -> tuple[PositionComparison, ...]:
    """Match the two reports' lines position by position, keeping the portfolio's order.

    Both reports list their lines in the portfolio's order and each skips only expired
    positions, so each is a subsequence of the same book. Equal positions are legal in a
    portfolio and are matched in order, which is exact: two equal positions meet the same
    surface identically, so either both or neither of them appear in a given report.
    """
    lines: list[PositionComparison] = []
    remaining = list(challenger.positions)
    for line in baseline.positions:
        for index, other in enumerate(remaining):
            if other.position == line.position:
                lines.append(PositionComparison(baseline=line, challenger=other))
                del remaining[: index + 1]
                break
    return tuple(lines)
