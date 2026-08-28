"""How Risk reads a point that is not on the grid: bilinear in total variance, flat outside.

ADR-001 gave the consumer a mesh of numbers instead of an evaluable surface, and this module is
where that trade is settled. The producer knows the exact smile -- five SVI parameters or a
network -- but that knowledge cannot cross a process boundary, cannot be written to a recording
and read back tomorrow, and cannot be serialized without shipping the producer's numerical stack
to everyone downstream. So the surface arrives sampled, and the consumer states its own
interpolation, out loud, in one module, where its limitations can be written down next to it.

**Bilinear in total variance**, fixed here in the domain rather than left to a use case
(Design 7.1). Total variance is the right quantity to interpolate in, and not merely the one
:class:`SurfaceView` happens to store: ``w`` is the accumulated quantity, additive along the
tenor axis in the sense that a calendar-arbitrage-free surface has ``w`` non-decreasing in ``T``,
so a linear step between two tenors preserves that ordering exactly. Interpolating volatilities
instead would not: two nodes whose ``w`` rises with tenor can have a *falling* vol between them,
and a straight line in vol space crosses into a total variance that dips -- a calendar arbitrage
manufactured by the consumer, out of a surface the producer's hard gate had certified clean.

Two limitations, and both are stated where they bite rather than in a footnote.

*Between the nodes, nothing guarantees the absence of arbitrage.* The producers judge
no-arbitrage on their own analytic or meshed conditions before publishing (ADR-010), and what
they certify is the surface they fitted. A bilinear patch through four of its samples is a
different function: it is piecewise flat in curvature within each cell and kinked at every node
line, so Durrleman's condition -- which is a statement about the second derivative in ``k`` --
is simply not inherited. In practice the mesh is sized so the interpolation error stays well
below the bid-ask noise, which bounds how much arbitrage can hide in a cell; it does not
eliminate it, and a greek computed by bumping across a node line will feel the kink.

*Outside the nodes, the surface goes flat*, on all four sides, and the tenor side of that is the
surprising half. Flat in **total variance** past the last tenor means ``w(T) = w(T_last)``, so
the implied volatility decays as ``sqrt(w_last / T)`` -- proportional to ``1 / sqrt(T)``. A
two-year option valued off a one-year grid is therefore priced at about 71% of the one-year vol,
not at the one-year vol. That is a conservative and defensible reading, since a flat ``w`` is the
boundary case of the calendar condition and cannot create a calendar arbitrage; it is also
emphatically not a forecast, and a book with real long-dated risk needs a grid that reaches it.
On the moneyness axis the same clamp is the ordinary one: the wing volatility stops rising past
the last quoted strike.

**Plain Python and ``bisect``, no numpy, and that is a decision rather than an oversight.** The
work here is two binary searches over axes of a few dozen entries and four multiplications. numpy
would add an array allocation per call to a function called once per position and six more times
per position's greeks, and it would bring the truth-value-of-an-array trap into a module whose
guards are all scalar comparisons -- the one trap this codebase keeps rediscovering. The domain is
allowed numpy; it earns its place in ``neural_surface``'s gate, which evaluates a surface over a
mesh of thousands of points. It does not earn it here.
"""

from __future__ import annotations

import math
from bisect import bisect_right

from volengine.risk.domain.surface_view import SurfaceView


def _bracket(axis: tuple[float, ...], value: float) -> tuple[int, int, float]:
    """Locate ``value`` on a strictly increasing axis, clamped to its ends.

    Returns the two node indices the value sits between and the weight of the second one, so that
    the caller's interpolation is a single expression with no special cases left in it. The three
    situations that would otherwise each need their own branch at every call site collapse here:

    * A single-node axis, and a value below the first node or above the last one, all return a
      degenerate bracket ``(i, i, 0.0)``. That *is* the flat extrapolation: both corners of the
      cell are the same node, so any weighted average of them is that node's value, exactly.
    * Anywhere inside, ``bisect_right`` gives the first node strictly greater than the value, so
      the weight lies in ``[0, 1)`` -- and it is exactly ``0.0`` when the value lands on a node,
      which is what makes a lookup at a grid point reproduce that point bit for bit rather than
      to within an ulp.

    The axis is assumed strictly increasing and finite; :class:`SurfaceView` guarantees both at
    construction, and re-checking here would be re-deriving an invariant the type already owns.
    """
    if value <= axis[0]:
        return 0, 0, 0.0
    last = len(axis) - 1
    if value >= axis[last]:
        return last, last, 0.0

    upper = bisect_right(axis, value)
    lower = upper - 1
    return lower, upper, (value - axis[lower]) / (axis[upper] - axis[lower])


def bilinear_total_variance(view: SurfaceView, k: float, tenor_years: float) -> float:
    """Total variance at an arbitrary point of the surface, bilinearly, flat outside the grid.

    The four corners of the cell containing ``(tenor_years, k)`` are combined with the two
    weights: linearly in ``k`` along each of the two bracketing tenors, then linearly in ``T``
    between the two results. Whether that is read as "interpolate in ``k``, then in ``T``" or the
    other way round makes no difference -- the bilinear form is symmetric in the two axes -- which
    is worth knowing, because it means no ordering convention is hiding in the implementation.

    Outside the grid the value is clamped on all four sides: the corners degenerate to an edge
    node, or to the single corner node, and the combination returns it unchanged. The module
    docstring argues both halves of that choice, and the tenor half is the one to read: flat
    ``w`` past the last node is a volatility falling as ``1 / sqrt(T)``, not a flat volatility.

    The result is strictly positive without needing a guard: every node of the grid is strictly
    positive by construction and the four weights are a convex combination, so the value is
    bounded below by the smallest corner. That is exactly the property
    :func:`implied_vol` needs to take a square root, and it is why the grid's positivity invariant
    lives on the type rather than being re-checked per lookup.

    Args:
        view: The surface to read. Its axes carry every invariant this function assumes.
        k: Log-moneyness, ``ln(K / F)``. Finite; ``0.0`` is at-the-money forward and perfectly
            ordinary, so it is never tested for truthiness.
        tenor_years: Year fraction, strictly positive and finite. Normally
            :meth:`SurfaceView.tenor_of`'s output.

    Returns:
        Total variance ``w = vol**2 * T`` at that point, strictly positive.

    Raises:
        ValueError: If ``k`` is not finite, or ``tenor_years`` is not positive and finite. Both
            non-finite cases have to be refused at the door, and they fail differently, which is
            why neither can be left to the lookup. An infinity satisfies the clamp test, so it
            would sail through :func:`_bracket` and come back as the edge of the grid -- a
            perfectly plausible wing volatility, reported for a coordinate that does not exist. A
            NaN compares ``False`` against every bound instead, so it misses both clamps, drives
            ``bisect_right`` to the far end of the axis and indexes one past it: an ``IndexError``
            raised from inside a binary search, with nothing in the message naming the tenor the
            caller actually passed.
    """
    if not math.isfinite(k):
        raise ValueError(f"The log-moneyness must be finite, got {k}")
    if not math.isfinite(tenor_years) or tenor_years <= 0:
        raise ValueError(f"The tenor in years must be positive and finite, got {tenor_years}")

    near, far, weight_tenor = _bracket(view.tenors, tenor_years)
    low, high, weight_k = _bracket(view.log_moneyness, k)

    near_row = view.total_variance[near]
    far_row = view.total_variance[far]
    w_near = (1.0 - weight_k) * near_row[low] + weight_k * near_row[high]
    w_far = (1.0 - weight_k) * far_row[low] + weight_k * far_row[high]
    return (1.0 - weight_tenor) * w_near + weight_tenor * w_far


def implied_vol(view: SurfaceView, k: float, tenor_years: float) -> float:
    """The annualised volatility the surface implies at that point: ``sqrt(w / T)``.

    The inverse of the conversion the ACL performed once on the way in, applied at the one place
    where a volatility is genuinely needed -- pricing a position. Doing it here rather than
    storing vols is what keeps :class:`SurfaceView` a single source of truth and keeps the
    interpolation itself in the space it is correct in; see the module docstring on why a straight
    line in vol space can manufacture a calendar arbitrage that a straight line in ``w`` cannot.

    ``T`` here is the **requested** tenor, never the clamped one, and the difference is the whole
    of the ``1 / sqrt(T)`` decay past the last node: ``w`` stops growing at the edge of the grid
    while the divisor keeps going. Clamping the divisor too would flatten the volatility instead,
    which reads more natural and is a strictly worse answer -- it makes total variance grow
    linearly out to any maturity on the strength of a grid that ends years earlier.

    Args:
        view: The surface to read.
        k: Log-moneyness, finite.
        tenor_years: Year fraction, strictly positive and finite.

    Returns:
        The implied volatility in absolute terms -- ``0.65`` is 65% -- strictly positive.

    Raises:
        ValueError: If ``k`` or ``tenor_years`` is inadmissible, or -- and this one should be
            unreachable -- if the result is not finite. The guard is not redundant despite the
            argument that positive ``w`` over positive ``T`` is a positive square root: a
            volatility is the value that gets multiplied into a price and then into every greek,
            and a NaN there survives every ordering guard downstream (``nan <= 0`` is ``False``)
            to be reported as a number somebody trades on. This module refuses to be the place
            that let one through.
    """
    total_variance = bilinear_total_variance(view, k, tenor_years)
    vol = math.sqrt(total_variance / tenor_years)
    if not math.isfinite(vol) or vol <= 0:
        raise ValueError(
            f"The interpolated volatility must be positive and finite, got {vol} from a total "
            f"variance of {total_variance} at a tenor of {tenor_years}"
        )
    return vol
