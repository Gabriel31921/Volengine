from __future__ import annotations

import numpy as np
import pytest
from numpy.typing import NDArray

from tests.neural_surface.builders import (
    FAR_TENOR,
    MESH_K,
    NEAR_TENOR,
    CallableSurface,
    FlatVolSurface,
    TransposedSurface,
    make_mesh,
    make_report,
)
from tests.support import replace_field
from volengine.neural_surface.domain.errors import SurfaceEvaluationError
from volengine.neural_surface.domain.invariants import (
    ArbitrageMesh,
    ArbitrageReport,
    check_surface,
    durrleman_g,
)

FLAT_W = 0.65 * 0.65 * NEAR_TENOR
"""Total variance of the flat 65% surface at one month, the row every structural test bends."""


def _fine_mesh(n_points: int) -> tuple[float, ...]:
    """A uniform mesh across the same +-60% band, with as many points as a test needs.

    Built with ``linspace`` and converted point by point: the mesh has to satisfy the uniformity
    the gate enforces, and ``linspace`` drifts by about ``1e-13`` relatively over this band, which
    is four orders of magnitude inside the tolerance.
    """
    return tuple(float(k) for k in np.linspace(-0.60, 0.60, n_points))


# --- An SVI-shaped surface with a closed form, used as the independent answer.
# The gate has no analytic derivative of its own by construction, so its differences have to be
# pinned against a curve whose derivatives are known exactly. Raw SVI is the obvious such curve --
# and writing the five-line formula out here rather than importing the parametric context's is
# rule 6 in practice: this file may not reach across the boundary, and would not want to, since
# the whole point is an answer computed by algebra that shares nothing with the code under test.

SVI_A = 0.04
SVI_B = 0.80
SVI_RHO = -0.90
SVI_M = 0.0
SVI_SIGMA = 0.15
"""Steep wings around a tight bottom: a genuine butterfly violation, deep enough to be measured
rather than merely detected. ``min(g)`` is about -0.53 across the band."""


def _svi_total_variance(k: NDArray[np.float64]) -> NDArray[np.float64]:
    """``w = a + b (rho y + sqrt(y^2 + sigma^2))``, the curve the surfaces below sample."""
    y = k - SVI_M
    return SVI_A + SVI_B * (SVI_RHO * y + np.sqrt(y * y + SVI_SIGMA * SVI_SIGMA))


def _svi_analytic_g(k: NDArray[np.float64]) -> NDArray[np.float64]:
    """Durrleman's function of that curve, with ``w'`` and ``w''`` differentiated by hand."""
    y = k - SVI_M
    r = np.sqrt(y * y + SVI_SIGMA * SVI_SIGMA)
    w = SVI_A + SVI_B * (SVI_RHO * y + r)
    w_prime = SVI_B * (SVI_RHO + y / r)
    w_second = SVI_B * SVI_SIGMA * SVI_SIGMA / (r * r * r)
    return (
        (1.0 - k * w_prime / (2.0 * w)) ** 2
        - (w_prime * w_prime / 4.0) * (1.0 / w + 0.25)
        + w_second / 2.0
    )


def _svi_surface() -> CallableSurface:
    """The same smile at every tenor: butterfly-violating, and calendar-clean by construction.

    Total variance flat in ``T`` is weakly non-decreasing, which is the calendar condition
    satisfied with nothing to spare, so any calendar number this surface produces is a bug rather
    than a feature of the test data.
    """
    return CallableSurface(fn=lambda k, t: _svi_total_variance(k))


def _bumped_surface(centre: float, width: float, amplitude: float) -> CallableSurface:
    """A flat surface with one narrow Gaussian bump added to its total variance.

    A bump of width ``width`` has a curvature of about ``-2 * amplitude / width^2`` at its peak,
    so a small, narrow one drives ``w''`` violently negative over a band far shorter than a coarse
    mesh's spacing. That is the shape of the sampling caveat made concrete: real, local, and
    invisible to a mesh that does not step inside it.
    """

    def fn(k: NDArray[np.float64], t: NDArray[np.float64]) -> NDArray[np.float64]:
        return 0.65 * 0.65 * t + amplitude * np.exp(-(((k - centre) / width) ** 2))

    return CallableSurface(fn=fn)


# --- ArbitrageMesh


def test_mesh_rejects_a_non_uniformly_spaced_moneyness_axis() -> None:
    """The instinctive mesh -- fine at the money, coarse in the wings -- is exactly what is
    refused.

    Central differences are second-order accurate only on an equidistant stencil; on an unequal
    one the first-order error terms stop cancelling and w' quietly degrades to O(h), which shows
    up as a small positive or negative g in precisely the regime the gate has to be trusted in.
    """
    with pytest.raises(ValueError, match="uniformly spaced"):
        make_mesh(log_moneyness=(-0.60, -0.10, 0.0, 0.10, 0.60))


def test_mesh_accepts_an_axis_assembled_in_floating_point() -> None:
    """The tolerance has to survive the rounding of building a mesh at all, not just exact steps.

    Guards the point: the builders' mesh really does have spacings that differ from each other,
    so the test above is passing on a tolerance that is doing work rather than on an axis whose
    differences happen to be bit-identical.
    """
    spacings = np.diff(np.asarray(MESH_K))
    assert not np.all(spacings == spacings[0])

    assert make_mesh().k_array.shape == (25,)


def test_mesh_rejects_fewer_than_three_moneyness_points() -> None:
    """Two points have no interior, and only interior points are ever judged."""
    with pytest.raises(ValueError, match="at least three points"):
        make_mesh(log_moneyness=(-0.05, 0.05))


def test_mesh_rejects_a_moneyness_axis_running_backwards() -> None:
    with pytest.raises(ValueError, match="strictly increasing"):
        make_mesh(log_moneyness=(0.10, 0.05, 0.0))


def test_mesh_rejects_a_repeated_moneyness_point() -> None:
    """A zero spacing makes the second difference infinite, not merely inaccurate."""
    with pytest.raises(ValueError, match="strictly increasing"):
        make_mesh(log_moneyness=(-0.05, 0.0, 0.0, 0.05))


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_mesh_rejects_a_non_finite_moneyness_point(bad: float) -> None:
    """Finiteness is checked before the ordering: nan compares False against every bound."""
    with pytest.raises(ValueError, match="finite"):
        make_mesh(log_moneyness=(-0.05, bad, 0.05))


def test_mesh_accepts_a_single_tenor() -> None:
    """One expiry is a real market, and the butterfly condition says everything it has to say."""
    assert make_mesh(tenors=(NEAR_TENOR,)).tenor_array.shape == (1,)


def test_mesh_rejects_an_empty_tenor_axis() -> None:
    with pytest.raises(ValueError, match="at least one tenor"):
        make_mesh(tenors=())


@pytest.mark.parametrize("bad", [0.0, -0.25, float("nan"), float("inf")])
def test_mesh_rejects_a_tenor_that_is_not_positive_and_finite(bad: float) -> None:
    """A zero tenor has no total variance to speak of, and a NaN one poisons every comparison."""
    with pytest.raises(ValueError, match="positive and finite"):
        make_mesh(tenors=(bad, FAR_TENOR))


def test_mesh_rejects_unordered_tenors() -> None:
    """The calendar condition is directional; a shuffled axis would flip the verdict."""
    with pytest.raises(ValueError, match="strictly increasing"):
        make_mesh(tenors=(FAR_TENOR, NEAR_TENOR))


def test_mesh_does_not_require_uniformly_spaced_tenors() -> None:
    """Nothing is differentiated along this axis, so the real expiry ladder is admissible.

    Front expiries cluster and back ones spread out; requiring uniformity here would refuse every
    market's actual term structure for the sake of a symmetry the mathematics does not need.
    """
    ladder = ArbitrageMesh(log_moneyness=MESH_K, tenors=(0.02, 0.08, 0.50, 2.00))

    assert ladder.tenor_array.shape == (4,)


def test_mesh_arrays_do_not_alias_between_accesses() -> None:
    """Each access rebuilds the array, so a caller writing through one cannot change the mesh."""
    mesh = make_mesh()
    first = mesh.k_array
    first[0] = 99.0

    assert mesh.k_array[0] == pytest.approx(MESH_K[0])


# --- durrleman_g


def test_g_is_exactly_one_everywhere_for_a_flat_slice() -> None:
    """Constant w makes both differences exact floating-point zeros, so every term but the
    leading square vanishes and g == 1 on the nose.

    Equality rather than approximation, and it is the one case in this module with an answer
    nobody can get subtly wrong: w[i+1] - w[i-1] is a subtraction of two identical floats, not a
    small number, so there is no rounding left for a tolerance to hide.
    """
    k = np.asarray(MESH_K)
    w = np.full(len(MESH_K), FLAT_W)

    assert np.all(durrleman_g(w, k) == 1.0)


def test_g_is_returned_only_for_the_interior_points() -> None:
    """Two shorter than the mesh, because each stencil reaches one point either side."""
    k = np.asarray(MESH_K)
    w = np.full(len(MESH_K), FLAT_W)

    assert durrleman_g(w, k).shape == (len(MESH_K) - 2,)


def test_the_outermost_moneyness_points_enter_only_as_neighbours() -> None:
    """An edge value moves exactly one g; an interior value moves three. That is the whole claim.

    If the edges were judged -- by a one-sided difference, say -- poisoning w[0] would move a g
    of its own as well as its neighbour's, and the output would be as long as the mesh. Poisoning
    an interior point is the control: it feeds the stencils of three consecutive judged points,
    which is what a central difference aligned to k[1:-1] must look like.
    """
    k = np.asarray(MESH_K)
    flat = np.full(len(MESH_K), FLAT_W)
    baseline = durrleman_g(flat, k)

    at_the_edge = flat.copy()
    at_the_edge[0] += 0.01
    assert int(np.count_nonzero(durrleman_g(at_the_edge, k) != baseline)) == 1

    inside = flat.copy()
    inside[2] += 0.01
    assert int(np.count_nonzero(durrleman_g(inside, k) != baseline)) == 3


def test_g_matches_the_analytic_answer_for_a_curve_with_a_closed_form() -> None:
    """The differences are pinned against algebra, not against their own plausibility.

    A sign slip in w' still yields a smooth, entirely reasonable-looking g, and every threshold
    test in this file would accept it. Raw SVI is differentiated by hand at the top of this module
    precisely so the comparison shares none of the code under test.
    """
    k = np.asarray(_fine_mesh(1201))
    differenced = durrleman_g(_svi_total_variance(k), k)

    assert differenced == pytest.approx(_svi_analytic_g(k[1:-1]), abs=1e-4)


def test_g_converges_on_the_analytic_answer_at_second_order() -> None:
    """Halving the step quarters the error, which is what makes the stencil central.

    Not decoration: a forward difference agrees with the analytic answer too, just half as fast,
    so the test above alone would pass on a first-order formula with a looser tolerance. The rate
    is the property that distinguishes them, and it is the rate the module's accuracy claim --
    and therefore its truncation error budget near zero -- actually rests on.
    """
    errors = []
    for n_points in (241, 481, 961):
        k = np.asarray(_fine_mesh(n_points))
        differenced = durrleman_g(_svi_total_variance(k), k)
        errors.append(float(np.max(np.abs(differenced - _svi_analytic_g(k[1:-1])))))

    assert errors[0] / errors[1] == pytest.approx(4.0, rel=0.05)
    assert errors[1] / errors[2] == pytest.approx(4.0, rel=0.05)


def test_g_rejects_a_total_variance_row_of_the_wrong_length() -> None:
    """One value per mesh point, or the stencils pair a variance with someone else's moneyness."""
    k = np.asarray(MESH_K)
    with pytest.raises(ValueError, match="one value per mesh point"):
        durrleman_g(np.full(len(MESH_K) - 1, FLAT_W), k)


def test_g_rejects_a_vanishing_total_variance() -> None:
    """g divides by w twice, so a zero there returns inf or a NaN -- and both read as clean.

    max(0, -min(g)) turns either one into 0.0, because nan < 0 is False. A collapsed distribution
    reported as arbitrage-free is the worst failure a gate has, so the function refuses to answer.
    """
    k = np.asarray(MESH_K)
    w = np.full(len(MESH_K), FLAT_W)
    w[5] = 0.0

    with pytest.raises(ValueError, match="strictly positive"):
        durrleman_g(w, k)


@pytest.mark.parametrize("bad", [float("nan"), float("inf")])
def test_g_rejects_a_non_finite_total_variance(bad: float) -> None:
    """The ordering-guard trap in its numerical form: isfinite has to be tested first."""
    k = np.asarray(MESH_K)
    w = np.full(len(MESH_K), FLAT_W)
    w[3] = bad

    with pytest.raises(ValueError, match="finite"):
        durrleman_g(w, k)


def test_g_rejects_a_non_uniform_mesh_handed_to_it_directly() -> None:
    """The mesh object enforces uniformity, but this function is public and takes raw arrays."""
    k = np.array([-0.10, -0.05, 0.20, 0.25])
    with pytest.raises(ValueError, match="uniformly spaced"):
        durrleman_g(np.full(4, FLAT_W), k)


# --- check_surface, butterfly


def test_a_flat_surface_is_judged_arbitrage_free() -> None:
    """The reference verdict: g == 1 everywhere and total variance rising linearly in T.

    Guards the point: the same call on a violating surface returns a positive depth, so the zeros
    below are a verdict rather than a function that reports zero whatever it is handed.
    """
    report = check_surface(FlatVolSurface(), make_mesh())

    assert report.butterfly_violation == 0.0
    assert report.n_butterfly_violations == 0
    assert report.calendar_violation == 0.0
    assert report.n_calendar_violations == 0

    violating = check_surface(_svi_surface(), make_mesh(log_moneyness=_fine_mesh(1201)))
    assert violating.butterfly_violation > 0


def test_the_butterfly_depth_is_the_worst_negative_g_on_the_mesh() -> None:
    """Pinned to the independently computed depth, not merely to the sign.

    A formula that finds *a* negative value for the wrong reason would pass a sign test. The
    expected number comes from the analytic g at the top of this file, so agreement here means
    the differences, the interior slicing and the clamp all line up with the algebra at once.
    """
    mesh = make_mesh(log_moneyness=_fine_mesh(1201))
    expected = max(0.0, -float(np.min(_svi_analytic_g(mesh.k_array[1:-1]))))

    assert check_surface(_svi_surface(), mesh).butterfly_violation == pytest.approx(
        expected, abs=1e-4
    )
    assert expected == pytest.approx(0.5308, abs=1e-3)


def test_the_butterfly_count_spans_every_tenor_row() -> None:
    """The count is over judged points, not over moneyness: two identical rows count twice.

    The surface is the same smile at both tenors, so anything else would mean the second row was
    dropped, averaged, or silently overwritten by the first.
    """
    mesh = make_mesh(log_moneyness=_fine_mesh(1201))
    one_row = durrleman_g(_svi_total_variance(mesh.k_array), mesh.k_array)
    per_row = int(np.count_nonzero(one_row < 0.0))

    assert per_row > 0
    assert check_surface(_svi_surface(), mesh).n_butterfly_violations == 2 * per_row


def test_a_violation_narrower_than_the_mesh_spacing_is_invisible() -> None:
    """The gate sees what the mesh samples, and cannot tell a missed dip from a clean surface.

    A bump 0.004 wide sitting halfway between two points of a mesh stepping by 0.05 leaves every
    sampled value untouched to within a factor of e^-39, so the coarse mesh returns the flat
    surface's verdict exactly: g == 1 everywhere and a depth of 0.0. The same surface on a mesh
    stepping by 0.001 reports a depth near 60.

    Guards the point, and is the reason this test exists as a fact rather than as a paragraph:
    the caveat in the docstring is a real, measurable blind spot with a two-order-of-magnitude
    consequence, not a hedge. Mesh density is the caller's decision and a consequential one.
    """
    surface = _bumped_surface(centre=0.025, width=0.004, amplitude=0.001)

    assert check_surface(surface, make_mesh()).butterfly_violation == 0.0

    resolved = check_surface(surface, make_mesh(log_moneyness=_fine_mesh(1201)))
    assert resolved.butterfly_violation > 50.0


def test_the_reported_violation_sits_where_the_defect_is() -> None:
    """g is aligned to k[1:-1]; an off-by-one would place the worst point one step away.

    The bump is centred on a mesh point by construction, so the deepest g has an exact expected
    location rather than an approximate one.
    """
    mesh = make_mesh(log_moneyness=_fine_mesh(1201))
    k = mesh.k_array
    surface = _bumped_surface(centre=0.025, width=0.004, amplitude=0.001)
    g = durrleman_g(surface.total_variance(k, mesh.tenor_array)[0], k)

    assert k[1:-1][int(np.argmin(g))] == pytest.approx(0.025, abs=1e-9)


def test_the_number_of_judged_points_drops_the_two_edge_columns() -> None:
    """23 interior points at each of two tenors, never the 50 the mesh holds.

    Reporting the mesh size here would overstate the coverage by exactly the band where the wings
    are, which is the band the caller most needs to know was not examined.
    """
    report = check_surface(FlatVolSurface(), make_mesh())

    assert report.n_points_judged == 2 * (len(MESH_K) - 2) == 46


def test_a_single_tenor_mesh_still_judges_the_butterfly_condition() -> None:
    """One slice is all the butterfly condition ever needed; the tenor is already inside w."""
    report = check_surface(FlatVolSurface(), make_mesh(tenors=(NEAR_TENOR,)))

    assert report.n_points_judged == len(MESH_K) - 2


# --- check_surface, calendar


def test_a_calendar_crossing_is_caught_with_its_depth() -> None:
    """Total variance that falls with maturity prices a calendar spread at a negative cost.

    The surface is flat in moneyness, so g stays exactly 1 and nothing but the calendar condition
    can produce a number here. The expected depth is the two total variances subtracted by hand:
    0.65^2/12 at one month against 0.65^2/4 - 0.09 at three.
    """
    surface = CallableSurface(fn=lambda k, t: 0.65 * 0.65 * t - np.where(t > 0.15, 0.09, 0.0))
    report = check_surface(surface, make_mesh())

    assert report.butterfly_violation == 0.0
    assert report.calendar_violation == pytest.approx(
        0.65 * 0.65 * NEAR_TENOR - (0.65 * 0.65 * FAR_TENOR - 0.09)
    )
    assert report.calendar_violation == pytest.approx(0.0195833, abs=1e-6)


def test_the_calendar_condition_is_judged_at_every_moneyness_point_including_the_edges() -> None:
    """Nothing is differentiated across tenors, so nothing is lost at the ends of the axis.

    The crossing here exists only at the very first mesh point. A calendar check that reused the
    butterfly condition's interior slice would report this surface as clean, which is the wing
    thrown away for no reason at all.
    """

    def fn(k: NDArray[np.float64], t: NDArray[np.float64]) -> NDArray[np.float64]:
        edge = np.isclose(k, MESH_K[0])
        return 0.65 * 0.65 * t - np.where(edge & (t > 0.15), 0.09, 0.0)

    report = check_surface(CallableSurface(fn=fn), make_mesh())

    assert report.calendar_violation > 0.0
    assert report.n_calendar_violations == 1


def test_the_calendar_count_spans_every_moneyness_point_of_every_pair() -> None:
    """One pair of tenors crossing everywhere is 25 breaches, not one."""
    surface = CallableSurface(fn=lambda k, t: 0.65 * 0.65 * t - np.where(t > 0.15, 0.09, 0.0))

    assert check_surface(surface, make_mesh()).n_calendar_violations == len(MESH_K)


def test_a_single_tenor_mesh_reports_no_calendar_violation_at_all() -> None:
    """There is no consecutive pair, so there is nothing to check and nothing to report.

    Guards the point: the very same surface crosses badly once a second tenor is on the mesh, so
    the 0.0 here is "no condition applies" rather than "this surface is fine" -- the two readings
    share a number, and only the tenor count tells them apart.
    """
    surface = CallableSurface(fn=lambda k, t: 0.65 * 0.65 * t - np.where(t > 0.15, 0.09, 0.0))

    lonely = check_surface(surface, make_mesh(tenors=(NEAR_TENOR,)))
    assert lonely.calendar_violation == 0.0
    assert lonely.n_calendar_violations == 0

    assert check_surface(surface, make_mesh()).calendar_violation > 0.0


def test_a_flat_term_structure_is_not_a_calendar_violation() -> None:
    """The condition is non-decreasing, not increasing: equal total variance is admissible."""
    surface = CallableSurface(fn=lambda k, t: np.full_like(t, FLAT_W))
    report = check_surface(surface, make_mesh())

    assert report.calendar_violation == 0.0
    assert report.n_calendar_violations == 0


# --- check_surface, a model that has stopped being a surface


def test_a_diverged_surface_raises_rather_than_reporting_a_clean_verdict() -> None:
    """A grid of NaN would score a butterfly violation of exactly zero, because nan < 0 is False.

    That is the failure this whole module is written around: the cleanest report of the session,
    produced by a network with no numbers left in it. The exception belongs on the evaluation
    path, not in a metric series that is supposed to mean "how close to arbitrage-free is this".
    """
    with pytest.raises(SurfaceEvaluationError):
        check_surface(CallableSurface(fn=lambda k, t: np.full_like(t, np.nan)), make_mesh())


def test_a_surface_answering_with_transposed_axes_raises() -> None:
    """The mistake a real adapter makes on its first day, and one no arbitrage number describes."""
    with pytest.raises(SurfaceEvaluationError):
        check_surface(TransposedSurface(), make_mesh())


# --- ArbitrageReport


def test_report_exceeds_is_false_exactly_at_the_tolerance() -> None:
    """The comparison is >, so a breach landing on the threshold passes.

    With >= a tolerance of 0.0 would refuse every surface including the arbitrage-free ones, since
    a clean report holds exactly 0.0 -- the gate would be shut permanently by its strictest and
    most obvious configuration.
    """
    report = make_report(butterfly_violation=0.25)

    assert report.exceeds(butterfly_tol=0.25, calendar_tol=1.0) is False
    assert report.exceeds(butterfly_tol=0.2499, calendar_tol=1.0) is True


def test_report_exceeds_on_a_calendar_breach_alone() -> None:
    """The two conditions are combined with or: a clean smile does not excuse a crossing."""
    report = make_report(calendar_violation=0.05)

    assert report.exceeds(butterfly_tol=1.0, calendar_tol=0.04) is True


def test_report_exceeds_is_false_for_a_clean_report_at_zero_tolerance() -> None:
    """Zero tolerance is the meaningful strictest setting, not a degenerate one."""
    assert make_report().exceeds(butterfly_tol=0.0, calendar_tol=0.0) is False


@pytest.mark.parametrize("bad", [-0.01, float("nan"), float("inf"), float("-inf")])
def test_report_exceeds_rejects_an_unusable_tolerance(bad: float) -> None:
    """A NaN tolerance is the dangerous one: every > against it is False.

    A misconfigured gate would not fail loudly, it would pass every surface for the rest of the
    session, which is exactly the silent-and-downstream failure ADR-010 rejects soft constraints
    alone for.
    """
    with pytest.raises(ValueError, match="tolerance"):
        make_report().exceeds(butterfly_tol=bad, calendar_tol=0.0)
    with pytest.raises(ValueError, match="tolerance"):
        make_report().exceeds(butterfly_tol=0.0, calendar_tol=bad)


@pytest.mark.parametrize("field", ["butterfly_violation", "calendar_violation"])
@pytest.mark.parametrize("bad", [-0.01, float("nan"), float("inf")])
def test_report_rejects_a_depth_that_is_not_non_negative_and_finite(field: str, bad: float) -> None:
    """Depths are clamped at zero by construction, so a negative one means the clamp was lost."""
    with pytest.raises(ValueError, match="must be non-negative and finite"):
        replace_field(make_report(), field, bad)


def test_report_rejects_judging_no_points_at_all() -> None:
    """Nothing was found and nothing was looked at are different verdicts, and only one scores 0."""
    with pytest.raises(ValueError, match="at least one point"):
        replace_field(make_report(), "n_points_judged", 0)


def test_report_rejects_more_butterfly_violations_than_judged_points() -> None:
    """A count above its own denominator means the two were computed over different populations."""
    with pytest.raises(ValueError, match="between zero and the number of judged points"):
        make_report(n_points_judged=46, n_butterfly_violations=47)


def test_report_rejects_a_negative_calendar_count() -> None:
    with pytest.raises(ValueError, match="cannot be negative"):
        replace_field(make_report(), "n_calendar_violations", -1)


def test_report_allows_more_calendar_violations_than_judged_points() -> None:
    """Deliberately unbounded: the calendar check runs over a different population.

    Every moneyness point including the two edges, against one fewer tenor row -- so a ceiling
    borrowed from the butterfly count would be wrong in both directions.
    """
    lopsided = ArbitrageReport(
        butterfly_violation=0.0,
        calendar_violation=0.01,
        n_points_judged=1,
        n_butterfly_violations=0,
        n_calendar_violations=50,
    )

    assert lopsided.n_calendar_violations == 50
