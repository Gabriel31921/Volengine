from __future__ import annotations

import math

import pytest

from tests.risk.builders import (
    EXPIRIES,
    FORWARDS,
    K_AXIS,
    TENORS,
    make_view,
    smile_vol,
    total_variance_grid,
)
from volengine.risk.domain.interpolation import bilinear_total_variance, implied_vol

GRID = total_variance_grid()
"""The builder's grid, spelled out once so every expectation below is written against the nodes
rather than against the implementation that reads them."""

ATM = K_AXIS.index(0.0)
"""Index of the at-the-money-forward column. ``0.0`` is a legitimate moneyness, so this is looked
up rather than tested for truthiness anywhere."""


# --- On the nodes


def test_every_node_is_reproduced_exactly() -> None:
    """Bit for bit at all fifteen nodes, not to within a tolerance.

    Both weights are exactly ``0.0`` on a node, and ``(1 - 0.0) * a + 0.0 * b`` is exactly ``a``,
    so the lookup that happens most often -- a listed expiry at a quoted strike -- introduces no
    error of its own. An implementation that drifted here would put an interpolation error into
    the one place there is real data.
    """
    view = make_view()

    for i, tenor in enumerate(TENORS):
        for j, k in enumerate(K_AXIS):
            assert bilinear_total_variance(view, k, tenor) == GRID[i][j]


def test_the_implied_vol_at_a_node_is_the_volatility_that_built_it() -> None:
    """``sqrt(w / T)`` undoes the ACL's one conversion, so the round trip returns the smile."""
    view = make_view()

    for tenor in TENORS:
        for k in K_AXIS:
            assert implied_vol(view, k, tenor) == pytest.approx(smile_vol(tenor, k), rel=1e-12)


# --- Between the nodes: in total variance, and demonstrably not in volatility


def test_the_midpoint_of_two_moneyness_nodes_is_the_mean_of_their_total_variances() -> None:
    """Halfway between two strikes, the answer is the average of the two ``w``, exactly.

    Stated in total-variance space on purpose. The same query written in volatility space --
    "assert the vol is the average of the two vols" -- would pass here to within any tolerance
    anyone would think to write, because along a single tenor the two averages differ in the
    seventh decimal. A test that cannot fail on the wrong interpolation is decoration; see the
    tenor-direction test below, which is where the two conventions genuinely separate.
    """
    view = make_view()
    mid_k = (K_AXIS[1] + K_AXIS[2]) / 2.0
    expected = (GRID[1][1] + GRID[1][2]) / 2.0

    # The vacuity guard: the smile has to actually move between those two nodes, or a flat
    # surface would satisfy this assertion under any interpolation at all.
    assert GRID[1][1] != GRID[1][2]

    assert bilinear_total_variance(view, mid_k, TENORS[1]) == pytest.approx(expected, rel=1e-12)


def test_the_midpoint_of_two_tenors_is_linear_in_total_variance_and_not_in_volatility() -> None:
    """The test the whole storage decision rests on: ``w`` is what is interpolated, not ``vol``.

    Between the one- and three-month nodes the two conventions disagree by about 4e-4 in total
    variance, which is a fifth of a volatility point -- small enough to be invisible in a report
    and large enough to pin here. Interpolating volatilities would also be wrong in a way that
    matters beyond accuracy: total variance rising with tenor is the calendar-arbitrage condition,
    and a straight line in vol space between two nodes whose ``w`` rises can dip below it,
    manufacturing an arbitrage in the consumer out of a surface the producer's gate had certified.
    """
    view = make_view()
    mid_tenor = (TENORS[0] + TENORS[1]) / 2.0
    in_total_variance = (GRID[0][ATM] + GRID[1][ATM]) / 2.0
    mid_vol = (smile_vol(TENORS[0], 0.0) + smile_vol(TENORS[1], 0.0)) / 2.0
    in_volatility = mid_vol * mid_vol * mid_tenor

    # The vacuity guard: the two conventions have to give different answers on this grid, or the
    # assertion below would hold whichever one had been implemented.
    assert abs(in_total_variance - in_volatility) > 1e-4

    assert bilinear_total_variance(view, 0.0, mid_tenor) == pytest.approx(
        in_total_variance, rel=1e-12
    )
    assert implied_vol(view, 0.0, mid_tenor) != pytest.approx(mid_vol, abs=1e-4)


def test_the_centre_of_a_cell_is_the_mean_of_its_four_corners() -> None:
    """Bilinear, not two independent one-dimensional lookups with an axis silently preferred.

    Halfway along both axes every corner carries a weight of a quarter, which is a statement about
    the whole cell rather than about either edge of it, and it is the assertion an implementation
    that interpolated in ``k`` and then simply clamped the tenor would fail.
    """
    view = make_view()
    mid_k = (K_AXIS[2] + K_AXIS[3]) / 2.0
    mid_tenor = (TENORS[1] + TENORS[2]) / 2.0
    expected = (GRID[1][2] + GRID[1][3] + GRID[2][2] + GRID[2][3]) / 4.0

    assert len({GRID[1][2], GRID[1][3], GRID[2][2], GRID[2][3]}) == 4

    assert bilinear_total_variance(view, mid_k, mid_tenor) == pytest.approx(expected, rel=1e-12)


# --- Outside the nodes: flat on all four sides


def test_the_surface_is_flat_below_the_first_moneyness_node() -> None:
    """The put wing stops moving past the last quoted strike."""
    view = make_view()

    # The vacuity guard: the smile has to vary along this axis for a clamp to be a real choice.
    assert GRID[1][0] != GRID[1][1]

    assert bilinear_total_variance(view, K_AXIS[0] - 0.50, TENORS[1]) == GRID[1][0]


def test_the_surface_is_flat_above_the_last_moneyness_node() -> None:
    """And the call wing likewise."""
    view = make_view()

    assert GRID[1][-1] != GRID[1][-2]

    assert bilinear_total_variance(view, K_AXIS[-1] + 0.50, TENORS[1]) == GRID[1][-1]


def test_the_surface_is_flat_below_the_first_tenor() -> None:
    """Total variance stops falling below the front node, so the implied volatility of a very
    short expiry runs *up* as ``sqrt(w_first / T)``. The mirror image of the far-end behaviour,
    and the reason a grid should reach the front of the book it is used on."""
    view = make_view()

    assert GRID[0][ATM] != GRID[1][ATM]

    assert bilinear_total_variance(view, 0.0, TENORS[0] / 4.0) == GRID[0][ATM]


def test_the_surface_is_flat_above_the_last_tenor() -> None:
    view = make_view()

    assert GRID[-1][ATM] != GRID[-2][ATM]

    assert bilinear_total_variance(view, 0.0, TENORS[-1] * 3.0) == GRID[-1][ATM]


def test_the_corner_is_flat_in_both_directions_at_once() -> None:
    """Past the end of both axes the cell degenerates to a single node, and the answer is that
    node -- not an extrapolated diagonal, which is what a bilinear form written without the clamp
    would produce."""
    view = make_view()

    assert bilinear_total_variance(view, K_AXIS[-1] + 1.0, TENORS[-1] + 5.0) == GRID[-1][-1]


def test_the_implied_vol_decays_as_one_over_sqrt_t_past_the_last_tenor() -> None:
    """The surprising half of flat extrapolation, asserted numerically.

    Flat in **total variance** is not flat in volatility: ``w`` stops growing at the last node
    while the divisor keeps going, so the vol falls as ``1 / sqrt(T)``. A two-year option valued
    off a one-year grid is priced at about 71% of the one-year volatility, and a four-year one at
    exactly half. That is conservative and defensible -- a flat ``w`` is the boundary case of the
    calendar condition and cannot create an arbitrage -- but it is emphatically not a forecast.
    """
    view = make_view()
    at_the_last_node = implied_vol(view, 0.0, TENORS[-1])

    # The vacuity guard: the decay must come from the divisor and not from the grid, so pin that
    # the total variance really is unchanged out there.
    assert bilinear_total_variance(view, 0.0, TENORS[-1] * 4.0) == bilinear_total_variance(
        view, 0.0, TENORS[-1]
    )

    assert implied_vol(view, 0.0, TENORS[-1] * 2.0) == pytest.approx(
        at_the_last_node / math.sqrt(2.0), rel=1e-12
    )
    assert implied_vol(view, 0.0, TENORS[-1] * 4.0) == pytest.approx(
        at_the_last_node / 2.0, rel=1e-12
    )


# --- Degenerate grids


def test_a_single_moneyness_node_answers_everywhere() -> None:
    """A market quoting one strike per expiry is a legal surface with no smile at all, and the
    lookup has to return that column rather than fail to find a bracket for it."""
    view = make_view(log_moneyness=(0.0,))
    row = view.total_variance[1]

    assert bilinear_total_variance(view, 0.0, TENORS[1]) == row[0]
    assert bilinear_total_variance(view, -1.5, TENORS[1]) == row[0]
    assert bilinear_total_variance(view, 1.5, TENORS[1]) == row[0]


def test_a_single_tenor_answers_everywhere() -> None:
    """The same on the other axis: one expiry, no term structure, and the total variance held
    flat in ``T`` -- which is exactly why the vol at four times the tenor is halved."""
    view = make_view(tenors=(TENORS[1],), expiries=(EXPIRIES[1],), forwards=(FORWARDS[1],))
    row = view.total_variance[0]

    assert bilinear_total_variance(view, K_AXIS[1], TENORS[1] / 10.0) == row[1]
    assert bilinear_total_variance(view, K_AXIS[1], TENORS[1] * 10.0) == row[1]
    assert implied_vol(view, 0.0, TENORS[1] * 4.0) == pytest.approx(
        implied_vol(view, 0.0, TENORS[1]) / 2.0, rel=1e-12
    )


def test_a_one_by_one_grid_is_a_constant_surface() -> None:
    """Both axes degenerate at once: the smallest surface anyone can publish still answers."""
    view = make_view(
        log_moneyness=(0.0,),
        tenors=(TENORS[1],),
        expiries=(EXPIRIES[1],),
        forwards=(FORWARDS[1],),
    )
    only = view.total_variance[0][0]

    assert bilinear_total_variance(view, -0.80, TENORS[1] * 7.0) == only


# --- Rejections


@pytest.mark.parametrize("bad_k", [math.nan, math.inf, -math.inf])
def test_the_lookup_rejects_a_non_finite_moneyness(bad_k: float) -> None:
    with pytest.raises(ValueError, match="log-moneyness must be finite"):
        bilinear_total_variance(make_view(), bad_k, TENORS[1])


@pytest.mark.parametrize("bad_tenor", [0.0, -0.5, math.nan, math.inf])
def test_the_lookup_rejects_a_tenor_that_is_not_positive_and_finite(bad_tenor: float) -> None:
    """Zero because ``implied_vol`` divides by it; the two non-finite cases because they fail in
    two different silent ways, neither of which announces itself."""
    with pytest.raises(ValueError, match="tenor in years must be positive and finite"):
        bilinear_total_variance(make_view(), 0.0, bad_tenor)


def test_an_infinite_coordinate_really_would_have_been_clamped_silently() -> None:
    """The vacuity guard for the two rejections above: they are not merely defensive.

    A very large but finite tenor is clamped to the last node and comes back as a perfectly
    ordinary total variance -- which is the right answer for a coordinate that exists. An infinity
    satisfies exactly the same clamp test, so without the finiteness guard it would return that
    same plausible number for a coordinate that does not, and nothing downstream could tell the
    two apart.
    """
    view = make_view()

    assert bilinear_total_variance(view, 1e9, 1e9) == GRID[-1][-1]


@pytest.mark.parametrize("bad_tenor", [0.0, math.nan])
def test_the_volatility_rejects_what_the_lookup_rejects(bad_tenor: float) -> None:
    """``implied_vol`` delegates its validation rather than repeating it, so the same coordinate
    is refused by both and with the same message."""
    with pytest.raises(ValueError, match="tenor in years must be positive and finite"):
        implied_vol(make_view(), 0.0, bad_tenor)


def test_the_volatility_is_strictly_positive_everywhere_it_is_defined() -> None:
    """Positivity is inherited, not imposed: every node is strictly positive and the four weights
    are a convex combination, so no combination of them can reach zero.

    Checked across the grid and well outside it, because the clamped regions are where a lookup
    that had gone wrong would be least likely to be noticed.
    """
    view = make_view()

    for tenor in (1e-4, TENORS[0], TENORS[1], TENORS[2], 25.0):
        for k in (-2.0, -0.15, 0.0, 0.15, 2.0):
            vol = implied_vol(view, k, tenor)
            assert math.isfinite(vol)
            assert vol > 0.0
