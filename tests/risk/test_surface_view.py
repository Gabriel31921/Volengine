from __future__ import annotations

import math
from dataclasses import FrozenInstanceError, replace
from datetime import timedelta

import pytest

from tests.risk.builders import (
    EXPIRIES,
    FORWARDS,
    K_AXIS,
    NAIVE,
    NOW,
    TENORS,
    make_view,
    total_variance_grid,
)
from tests.support import replace_field
from volengine.risk.domain.errors import ExpiredPositionError
from volengine.risk.domain.interpolation import bilinear_total_variance

# --- The fields it keeps


def test_the_view_keeps_every_field_as_given() -> None:
    """Nothing is normalised, sorted or rescaled on the way in: the ACL already did that."""
    view = make_view()

    assert view.market_id == "BTC-DERIBIT"
    assert view.producer_id == "svi-scipy"
    assert view.surface_id == "01JZQ0T4M2"
    assert view.ts_snapshot == NOW
    assert view.log_moneyness == K_AXIS
    assert view.tenors == TENORS
    assert view.expiries == EXPIRIES
    assert view.forwards == FORWARDS
    assert view.total_variance == total_variance_grid()


@pytest.mark.parametrize("field", ["ts_snapshot", "total_variance"])
def test_the_view_is_frozen_and_slotted(field: str) -> None:
    """A surface is a value. Rewriting a node in place would change what a report was computed on
    after the report was written, and nothing would say so."""
    view = make_view()
    assert not hasattr(view, "__dict__")

    with pytest.raises(FrozenInstanceError):
        setattr(view, field, None)


def test_the_view_carries_no_status_field() -> None:
    """The deliberate absence, pinned so that adding one means deleting a test that says why.

    ADR-006 republishes a stale surface with its **original** ``ts_snapshot``, so the single
    timestamp the freshness policy already reads carries the staleness as a comparable number. A
    fourth spelling of a status enum in this context would be a second channel for the same fact,
    and the first thing a second channel does is disagree with the first.
    """
    assert not hasattr(make_view(), "status")


def test_the_view_stores_total_variance_rather_than_volatilities() -> None:
    """The grid really is ``w = vol**2 * T`` and not a grid of vols, which the numbers can tell
    apart: at 65% and a month, the two differ by a factor of eight."""
    view = make_view()
    at_the_money = view.total_variance[0][K_AXIS.index(0.0)]

    assert at_the_money == pytest.approx(0.6458904**2 * TENORS[0], rel=1e-6)
    assert at_the_money < 0.05
    assert not hasattr(view, "vols")


# --- Construction: identity and instants


@pytest.mark.parametrize("field", ["market_id", "producer_id", "surface_id"])
def test_the_view_rejects_an_empty_identifier(field: str) -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        replace_field(make_view(), field, "")


def test_the_view_rejects_a_naive_snapshot_instant() -> None:
    """Staleness is a subtraction of datetimes, and mixing the two kinds raises ``TypeError`` far
    from whoever built the value."""
    with pytest.raises(ValueError, match="ts_snapshot must be timezone-aware"):
        replace(make_view(), ts_snapshot=NAIVE)


def test_the_view_rejects_a_naive_expiry() -> None:
    with pytest.raises(ValueError, match="expiry at index 0 must be timezone-aware"):
        replace(make_view(), expiries=(NAIVE, EXPIRIES[1], EXPIRIES[2]))


# --- Construction: the axes


def test_the_view_rejects_an_empty_moneyness_axis() -> None:
    with pytest.raises(ValueError, match="at least one moneyness node"):
        make_view(log_moneyness=())


def test_the_view_rejects_an_empty_tenor_axis() -> None:
    with pytest.raises(ValueError, match="at least one tenor"):
        make_view(tenors=(), expiries=(), forwards=())


def test_the_view_rejects_a_non_finite_moneyness_node() -> None:
    with pytest.raises(ValueError, match="log-moneyness at index 1 must be finite"):
        make_view(log_moneyness=(-0.10, math.nan, 0.10))


def test_the_view_rejects_an_unordered_moneyness_axis() -> None:
    """Every lookup binary-searches this axis, and a binary search over unsorted data does not
    fail, it answers wrongly."""
    with pytest.raises(ValueError, match="moneyness axis must be strictly increasing"):
        make_view(log_moneyness=(-0.10, 0.10, 0.0))


def test_the_view_rejects_a_repeated_moneyness_node() -> None:
    """Strictly increasing, not merely sorted: two nodes at one ``k`` are two answers to one
    question, and the interpolation would divide by their zero spacing."""
    with pytest.raises(ValueError, match="moneyness axis must be strictly increasing"):
        make_view(log_moneyness=(-0.10, 0.0, 0.0))


@pytest.mark.parametrize("bad", [0.0, -0.25, math.nan, math.inf])
def test_the_view_rejects_a_tenor_that_is_not_positive_and_finite(bad: float) -> None:
    """Zero included: a tenor of zero is an expired option, which has no total variance to hold
    and is refused by ``tenor_of`` rather than represented on the axis."""
    with pytest.raises(ValueError, match="tenor at index 0 must be positive and finite"):
        make_view(tenors=(bad, TENORS[1], TENORS[2]))


def test_the_view_rejects_an_unordered_tenor_axis() -> None:
    with pytest.raises(ValueError, match="tenor axis must be strictly increasing"):
        make_view(tenors=(TENORS[1], TENORS[0], TENORS[2]))


# --- Construction: the calendar


def test_the_view_rejects_a_calendar_that_does_not_match_the_tenor_axis() -> None:
    with pytest.raises(ValueError, match="one expiry per tenor"):
        make_view(expiries=EXPIRIES[:2])


def test_the_view_rejects_unordered_expiries() -> None:
    with pytest.raises(ValueError, match="expiries must be strictly increasing"):
        make_view(expiries=(EXPIRIES[1], EXPIRIES[0], EXPIRIES[2]))


def test_the_view_rejects_a_first_expiry_at_the_snapshot_instant() -> None:
    """The tenor axis has to start after the instant the surface describes.

    A node at ``ts_snapshot`` is an expiry with no time left, and it would leave the
    below-the-first-node extrapolation of ``tenor_of`` with an interval of zero seconds to divide
    by.
    """
    with pytest.raises(ValueError, match="first expiry must be strictly after the snapshot"):
        make_view(expiries=(NOW, EXPIRIES[1], EXPIRIES[2]))


def test_the_view_rejects_a_first_expiry_before_the_snapshot_instant() -> None:
    with pytest.raises(ValueError, match="first expiry must be strictly after the snapshot"):
        make_view(expiries=(NOW - timedelta(days=1), EXPIRIES[1], EXPIRIES[2]))


# --- Construction: the forwards


def test_the_view_rejects_a_forward_curve_that_does_not_match_the_tenor_axis() -> None:
    with pytest.raises(ValueError, match="one forward per tenor"):
        make_view(forwards=FORWARDS[:2])


@pytest.mark.parametrize("bad", [0.0, -1.0, math.nan, math.inf])
def test_the_view_rejects_a_forward_that_is_not_positive_and_finite(bad: float) -> None:
    """A NaN forward is the one the ordering test alone would miss: ``float("nan") <= 0`` is
    ``False``, and it would come back out of ``forward_at`` as a strike nobody can place."""
    with pytest.raises(ValueError, match="forward at index 1 must be positive and finite"):
        make_view(forwards=(FORWARDS[0], bad, FORWARDS[2]))


# --- Construction: the grid


def test_the_view_rejects_a_grid_with_the_wrong_number_of_rows() -> None:
    with pytest.raises(ValueError, match="one row of total variance per tenor"):
        make_view(total_variance=total_variance_grid()[:2])


def test_the_view_rejects_a_ragged_grid() -> None:
    """A ragged row makes a bilinear lookup ambiguous exactly where the surface is thinnest."""
    grid = total_variance_grid()
    with pytest.raises(ValueError, match="one total variance per moneyness node"):
        make_view(total_variance=(grid[0], grid[1][:3], grid[2]))


@pytest.mark.parametrize("bad", [0.0, -0.01, math.nan, math.inf])
def test_the_view_rejects_a_total_variance_that_is_not_positive_and_finite(bad: float) -> None:
    """Strict positivity is what lets ``implied_vol`` take a square root anywhere on the grid, and
    a convex combination of positive corners is positive, so it survives the interpolation."""
    grid = total_variance_grid()
    poisoned = (grid[0], (bad, *grid[1][1:]), grid[2])

    with pytest.raises(ValueError, match="total variance at tenor"):
        make_view(total_variance=poisoned)


def test_a_poisoned_grid_entry_really_would_have_reached_the_interpolation() -> None:
    """The guard above is not vacuous: the poisoned corner is one the lookup actually reads.

    Without this, a rejection test proves only that a constructor is fussy. The entry bent above
    is ``[1][0]``, and it is the corner a lookup at the first moneyness node and the second tenor
    returns -- so a surface built with a NaN there would answer a perfectly ordinary query with a
    NaN, and every ordering guard downstream would pass it (``nan <= 0`` is ``False``).
    """
    view = make_view()
    assert bilinear_total_variance(view, K_AXIS[0], TENORS[1]) == view.total_variance[1][0]


# --- tenor_of: the payoff of carrying the expiries


def test_tenor_of_reproduces_the_tenor_axis_exactly_at_every_node() -> None:
    """Bit for bit, not to within a tolerance.

    A position on a listed expiry is the single most common lookup this context performs, and an
    interpolated tenor that drifted by an ulp there would make every "on a node" test in the
    context approximate for no reason. It is also the check that the grid's own calendar is being
    read rather than a daycount being guessed: ``TENORS`` is ACT/365F and nothing in ``risk/``
    knows that.
    """
    view = make_view()

    for expiry, tenor in zip(EXPIRIES, TENORS, strict=True):
        assert view.tenor_of(expiry) == tenor


def test_tenor_of_interpolates_linearly_in_calendar_time_between_two_nodes() -> None:
    """Sixty days sits halfway between the thirty- and ninety-day nodes, so its tenor is 60/365.

    Computed here by division rather than copied from the implementation: the point of the
    interpolation is that ``expiries`` and ``tenors`` together *state* the daycount at the nodes,
    so a query in between must land where that straight line says, and this test writes the line
    out independently.
    """
    view = make_view()

    assert view.tenor_of(NOW + timedelta(days=60)) == pytest.approx(60 / 365.0, rel=1e-12)


def test_tenor_of_extrapolates_from_the_snapshot_below_the_first_node() -> None:
    """A front-week option is priceable even when the nearest grid node is a month out.

    The line runs from ``ts_snapshot`` at tenor zero to the first node, so fifteen days is half of
    the thirty-day node's tenor. Clamping to the first tenor instead would value a Friday expiry
    as a month-long one, which is the whole front of a crypto book mispriced.
    """
    view = make_view()

    assert view.tenor_of(NOW + timedelta(days=15)) == pytest.approx(15 / 365.0, rel=1e-12)


def test_tenor_of_clamps_beyond_the_last_node() -> None:
    """A long-dated position is reported with a documented weakness rather than dropped."""
    view = make_view()

    assert view.tenor_of(NOW + timedelta(days=500)) == TENORS[-1]
    assert view.tenor_of(EXPIRIES[-1]) == TENORS[-1]


def test_tenor_of_is_not_a_daycount_in_disguise() -> None:
    """The vacuity guard for the three tests above: the calendar and the tenor axis really are
    two independent statements, so reading one off the other is a genuine interpolation.

    If ``tenor_of`` computed ``(expiry - ts_snapshot) / 365 days`` it would agree with every
    assertion above, since the builder's numbers are ACT/365F. Rebuild the same view with a tenor
    axis a fifth of the size -- an absurd daycount, but a self-consistent one -- and the answer has
    to follow the grid rather than the calendar. That is ADR-002 in one assertion.
    """
    squashed = tuple(tenor / 5.0 for tenor in TENORS)
    view = make_view(tenors=squashed)

    assert view.tenor_of(EXPIRIES[1]) == squashed[1]
    assert view.tenor_of(NOW + timedelta(days=60)) == pytest.approx(60 / 365.0 / 5.0, rel=1e-12)


def test_tenor_of_works_on_a_single_tenor_grid() -> None:
    """One node is a legal surface -- a market quoting a single expiry -- and both methods have to
    work on it: below the node the line still runs from the snapshot, above it the clamp holds."""
    view = make_view(tenors=(TENORS[1],), expiries=(EXPIRIES[1],), forwards=(FORWARDS[1],))

    assert view.tenor_of(EXPIRIES[1]) == TENORS[1]
    assert view.tenor_of(NOW + timedelta(days=45)) == pytest.approx(45 / 365.0, rel=1e-12)
    assert view.tenor_of(NOW + timedelta(days=500)) == TENORS[1]


def test_tenor_of_raises_for_a_position_expiring_at_the_snapshot_instant() -> None:
    """Zero is not an honest answer: it is a tenor the interpolation divides by."""
    with pytest.raises(ExpiredPositionError, match="at or before"):
        make_view().tenor_of(NOW)


def test_tenor_of_raises_for_a_position_that_has_already_expired() -> None:
    """An ordinary runtime condition rather than a bug. A portfolio file is configuration, it
    outlives its positions, and the first run after a roll names an expiry that has passed."""
    with pytest.raises(ExpiredPositionError, match="at or before"):
        make_view().tenor_of(NOW - timedelta(days=1))


def test_tenor_of_rejects_a_naive_expiry() -> None:
    """Caught at the door rather than three frames deep in ``bisect``, where comparing a naive
    datetime against the grid's aware nodes raises a ``TypeError`` naming neither."""
    with pytest.raises(ValueError, match="expiry must be timezone-aware"):
        make_view().tenor_of(NAIVE)


# --- forward_at: constant carry between the nodes


def test_forward_at_reproduces_the_forward_curve_exactly_at_every_node() -> None:
    view = make_view()

    for tenor, forward in zip(TENORS, FORWARDS, strict=True):
        assert view.forward_at(tenor) == forward


def test_forward_at_is_log_linear_and_not_linear_between_two_nodes() -> None:
    """Halfway between two nodes the answer is their geometric mean, not their arithmetic one.

    Constant carry means ``F(T) = F0 * exp(r * (T - T0))``, so the midpoint of the interval is
    ``sqrt(F0 * F1)``. Interpolating ``F`` itself would give ``(F0 + F1) / 2`` instead and imply
    an instantaneous rate that drifts across the interval and jumps at the node.

    The two conventions differ by about a third of a dollar on this curve, which is small in
    absolute terms and is exactly why the test pins it: at that size the wrong interpolation is
    invisible in a report and would be caught by nothing else.
    """
    view = make_view()
    mid_tenor = (TENORS[0] + TENORS[1]) / 2.0
    geometric = math.sqrt(FORWARDS[0] * FORWARDS[1])
    arithmetic = (FORWARDS[0] + FORWARDS[1]) / 2.0

    # The vacuity guard: on a flat forward curve the two conventions coincide and this test would
    # prove nothing, so pin that the builder's curve really does separate them.
    assert abs(geometric - arithmetic) > 0.3

    assert view.forward_at(mid_tenor) == pytest.approx(geometric, rel=1e-12)
    assert view.forward_at(mid_tenor) != pytest.approx(arithmetic, abs=1e-3)


def test_forward_at_clamps_flat_below_the_first_tenor() -> None:
    """Extending the first segment's carry downwards would compound a two-node slope into a
    region with no data at all."""
    view = make_view()

    # The vacuity guard: the curve is not flat, so a clamp is a real choice with a real cost.
    assert FORWARDS[0] != FORWARDS[1]

    assert view.forward_at(TENORS[0] / 2.0) == FORWARDS[0]
    assert view.forward_at(1e-9) == FORWARDS[0]


def test_forward_at_clamps_flat_above_the_last_tenor() -> None:
    view = make_view()

    assert FORWARDS[-2] != FORWARDS[-1]
    assert view.forward_at(TENORS[-1] * 5.0) == FORWARDS[-1]


def test_forward_at_returns_the_only_forward_of_a_single_tenor_grid() -> None:
    """One node states a level and says nothing whatsoever about a carry, so there is nothing to
    interpolate and nothing to extrapolate."""
    view = make_view(tenors=(TENORS[1],), expiries=(EXPIRIES[1],), forwards=(FORWARDS[1],))

    assert view.forward_at(TENORS[1]) == FORWARDS[1]
    assert view.forward_at(0.01) == FORWARDS[1]
    assert view.forward_at(10.0) == FORWARDS[1]


@pytest.mark.parametrize("bad", [0.0, -1.0, math.nan, math.inf])
def test_forward_at_rejects_a_tenor_that_is_not_positive_and_finite(bad: float) -> None:
    """An infinity would clamp silently to the last node and report a plausible forward for a
    coordinate that does not exist; a NaN would miss both clamps and index off the axis."""
    with pytest.raises(ValueError, match="tenor in years must be positive and finite"):
        make_view().forward_at(bad)
