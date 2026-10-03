"""The shape of a comparison between two producers, and the distance between two surfaces.

Pure domain: no clock, no cache, no use case. What is pinned here is the sign convention, the
refusal to compare a refusal, and that the surface distance measures the producers rather than
this context's own interpolation.
"""

from __future__ import annotations

import math
from dataclasses import replace

import pytest

from tests.risk.builders import (
    EXPIRIES,
    K_AXIS,
    TENORS,
    make_flat_view,
    make_position,
    make_position_risk,
    make_report,
    make_view,
    total_variance_grid,
)
from volengine.risk.domain.comparison import (
    ComparativeReport,
    PositionComparison,
    SurfaceDistance,
    pair_lines,
    surface_distance,
)
from volengine.risk.domain.freshness_policy import FreshnessDecision
from volengine.risk.domain.interpolation import implied_vol
from volengine.risk.domain.pricing import OptionKindR

# --- one position, twice


def test_every_difference_is_challenger_minus_baseline() -> None:
    line = PositionComparison(
        baseline=make_position_risk(vol=0.60, value=100.0, delta=5.0, gamma=0.1, vega=20.0),
        challenger=make_position_risk(vol=0.65, value=130.0, delta=4.0, gamma=0.4, vega=25.0),
    )

    assert line.vol_diff == pytest.approx(0.05)
    assert line.value_diff == pytest.approx(30.0)
    assert line.delta_diff == pytest.approx(-1.0)
    assert line.gamma_diff == pytest.approx(0.3)
    assert line.vega_diff == pytest.approx(5.0)


def test_two_different_positions_cannot_be_paired() -> None:
    with pytest.raises(ValueError, match="pairs one position with itself"):
        PositionComparison(
            baseline=make_position_risk(),
            challenger=make_position_risk(position=make_position(kind=OptionKindR.PUT)),
        )


# --- pairing the two reports' lines


def test_lines_are_paired_in_the_portfolios_order() -> None:
    near = make_position(expiry=EXPIRIES[0])
    far = make_position(expiry=EXPIRIES[2])
    baseline = make_report(positions=(make_position_risk(near), make_position_risk(far)))
    challenger = make_report(
        producer_id="mlp-torch",
        positions=(make_position_risk(near, value=1.0), make_position_risk(far, value=2.0)),
    )

    lines = pair_lines(baseline, challenger)

    assert [line.baseline.position for line in lines] == [near, far]
    assert [line.challenger.value for line in lines] == [1.0, 2.0]


def test_a_position_only_one_side_valued_gets_no_line() -> None:
    """It expired between the two snapshots; a line for it would be a difference against nothing."""
    near = make_position(expiry=EXPIRIES[0])
    far = make_position(expiry=EXPIRIES[2])
    baseline = make_report(positions=(make_position_risk(near), make_position_risk(far)))
    challenger = make_report(producer_id="mlp-torch", positions=(make_position_risk(far),))

    lines = pair_lines(baseline, challenger)

    assert [line.baseline.position for line in lines] == [far]


def test_equal_positions_are_paired_one_to_one() -> None:
    """Duplicates are legal in a book; each must meet its own counterpart, not the first one."""
    same = make_position()
    baseline = make_report(positions=(make_position_risk(same), make_position_risk(same)))
    challenger = make_report(
        producer_id="mlp-torch",
        positions=(make_position_risk(same, value=1.0), make_position_risk(same, value=2.0)),
    )

    assert [line.challenger.value for line in pair_lines(baseline, challenger)] == [1.0, 2.0]


# --- the comparative report


def make_comparison(
    baseline_freshness: FreshnessDecision = FreshnessDecision.NORMAL,
    challenger_freshness: FreshnessDecision = FreshnessDecision.NORMAL,
) -> ComparativeReport:
    """Two healthy reports on one market from two producers, one line apart by 30 in value."""

    baseline = make_report(positions=(make_position_risk(value=100.0),))
    challenger = make_report(producer_id="mlp-torch", positions=(make_position_risk(value=130.0),))
    if baseline_freshness is FreshnessDecision.REJECT:
        baseline = make_report(freshness=baseline_freshness, positions=(), message="stale")
    if challenger_freshness is FreshnessDecision.REJECT:
        challenger = make_report(
            producer_id="mlp-torch", freshness=challenger_freshness, positions=(), message="stale"
        )
    comparable = FreshnessDecision.REJECT not in (baseline_freshness, challenger_freshness)
    distance = SurfaceDistance(rms_vol_bp=1.0, max_vol_bp=2.0, n_points=3)
    return ComparativeReport(
        market_id="BTC-DERIBIT",
        baseline=baseline,
        challenger=challenger,
        lines=pair_lines(baseline, challenger) if comparable else (),
        distance=distance if comparable else None,
    )


def test_the_total_difference_sums_the_paired_lines() -> None:
    assert make_comparison().total_value_diff == pytest.approx(30.0)


def test_two_reports_with_numbers_are_comparable() -> None:
    assert make_comparison().comparable


@pytest.mark.parametrize("side", ["baseline", "challenger"])
def test_a_rejected_side_makes_the_comparison_incomparable(side: str) -> None:
    reject = FreshnessDecision.REJECT
    comparison = (
        make_comparison(baseline_freshness=reject)
        if side == "baseline"
        else make_comparison(challenger_freshness=reject)
    )

    assert not comparison.comparable
    assert comparison.lines == ()
    assert comparison.total_value_diff == 0.0


def test_a_comparison_against_a_refusal_cannot_carry_lines() -> None:
    """A difference against a refusal would be one report's numbers relabelled as disagreement."""
    healthy = make_comparison()
    rejected = make_report(
        producer_id="mlp-torch", freshness=FreshnessDecision.REJECT, positions=(), message="old"
    )

    with pytest.raises(ValueError, match="rejected report"):
        replace(healthy, challenger=rejected)


def test_a_comparison_against_a_refusal_cannot_carry_a_distance() -> None:
    rejected = make_report(
        producer_id="mlp-torch", freshness=FreshnessDecision.REJECT, positions=(), message="old"
    )

    with pytest.raises(ValueError, match="rejected report"):
        replace(make_comparison(), challenger=rejected, lines=())


def test_a_producer_cannot_be_compared_with_itself() -> None:
    with pytest.raises(ValueError, match="two producers"):
        replace(make_comparison(), challenger=make_report(producer_id="svi-scipy"))


def test_both_reports_must_price_the_comparisons_market() -> None:
    with pytest.raises(ValueError, match="Both reports must price"):
        replace(make_comparison(), market_id="ETH-DERIBIT")


# --- the distance itself


def test_a_surface_is_at_zero_distance_from_itself() -> None:
    distance = surface_distance(make_view(), make_view(producer_id="mlp-torch"))

    assert distance is not None
    assert distance.rms_vol_bp == 0.0
    assert distance.max_vol_bp == 0.0


def test_identical_grids_are_compared_once_per_node() -> None:
    """The union of two equal node sets is one copy of it, not two."""
    distance = surface_distance(make_view(), make_view())

    assert distance is not None
    assert distance.n_points == len(TENORS) * len(K_AXIS)


def test_a_parallel_shift_is_measured_in_basis_points() -> None:
    """65% against 66% everywhere is 100 bp, at every point and in the RMS."""
    distance = surface_distance(make_flat_view(0.65), make_flat_view(0.66))

    assert distance is not None
    assert distance.rms_vol_bp == pytest.approx(100.0)
    assert distance.max_vol_bp == pytest.approx(100.0)


def test_the_distance_is_symmetric_when_the_grids_differ() -> None:
    """The union of both node sets is what makes swapping the roles give the same number."""
    coarse = make_view()
    fine_k = (-0.20, -0.15, -0.10, -0.05, 0.0, 0.05, 0.10, 0.15, 0.20)
    fine = make_view(
        producer_id="mlp-torch",
        log_moneyness=fine_k,
        total_variance=tuple(
            tuple(w * 1.02 for w in row) for row in total_variance_grid(log_moneyness=fine_k)
        ),
    )

    there = surface_distance(coarse, fine)
    back = surface_distance(fine, coarse)

    assert there is not None and back is not None
    assert there.rms_vol_bp == pytest.approx(back.rms_vol_bp, rel=1e-12)
    assert there.n_points == back.n_points == len(TENORS) * len(fine_k)


def test_points_outside_the_other_surface_are_not_compared() -> None:
    """A wing one surface does not reach would be measured against the consumer's own clamp."""
    narrow_k = (-0.10, 0.0, 0.10)
    narrow = make_view(producer_id="mlp-torch", log_moneyness=narrow_k)

    distance = surface_distance(make_view(), narrow)

    assert distance is not None
    assert distance.n_points == len(TENORS) * len(narrow_k)


def test_a_wing_outside_the_overlap_does_not_move_the_distance() -> None:
    """The guard on the test above: poison the wing only the wide surface has, and nothing moves.

    Without it, the exclusion could be vacuous -- a poisoned wing that was compared after all would
    show up here as a distance far from zero.
    """
    narrow = make_view(producer_id="mlp-torch", log_moneyness=(-0.10, 0.0, 0.10))
    poisoned_rows = tuple(
        (row[0] * 4.0, *row[1:-1], row[-1] * 4.0) for row in total_variance_grid()
    )
    poisoned = make_view(total_variance=poisoned_rows)
    wing = implied_vol(make_view(), -0.20, TENORS[0])
    assert implied_vol(poisoned, -0.20, TENORS[0]) > 1.5 * wing

    distance = surface_distance(poisoned, narrow)

    assert distance is not None
    assert distance.max_vol_bp == pytest.approx(0.0, abs=1e-9)


def test_surfaces_with_no_overlap_have_no_distance() -> None:
    """Not zero: "we could not look" is not "we looked and they agree"."""
    left = make_view(log_moneyness=(-0.40, -0.35, -0.30))
    right = make_view(producer_id="mlp-torch", log_moneyness=(0.30, 0.35, 0.40))

    assert surface_distance(left, right) is None


@pytest.mark.parametrize(
    ("rms", "maximum", "n_points", "fragment"),
    [
        (math.nan, 1.0, 1, "RMS distance"),
        (-1.0, 1.0, 1, "RMS distance"),
        (2.0, 1.0, 1, "at least the RMS"),
        (1.0, math.inf, 1, "at least the RMS"),
        (1.0, 1.0, 0, "at least one point"),
    ],
)
def test_a_distance_refuses_inconsistent_numbers(
    rms: float, maximum: float, n_points: int, fragment: str
) -> None:
    with pytest.raises(ValueError, match=fragment):
        SurfaceDistance(rms_vol_bp=rms, max_vol_bp=maximum, n_points=n_points)
