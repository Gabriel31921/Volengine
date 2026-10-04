"""Risk's comparative report rendered as Markdown, and what each side was valued off."""

from __future__ import annotations

from benchmarks.comparative import render_comparison, render_last_surfaces

from tests.risk.builders import (
    make_calibrated_surface,
    make_position_risk,
    make_report,
)
from volengine.contracts.calibrated_surface import SurfaceStatus
from volengine.risk.domain.comparison import ComparativeReport, SurfaceDistance, pair_lines
from volengine.risk.domain.freshness_policy import FreshnessDecision


def comparable() -> ComparativeReport:
    baseline = make_report(producer_id="svi-scipy")
    challenger = make_report(
        producer_id="mlp-torch",
        positions=(make_position_risk(vol=0.6475, value=61_250.0),),
    )
    return ComparativeReport(
        market_id="BTC-DERIBIT",
        baseline=baseline,
        challenger=challenger,
        lines=pair_lines(baseline, challenger),
        distance=SurfaceDistance(rms_vol_bp=64.7, max_vol_bp=292.9, n_points=34),
    )


def test_a_comparable_report_has_one_row_per_paired_position_with_its_differences() -> None:
    text = render_comparison(comparable())
    rows = [line for line in text.splitlines() if line.startswith("| ") and "BTC" in line]

    assert len(rows) == 1
    assert "+100.0" in rows[0]  # 0.6475 - 0.6375 = one vol point, in basis points
    assert "+250.0000" in rows[0]


def test_a_comparable_report_states_the_total_and_the_surface_distance() -> None:
    text = render_comparison(comparable())

    assert "Total value difference over paired lines: +250.0000" in text
    assert "RMS 64.7 bp, max 292.9 bp" in text


def test_a_comparison_against_a_refusal_prints_the_reason_instead_of_a_table() -> None:
    report = ComparativeReport(
        market_id="BTC-DERIBIT",
        baseline=make_report(producer_id="svi-scipy"),
        challenger=make_report(
            producer_id="mlp-torch",
            freshness=FreshnessDecision.REJECT,
            positions=(),
            message="no surface yet",
        ),
        lines=(),
        distance=None,
    )

    text = render_comparison(report)

    assert "`mlp-torch`: no surface yet" in text
    assert "No comparison" in text
    assert "| Position" not in text


def test_the_last_surfaces_say_which_side_was_degraded_and_which_tenors_it_carried() -> None:
    degraded = make_calibrated_surface(status=SurfaceStatus.DEGRADED)

    text = render_last_surfaces({"svi-scipy": degraded})

    assert "`DEGRADED`" in text
    assert "0.082, 0.247, 1.000" in text
