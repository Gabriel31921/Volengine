"""The report, and the three ways it can honestly have nothing to say.

Design 7.2 made testable: two thresholds, one clock read by hand, and a verdict that decides
whether there are numbers at all. Every test here runs on a ``ManualClock``, which is what ADR-004
promised would make the freshness decision reproducible -- a wall clock would make ``DEGRADED``
appear or not according to how busy the machine was.
"""

from __future__ import annotations

from datetime import timedelta

import pytest

from tests.risk.builders import (
    EXPIRIES,
    NOW,
    StubSurfaceProvider,
    make_bumps,
    make_policy,
    make_portfolio,
    make_position,
    make_view,
)
from tests.support import RecordingMetrics
from volengine.platform.clock import ManualClock
from volengine.risk.application.compute_report import ComputeReportUseCase, ReportSettings
from volengine.risk.domain.freshness_policy import FreshnessDecision
from volengine.risk.domain.portfolio import Portfolio
from volengine.risk.domain.pricing import OptionKindR
from volengine.risk.domain.surface_view import SurfaceView

MARKET = "BTC-DERIBIT"


def make_use_case(
    view: SurfaceView | None = None,
    portfolio: Portfolio | None = None,
    now: float = 2.0,
    warn_seconds: float = 5.0,
    reject_seconds: float = 30.0,
    expected_producer_id: str = "svi-scipy",
) -> tuple[ComputeReportUseCase, RecordingMetrics]:
    """The use case with a clock ``now`` seconds after the surface's snapshot instant."""
    metrics = RecordingMetrics()
    use_case = ComputeReportUseCase(
        provider=StubSurfaceProvider(view),
        portfolio=portfolio if portfolio is not None else make_portfolio(),
        policy=make_policy(warn_seconds=warn_seconds, reject_seconds=reject_seconds),
        clock=ManualClock(NOW + timedelta(seconds=now)),
        metrics=metrics,
        settings=ReportSettings(bumps=make_bumps()),
        expected_producer_id=expected_producer_id,
    )
    return use_case, metrics


# --- the healthy report


def test_a_fresh_surface_produces_one_line_per_position() -> None:
    use_case, _ = make_use_case(view=make_view())

    report = use_case.compute(MARKET)

    assert report.freshness is FreshnessDecision.NORMAL
    assert len(report.positions) == 2


def test_the_report_names_the_producer_that_supplied_the_surface() -> None:
    """Identity travels in the answer, not on the port."""
    use_case, _ = make_use_case(view=make_view(producer_id="mlp-torch"))

    assert use_case.compute(MARKET).producer_id == "mlp-torch"


def test_the_report_carries_the_surfaces_instant_and_its_own() -> None:
    """A fresh report of a stale market is a state that has to remain reportable."""
    use_case, _ = make_use_case(view=make_view(), now=2.0)

    report = use_case.compute(MARKET)

    assert report.ts_snapshot == NOW
    assert report.ts_report == NOW + timedelta(seconds=2)


def test_the_total_nets_a_long_against_a_short() -> None:
    """A book cancels by design, which is why the total is an fsum over the lines."""
    use_case, _ = make_use_case(view=make_view())

    report = use_case.compute(MARKET)

    assert report.total_value == pytest.approx(
        sum(line.value for line in report.positions), rel=1e-12
    )


def test_a_short_position_reports_a_negative_value() -> None:
    """The vacuous-pass guard on the netting above: the sign has to reach the line."""
    book = Portfolio(
        positions=(make_position(quantity=-4.0),),
    )
    use_case, _ = make_use_case(view=make_view(), portfolio=book)

    assert use_case.compute(MARKET).positions[0].value < 0


# --- freshness


def test_a_surface_past_the_warning_threshold_is_reported_degraded() -> None:
    use_case, _ = make_use_case(view=make_view(), now=10.0, warn_seconds=5.0)

    report = use_case.compute(MARKET)

    assert report.freshness is FreshnessDecision.DEGRADED
    assert report.positions


def test_a_degraded_report_says_how_old_the_data_is() -> None:
    use_case, _ = make_use_case(view=make_view(), now=10.0, warn_seconds=5.0)

    message = use_case.compute(MARKET).message

    assert message is not None
    assert "10.0" in message


def test_a_normal_report_needs_no_message() -> None:
    use_case, _ = make_use_case(view=make_view(), now=1.0)

    assert use_case.compute(MARKET).message is None


def test_a_surface_past_the_rejection_threshold_carries_no_numbers() -> None:
    """The label and the content cannot come apart; ``RiskReport`` enforces it, this proves it."""
    use_case, _ = make_use_case(view=make_view(), now=120.0, reject_seconds=30.0)

    report = use_case.compute(MARKET)

    assert report.freshness is FreshnessDecision.REJECT
    assert report.positions == ()


def test_a_rejected_report_still_names_the_instant_it_refused() -> None:
    """The age is the reason, and a refusal without the evidence cannot be checked."""
    use_case, _ = make_use_case(view=make_view(), now=120.0, reject_seconds=30.0)

    report = use_case.compute(MARKET)

    assert report.ts_snapshot == NOW
    assert report.message is not None
    assert MARKET in report.message


def test_a_surface_stamped_in_the_future_is_reported_normal() -> None:
    """A venue clock a second fast must not take the whole risk report down."""
    use_case, _ = make_use_case(view=make_view(), now=-1.0)

    assert use_case.compute(MARKET).freshness is FreshnessDecision.NORMAL


def test_a_report_is_never_stamped_before_the_surface_it_describes() -> None:
    """Where two rules of this context's own domain disagree: the policy allows what the report
    refuses, and the use case is the only place that sees both."""
    use_case, _ = make_use_case(view=make_view(), now=-1.0)

    assert use_case.compute(MARKET).ts_report == NOW


def test_the_skew_stays_visible_in_the_age_even_though_the_stamp_was_clamped() -> None:
    """The vacuous-pass guard: clamping must not also hide the anomaly it absorbed."""
    use_case, metrics = make_use_case(view=make_view(), now=-1.0)

    use_case.compute(MARKET)

    assert metrics.gauge_value("risk.surface_age_seconds") == pytest.approx(-1.0)


# --- nothing to value


def test_an_empty_provider_produces_a_rejection_rather_than_an_exception() -> None:
    """Ordinary at start-up: no producer has published anything yet."""
    use_case, _ = make_use_case(view=None)

    report = use_case.compute(MARKET)

    assert report.freshness is FreshnessDecision.REJECT
    assert report.message is not None


def test_a_report_with_no_surface_has_no_snapshot_instant() -> None:
    """``None`` is an answer. Inventing one would read as perfectly fresh or absurdly stale."""
    use_case, _ = make_use_case(view=None)

    assert use_case.compute(MARKET).ts_snapshot is None


def test_a_report_with_no_surface_is_attributed_to_the_expected_producer() -> None:
    """``RiskReport`` needs a producer and there is nobody to name, so the label fills in."""
    use_case, _ = make_use_case(view=None, expected_producer_id="mlp-torch")

    assert use_case.compute(MARKET).producer_id == "mlp-torch"


def test_an_empty_expected_producer_is_refused_at_construction() -> None:
    with pytest.raises(ValueError, match="producer id"):
        make_use_case(view=make_view(), expected_producer_id="")


# --- expired positions


def test_an_expired_position_is_skipped_rather_than_fatal() -> None:
    """A portfolio file is configuration and outlives its contracts."""
    book = Portfolio(
        positions=(
            make_position(expiry=NOW - timedelta(days=1)),
            make_position(),
        )
    )
    use_case, metrics = make_use_case(view=make_view(), portfolio=book)

    report = use_case.compute(MARKET)

    assert len(report.positions) == 1
    assert "risk.position.expired" in metrics.counter_names()


def test_a_book_that_has_entirely_expired_is_a_rejection() -> None:
    """A total of ``0.0`` over no lines is indistinguishable from a book that lost everything."""
    book = Portfolio(positions=(make_position(expiry=NOW - timedelta(days=1)),))
    use_case, _ = make_use_case(view=make_view(), portfolio=book)

    report = use_case.compute(MARKET)

    assert report.freshness is FreshnessDecision.REJECT
    assert report.message is not None
    assert "expired" in report.message


def test_a_position_past_the_last_tenor_is_still_valued() -> None:
    """Flat extrapolation is a documented weakness, not a reason to drop a line from the book."""
    book = Portfolio(positions=(make_position(expiry=EXPIRIES[-1] + timedelta(days=400)),))
    use_case, _ = make_use_case(view=make_view(), portfolio=book)

    assert len(use_case.compute(MARKET).positions) == 1


# --- what the report reports


def test_the_staleness_actually_observed_is_gauged() -> None:
    """The one measurement that includes every hop at once, including the uninstrumented ones."""
    use_case, metrics = make_use_case(view=make_view(), now=3.0)

    use_case.compute(MARKET)

    assert metrics.gauge_value("risk.surface_age_seconds") == pytest.approx(3.0)


def test_the_freshness_decision_is_counted_so_a_session_can_be_split_by_it() -> None:
    use_case, metrics = make_use_case(view=make_view(), now=10.0, warn_seconds=5.0)

    use_case.compute(MARKET)

    counted = [tags for name, _, tags in metrics.counters if name == "risk.report.freshness"]
    assert counted[0]["decision"] == "DEGRADED"


def test_a_refusal_is_counted() -> None:
    use_case, metrics = make_use_case(view=None)

    use_case.compute(MARKET)

    assert "risk.report.rejected" in metrics.counter_names()


# --- the greeks reach the report


def test_a_call_reports_a_positive_delta_scaled_by_quantity() -> None:
    book = Portfolio(positions=(make_position(quantity=10.0, kind=OptionKindR.CALL),))
    use_case, _ = make_use_case(view=make_view(), portfolio=book)

    line = use_case.compute(MARKET).positions[0]

    assert 0 < line.delta < 10.0


def test_the_volatility_used_is_reported_beside_the_value() -> None:
    """Without it, a line cannot be checked against the surface it came from."""
    use_case, _ = make_use_case(view=make_view())

    assert use_case.compute(MARKET).positions[0].vol > 0


# --- configuration


@pytest.mark.parametrize("bad", [0.0, -1.0, float("nan"), float("inf")])
def test_an_unusable_discount_is_refused(bad: float) -> None:
    with pytest.raises(ValueError, match="discount"):
        ReportSettings(bumps=make_bumps(), discount=bad)
