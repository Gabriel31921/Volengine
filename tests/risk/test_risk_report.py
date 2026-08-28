from __future__ import annotations

import math
from dataclasses import FrozenInstanceError, replace
from datetime import timedelta

import pytest

from tests.risk.builders import NAIVE, NOW, make_position, make_position_risk, make_report
from tests.support import replace_field
from volengine.risk.domain.freshness_policy import FreshnessDecision
from volengine.risk.domain.pricing import OptionKindR

# --- PositionRisk: the volatility


@pytest.mark.parametrize("bad", [0.0, -0.65, float("nan"), float("inf")])
def test_position_risk_rejects_a_non_positive_vol(bad: float) -> None:
    with pytest.raises(ValueError, match="implied volatility must be positive"):
        replace(make_position_risk(), vol=bad)


def test_position_risk_rejects_a_nan_vol_that_slips_past_a_bare_ordering_guard() -> None:
    """The trap this codebase keeps rediscovering, asserted rather than trusted.

    ``float("nan") <= 0`` is ``False``, so a guard written as ``if vol <= 0`` would admit the NaN
    and print it in the report, which is the last place anything can catch it.
    """
    nan = float("nan")
    assert not nan <= 0

    with pytest.raises(ValueError, match="positive and finite"):
        replace(make_position_risk(), vol=nan)


# --- PositionRisk: the value, which may be of either sign


def test_position_risk_accepts_a_negative_value() -> None:
    """A short position is a liability and its line in the report is a negative number."""
    line = make_position_risk(position=make_position(quantity=-4.0), value=-25_000.0)

    assert line.value == -25_000.0


def test_position_risk_accepts_a_zero_value() -> None:
    """``not 0.0`` is ``True``: a leg flattened intraday is worth nothing and still gets a line."""
    line = make_position_risk(value=0.0)

    assert line.value == 0.0


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_position_risk_rejects_a_non_finite_value(bad: float) -> None:
    with pytest.raises(ValueError, match="position value must be finite"):
        replace(make_position_risk(), value=bad)


# --- PositionRisk: the greeks


@pytest.mark.parametrize("field", ["delta", "gamma", "vega"])
@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_position_risk_rejects_a_non_finite_greek(field: str, bad: float) -> None:
    with pytest.raises(ValueError, match=f"{field} must be finite"):
        replace_field(make_position_risk(), field, bad)


@pytest.mark.parametrize("field", ["delta", "gamma", "vega"])
def test_position_risk_accepts_a_zero_greek(field: str) -> None:
    """A position with no quantity left has no sensitivities, and zero is not falsy data."""
    line = replace_field(make_position_risk(), field, 0.0)

    assert getattr(line, field) == 0.0


@pytest.mark.parametrize("field", ["delta", "gamma", "vega"])
def test_position_risk_accepts_a_negative_greek(field: str) -> None:
    """The greeks are already scaled by quantity, so a short leg is short all three of them."""
    line = replace_field(make_position_risk(), field, -2.5)

    assert getattr(line, field) == -2.5


def test_position_risk_carries_the_position_it_describes() -> None:
    """The line is self-describing: nobody needs the portfolio file to read it."""
    position = make_position(strike=58_000.0, kind=OptionKindR.PUT, quantity=-4.0)

    assert make_position_risk(position=position).position is position


# --- RiskReport: identity


@pytest.mark.parametrize("field", ["market_id", "producer_id"])
def test_report_rejects_an_empty_identity_field(field: str) -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        replace_field(make_report(), field, "")


# --- RiskReport: the two instants


def test_report_accepts_a_report_stamped_at_the_snapshot_instant() -> None:
    """Equality is allowed for the ``ManualClock`` of ADR-004: replay values at the same instant."""
    report = make_report(ts_snapshot=NOW, ts_report=NOW)

    assert report.ts_report == report.ts_snapshot


def test_report_rejects_a_report_stamped_before_its_snapshot() -> None:
    with pytest.raises(ValueError, match="cannot be stamped before the snapshot"):
        make_report(ts_snapshot=NOW, ts_report=NOW - timedelta(seconds=1))


def test_report_accepts_an_absent_snapshot_instant() -> None:
    """No surface at all: the provider had nothing, so there is no instant to report."""
    report = make_report(
        ts_snapshot=None,
        freshness=FreshnessDecision.REJECT,
        positions=(),
        message="no valid surface for BTC-DERIBIT",
    )

    assert report.ts_snapshot is None


def test_report_skips_the_ordering_check_when_there_is_no_snapshot() -> None:
    """With nothing to be older than, any report instant is admissible."""
    report = make_report(
        ts_snapshot=None,
        ts_report=NOW - timedelta(days=1),
        freshness=FreshnessDecision.REJECT,
        positions=(),
        message="no valid surface for BTC-DERIBIT",
    )

    assert report.ts_report == NOW - timedelta(days=1)


def test_report_rejects_a_naive_report_instant() -> None:
    with pytest.raises(ValueError, match="ts_report must be timezone-aware"):
        make_report(ts_report=NAIVE)


def test_report_rejects_a_naive_snapshot_instant() -> None:
    with pytest.raises(ValueError, match="ts_snapshot must be timezone-aware"):
        make_report(ts_snapshot=NAIVE)


# --- RiskReport: the rejection invariants, which are Design 7.2 made structural


def test_report_rejects_a_rejected_report_that_carries_positions() -> None:
    """Publishing a number under a REJECT label is the failure this context exists to prevent."""
    with pytest.raises(ValueError, match="rejected report must not carry any position"):
        make_report(
            freshness=FreshnessDecision.REJECT,
            positions=(make_position_risk(),),
            message="no valid surface for BTC-DERIBIT",
        )


@pytest.mark.parametrize("empty", [None, ""])
def test_report_rejects_a_rejected_report_without_a_message(empty: str | None) -> None:
    """A refusal that does not say why is not the explicit statement Design 7.2 asks for."""
    with pytest.raises(ValueError, match="rejected report must carry a message"):
        make_report(freshness=FreshnessDecision.REJECT, positions=(), message=empty)


def test_report_accepts_a_rejection_with_no_positions_and_a_message() -> None:
    report = make_report(
        freshness=FreshnessDecision.REJECT,
        positions=(),
        message="no valid surface: the last snapshot is 94 seconds old",
    )

    assert report.positions == ()
    assert report.message == "no valid surface: the last snapshot is 94 seconds old"


@pytest.mark.parametrize("decision", [FreshnessDecision.NORMAL, FreshnessDecision.DEGRADED])
def test_report_rejects_a_valued_report_with_no_positions(decision: FreshnessDecision) -> None:
    """A ``Portfolio`` cannot be empty, so an empty valued report means the book was dropped."""
    with pytest.raises(ValueError, match="must carry at least one position"):
        make_report(freshness=decision, positions=())


@pytest.mark.parametrize("decision", [FreshnessDecision.NORMAL, FreshnessDecision.DEGRADED])
def test_report_accepts_a_valued_report_with_a_message(decision: FreshnessDecision) -> None:
    """A message is optional outside a rejection: DEGRADED is the natural place to say how old."""
    report = make_report(freshness=decision, message="the snapshot is 11 seconds old")

    assert report.message == "the snapshot is 11 seconds old"


def test_report_is_frozen() -> None:
    """The verdict invariants are checked at construction, so they hold only while nothing moves."""
    report = make_report()

    with pytest.raises(FrozenInstanceError):
        report.freshness = FreshnessDecision.REJECT  # type: ignore[misc]


# --- RiskReport: the derived total


def test_total_value_sums_the_position_values() -> None:
    report = make_report(
        positions=(
            make_position_risk(value=61_000.0),
            make_position_risk(value=12_500.0),
        )
    )

    assert report.total_value == pytest.approx(73_500.0)


def test_total_value_nets_a_short_position_against_a_long_one() -> None:
    report = make_report(
        positions=(
            make_position_risk(value=61_000.0),
            make_position_risk(position=make_position(quantity=-4.0), value=-25_000.0),
        )
    )

    assert report.total_value == pytest.approx(36_000.0)


def test_total_value_would_differ_if_the_signs_were_ignored() -> None:
    """Vacuity guard for the test above: the two legs neither cancel nor share a magnitude.

    Without this, a total of ``36_000`` would also be produced by a sum that took absolute values
    of legs that happened to be chosen symmetrically, and the netting assertion would prove
    nothing about the sign ever reaching the sum.
    """
    values = (61_000.0, -25_000.0)
    unsigned = math.fsum(abs(one) for one in values)

    assert math.fsum(values) != unsigned
    assert abs(values[0]) != abs(values[1])


def test_total_value_is_zero_for_a_rejected_report() -> None:
    """The one number this object states with no surface behind it, and only alongside REJECT."""
    report = make_report(
        freshness=FreshnessDecision.REJECT,
        positions=(),
        message="no valid surface for BTC-DERIBIT",
    )

    assert report.total_value == 0.0


def test_total_value_is_derived_rather_than_stored() -> None:
    """An architectural assertion: there is no field to disagree with the lines above it."""
    report = make_report()

    assert "total_value" not in {field.name for field in report.__dataclass_fields__.values()}
    assert report.total_value == pytest.approx(report.positions[0].value)
