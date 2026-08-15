from __future__ import annotations

from datetime import UTC, date, datetime, time, timedelta, timezone

import pytest

from tests.market_data.builders import NAIVE, NOW, make_conventions
from tests.support import replace_field
from volengine.market_data.domain.errors import ExpiredInstrumentError, MarketDataError
from volengine.market_data.domain.market_conventions import (
    DayCount,
)

# --- construction


@pytest.mark.parametrize("field", ["market_id", "underlying"])
@pytest.mark.parametrize("bad", ["", " ", "   "])
def test_conventions_reject_a_blank_identifier(field: str, bad: str) -> None:
    with pytest.raises(ValueError, match=field.replace("_", " ")):
        replace_field(make_conventions(), field, bad)


def test_conventions_reject_an_expiry_time_in_another_zone() -> None:
    """The field is named _utc and there must be exactly one way to spell it."""
    with pytest.raises(ValueError, match="UTC"):
        make_conventions(expiry_time_utc=time(8, 0, tzinfo=timezone(timedelta(hours=2))))


def test_conventions_accept_an_explicitly_utc_expiry_time() -> None:
    assert make_conventions(expiry_time_utc=time(8, 0, tzinfo=UTC)).expiry_time_utc.hour == 8


# --- expiry_instant


def test_expiry_instant_applies_the_venue_time_of_day() -> None:
    """A symbol encodes a date; everything downstream reasons in instants."""
    assert make_conventions().expiry_instant(date(2026, 6, 27)) == datetime(
        2026, 6, 27, 8, 0, tzinfo=UTC
    )


def test_expiry_instant_is_timezone_aware() -> None:
    """InstrumentId rejects a naive expiry, so this is the seam that must not produce one."""
    assert make_conventions().expiry_instant(date(2026, 6, 27)).tzinfo is not None


def test_expiry_instant_normalises_an_aware_expiry_time() -> None:
    aware = make_conventions(expiry_time_utc=time(8, 0, tzinfo=UTC))
    assert aware.expiry_instant(date(2026, 6, 27)) == datetime(2026, 6, 27, 8, 0, tzinfo=UTC)


# --- tenor_years


def test_tenor_years_is_calendar_days_over_365_under_act_365f() -> None:
    tenor = make_conventions().tenor_years(NOW + timedelta(days=30), NOW)
    assert tenor == pytest.approx(30 / 365)


def test_tenor_years_counts_a_weekend_like_any_other_days() -> None:
    """The point of ACT/365F: a 24/7 market accumulates variance on Saturday too.

    This is the assertion ACT/252 would have to fail, which is why the two cannot share an
    implementation and why the daycount is a declared convention rather than a constant.
    """
    friday = datetime(2026, 7, 24, 8, 0, tzinfo=UTC)
    monday = friday + timedelta(days=3)
    assert make_conventions().tenor_years(monday, friday) == pytest.approx(3 / 365)


def test_tenor_years_resolves_below_one_day() -> None:
    """A six-hour option has a real tenor; rounding it to a whole day breaks the short end."""
    tenor = make_conventions().tenor_years(NOW + timedelta(hours=6), NOW)
    assert tenor == pytest.approx(0.25 / 365)


def test_tenor_years_rejects_an_expiry_in_the_past() -> None:
    with pytest.raises(ExpiredInstrumentError, match="no tenor"):
        make_conventions().tenor_years(NOW - timedelta(seconds=1), NOW)


def test_tenor_years_rejects_an_expiry_exactly_now() -> None:
    """Zero is not a tenor: SliceData requires it positive and the pricing formulas divide."""
    with pytest.raises(ExpiredInstrumentError):
        make_conventions().tenor_years(NOW, NOW)


def test_expired_instrument_error_belongs_to_the_context_hierarchy() -> None:
    """A caller catching MarketDataError must catch this one -- one hierarchy per context."""
    assert issubclass(ExpiredInstrumentError, MarketDataError)


@pytest.mark.parametrize("field", ["expiry", "now"])
def test_tenor_years_rejects_naive_instants(field: str) -> None:
    args = {"expiry": NOW + timedelta(days=30), "now": NOW, field: NAIVE}
    with pytest.raises(ValueError, match="timezone-aware"):
        make_conventions().tenor_years(**args)


def test_tenor_years_refuses_act_252_instead_of_approximating_it() -> None:
    """Declared but not implemented (ADR-007). Silently falling back to ACT/365F would give a
    plausible number that is wrong by the weekends, which nothing downstream could detect.
    """
    with pytest.raises(NotImplementedError, match="calendar"):
        make_conventions(day_count=DayCount.ACT_252).tenor_years(NOW + timedelta(days=30), NOW)


# --- is_expired


def test_is_expired_agrees_with_tenor_years() -> None:
    conventions = make_conventions()
    expired = NOW - timedelta(seconds=1)
    assert conventions.is_expired(expired, NOW)
    with pytest.raises(ExpiredInstrumentError):
        conventions.tenor_years(expired, NOW)


def test_is_expired_is_false_for_a_live_instrument() -> None:
    assert not make_conventions().is_expired(NOW + timedelta(seconds=1), NOW)


# --- the enums are a declared vocabulary


def test_daycount_values_are_readable_labels() -> None:
    """Domain enums, not wire format: nothing serialises these, the ACL maps them."""
    assert DayCount.ACT_365F.value == "ACT/365F"
