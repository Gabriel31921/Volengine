from __future__ import annotations

from datetime import UTC, datetime, timedelta, timezone

import pytest

from volengine.shared_kernel.domain.instants import require_aware


def test_an_aware_instant_passes() -> None:
    require_aware(datetime(2026, 7, 27, 8, 0, tzinfo=UTC), "ts")


def test_a_non_utc_zone_is_still_aware() -> None:
    """The rule is that a zone exists, not which one: arithmetic works across zones."""
    require_aware(datetime(2026, 7, 27, 10, 0, tzinfo=timezone(timedelta(hours=2))), "ts")


def test_a_naive_instant_is_refused_by_its_field_name() -> None:
    """The field name is the whole point: deferred, this failure surfaces as a TypeError from
    inside an unrelated subtraction, with nothing in the message saying what was wrong.
    """
    with pytest.raises(ValueError, match="ts_exchange must be timezone-aware"):
        require_aware(datetime(2026, 7, 27, 8, 0), "ts_exchange")
