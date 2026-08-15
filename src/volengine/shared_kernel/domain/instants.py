"""Shared vocabulary about instants.

The shared kernel is kept deliberately tiny -- growing it is how a shared kernel turns into
the canonical model DDD exists to prevent. This earns its place because it is not anyone's
business logic: no context owns the rule that a timestamp must carry a zone, every context
needs it, and it had been copied into six modules before it was worth a name.

Importing stdlib only, it can be reached from ``contracts/`` and from any ``domain/`` without
breaking a single import rule.
"""

from __future__ import annotations

from datetime import datetime


def require_aware(ts: datetime, field: str) -> None:
    """Reject a naive datetime at construction, while the offending field is still known.

    Staleness, tenors and latencies are all subtractions of datetimes, and subtracting a naive
    one from an aware one raises ``TypeError``. Deferred, that surfaces from inside arithmetic
    far away from whoever built the bad value, with nothing in the message naming the field.

    Raises:
        ValueError: If ``ts`` has no timezone.
    """
    if ts.tzinfo is None:
        raise ValueError(f"{field} must be timezone-aware, got naive {ts}")
