"""Business rules of risk: the book, interpolation, greeks, staleness policy.

Imports are stdlib, ``shared_kernel/`` and numpy. Never ``contracts/`` -- so this layer works
on its own surface representation, and ``application/acl.py`` builds it from the incoming
``CalibratedSurface``. The awkward-looking consequence is the useful one: the interpolation
and the greeks can be tested against a hand-built surface with no bus, no calibrator and no
snapshot anywhere in sight.

Freshness is a rule here, not a detail. A surface arriving with ``STALE_REPUBLISH`` (ADR-006)
is still usable and must be treated differently from a fresh one, and how much staleness is
tolerable is configuration (ADR-012) rather than a constant, because the answer differs per
market and per use.
"""

from __future__ import annotations
