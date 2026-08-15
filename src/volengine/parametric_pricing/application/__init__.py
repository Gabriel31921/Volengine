"""Use cases of parametric calibration, and the anticorruption layer.

Receives a snapshot, prepares the fit, runs it through the calibrator port, applies the
acceptance criteria and publishes the result. Imports the domain plus ``contracts/``, never
``adapters/``.

``acl.py`` translates in both directions here: incoming ``MarketSnapshot`` into whatever the
domain works with, and the resulting parameters into a ``VolGrid``. Evaluating the fitted
smile onto the published grid happens on this side of the boundary, because the contract is
data and not an evaluable object (ADR-001).

This is also where a failed calibration becomes a decision rather than an exception:
republishing the previous surface marked ``STALE_REPUBLISH`` (ADR-006) keeps the consumer
supplied while telling it plainly that the number is old -- silence would be indistinguishable
from a healthy market that simply had not moved.
"""

from __future__ import annotations
