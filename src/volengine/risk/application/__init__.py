"""Use cases of risk, and the anticorruption layer.

Receives a calibrated surface, decides whether it is fresh enough to use, values the book and
reports. Imports the domain plus ``contracts/``, never ``adapters/``.

``acl.py`` translates ``CalibratedSurface`` into the domain's own representation, which is
where the ``VolGrid`` stops being nested tuples (ADR-011) and becomes whatever the
interpolation actually wants. The conversion is not free, and doing it once at the boundary
rather than repeatedly inside the valuation is the reason it lives here.

Conflation means this context may never see some surfaces at all (ADR-003): under a slow
consumer the mailbox keeps only the most recent one. That is intended -- a stale surface has
no value once a newer one exists -- but it does mean nothing here may assume it has observed
every calibration, which is why ``source_snapshot_id`` and not arrival order is what ties a
result back to its input.
"""

from __future__ import annotations
