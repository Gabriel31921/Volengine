# ADR-021: ACL and event decisions of F1-06

**Status:** Accepted · 2026-08 · taken while building the four ACLs and five use cases

## Context

The ACLs and use cases are where the domain meets the published language, and `Implementation.md`
specifies neither in the detail the boundary turned out to need. The decisions below were taken
while building them. The shape of the use cases themselves is ADR-016; the weighting is ADR-018;
the out-of-the-money rule is ADR-017.

---

## `handle` returns a tuple of events, never an empty one

A refused calibration is **two facts at once** — `CalibrationFailed`, *and* the previous surface
republished as `STALE_REPUBLISH` (ADR-006). Either one dropped loses something the pipeline was
built to carry.

Never an empty tuple, either: a cycle that ran and said nothing is indistinguishable downstream
from one that never ran.

## `snapshot_id` and `surface_id` are derived, never random

`"BTC-DERIBIT:00000042"` from a sequence the use case owns, and `"{snapshot_id}/{producer_id}"` for
a surface.

A `uuid4` would make two runs of one recording incomparable line by line while proving nothing a
derivation does not — and ADR-004's determinism is the whole point of the replay. The surface's
form is derived from the snapshot's because **two producers fit the same snapshot by design**, and
the identifier should say so.

## Clock skew is reconciled inside a configured tolerance, in both directions

Within the tolerance the **venue's** stamp is kept: that lag is the latency the freshness policy
must be able to see, and clamping it would hide the thing being measured.

Outside it, `ts_local` is used. A venue far ahead breaks `CalibratedSurface`'s ordering invariant
(`ts_calibrated < ts_snapshot`) and fails loudly. A venue far behind is **worse**, because nothing
fails at all: Risk simply reports `DEGRADED` and then `REJECT` for a market that is quoting
perfectly.

## Acceptance is per slice

A surface may publish the expiries that fitted and omit one that did not, marked `DEGRADED`. This
is what `CalibrationResult`'s own docstring means by "the ACL turns the accepted slices into the
published surface".

## `converged` and `at_bound` are not configurable — only the RMSE threshold is

A deployment that could switch either check off is a deployment that eventually does.

## `LastValueSurfaceProvider` lives in `risk/application/`, not `adapters/`

It consumes DTOs, so rule 5 puts it beside the ACL, and it talks to no technology whatsoever.

## `ComputeReportUseCase` takes an `expected_producer_id`

Used in exactly one case: the report with no surface at all, where `RiskReport` demands a producer
and there is nobody to name. Every other report takes it from `SurfaceView.producer_id`.

## The report is stamped `max(now, ts_snapshot)`

This is where two rules of Risk's own domain disagree. `FreshnessPolicy` calls a negative age
`NORMAL`; `RiskReport` refuses to be stamped before its input. Only the use case sees both.

The age stays negative in the metric, so the skew is reported rather than hidden.

## The neural replay draw happens *before* the fresh points are added to the buffer

Otherwise a snapshot's own quotes come straight back out of the buffer and train the model twice at
double weight — guaranteed on the very first cycle.

## Consequences

- The identifiers make a recorded run diffable line by line against a replay of it, which is what
  ADR-004 is for.
- The skew tolerance is configuration (ADR-012), because how far a venue's clock may drift is a
  judgement about a venue.
- Pruning before the draw has a price, recorded in `docs/SEAMS.md`.
