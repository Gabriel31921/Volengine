# ADR-019: The two producers are one function twice

**Status:** Accepted · 2026-08 · taken while building `neural_surface/` (F1-05)

## Context

`Calibrator.calibrate(previous, task)` is a pure function: state in, state out, chained by the use
case (Design §5.6). `SurfaceLearner` as drafted was not — it had an `evaluate` method, an `update`
with no history parameter, and a separate `restart`.

Design §5.7 and §6.5 compare the parametric and neural producers on the same market. A comparison
between two things shaped differently is a comparison of the shapes.

## Decision

Three changes, one idea: **the learner is the calibrator's function with different mathematics
inside.**

**`SurfaceLearner.evaluate` is deleted.** `LearnedSurface` is itself the evaluable abstraction
(Design §6.1). A learner evaluating on its behalf would make evaluation a capability of the
*learner*, and the ADR-010 hard gate would then need a live learner to judge anything — which is
the coupling that drags torch back within reach of the domain.

**`update` gains `previous: LearnedSurface | None`.** The learner becomes a pure function exactly
like `Calibrator.calibrate(previous, task)`. The two producers are then the same function twice,
differing only in the mathematics, which is what makes the comparisons mean anything.

**Consequently there is no `restart` method.** The scheduled restart of §6.4 is
`update(None, batch_from_buffer.snapshot())` — the same operation without a history. A separate
method would be the rarest and least exercised path in the system.

## Consequences

- Both producers are driven by handlers of the same shape (ADR-016), and the composition root wires
  them identically.
- The ADR-010 hard gate runs against a `LearnedSurface`, with no learner in scope.
- `SurfaceLearner.update` still returns a bare `LearnedSurface` where `Calibrator.calibrate`
  returns a metrics-carrying `CalibrationResult`. The asymmetry that remains is recorded in
  `docs/SEAMS.md`.
