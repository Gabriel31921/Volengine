# ADR-023: `FlatVolCalibrator`'s departure from `Plan.md` in F1-08

**Status:** Accepted · 2026-08 · taken while building the walking skeleton

## Context

ADR-020, ADR-021 and ADR-022 collect the departures from `Implementation.md` found while building
the domain layer, the ACLs and use cases, and the composition root. F1-08 produced exactly one
departure of its own, against `Plan.md` rather than `Implementation.md`, and it needed the same
kind of record for the same reason: `CLAUDE.md` says a phase's departures are collected in an ADR,
and editing one of the three existing records to carry it would misdate a decision taken in a
different phase and make that record's own heading false.

This departure is one item, not a batch, but it is not self-explanatory from the port signature
alone — `Plan.md` and `Implementation.md` both describe the calibrator in terms a reader could
reasonably expect the code to match, and it does not — so it earns the same kind of record as the
other three rather than being left to the docstring alone.

---

## `FlatVolCalibrator` returns a `CalibrationResult`, not "a constant vol grid"

`Plan.md`'s F1-08 section describes `FlatVolCalibrator` as implementing `Calibrator` and
"returning a constant vol grid (pure numpy)". The `Calibrator` port returns `CalibrationResult`,
and that is what the adapter returns here too: each slice's weighted-mean volatility is expressed
as the degenerate SVI parameters `w(k) = a`, `b = 0`, alongside the RMSE that mean actually makes
against the quotes it was fitted to.

The grid a report eventually carries is built one layer up, by
`parametric_pricing/application/acl.to_calibrated_surface`, which evaluates a producer's
`SVIParams` on the configured mesh (ADR-011). Flat parameters are how this calibrator asks for a
constant grid; a port that returned the grid directly would put the mesh — and therefore
configuration — inside every calibrator instead of behind the one ACL that already owns the
translation to the published contract.

The code is right and `Plan.md`'s wording is loose, not wrong in intent: a flat fit *is* a constant
grid, just not a constant grid returned directly. Nothing changes about the plan's goal for F1-08.

## Consequences

- `Plan.md` is not amended — a plan and an ADR disagreeing is exactly what this record exists to
  resolve, and `CLAUDE.md` says the ADR wins.
- The reasoning also lives in `FlatVolCalibrator`'s own docstring
  (`parametric_pricing/adapters/flat_vol.py`), which is where a reader meets the departure first;
  this record is where `CLAUDE.md` says it must also be collected.
- A future calibrator that fits a real smile returns `CalibrationResult` the same way, so this is
  not a one-off shape for the trivial adapter — F2-05's scipy calibrator inherits it rather than
  reopening the question.
