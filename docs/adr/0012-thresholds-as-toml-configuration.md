# ADR-012: Thresholds are configuration data, not constants

**Status:** Accepted · 2026-07 · project decision, not in Design.md

## Context

The system is full of numbers that decide behaviour: snapshot cadence, the material-move
threshold, admissibility limits on spread, age and moneyness, the calibration acceptance
RMSE, the two freshness bands. They differ per market, and every one of them will be retuned
once real Deribit data shows how bad the wings actually are.

## Decision

Frozen dataclasses in `entrypoints/config.py`, loaded from TOML with `tomllib`, injected from
the composition root down into the use cases.

## Alternatives considered

**Module-level constants.** Retuning a threshold becomes a code change, and per-market values
degenerate into conditionals inside the domain.

**Environment variables.** Flat, untyped, and hostile to the nested per-market structure this
configuration actually has.

## Consequences

- Tuning is a configuration change, not a code change — which matters because tuning against
  real data is an explicit task of phase 3.
- Multi-market becomes a second table in the TOML file rather than a second code path.
- Tests construct configuration objects directly and never read a file.
- `tomllib` is in the standard library since 3.11, so this adds no dependency.
- Nothing in `*/domain/` reads configuration: it receives already-built threshold objects,
  which keeps the domain free of I/O.
