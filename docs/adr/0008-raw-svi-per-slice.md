# ADR-008: Raw SVI per slice in v1

**Status:** Accepted · 2026-07 · Design.md §5.2

## Context

SVI parameterizes the total variance of a single expiry with five parameters. A surface is
many expiries. They can be calibrated independently, one slice at a time, or jointly through
a global parameterization that ties the tenors together.

## Decision

Raw SVI per slice, five parameters per expiry, independent calibrations.

## Alternatives considered

**SSVI / eSSVI** (Gatheral–Jacquier), where calendar and butterfly no-arbitrage hold by
construction. Deferred to an extension: the global fit couples every tenor, so warm start
goes from per-slice to per-surface and the whole calibration cycle changes shape. It is worth
doing, but as its own milestone with a comparison against this baseline.

## Consequences

- Calendar arbitrage risk **between** slices is accepted. It is measured and reported
  (`calendar_violation`), not silently ignored, and it becomes analysis material.
- Poor crypto slices with few liquid strikes are ill-conditioned with five free parameters.
  Mitigation: regularize toward the neighbouring slice, or fix `m` and `sigma` below N quotes.
- The contract does not change if SSVI arrives later. Only the domain and the adapter of
  `parametric_pricing` are affected, because the ACL is the single place that knows the
  parameterization.
