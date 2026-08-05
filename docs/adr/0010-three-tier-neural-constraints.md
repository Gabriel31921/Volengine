# ADR-010: Three tiers of constraints on the neural surface

**Status:** Accepted · 2026-07 · Design.md §6.3

## Context

An MLP fits data. Nothing in it prevents producing a surface that admits arbitrage — a
butterfly violation in the wings, or total variance decreasing in maturity. The question is
how a domain invariant governs an artefact produced by machine learning.

## Decision

Three tiers, with different jobs.

1. **Soft — they train.** Durrleman's condition on a dense moneyness mesh, plus monotonicity
   of total variance in T, penalized inside the loss.
2. **Architectural — not built in v1.** Constructions with guaranteed partial monotonicity or
   convexity (non-negative weights on selected paths, partial ICNNs). An advanced extension.
3. **Hard — they govern.** The domain invariant is checked on the grid *before publishing*.
   If it fails, nothing is published and the failure is reported.

## Alternatives considered

**Soft constraints only.** You publish arbitrageable surfaces and hope the penalty was strong
enough. The failure mode is silent and downstream.

**Hard constraints only.** Nothing trains: the gradient has no signal pushing it toward the
feasible region.

## Consequences

- The slogan is the design: *soft constraints train, hard constraints govern*.
- The hard gate lives in `neural_surface/domain` and imports no torch, so the invariant is
  testable without a model.
- A rejected publication is observable as a metric, which makes the failure rate an honest
  quality signal for the neural producer.
- The Durrleman machinery is conceptually shared with the parametric context but duplicated
  per context on purpose — unifying it would be the canonical-model anti-pattern.
