# ADR-007: European index options before American single names

**Status:** Accepted · 2026-07 · Design.md §3

## Context

Extending beyond crypto means equities. SPX and ESTX50 options are European, and so are the
Deribit crypto options already supported. Single-name US equity options are American: their
price includes an early-exercise premium, so extracting an equivalent European implied vol
requires de-americanization (a binomial tree or a Ju-Zhong approximation, iterating on vol).

## Decision

Equities in v1 means European indices only. They enter with zero changes in the calibrators.
American options are an advanced extension with its own milestone.

## Alternatives considered

Support American options from the start. Rejected: the de-americanization error would be
indistinguishable from calibration error, and it contaminates the central promise of the
project — that the calibrators do not change when the market changes.

## Consequences

- `IBProvider` and equities stay outside phases 1 to 3 of the plan. The multi-market claim is
  demonstrated instead with two crypto markets, which is the same architectural proof without
  the API and account friction.
- When de-americanization arrives, it lives in the Market Data domain of the equities market,
  **before** the boundary. The snapshot keeps publishing homogeneous European implied vols
  and the calibrators never find out. That is the definitive test of whether the contract was
  designed correctly.
