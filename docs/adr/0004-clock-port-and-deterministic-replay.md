# ADR-004: Clock port from day one, and deterministic replay

**Status:** Accepted · 2026-07 · Design.md §3

## Context

Almost every rule in this system depends on time: the snapshot cadence, quote staleness, the
freshness policy that decides whether a risk report is valid. A system whose behaviour
depends on the wall clock and on a live exchange feed cannot be tested reproducibly, and a
bug seen once during a volatile session cannot be reproduced at all.

## Decision

A `Clock` port from the first commit, with three implementations: `SystemClock` for
production, `ManualClock` for tests, and `SimulatedClock` driven by recorded timestamps.
Together with `SyntheticProvider` and `RecordedProvider` this gives deterministic replay: a
recorded file of exchange messages produces the same final risk report, every time.

## Alternatives considered

Call `datetime.now()` directly where time is needed. Every time-dependent rule becomes
untestable except by sleeping, and replay becomes impossible.

## Consequences

- No module outside `platform/` reads the system clock. Time is an injected dependency.
- Time-dependent tests run instantly: `ManualClock.advance(3600)` instead of waiting.
- One recorded hour of real Deribit traffic becomes a golden fixture for a permanent
  end-to-end test.
- The calibrator adapters must be pure functions for this to hold — state is chained by the
  use case, not kept inside the adapter (see ADR-008 and Design.md §5.6).
