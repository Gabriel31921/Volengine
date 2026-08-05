# ADR-006: An honest old surface beats a broken new one

**Status:** Accepted · 2026-07 · Design.md §3

## Context

A calibration can fail its acceptance criteria: RMSE above the threshold, a parameter pinned
at a bound, the optimizer not converging. The chain may be degraded, the market may be
gapping. Something has to be published, or explicitly not published.

## Decision

On failure, publish a `CalibrationFailed` event and re-publish the last good surface with
`status=STALE_REPUBLISH`, keeping its **original** `ts_snapshot`.

## Alternatives considered

**Publish the bad surface anyway.** Downstream gets numbers that look fine and are wrong.
The worst outcome available.

**Publish nothing.** Risk cannot distinguish "the calibration failed" from "nothing has
started yet", and the failure becomes invisible.

## Consequences

- Degradation is always explicit and observable downstream.
- Risk derives freshness from `ts_snapshot`, never from `ts_calibrated`. That is precisely
  what makes a republished surface look as old as it really is: the calculation is recent,
  the market data is not.
- The calibration use case must retain the last good surface per market, which is one more
  reason the state lives in `application/` and not in the adapter.
- Combined with the freshness policy (Design.md §7.2), a market that freezes produces
  reports marked DEGRADED and eventually rejected, rather than confident stale numbers.
