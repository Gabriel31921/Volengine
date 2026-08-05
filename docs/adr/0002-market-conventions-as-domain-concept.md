# ADR-002: MarketConventions is a first-class domain concept

**Status:** Accepted · 2026-07 · Design.md §3

## Context

BTC on Deribit trades 24/7, uses ACT/365 fixed, and expires at exactly 08:00 UTC. SPX has
trading sessions, holiday calendars, discrete dividends with ex-dates, another expiry time,
and possibly a 252-business-day count. These differences decide the tenor, the forward and
therefore every number a calibrator sees.

If each adapter bakes in its own conventions, a difference between the BTC surface and the
SPX surface becomes impossible to attribute: is it the market, or is it our daycount?

## Decision

A declarative object per market: daycount, exact expiry instant, holiday and session
calendar, numeraire, and forward construction method. It is injected when the Market Data
context instance is built. Adapters *use* the conventions; they never *define* them.

## Alternatives considered

Bury the conventions inside each adapter, next to the parsing code that needs them. Cheaper
to write and impossible to audit.

## Consequences

- Everything convention-dependent — `tenor_years`, `forward`, the exact expiry instant — is
  computed **before** the contract boundary. Calibrators receive homogeneous numbers and
  never learn which market produced them.
- Adding a market means adding a conventions object, not a code path.
- The conventions object is the natural place for de-americanization to hook in later
  (ADR-007), still before the boundary.
- Daycount and the 08:00 UTC expiry are a known risk: small tenor errors are invisible at
  one year and very visible at one day.
