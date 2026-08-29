# ADR-015: `VolGrid` carries expiries and forwards

**Status:** Accepted · 2026-08 · taken while building `risk/` (F1-05)

## Context

`contracts/` was frozen in F1-03. Building `risk/` reopened it, and this is the **one place F1-03
changed**.

A `Position` is an option with a contractual expiry and a strike in currency. `VolGrid` is indexed
by year fraction and log-moneyness. Placing the first on the second needs a daycount and a forward
— and ADR-002 forbids Risk from owning either, because every convention belongs to Market Data and
is resolved before the boundary.

Without them the contract's only consumer could not use it. The tempting workaround,
`ts_snapshot + tenor * 365 days`, is a daycount assumption wearing the clothes of arithmetic: right
under ACT/365F, days out under a business-day count, and with nothing in the data to say which.

## Decision

`VolGrid` gains `expiries: tuple[datetime, ...]` and `forwards: tuple[float, ...]`, one per tenor
node. `SliceData` already carried both, for the same reason.

## Consequences

- `SurfaceView.tenor_of` interpolates in calendar time between the grid's own `(expiry, tenor)`
  pairs, and so recovers a year fraction without Risk ever naming a convention.
- `SurfaceView.forward_at` reads the curve the producer published rather than inventing one.
- The grid stays data (ADR-001): these are two more axes of numbers, not behaviour.
- ADR-002 holds unweakened — the convention is still computed once, upstream, and travels as its
  result.
