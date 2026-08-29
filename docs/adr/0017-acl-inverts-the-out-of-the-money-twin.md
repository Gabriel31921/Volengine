# ADR-017: The ACL inverts the out-of-the-money twin

**Status:** Accepted · 2026-08 · implemented in F1-06, both producers

## Context

Black-76 inversion is well-conditioned only out of the money. Deep in the money the time value
falls below the ulp of the intrinsic value, so the price carries no information about volatility at
all.

Measured, not assumed. A 200k-case sweep at realistic crypto ranges gave up to **0.17 of silent vol
error** on in-the-money quotes against **0.003** out of the money. A 0.1% error in the forward
becomes **4.2 vol points** through an ITM inversion, versus **0.08** through its OTM twin.

## Decision

Both ACLs invert the **out-of-the-money leg** of every strike and discard the in-the-money twin.

**When a strike has no OTM twin, the ITM leg is inverted anyway and down-weighted** by a configured
`unpaired_itm_factor`.

## Why nothing is lost by the rule

Put-call parity makes the ITM option's time value *equal* to the OTM option's price, and vega is
identical on both sides. What the discarded quote carried was a **parity residual** — a statement
about the forward and the discount, never about the smile — and that is already measured upstream
by `forward.crosscheck_error` → `QualityBlock.forward_crosscheck_error`. Inverting the ITM side
would not surface that anomaly, it would launder it into the wing of the surface.

## Why the unpaired ITM leg is kept rather than dropped

A one-sided book is ordinary in the wings, and dropping the strike would leave the fit blind
exactly where the market is thinnest. That is worse for the network than for SVI: a hole in
training data is unconstrained across the gap, where five SVI parameters still span it.

**The weight is the mark.** `SliceTask` and `TrainingSample` carry no flags by design, because a
loss consumes parallel arrays of numbers. The marking and its consequence are therefore one number
and cannot be left inconsistent with each other.

## Consequences

- Roughly half of every chain is discarded by construction, and that is the intent.
- `unpaired_itm_factor` is configuration (ADR-012), because how much to trust a one-sided wing is a
  judgement about a market.
- Both ACLs implement this rule independently, in prose rather than in shared arithmetic, which is
  what the cross-context test in `tests/neural_surface/test_acl.py` now exists to guard (ADR-014).
