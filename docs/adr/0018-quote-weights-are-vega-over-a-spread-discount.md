# ADR-018: Quote weights are vega over a spread discount

**Status:** Accepted · 2026-08 · taken in F1-06, both ACLs

## Context

Both producers fit in volatility space and both need a weight per quote. Design §5.3 asks for an
error weighted by vega with weight proportional to the inverse relative spread. Design §6.5
compares the two producers on one market, and **the weights are what that comparison holds
constant**.

The obvious reading — inverse-variance weighting, `(vega / spread)²` — is a trap. It makes
influence scale like the *fourth* power of vega: one tight at-the-money quote carries a whole
slice, and the wings stop constraining the shape they are the only evidence for.

## Decision

```
weight = vega / (1 + spread_rel / spread_scale)
```

Deliberately **not** inverse-variance. Vega enters linearly; the spread is a discount rather than a
divisor, so a wide quote is de-emphasised without a narrow one running away.

**Pricing normalises weights per slice; Neural normalises across the whole snapshot.** SVI fits each
smile independently, so a slice is its own optimisation and its own scale. One gradient step spans
every point of the snapshot, so the normalisation has to as well.

**A crossed quote publishes `spread_rel = 0.0`.** `ChainQuote` keeps the negative sign as evidence
and `QuoteData` refuses it outright; the evidence survives in `QuoteFlag.CROSSED`, which is the
channel the published language provides. A negative spread would otherwise become a negative weight
— an optimiser instructed to move *away* from a quote.

## Consequences

- The wings keep a voice proportional to the information they carry.
- The two producers weight one market identically by construction, which is what the end-to-end
  cross-context test in `tests/neural_surface/test_acl.py` asserts through both ACLs.
- `spread_scale` is configuration (ADR-012): it is the spread at which a quote's weight halves, and
  that is a judgement about a venue.
- The weight also carries the `unpaired_itm_factor` of ADR-017, because a loss consumes numbers and
  a mark that is not a number cannot reach it.
