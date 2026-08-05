# ADR-009: Fixed shape by padding plus a boolean mask

**Status:** Accepted · 2026-07 · Design.md §5.5

## Context

`jax.jit` compiles one specialization per input **shape**. A recompilation costs hundreds of
milliseconds to seconds. The composition of an option chain changes constantly: strikes are
born and die as the underlying moves, and expiries roll off. An array shaped naturally to the
current chain would therefore trigger a recompilation on nearly every snapshot, which in a
streaming system is unacceptable.

## Decision

Pad to a maximum grid — sized with margin against the real chain, on the order of 64 strikes
by 16 expiries — and carry a boolean mask. The loss becomes
`where(mask, err**2, 0).sum() / mask.sum()`. The function is compiled **once** at start-up;
every later calibration is execution only, microseconds to milliseconds per iteration on CPU.

## Alternatives considered

**Recompile per shape.** Correct and unusably slow.

**Bucket the shapes into a few sizes.** Reduces recompilation without eliminating it, and
adds a bucketing policy to maintain.

## Consequences

- A permanent **non-recompilation test**: calibrate under different chain compositions with
  the same padded shape and assert that JAX's compilation counter does not rise. Silent
  recompilation is a known risk and this test is its only guard.
- A property test asserting that perturbing values under the mask does not change the result.
- `ChainCompositionChanged` has to be handled so the padding stays valid when the chain grows
  past the reserved size.
- Wasted memory and arithmetic on padded cells, both negligible at this size.
