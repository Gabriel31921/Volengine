# ADR-005: One thread pool per calibrator

**Status:** Accepted · 2026-07 · Design.md §3

## Context

Two calibrators run in the same process against the same stream of snapshots. Both JAX and
PyTorch release the GIL while doing numerical work, so real parallelism is available — but
only if they are not queued behind each other on the same executor.

## Decision

Concurrency in v1 is one named thread pool per calibrator, created by
`platform/executors.py` and wired in the composition root.

## Alternatives considered

**A single shared executor.** A slow neural fine-tuning step would delay parametric
calibration, and the two producers' latencies would become impossible to attribute.

**Separate processes.** Kept as an extension. The bus design already permits it, since
nothing on it assumes shared memory (ADR-003), so the decision can be revisited without
touching the contracts.

## Consequences

- The latency of each producer is measurable independently, which is a prerequisite for the
  comparative benchmark.
- A slow consumer is already tolerated by the bus through conflation, so a lagging
  calibrator degrades its own freshness rather than the system's.
- GPU is out of scope for v1, so thread-level parallelism on CPU is the whole story.
