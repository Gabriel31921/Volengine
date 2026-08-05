# ADR-003: In-process event bus with conflation

**Status:** Accepted · 2026-07 · Design.md §3

## Context

Bounded contexts are forbidden from importing each other, so they need an intermediary to
communicate. The rates involved are wildly different: the exchange pushes ticker updates
every 100 ms, while a calibration takes milliseconds to seconds. A fast producer will
routinely outrun a slow consumer.

## Decision

An asyncio in-memory bus. Events (`SnapshotReady`, `SurfaceCalibrated`,
`ChainCompositionChanged`, `CalibrationFailed`) carry DTOs from `contracts/`. Each consumer
has a **mailbox of size one with last-write-wins semantics** — conflation. The bus interface
does not assume shared memory: everything it carries is serializable.

## Alternatives considered

**Kafka or Redis.** Out of all proportion at this scale, and they would add operational
weight that teaches nothing about the actual problem.

**An unbounded queue.** Under a slow consumer the backlog grows without bound and the system
spends its time processing data that is already obsolete. Conflation bounds the lag by
construction: you always process the most recent state.

## Consequences

- Dropping messages is the intended behaviour here, not a defect: a stale snapshot has no
  value once a newer one exists.
- Every drop increments a counter, so degradation is observable and never silent.
- Migrating to one process per context becomes a change of transport, not a change of
  contracts.
- Every DTO needs a round-trip test through a real `json.dumps`. In-memory equality is not
  enough — it would pass with a `datetime` object smuggled into the payload.
