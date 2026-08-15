# ADR-013: Composition events carry state, not deltas

**Status:** Accepted · 2026-07 · project decision, not in Design.md

## Context

ADR-003 gives every subscriber a size-one mailbox with last-write-wins semantics. That is
correct for messages which *replace* one another: a `MarketSnapshot` is the state of the
market right now, so snapshot 5 makes snapshot 2 worthless and dropping it costs nothing.

`ChainCompositionChanged` was originally specified with `added` and `removed` fields — a
**delta**. Deltas do not replace one another, they accumulate. If "strike 70000 was born" and
"strike 55000 died" are published back to back and the consumer only reads the second, the
birth is gone for good. And this event matters downstream: it is what tells the JAX
calibrator that the padded shape of ADR-009 may need to change.

So the bus semantics and this one event contradicted each other.

## Decision

The event carries the **full set of live instruments**, sorted, instead of the delta. A
consumer that wants to know what changed compares against the set it already had.

```python
@dataclass(frozen=True, slots=True)
class ChainCompositionChanged:
    market_id: str
    ts: datetime
    instruments: tuple[str, ...]  # every live instrument, sorted
```

## Alternatives considered

**A per-topic policy**: a normal queue for composition, a size-one mailbox for everything
else. Correct, but the bus stops being uniform. Every subscription then carries a policy
decision, and every reader of the code has to ask which semantics apply here — for a
component whose whole value is being simple and predictable.

**Treat the event as a hint.** The next snapshot carries the whole chain anyway, so losing
the event only delays the reaction. It works, but it makes correctness depend on the snapshot
cadence and leaves the padding reacting one cycle late, exactly when the chain is changing
fastest.

## Consequences

- Conflating this event is safe by construction rather than by convention. The bus keeps one
  semantics and one explanation.
- The event becomes idempotent: receiving it twice, or receiving only the last of five, leaves
  the consumer in the same correct state. That is the property that makes conflation sound.
- Consumers that want the delta must keep the previous set. That is more robust anyway — it
  does not assume they saw every intermediate event.
- The payload grows from a handful of names to the whole chain: a few hundred short strings,
  at most a few times per hour. Negligible.
- Sorting at construction makes two events with the same instruments equal and identically
  serialized, which the round-trip test relies on.
- Ties into ADR-009: the padding must handle a chain that grows past the reserved shape, and
  this event is where that is noticed.
