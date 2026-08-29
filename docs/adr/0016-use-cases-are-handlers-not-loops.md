# ADR-016: Use cases are handlers, not loops

**Status:** Accepted · 2026-08 · the shape decision of F1-06

## Context

`Implementation.md` names the use cases `CalibrateStreamUseCase`, `TrainStreamUseCase` and
`ComputeReportUseCase`, and pictures objects that subscribe to the bus and consume from it. Each
would own an event loop, a subscription and a lifecycle.

## Decision

Three of the five are **synchronous handlers** — `CalibrateOnSnapshot.handle(snapshot) ->
tuple[Event, ...]`, `TrainOnSnapshot.handle`, `ComputeReportUseCase.compute(market_id)` — with the
subscription living in `platform/runner.py`'s `BusRunner`, one generic class the composition root
wires per consumer.

The system is no less event-driven. What moved is the boundary between reacting and calculating.

**No use case touches the bus.** They return events; the composition root publishes them.

The reasons, in increasing weight:

1. A handler is tested by calling it. A loop needs an event loop, a fake bus, and a non-racy
   spelling of "it has processed now" — which does not exist.
2. ADR-004's replay becomes a `for` over recorded events instead of a simulated bus.
3. ADR-005's thread pool belongs to whoever owns the loop, so a handler reaches a named executor
   via `runner.in_executor` without ever learning that threads exist.

## Market Data is the exception, and had no choice

`IngestStreamUseCase.run()` is an `AsyncIterator[Event]`, because it owns `provider.stream()` and
`QuoteUpdate` is a domain type nobody outside the context may iterate (rule 3). `BuildSnapshot`
stays synchronous and is called by that loop after every update.

## Consequences

- `application/` never imports `platform/`, and every test drives a use case with a list.
- One `BusRunner` is the single place subscription semantics live, so conflation, lag metrics and
  executor dispatch are configured once rather than per consumer.
- The composition root grows: it is what wires runners to handlers and publishes what they return.
  That is where such code belongs (rule 8).
- `handle` returning a **tuple** of events is what makes this work at all — see ADR-021.
