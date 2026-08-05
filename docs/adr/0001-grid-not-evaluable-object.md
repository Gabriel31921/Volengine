# ADR-001: The published surface is a grid, not an evaluable object

**Status:** Accepted · 2026-07 · Design.md §2.2

## Context

The calibrators produce an implied volatility surface. Risk needs to evaluate it at the
strikes and expiries of its own book. Something has to cross the boundary between the
producing context and the consuming one, and the shape of that something decides how
replaceable the producer is.

## Decision

The canonical contract is a grid of pure data: a log-moneyness axis, a tenor axis, and a
matrix of already-evaluated implied vols. The consumer interpolates between nodes with its
own method.

## Alternatives considered

Expose `implied_vol(K, T)` as a method of the contract. More faithful to the concept and
exact between nodes, but it drags the producer's behaviour and implementation across the
boundary: the object cannot be serialized, cannot cross a process, cannot be written to a
recording and read back tomorrow. A contract you can only use by holding a reference to the
producer's memory is not a contract.

## Consequences

- Accuracy is lost between nodes. Accepted: the mesh density is sized so the interpolation
  error stays far below the bid-ask noise of the market.
- The contract survives a `json.dumps` hop, which is what ADR-003 requires of everything on
  the bus.
- Risk fixes its interpolation method (bilinear in total variance) in its own domain and
  documents the limitation.
- Greeks do **not** travel in the contract. Risk computes them on its book; including them
  would couple the published language to the needs of one particular consumer.
- A parametric and a neural producer become trivially comparable, because both publish the
  same kind of object.
