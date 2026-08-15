"""Real-time implied volatility surface calibration engine.

Four bounded contexts -- Market Data, Parametric Pricing, Neural Surface and Risk -- built
hexagonally, with two competing calibrators (SVI on JAX, an MLP on PyTorch) behind one output
contract. Whether those two are genuinely interchangeable is the question the whole design
exists to answer, so nothing is allowed to make swapping them easier for one than the other.

The one rule everything else protects: **contexts communicate only through the immutable DTOs
in** ``contracts/``. No tensor, no exchange type, no domain object crosses a boundary. A
context able to reach into another's model would make the two impossible to evolve, test or
replace separately, and the comparison would stop meaning anything.

Layout:

- ``shared_kernel/`` -- the handful of concepts every context agrees on. Stdlib only.
- ``contracts/`` -- the published language between contexts. Data, never behaviour.
- ``market_data/``, ``parametric_pricing/``, ``neural_surface/``, ``risk/`` -- the contexts,
  each split into ``domain/`` (business rules), ``application/`` (use cases and the ACL) and
  ``adapters/`` (the outside world).
- ``platform/`` -- technical machinery shared by everyone: bus, clock, metrics, executors.
- ``entrypoints/`` -- the composition root, the only place allowed to import everything.

Architecture decisions are recorded in ``docs/adr/``; docstrings across the codebase cite
them by number rather than restating them.
"""

from __future__ import annotations
