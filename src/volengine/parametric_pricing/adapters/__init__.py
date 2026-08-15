"""The optimisers: scipy as the reference, JAX for speed.

Both implement the same calibrator port, so the use case cannot tell them apart and the
comparison between them is a swap in the composition root rather than a branch in the code.
This package is the only place in the context allowed to import ``jax`` or ``scipy``, which
is also why both are optional extras rather than base dependencies -- the domain and its
tests run without either installed.

The JAX side carries the constraints that make its speed real: fixed input shapes with
padding and a mask, so a slice whose quote count changes between snapshots does not retrigger
a JIT compilation (ADR-009). The recompilation counter in the tests is there because that
failure is silent -- everything still produces correct numbers, just slowly enough to defeat
the entire reason for using JAX.
"""

from __future__ import annotations
