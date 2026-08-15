"""Business rules of parametric calibration: SVI parameters, admissibility, acceptance.

Imports are stdlib, ``shared_kernel/`` and numpy. **Never jax, never scipy, never
``contracts/``.** The rule looks harsh for a calibration context and it is deliberate: what
lives here is what SVI *means* -- the parameter vector, the conditions under which a slice is
admissible, whether a fit is good enough to publish -- and none of that depends on which
library computed the numbers.

The consequence is that these rules stay testable without a GPU, without a JIT warm-up and
without an optimiser, and that swapping scipy for JAX cannot quietly change what "admissible"
means. The optimisers themselves are adapters.
"""

from __future__ import annotations
