"""Business rules of the neural calibrator: constraints, acceptance, what a surface must obey.

Imports are stdlib, ``shared_kernel/`` and numpy. **Never torch, never ``contracts/``.**
Keeping PyTorch out of this layer is what stops "the model says so" from becoming the
definition of correct: the no-arbitrage conditions, the acceptance thresholds and the hard
publication gate are stated here in ordinary arrays, so they can be checked against a surface
of any origin -- including one produced by SVI, or one written by hand in a test.

That is also what makes the three-tier scheme (ADR-010) auditable. The architectural tier
lives in the adapter, the soft penalty lives in the training loop, but the hard gate lives
here, where nothing can train its way around it.
"""

from __future__ import annotations
