"""The network itself: PyTorch modules, the training loop, checkpoint storage.

The only place in the context allowed to import ``torch``, which is why it is an optional
extra rather than a base dependency -- the domain and its tests run without it installed.

The architectural tier of ADR-010 lives here: choices of activation, output transform and
monotonicity-by-construction that make a property true by the shape of the network rather
than by penalising its violation. What this package cannot do is decide whether the result is
publishable; that gate is in ``domain/``, out of reach of the optimiser.
"""

from __future__ import annotations
