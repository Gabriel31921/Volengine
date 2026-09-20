"""The network itself: the PyTorch learner and the surface it produces.

The only place in the context allowed to import ``torch``, which is why it is an optional extra
(the ``neural`` group) rather than a base dependency -- the domain and its tests run without it
installed, and so does the whole engine on the scipy baseline.

Of ADR-010's three tiers, **the soft one lives here**: ``torch_learner.py`` penalises Durrleman's
condition and the calendar crossing inside its loss. The **architectural** tier -- monotonicity or
convexity made true by the shape of the network -- is deliberately *not* built; that is Design
§10.2, an extension, and the network shipped is a plain MLP. What this package can never do is
decide whether the result is publishable; that hard gate is in ``domain/``, out of reach of the
optimiser, and it judges every surface this package returns.
"""

from __future__ import annotations
