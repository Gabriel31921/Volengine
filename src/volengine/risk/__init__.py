"""Risk context: the consumer, and the reason the whole contract has the shape it does.

Subscribes to ``SurfaceCalibrated`` and evaluates the surface at the strikes and expiries of
its own book. Publishes nothing back: the pipeline ends here.

It receives a grid of already-evaluated vols and **interpolates with its own method**
(ADR-001). That is a real cost -- accuracy is lost between nodes, and the mesh is sized so
the error stays well under the market's bid-ask noise -- bought in exchange for a producer
that can be replaced, serialized, recorded and replayed. Risk deliberately does not know
which calibrator produced the surface it is holding, beyond a ``producer_id`` it must not
branch on.

Greeks are computed here and are absent from the contract on purpose. A greek depends on the
differentiation convention, the bump size and the book's own model; publishing one would ship
this context's assumptions to every other consumer as though they were facts.
"""

from __future__ import annotations
