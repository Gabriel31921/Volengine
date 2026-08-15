"""Neural Surface context: the MLP calibrator.

Consumes ``SnapshotReady``, fits a neural representation of the surface and publishes
``SurfaceCalibrated`` carrying a ``CalibratedSurface``, or ``CalibrationFailed``. The second
of the two competitors, and it emits **the same contract** as ``parametric_pricing/`` -- if
the consumer could tell which one produced a surface, the comparison would be rigged.

The interesting difficulty is that a network has no built-in notion of no-arbitrage, while
SVI's parameterisation encodes part of it for free. The answer here is three tiers of
constraint (ADR-010): architectural where the shape of the network can guarantee a property
outright, a soft penalty in the loss where it can only be encouraged, and a hard gate before
publishing that refuses a surface violating the conditions outright. The gate exists because
a penalty is a preference, not a guarantee, and a downstream consumer cannot be handed a
surface that merely tried to be arbitrage-free.

Bounded to European index options for now (ADR-007), same as its competitor.
"""

from __future__ import annotations
