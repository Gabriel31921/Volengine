"""Parametric Pricing context: the SVI calibrator.

Consumes ``SnapshotReady``, fits a raw SVI parameterisation per slice (ADR-008) and publishes
``SurfaceCalibrated`` carrying a ``CalibratedSurface``, or ``CalibrationFailed`` when it
cannot. One of the two competitors behind the same output contract; the other is
``neural_surface/``.

Per slice, and not a single surface-wide fit, because raw SVI is a five-parameter smile model
for one expiry. Calendar coherence between slices is then imposed as a constraint across
fits, rather than assumed by the parameterisation.

Two numerical implementations live behind the same port: scipy as the readable reference, JAX
for speed. They must agree to within tolerance, which is what makes the JAX version's speed
claim believable and gives the guard on recompilation (ADR-009) something to protect. The
fixed-shape padding and mask that guard requires are why a slice with a changing quote count
does not retrigger a JIT compile on every snapshot.
"""

from __future__ import annotations
