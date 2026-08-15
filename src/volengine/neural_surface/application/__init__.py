"""Use cases of the neural calibrator, and the anticorruption layer.

Receives a snapshot, prepares the training or fine-tuning step, runs it through the port,
applies the hard gate and publishes. Imports the domain plus ``contracts/``, never
``adapters/``.

``acl.py`` is where a tensor stops being a tensor. Nothing torch-shaped is allowed past this
point: the grid published in ``CalibratedSurface`` is built from plain Python floats, so a
consumer never inherits a dependency on the producer's numerical stack (ADR-011). This is the
narrowest and most important part of the context -- an ACL that leaked an array type would
defeat the whole comparison, since only one of the two calibrators would be swappable.

Warm-starting from the previous fit happens on this side too, which is why calibrations of a
producer must stay sequential (ADR-005): two racing fits would interleave that state.
"""

from __future__ import annotations
