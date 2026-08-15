"""The few concepts every context agrees on, verbatim.

A shared kernel is a deliberate exception to context autonomy, and it is dangerous in
proportion to its size: everything here is coupling all four contexts pay for and none of
them can evolve alone. So it stays tiny, holds only notions that genuinely mean the same
thing everywhere -- a strike is a strike in ingestion, in calibration and in risk -- and
grows only when leaving something out would be a lie.

Imports are stdlib only. Anything needing numpy, an exchange or a tensor is by definition not
shared.

When two contexts appear to need "the same" concept but with different invariants, that is
the signal to duplicate rather than to promote it here. ``QuoteObservation`` and ``QuoteData``
are the worked example.
"""

from __future__ import annotations
