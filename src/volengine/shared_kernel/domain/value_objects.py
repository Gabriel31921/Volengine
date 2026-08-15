"""The three quantities that mean the same thing in every context.

Small on purpose. A shared kernel is coupling all four contexts pay for, so it holds only
notions no context would define differently: a strike is a strike whether it is being
ingested, fitted or valued. Anything with context-specific invariants gets duplicated
instead -- that is why ``QuoteObservation`` and ``QuoteData`` are separate types and these
three are not.

Each wraps a single float, which buys two things a bare ``float`` cannot. A function taking a
``Tenor`` cannot silently be handed a moneyness, even though both are floats and the mistake
would otherwise produce plausible numbers rather than an error. And "positive and finite" is
enforced once at construction rather than re-checked at every call site.

Note the guard shape used throughout: ``not math.isfinite(x)`` **or** ``x <= 0``, joining the
bad conditions with ``or``. NaN fails every ordering comparison, so ``nan <= 0`` is ``False``
and a NaN walks straight through a naive check.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Strike:
    """The exercise price of an option, in the underlying's quote currency."""

    value: float
    """Strictly positive and finite. A zero or negative strike is not a strike."""

    def __post_init__(self) -> None:
        if not math.isfinite(self.value):
            raise ValueError(f"Strike must exist, got {self.value}")
        if self.value <= 0.0:
            raise ValueError(f"Strike must be positive, got {self.value}")


@dataclass(frozen=True, slots=True)
class Tenor:
    """Time to expiry, expressed as a year fraction.

    Already resolved under the producing market's daycount (ADR-002), so it is a plain number
    that carries no memory of the convention that produced it -- which is exactly what lets a
    calibrator treat BTC and SPX identically.
    """

    years: float
    """Strictly positive and finite. Zero would mean an expired option, not a short-dated one,
    and it divides into every variance formula downstream."""

    def __post_init__(self) -> None:
        if not math.isfinite(self.years):
            raise ValueError(f"Tenor must exist, got {self.years}")
        if self.years <= 0.0:
            raise ValueError(f"Tenor must be greater than 0, got {self.years}")


@dataclass(frozen=True, slots=True)
class Moneyness:
    """How far an option is from the money, as ``log(K / F)``.

    Measured against the forward rather than spot, and in logs rather than as a ratio. The
    forward is the arbitrage-free reference for an expiry, so this puts smiles of different
    expiries and different underlyings on one comparable axis; the log makes it roughly
    symmetric between calls and puts, which is what the smile is actually shaped in.
    """

    log_forward: float
    """Finite, and unbounded in both directions. **Zero is at-the-money forward** -- a
    perfectly legitimate and central value, so never test this for truthiness."""

    def __post_init__(self) -> None:
        if not math.isfinite(self.log_forward):
            raise ValueError(f"Moneyness must be finite, got {self.log_forward}")
