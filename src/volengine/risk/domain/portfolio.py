"""What somebody owns, in this context's own words: a book of option positions.

Two value objects and nothing else. ``Position`` is one line of a portfolio file -- an option
contract and how much of it is held -- and ``Portfolio`` is the whole book. Everything that
*happens* to them lives elsewhere: the surface is looked up in ``surface_view.py``, the vol is
interpolated in ``interpolation.py``, the option is priced in ``pricing.py`` and the greeks are
bumped in ``valuation.py``. These types only have to be able to say, correctly and once, what is
held.

**This is Risk's own third spelling of an option, and the duplication is the architecture.**
``InstrumentId`` in ``market_data/`` names the same four things -- underlying, expiry, strike,
side -- and ``SliceTask`` in ``parametric_pricing/`` has already dissolved them into a
log-moneyness axis. None of the three may import either of the others (rule 6), and none of them
should want to: Market Data identifies an instrument so that a stream of updates lands on the
same slot, Pricing has thrown the identity away because a fit only sees numbers, and Risk holds a
contract somebody actually bought, with a signed size attached. A single canonical ``Option``
shared across the three would make every context's vocabulary hostage to every other's, which is
the failure DDD's bounded contexts exist to prevent. ``OptionKindR`` is imported from
``pricing.py`` for the same reason ``OptionKindP`` lives in ``black76.py``: the pricer is what
gives the side its meaning, so the side is declared next to the formula that reads it.

**No ``market_id`` on either type, deliberately.** A portfolio is a statement about ownership,
and which market prices it is a routing question the composition root answers by handing the use
case a portfolio and a ``SurfaceView`` together. Putting a market identifier on the book would
make it look as though the book knew, and would invite a check nothing in this layer can perform
-- see the note under :attr:`Position.underlying`, which is the same argument one field down.

Rule 3 applies here as everywhere in this layer: no ``contracts/`` import, so there is no
``to_dict``, no ``schema_version`` and no wire format below. A portfolio is read from a file by an
adapter and translated in; nothing here ever crosses a process boundary.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime

from volengine.risk.domain.pricing import OptionKindR
from volengine.shared_kernel.domain.instants import require_aware


@dataclass(frozen=True, slots=True)
class Position:
    """One option contract and the signed size held in it.

    Frozen and compared by value, so two lines built from the same five fields are equal. That
    is what lets a test assert on a position without reaching for identity, and it is also what
    makes the duplicate-position rule below a deliberate decision rather than an accident of
    hashing: equal positions are legal precisely because the type refuses to treat equality as
    sameness of *lot*.

    A position is only ever read here. Whether it can be valued at all is a question about the
    surface it is paired with, and it is answered later: an expiry the surface has already passed
    raises ``ExpiredPositionError`` from ``SurfaceView.tenor_of``, not from this constructor,
    because a portfolio file outlives its positions and "this option expired last week" is an
    ordinary runtime condition rather than a malformed line.
    """

    underlying: str
    """Symbol of the underlying, e.g. ``"BTC"``. Non-empty and not just whitespace.

    **Carried, never cross-checked against the surface, and that is an open seam.**
    ``CalibratedSurface`` publishes a ``market_id`` -- ``"BTC-DERIBIT"`` -- and no underlying
    anywhere, so ``SurfaceView`` has nothing this string can be compared against: the domain
    cannot tell whether a BTC position is being valued off an ETH surface. Pairing a portfolio
    with the right market is therefore the use case's job, and it does it by construction, by
    fetching the surface for the market it was asked to report on. The field stays because a risk
    line that does not name its underlying is unreadable to the person the report is for, and
    because the day the contract grows an underlying this becomes the check it looks like.

    Deriving a market id from it here would be worse than the seam: it would put a venue naming
    convention inside the domain, which is exactly what ADR-002 keeps out of this layer.
    """

    expiry: datetime
    """Exact expiry instant, timezone-aware.

    The instant and not the date, because a venue's expiry time of day is part of the contract:
    two options expiring on the same calendar day at 08:00 and at 16:00 have measurably different
    tenors on a one-week option. Aware because the tenor is recovered by subtracting this from the
    surface's ``ts_snapshot``, and mixing a naive datetime into that arithmetic raises
    ``TypeError`` deep inside the interpolation rather than here, where the field still has a
    name. Validated with the shared kernel's ``require_aware`` rather than a private copy of the
    same two lines.

    No upper or lower bound is imposed. An expiry in the past is representable on purpose: the
    report has to be able to *say* that a line has expired, which it cannot do for a position it
    was never allowed to build.
    """

    strike: float
    """Absolute strike in the currency the option is quoted in. Positive and finite.

    Absolute rather than a moneyness, because that is what a contract says and what a portfolio
    file records. Turning it into the ``k = ln(K / F)`` the grid is indexed by needs the forward
    at this position's tenor, which is a property of the surface and not of the book, so the
    conversion happens in ``valuation.py`` where both are in scope.

    The guard tests ``isfinite`` first and joins the bad cases with ``or``: ``float("nan") <= 0``
    is ``False``, so a NaN strike walks straight through any ordering test written the other way
    round and only surfaces as a NaN value at the bottom of the report.
    """

    kind: OptionKindR
    """Call or put. The same strike and expiry on the two sides are different positions.

    An enum member rather than a string so that a typo is a construction failure instead of a
    price that silently comes back as a put. It carries no sign of its own: a short call is this
    field set to ``CALL`` with a negative :attr:`quantity`, never a ``kind`` of ``SHORT_CALL``.
    """

    quantity: float
    """Signed size held, in the venue's contract units. Finite; **zero and negative are legal**.

    Negative is short, and that is the only place a direction is recorded: value, delta, gamma and
    vega are all multiplied by this number, so a short position reports negative value and flipped
    greeks with no branch anywhere in ``valuation.py``.

    **Zero is legal and load-bearing.** A leg flattened intraday stays in the portfolio file so
    that its risk line still appears -- with a value and greeks of exactly zero, which is the
    honest report on a position that is genuinely flat -- and deleting the line instead would make
    a flattened leg indistinguishable from one that was never there. The consequence for this
    guard is that it tests finiteness *only*: no ``if not self.quantity``, because ``not 0.0`` is
    ``True`` and a truthiness test would reject the legitimate case it was written to allow, and
    no ordering test either, because there is no bad side to be on.

    A float rather than an int because venues quote fractional contract sizes, and because a
    hedge ratio applied to a book produces one.
    """

    def __post_init__(self) -> None:
        # Truthiness is safe on a string -- it asks whether the text is empty, which is the
        # question -- and `strip()` extends that to a line of whitespace read out of a file.
        # It would be a bug on any number in this class.
        if not self.underlying.strip():
            raise ValueError(f"The underlying must not be empty, got {self.underlying!r}")
        require_aware(self.expiry, "expiry")
        if not math.isfinite(self.strike) or self.strike <= 0:
            raise ValueError(f"The strike must be positive and finite, got {self.strike}")
        # Finiteness only. Zero is a flattened leg and negative is a short, so the only thing a
        # quantity can be that this context cannot use is NaN or an infinity.
        if not math.isfinite(self.quantity):
            raise ValueError(f"The quantity must be finite, got {self.quantity}")


@dataclass(frozen=True, slots=True)
class Portfolio:
    """The whole book: the positions one risk report is computed over.

    An aggregate in the thinnest possible sense. It owns one rule -- that there is something to
    report on -- and refuses every other temptation: it does not net, does not sort, does not
    group by expiry and does not know which market prices it. Each of those is a presentation or
    routing decision belonging to whoever reads the report, and freezing one of them into the type
    would silently impose it on every reader.
    """

    positions: tuple[Position, ...]
    """The positions held. **Non-empty**; duplicates are legal and order is preserved.

    **An empty portfolio is refused**, and this is the one judgement the type makes. The honest
    report over an empty book is a total of exactly ``0.0`` with no position lines -- which is
    indistinguishable, on the page, from a book whose every option expired worthless overnight.
    Both read as "you have nothing", and only one of them is a market event; the other is a
    portfolio file that failed to load, a filter that matched no rows, or a market id typed
    wrongly. A misconfiguration that produces a plausible number is the worst failure mode
    available to a risk report, so it is turned into a loud ``ValueError`` at the boundary where
    the book is built, long before anything is valued.

    **Duplicate positions are legal.** Two lots of the same option bought at different times, or
    the same option arriving twice from two sub-books being merged, are two positions and are
    reported as two lines. Netting them into one is a decision -- it discards the fact that there
    were two lots, and it is only ever correct if nothing downstream cared -- so it belongs to
    whoever reads the report, not to the type that holds it. Value and greeks are linear in
    quantity, so the total is identical either way and nothing is lost by keeping the detail.
    That also means no ``set``, no dictionary keyed by position, and a ``tuple`` rather than a
    ``frozenset``: the order the file was written in is the order the report is read in.
    """

    def __post_init__(self) -> None:
        # Truthiness on a collection, which is the emptiness question and nothing else. The
        # element type is enforced by `Position.__post_init__`, which has already run on
        # everything in here by the time this constructor sees the tuple.
        if not self.positions:
            raise ValueError("A portfolio must hold at least one position")
