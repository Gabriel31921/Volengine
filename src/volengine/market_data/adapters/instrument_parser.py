"""Deribit's symbol grammar, parsed away at the edge: the domain never sees the string.

``BTC-27MAR26-60000-C`` is how the venue spells one contract, and it is the venue's business: the
day has no leading zero (``BTC-2OCT26-…``), the year has two digits, a decimal strike is written
with a ``d`` (``0d5`` for 0.5, on the assets whose strikes need one), and the expiry *date* is all
the symbol carries -- the time of day is knowledge about the venue, resolved by
``MarketConventions.expiry_instant`` (ADR-002) and never by this module. What leaves here is an
``InstrumentId``: four typed fields the chain is keyed by, with nothing of the spelling left on it.

Pure and stdlib-only on purpose. It is the one piece of the Deribit adapter that has no network in
it, so it is the piece that runs on every CI leg, and the piece the recording seam in
``docs/SEAMS.md`` points at: a replay never re-runs this grammar, so its tests are the only thing
standing between a venue renaming its symbols and a chain full of ``ValueError``.

Failures are plain ``ValueError``, like every constructor in this context: a symbol that does not
parse is malformed input, and the caller -- :mod:`deribit_ws` -- counts it and moves on rather than
letting one odd listing take the session down.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date
from typing import Final

from volengine.market_data.domain.market_conventions import MarketConventions
from volengine.market_data.domain.option_quote import InstrumentId, OptionKindD

MONTHS: Final[dict[str, int]] = {
    "JAN": 1,
    "FEB": 2,
    "MAR": 3,
    "APR": 4,
    "MAY": 5,
    "JUN": 6,
    "JUL": 7,
    "AUG": 8,
    "SEP": 9,
    "OCT": 10,
    "NOV": 11,
    "DEC": 12,
}
"""The venue's month abbreviations, upper case, as they appear in a symbol."""

_SYMBOL: Final = re.compile(
    r"^(?P<underlying>[A-Z0-9]+(?:_[A-Z0-9]+)?)"
    r"-(?P<day>\d{1,2})(?P<month>[A-Z]{3})(?P<year>\d{2})"
    r"-(?P<strike>\d+(?:d\d+)?)"
    r"-(?P<kind>[CP])$"
)
"""The grammar, anchored at both ends.

``underlying`` admits one underscore so that the linear ``BTC_USDC`` listings parse; ``day`` admits
one or two digits because the venue writes ``2OCT26`` and ``27MAR26`` in the same chain; ``strike``
admits the ``d`` decimal marker. Anything else -- a future (``BTC-27MAR26``), a perpetual, a
combo, lower case -- fails the match, which is the correct answer for a chain of options.
"""

_KINDS: Final[dict[str, OptionKindD]] = {"C": OptionKindD.CALL, "P": OptionKindD.PUT}


@dataclass(frozen=True, slots=True)
class DeribitSymbol:
    """One symbol, decoded but not yet resolved: the expiry is still a date.

    A separate step from ``InstrumentId`` because the two questions are different. What the
    symbol *says* is a fact about the string and needs nothing else; what instant it expires at
    is a fact about the venue and needs the conventions. Keeping the first answerable on its own is
    what lets the grammar be tested without a market.
    """

    underlying: str
    expiry_date: date
    strike: float
    kind: OptionKindD


def parse_symbol(text: str) -> DeribitSymbol:
    """Decode ``BTC-27MAR26-60000-C`` into its four parts.

    Raises:
        ValueError: If the text does not match the grammar, names a month the venue does not
            use, or names a day the month does not have (``31FEB26``). The message carries the
            symbol, because a person reading a log of skipped listings needs to see which one.
    """
    match = _SYMBOL.match(text)
    if match is None:
        raise ValueError(f"Not a Deribit option symbol: {text!r}")
    month = MONTHS.get(match["month"])
    if month is None:
        raise ValueError(
            f"Not a Deribit option symbol: {text!r} names the month {match['month']!r}"
        )
    try:
        expiry_date = date(2000 + int(match["year"]), month, int(match["day"]))
    except ValueError as failure:
        raise ValueError(f"Not a Deribit option symbol: {text!r} ({failure})") from failure
    return DeribitSymbol(
        underlying=match["underlying"],
        expiry_date=expiry_date,
        # `0d5` is the venue's spelling of 0.5: a dot is not allowed in a symbol.
        strike=float(match["strike"].replace("d", ".")),
        kind=_KINDS[match["kind"]],
    )


def instrument_from_symbol(text: str, conventions: MarketConventions) -> InstrumentId:
    """Parse a symbol and resolve it into the identity the chain is keyed by.

    The one place the venue's expiry time of day is applied to a Deribit listing, through
    ``conventions.expiry_instant`` -- 08:00 UTC on this venue, and a third of the tenor of a
    one-day option if it were forgotten (risk 2 of Design §11).

    Raises:
        ValueError: If the symbol does not parse, or if ``InstrumentId`` refuses the result (a
            zero strike parses and is then refused there).
    """
    symbol = parse_symbol(text)
    return InstrumentId(
        underlying=symbol.underlying,
        expiry=conventions.expiry_instant(symbol.expiry_date),
        strike=symbol.strike,
        kind=symbol.kind,
    )
