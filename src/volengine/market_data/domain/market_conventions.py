"""How one market measures time, money and the forward (ADR-002).

Everything a quote means depends on a handful of conventions that the venue never sends and
that differ between markets. A tenor is not a subtraction of two instants: it is a subtraction
interpreted under a daycount. A premium is not a number of dollars until you know the
numeraire. A strike is not in or out of the money until you know how the forward was built.

Those choices live here, as one declarative object injected when the Market Data context is
built. Adapters *use* the conventions; they never define them. Buried in each adapter instead,
a difference between the BTC and the SPX smile would be unattributable -- market, or bookkeeping?
This module is the answer to that question, which is why it is domain rather than plumbing.

It is also the last place these choices are visible. Downstream of the boundary a calibrator
receives ``tenor_years`` and ``forward`` as plain numbers and cannot tell which market produced
them, which is exactly what makes the same calibrator work on both.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, date, datetime, time
from enum import StrEnum
from typing import Final

from volengine.market_data.domain.errors import ExpiredInstrumentError
from volengine.shared_kernel.domain.instants import require_aware

SECONDS_PER_DAY: Final[float] = 86_400.0
DAYS_PER_YEAR_ACT365F: Final[float] = 365.0


class DayCount(StrEnum):
    """The clock a year fraction is measured on.

    The choice is not cosmetic. Implied vol is recovered by inverting a price given a tenor,
    so two daycounts over the same two instants yield two different implied vols for the same
    quote. Getting this wrong does not produce an obvious failure -- it produces a smile that
    is subtly the wrong shape, and no individual quote looks suspicious.
    """

    ACT_365F = "ACT/365F"
    """Actual calendar days over a fixed 365. The right clock for a market that never closes.

    Crypto trades through weekends and holidays, so calendar time and market time are the same
    thing and variance accumulates uniformly.
    """

    ACT_252 = "ACT/252"
    """Business days over 252. Declared, not implemented -- equities is an extension (ADR-007).

    Cash equity markets accumulate variance during sessions, not overnight and not over a
    weekend: Friday to Monday is three calendar days but one trading day. Implementing it
    needs a holiday calendar, which is why it waits until there is a market that requires it.
    """


class Numeraire(StrEnum):
    """The unit a premium is quoted and settled in."""

    QUOTE = "QUOTE"
    """Premium in the same currency the underlying is quoted in. The ordinary case: an SPX
    option quoted and settled in USD."""

    INVERSE = "INVERSE"
    """Premium in units of the underlying itself, the Deribit convention.

    A BTC option priced at ``0.05`` costs 0.05 BTC, not 50,000 USD, so the premium's value in
    quote currency moves with the very thing the option is written on. Ingestion decides the
    working unit and declares it here; the internal work is done in forward moneyness and in
    prices normalised by the forward, so the choice never reaches a calibrator.
    """


class ForwardMethod(StrEnum):
    """How the forward is built for this market -- the reference moneyness is measured from.

    The forward is the anchor of the whole surface: a forward that is wrong by a fraction of a
    percent shifts every log-moneyness in the slice, tilting the smile without making any
    single quote look wrong. Hence a declared method plus an independent cross-check, whose
    disagreement is published as ``QualityBlock.forward_crosscheck_error``.
    """

    PROVIDER_UNDERLYING = "PROVIDER_UNDERLYING"
    """Take the underlying reference the venue publishes with the quote.

    On Deribit that is a synchronous future or perpetual, i.e. an actually traded forward
    rather than a derived one. Primary route in crypto; put-call parity is the cross-check.
    """

    PUT_CALL_PARITY = "PUT_CALL_PARITY"
    """Infer it from call and put prices at strikes near the money.

    Uses only quotes we already have, so it needs no rate or dividend input -- which makes it
    the primary method where those inputs are unreliable, and the cross-check everywhere else.
    """

    SPOT_RATES_DIVIDENDS = "SPOT_RATES_DIVIDENDS"
    """Carry the spot forward with rates and discrete dividends. The equities construction,
    and the one that needs ex-dividend dates to be right."""


@dataclass(frozen=True, slots=True)
class MarketConventions:
    """The bookkeeping rules of one market, fixed for the lifetime of the context instance.

    One instance per market, not per venue: the same underlying on two venues is two markets
    (4.7), because the expiry time and the daycount differ and their numbers are therefore not
    interchangeable. Running BTC-DERIBIT and SPX-IB side by side is two of these objects
    publishing onto the same bus, with nothing downstream needing to change.
    """

    market_id: str
    """Identity of the market, e.g. ``"BTC-DERIBIT"``. Non-empty; matches the snapshots'."""

    underlying: str
    """Symbol of the underlying, e.g. ``"BTC"``. Non-empty."""

    day_count: DayCount
    """Clock used to turn two instants into a year fraction."""

    expiry_time_utc: time
    """Time of day, in UTC, at which this market's options expire. Deribit: 08:00.

    A market-wide constant rather than a per-instrument field, because the venue publishes an
    expiry *date* in its symbols and the time of day is knowledge about the venue. It matters
    far more than its size suggests: on a one-day option, being eight hours out is a third of
    the tenor, and the short end is where vol actually moves (risk 2 of section 11).
    """

    numeraire: Numeraire
    """Unit premiums are quoted in -- see :class:`Numeraire`."""

    forward_method: ForwardMethod
    """Primary route to the forward -- see :class:`ForwardMethod`."""

    def __post_init__(self) -> None:
        if not self.market_id.strip():
            raise ValueError(f"The market id can't be empty, got {self.market_id!r}")
        if not self.underlying.strip():
            raise ValueError(f"The underlying can't be empty, got {self.underlying!r}")
        offset = self.expiry_time_utc.utcoffset()
        if offset is not None and offset.total_seconds() != 0:
            raise ValueError(
                f"The expiry time must be expressed in UTC, got {self.expiry_time_utc}"
            )

    def expiry_instant(self, expiry_date: date) -> datetime:
        """Resolve the venue's expiry date into the exact instant it expires, in UTC.

        The bridge between what a symbol encodes (``27JUN25``) and what ``InstrumentId``
        requires (an aware instant). Everything downstream reasons in instants, so this is the
        single place the venue's time of day is applied.
        """
        return datetime.combine(expiry_date, self.expiry_time_utc.replace(tzinfo=None), tzinfo=UTC)

    def tenor_years(self, expiry: datetime, now: datetime) -> float:
        """Year fraction from ``now`` to ``expiry`` under this market's daycount.

        Measured in seconds rather than whole days: an option expiring in six hours has a real
        tenor, and rounding it to zero or to one day is the difference between a fitted short
        slice and a broken one.

        Raises:
            ValueError: If either instant is naive. Staleness and tenors are subtractions of
                datetimes, and mixing an aware one with a naive one raises ``TypeError`` deep
                inside unrelated code.
            ExpiredInstrumentError: If ``expiry`` is not strictly after ``now``.
            NotImplementedError: For ``ACT/252``, which needs a holiday calendar (ADR-007).
        """
        require_aware(expiry, "expiry")
        require_aware(now, "now")
        if expiry <= now:
            raise ExpiredInstrumentError(
                f"{self.market_id}: expiry {expiry} is not after {now}, there is no tenor"
            )

        if self.day_count is DayCount.ACT_365F:
            days = (expiry - now).total_seconds() / SECONDS_PER_DAY
            return days / DAYS_PER_YEAR_ACT365F

        raise NotImplementedError(
            f"{self.day_count} needs a trading calendar; only {DayCount.ACT_365F} is "
            "implemented while the engine runs on 24/7 markets (ADR-007)"
        )

    def is_expired(self, expiry: datetime, now: datetime) -> bool:
        """Whether ``expiry`` has no tenor left, without raising.

        The predicate behind the exception, so a caller filtering a chain does not have to
        drive control flow with try/except over something that happens on schedule.
        """
        require_aware(expiry, "expiry")
        require_aware(now, "now")
        return expiry <= now
