"""The venue's symbol grammar: what it decodes, what it refuses, where the time of day comes from.

The domain never sees the string, so this is the only place the string is tested at all. The
symbols below are real ones -- the day without its leading zero and the twelve-expiry chain are
what the live venue listed on 2026-09-18 -- because a grammar tested only against examples the
author invented is a grammar tested against the author's assumptions.
"""

from __future__ import annotations

from datetime import UTC, date, datetime, time

import pytest

from tests.market_data.builders import make_conventions
from volengine.market_data.adapters.instrument_parser import (
    DeribitSymbol,
    instrument_from_symbol,
    parse_symbol,
)
from volengine.market_data.domain.option_quote import OptionKindD

LIVE_SYMBOLS = (
    "BTC-19SEP26-78000-C",
    "BTC-2OCT26-80000-P",
    "BTC-9OCT26-70000-C",
    "BTC-25DEC26-104000-C",
    "BTC-25JUN27-150000-P",
)
"""One symbol from each shape the live chain had: two-digit day, one-digit day, a year ahead."""


# --- what it decodes


def test_a_symbol_decodes_into_its_four_parts() -> None:
    assert parse_symbol("BTC-27MAR26-60000-C") == DeribitSymbol(
        underlying="BTC",
        expiry_date=date(2026, 3, 27),
        strike=60_000.0,
        kind=OptionKindD.CALL,
    )


def test_a_put_is_a_put() -> None:
    """The guard on the kind table: a decoder that mapped both letters to one member would pass
    the test above."""
    assert parse_symbol("BTC-27MAR26-60000-P").kind is OptionKindD.PUT


def test_the_day_has_no_leading_zero() -> None:
    assert parse_symbol("BTC-2OCT26-80000-P").expiry_date == date(2026, 10, 2)


@pytest.mark.parametrize("symbol", LIVE_SYMBOLS)
def test_every_shape_the_live_chain_lists_parses(symbol: str) -> None:
    parsed = parse_symbol(symbol)

    assert parsed.underlying == "BTC"
    assert parsed.strike > 0


def test_a_decimal_strike_is_spelled_with_a_d() -> None:
    assert parse_symbol("XRP_USDC-27MAR26-0d5-C").strike == 0.5


def test_an_underlying_may_carry_one_underscore() -> None:
    """The linear listings: ``BTC_USDC`` is one underlying, not two."""
    assert parse_symbol("BTC_USDC-27MAR26-60000-C").underlying == "BTC_USDC"


def test_the_year_is_this_century() -> None:
    assert parse_symbol("BTC-25JUN27-150000-P").expiry_date.year == 2027


# --- what it refuses


@pytest.mark.parametrize(
    "text",
    [
        "BTC-27MAR26",  # a future
        "BTC-PERPETUAL",
        "btc-27mar26-60000-c",  # lower case is not the venue's spelling
        "BTC-27MAA26-60000-C",  # no such month
        "BTC-31FEB26-60000-C",  # no such day
        "BTC-27MAR26-60000-X",  # no such kind
        "BTC-27MAR26-60000-C-extra",
        "BTC-27MAR26--C",
        "BTC-27MAR2026-60000-C",  # four-digit year
        "",
    ],
)
def test_anything_that_is_not_an_option_symbol_is_refused(text: str) -> None:
    with pytest.raises(ValueError, match="Not a Deribit option symbol"):
        parse_symbol(text)


def test_the_refusal_names_the_symbol() -> None:
    """A log of skipped listings has to say which one, or nobody can go and look at it."""
    with pytest.raises(ValueError, match="BTC-31FEB26-60000-C"):
        parse_symbol("BTC-31FEB26-60000-C")


# --- resolution against the market's conventions


def test_the_expiry_instant_comes_from_the_conventions() -> None:
    instrument = instrument_from_symbol("BTC-27MAR26-60000-C", make_conventions())

    assert instrument.expiry == datetime(2026, 3, 27, 8, 0, tzinfo=UTC)


def test_a_different_expiry_time_gives_a_different_instrument() -> None:
    """The time of day is part of the identity (ADR-002), and it is the conventions' to apply."""
    at_eight = instrument_from_symbol("BTC-27MAR26-60000-C", make_conventions())
    at_noon = instrument_from_symbol(
        "BTC-27MAR26-60000-C", make_conventions(expiry_time_utc=time(12, 0))
    )

    assert at_eight != at_noon
    assert at_noon.expiry.hour == 12


def test_the_instrument_carries_the_symbol_strike_and_kind() -> None:
    instrument = instrument_from_symbol("BTC-27MAR26-60000-P", make_conventions())

    assert (instrument.underlying, instrument.strike, instrument.kind) == (
        "BTC",
        60_000.0,
        OptionKindD.PUT,
    )


def test_a_zero_strike_parses_and_is_refused_by_the_identity() -> None:
    """The grammar admits ``0``; the domain does not, and the domain's refusal is the one kept."""
    with pytest.raises(ValueError, match="strike"):
        instrument_from_symbol("BTC-27MAR26-0-C", make_conventions())
