from __future__ import annotations

from dataclasses import FrozenInstanceError, replace
from datetime import timedelta

import pytest

from tests.market_data.builders import (
    NAIVE,
    NOW,
    make_instrument,
    make_observation,
    make_update,
)
from tests.support import replace_field

NOT_FINITE = [float("nan"), float("inf"), float("-inf")]


# --- InstrumentId


@pytest.mark.parametrize("bad", ["", " ", "   ", "\t"])
def test_instrument_rejects_a_blank_underlying(bad: str) -> None:
    with pytest.raises(ValueError, match="underlying"):
        make_instrument(underlying=bad)


@pytest.mark.parametrize("bad", [0.0, -1.0, *NOT_FINITE])
def test_instrument_rejects_an_invalid_strike(bad: float) -> None:
    with pytest.raises(ValueError, match="strike"):
        make_instrument(strike=bad)


def test_instrument_rejects_a_naive_expiry() -> None:
    """The tenor is computed by subtracting datetimes; a naive one raises far from here."""
    with pytest.raises(ValueError, match="timezone-aware"):
        replace(make_instrument(), expiry=NAIVE)


def test_instrument_is_usable_as_a_dict_key() -> None:
    """QuoteChain keys its per-instrument state by this object, so it must hash structurally.

    Two ids describing the same contract are the same key even when they are different
    objects -- which is what makes a reconnection idempotent instead of duplicating the chain.
    """
    chain = {make_instrument(): "state"}
    assert chain[make_instrument()] == "state"


def test_instrument_is_immutable() -> None:
    with pytest.raises(FrozenInstanceError):
        make_instrument().strike = 61_000.0  # type: ignore[misc]


# --- QuoteObservation: what is technically malformed


@pytest.mark.parametrize("side", ["bid", "ask"])
@pytest.mark.parametrize("bad", [-1.0, *NOT_FINITE])
def test_observation_rejects_an_invalid_price(side: str, bad: float) -> None:
    with pytest.raises(ValueError, match=f"{side} value"):
        replace_field(make_observation(), side, bad)


@pytest.mark.parametrize("side", ["bid_size", "ask_size"])
@pytest.mark.parametrize("bad", [-1.0, *NOT_FINITE])
def test_observation_rejects_an_invalid_size(side: str, bad: float) -> None:
    with pytest.raises(ValueError, match=side):
        replace_field(make_observation(), side, bad)


@pytest.mark.parametrize("bad", [0.0, -0.5, *NOT_FINITE])
def test_observation_rejects_an_invalid_exchange_iv(bad: float) -> None:
    with pytest.raises(ValueError, match="exchange_iv"):
        replace(make_observation(), exchange_iv=bad)


@pytest.mark.parametrize("field", ["ts_exchange", "ts_local"])
def test_observation_rejects_naive_timestamps(field: str) -> None:
    with pytest.raises(ValueError, match="timezone-aware"):
        replace_field(make_observation(), field, NAIVE)


# --- QuoteObservation: what ingestion must NOT reject
# Ingestion flags, the calibrator weights or excludes. Every rejection added here would be
# ingestion silently filtering -- a model criterion smuggled into a data criterion.


def test_observation_accepts_a_zero_bid() -> None:
    """A zero bid is information, not corruption: nobody wants the far wing at any price.

    Rejecting it would delete the quote before QuoteFlag.LOW_SIZE or the calibrator's weights
    ever get a say.
    """
    assert make_observation(bid=0.0).bid == 0.0


def test_observation_accepts_a_crossed_market() -> None:
    """bid > ask is QuoteFlag.CROSSED downstream, not a construction error."""
    crossed = make_observation(bid=0.060, ask=0.054)
    assert crossed.bid is not None and crossed.ask is not None
    assert crossed.bid > crossed.ask


@pytest.mark.parametrize("side", ["bid", "ask"])
def test_observation_accepts_a_missing_side(side: str) -> None:
    """A one-sided market is normal in the wings, and a None side is not a broken quote."""
    assert getattr(replace_field(make_observation(), side, None), side) is None


def test_observation_accepts_a_local_timestamp_before_the_exchange_one() -> None:
    """Venue clocks drift. The gap between the two stamps is a latency metric (4.5), and
    ordering them would turn a normal skew into a crash mid-session.
    """
    skewed = replace(make_observation(), ts_local=NOW - timedelta(milliseconds=40))
    assert skewed.ts_local < skewed.ts_exchange


def test_observation_accepts_zero_sizes() -> None:
    """Size below a threshold is QuoteFlag.LOW_SIZE, and the threshold is configuration."""
    assert replace(make_observation(), bid_size=0.0, ask_size=0.0).bid_size == 0.0


# --- QuoteObservation: derived quantities


def test_mid_is_the_midpoint_of_the_two_sides() -> None:
    assert make_observation(bid=0.050, ask=0.054).mid == pytest.approx(0.052)


def test_mid_is_computed_when_the_bid_is_zero() -> None:
    """Regression: a truthiness guard reads 0.0 as 'no bid' and erases a valid quote."""
    assert make_observation(bid=0.0, ask=0.100).mid == pytest.approx(0.050)


@pytest.mark.parametrize("side", ["bid", "ask"])
def test_mid_is_none_without_both_sides(side: str) -> None:
    assert replace_field(make_observation(), side, None).mid is None


def test_spread_rel_is_the_spread_over_the_mid() -> None:
    assert make_observation(bid=0.050, ask=0.054).spread_rel == pytest.approx(0.004 / 0.052)


def test_spread_rel_is_negative_on_a_crossed_market() -> None:
    """The sign is the evidence of the crossing, so it survives instead of being clamped.

    An abs() here would make a crossed book indistinguishable from a tight one, and CROSSED
    would be raised by a rule reading a quantity that no longer records the fact.
    """
    assert make_observation(bid=0.060, ask=0.054).spread_rel == pytest.approx(-0.006 / 0.057)


def test_spread_rel_is_zero_for_a_locked_market() -> None:
    assert make_observation(bid=0.052, ask=0.052).spread_rel == 0.0


@pytest.mark.parametrize("side", ["bid", "ask"])
def test_spread_rel_is_none_without_both_sides(side: str) -> None:
    assert replace_field(make_observation(), side, None).spread_rel is None


def test_spread_rel_is_none_when_the_mid_is_zero() -> None:
    """Both sides at zero is a dead instrument, and the relative spread is undefined there.

    Without the guard this is a ZeroDivisionError raised from a property, i.e. from whatever
    line happened to read it.
    """
    assert make_observation(bid=0.0, ask=0.0).spread_rel is None


# --- QuoteUpdate


@pytest.mark.parametrize("bad", [0.0, -1.0, *NOT_FINITE])
def test_update_rejects_an_invalid_underlying_price(bad: float) -> None:
    with pytest.raises(ValueError, match="underlying price"):
        make_update(underlying_price=bad)


def test_update_accepts_a_missing_underlying_price() -> None:
    """Not every ticker message carries one; the chain keeps the last value it saw."""
    assert make_update(underlying_price=None).underlying_price is None
