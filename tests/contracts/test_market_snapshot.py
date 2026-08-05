from __future__ import annotations

import json
from dataclasses import replace
from datetime import UTC, datetime, timedelta

import pytest

from volengine.contracts.market_snapshot import (
    SCHEMA_VERSION,
    MarketSnapshot,
    OptionKind,
    QualityBlock,
    QuoteData,
    QuoteFlag,
    SliceData,
)

TS = datetime(2026, 7, 27, 8, 0, tzinfo=UTC)
NEAR_EXPIRY = TS + timedelta(days=30)
FAR_EXPIRY = TS + timedelta(days=90)


# --- builders
# One valid object per class, with a single knob per test. A test that needs an
# invalid variant starts from a valid one and changes the minimum: the test then
# states what it is probing instead of drowning the intent in twenty fields.


def make_quote(strike: float = 60_000.0, kind: OptionKind = OptionKind.CALL) -> QuoteData:
    return QuoteData(
        strike=strike,
        kind=kind,
        mid=0.052,
        spread_rel=0.011,
        age_seconds=0.4,
        flags=(QuoteFlag.WIDE_SPREAD,),
        exchange_iv=0.62,
    )


def make_slice(expiry: datetime = NEAR_EXPIRY, tenor_years: float = 30 / 365.0) -> SliceData:
    return SliceData(
        expiry=expiry,
        tenor_years=tenor_years,
        forward=61_000.0,
        quotes=(make_quote(), make_quote(strike=65_000.0, kind=OptionKind.PUT)),
        flags=(),
    )


def make_quality() -> QualityBlock:
    return QualityBlock(
        coverage_ratio=0.93,
        max_age_seconds=1.2,
        n_quotes_admissible=180,
        n_quotes_total=194,
        forward_crosscheck_error=0.0004,
        degraded=False,
    )


def make_snapshot() -> MarketSnapshot:
    return MarketSnapshot(
        snapshot_id="01JZQ0S9K7",
        market_id="BTC-DERIBIT",
        ts_exchange=TS,
        ts_local=TS + timedelta(milliseconds=8),
        underlying="BTC",
        slices=(make_slice(NEAR_EXPIRY, 30 / 365.0), make_slice(FAR_EXPIRY, 90 / 365.0)),
        quality=make_quality(),
    )


# --- round trip


def test_quote_data_round_trip() -> None:
    quote = make_quote()
    assert QuoteData.from_dict(quote.to_dict()) == quote


def test_slice_data_round_trip() -> None:
    data = make_slice()
    assert SliceData.from_dict(data.to_dict()) == data


def test_quality_block_round_trip() -> None:
    quality = make_quality()
    assert QualityBlock.from_dict(quality.to_dict()) == quality


def test_market_snapshot_round_trip() -> None:
    snapshot = make_snapshot()
    assert MarketSnapshot.from_dict(snapshot.to_dict()) == snapshot


def test_market_snapshot_survives_a_real_json_hop() -> None:
    """ADR-003: the bus must not assume shared memory.

    This is the strong one: to_dict must emit primitives ONLY. A datetime or a
    dataclass smuggled into the dict passes the in-memory round trip and blows
    up right here.
    """
    snapshot = make_snapshot()
    wire = json.dumps(snapshot.to_dict())
    assert MarketSnapshot.from_dict(json.loads(wire)) == snapshot


def test_flags_deserialize_into_enum_members() -> None:
    """Blind spot of the round trip: a StrEnum IS a str, so ("STALE",) == (QuoteFlag.STALE,).

    Equality cannot tell them apart; the type can. Without this assertion, a
    from_dict that forgets to rebuild the enums passes every other test here.
    """
    quote = QuoteData.from_dict(make_quote().to_dict())
    assert all(isinstance(flag, QuoteFlag) for flag in quote.flags)
    assert isinstance(quote.kind, OptionKind)


def test_from_dict_does_not_bypass_validation() -> None:
    raw = make_quote().to_dict()
    raw["strike"] = -1.0
    with pytest.raises(ValueError):
        QuoteData.from_dict(raw)


def test_from_dict_rejects_a_schema_version_from_the_future() -> None:
    raw = make_snapshot().to_dict()
    raw["schema_version"] = SCHEMA_VERSION + 1
    with pytest.raises(ValueError, match="schema version"):
        MarketSnapshot.from_dict(raw)


# --- validation


@pytest.mark.parametrize("bad", [0.0, -1.0, float("nan"), float("inf"), float("-inf")])
def test_quote_rejects_invalid_strike(bad: float) -> None:
    with pytest.raises(ValueError, match="strike"):
        make_quote(strike=bad)


@pytest.mark.parametrize("bad", [0.0, -1.0, float("nan"), float("inf")])
def test_quote_rejects_invalid_mid(bad: float) -> None:
    with pytest.raises(ValueError, match="mid"):
        replace(make_quote(), mid=bad)


@pytest.mark.parametrize("bad", [-0.001, float("nan"), float("inf")])
def test_quote_rejects_invalid_spread(bad: float) -> None:
    with pytest.raises(ValueError, match="spread"):
        replace(make_quote(), spread_rel=bad)


def test_quote_accepts_zero_spread() -> None:
    """spread_rel = 0 is a market with no bid-ask: unusual, but not invalid."""
    assert replace(make_quote(), spread_rel=0.0).spread_rel == 0.0


@pytest.mark.parametrize("bad", [0.0, -1.0, float("nan")])
def test_slice_rejects_invalid_tenor(bad: float) -> None:
    with pytest.raises(ValueError, match="tenor"):
        replace(make_slice(), tenor_years=bad)


@pytest.mark.parametrize("bad", [0.0, -1.0, float("inf")])
def test_slice_rejects_invalid_forward(bad: float) -> None:
    with pytest.raises(ValueError, match="forward"):
        replace(make_slice(), forward=bad)


def test_slice_rejects_naive_expiry() -> None:
    with pytest.raises(ValueError, match="timezone-aware"):
        make_slice(expiry=datetime(2026, 8, 27, 8, 0))


@pytest.mark.parametrize("field", ["ts_exchange", "ts_local"])
def test_snapshot_rejects_naive_timestamps(field: str) -> None:
    naive = datetime(2026, 7, 27, 8, 0)
    with pytest.raises(ValueError, match=field):
        replace(make_snapshot(), **{field: naive})


def test_snapshot_rejects_empty_market_id() -> None:
    with pytest.raises(ValueError, match="market id"):
        replace(make_snapshot(), market_id="")


def test_snapshot_rejects_unsorted_slices() -> None:
    near = make_slice(NEAR_EXPIRY, 30 / 365.0)
    far = make_slice(FAR_EXPIRY, 90 / 365.0)
    with pytest.raises(ValueError, match="sorted by expiry"):
        replace(make_snapshot(), slices=(far, near))


def test_snapshot_accepts_an_empty_chain() -> None:
    """Zero slices is sorted vacuously, and a closed market is a real case."""
    assert replace(make_snapshot(), slices=()).slices == ()


@pytest.mark.parametrize("bad", [-0.01, 1.01])
def test_quality_rejects_coverage_out_of_range(bad: float) -> None:
    with pytest.raises(ValueError, match="coverage"):
        replace(make_quality(), coverage_ratio=bad)


def test_quality_rejects_negative_max_age() -> None:
    with pytest.raises(ValueError, match="negative"):
        replace(make_quality(), max_age_seconds=-1.0)


def test_quality_rejects_negative_total_quotes() -> None:
    with pytest.raises(ValueError, match="negative"):
        replace(make_quality(), n_quotes_total=-1)


def test_quality_rejects_negative_admissible_quotes() -> None:
    with pytest.raises(ValueError, match="negative"):
        replace(make_quality(), n_quotes_admissible=-1)


def test_quality_rejects_negative_forward_crosscheck_error() -> None:
    with pytest.raises(ValueError, match="negative"):
        replace(make_quality(), forward_crosscheck_error=-0.01)


def test_quality_rejects_more_admissible_than_total() -> None:
    with pytest.raises(ValueError, match="admissible"):
        replace(make_quality(), n_quotes_admissible=200, n_quotes_total=194)


# ---- design decisions


def test_contract_carries_no_greeks() -> None:
    """Design 2.2: greeks do NOT travel in the contract; Risk computes them on its book.

    This asserts no behaviour. It asserts that an architectural decision is still
    standing, which is the only thing that keeps it standing.
    """
    fields = set(MarketSnapshot.__annotations__) | set(QuoteData.__annotations__)
    assert not {f for f in fields if any(g in f for g in ("delta", "gamma", "vega", "theta"))}
