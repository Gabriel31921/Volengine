"""Valid Market Data objects, one per type, with a knob for every field a test bends.

The convention this repo tests by: a builder returns **one valid object**, and a test changes
the minimum needed to make its point -- either through a keyword here or through
``dataclasses.replace``, which re-runs ``__post_init__``. What a test says is then exactly what
it is probing, instead of twenty fields of noise around one poisoned value.

Shared from a module rather than from ``conftest.py``: conftest is where pytest looks for
fixtures and hooks it *injects*, and importing from it is discouraged because it is loaded by
collection magic rather than by an import anyone can follow. See ``tests/support.py``.
"""

from __future__ import annotations

from datetime import UTC, datetime, time, timedelta

from volengine.market_data.domain.admissibility import AdmissibilityThresholds
from volengine.market_data.domain.market_conventions import (
    DayCount,
    ForwardMethod,
    MarketConventions,
    Numeraire,
)
from volengine.market_data.domain.option_quote import (
    InstrumentId,
    OptionKindD,
    QuoteObservation,
    QuoteUpdate,
)
from volengine.market_data.domain.quote_chain import QuoteChain

NOW = datetime(2026, 7, 27, 8, 0, tzinfo=UTC)
NAIVE = datetime(2026, 7, 27, 8, 0)
"""Same instant with no zone. Every entry point in this context must refuse it."""

NEAR = datetime(2026, 8, 27, 8, 0, tzinfo=UTC)
FAR = datetime(2026, 10, 27, 8, 0, tzinfo=UTC)
"""Two live expiries at Deribit's 08:00 UTC, roughly one and three months out."""

FORWARD = 60_000.0
"""Forward for both expiries. The default strike sits on it, so k = 0 unless a test moves it."""


def make_conventions(
    day_count: DayCount = DayCount.ACT_365F,
    expiry_time_utc: time = time(8, 0),
) -> MarketConventions:
    return MarketConventions(
        market_id="BTC-DERIBIT",
        underlying="BTC",
        day_count=day_count,
        expiry_time_utc=expiry_time_utc,
        numeraire=Numeraire.INVERSE,
        forward_method=ForwardMethod.PROVIDER_UNDERLYING,
    )


def make_thresholds() -> AdmissibilityThresholds:
    """Wide enough that the default quote earns no flags, so one knob raises exactly one."""
    return AdmissibilityThresholds(
        max_spread_rel=0.5,
        max_age_seconds=5.0,
        moneyness_range=(-1.5, 1.5),
        max_iv_divergence_bp=500.0,
        convexity_tolerance=0.0005,
        min_size=1.0,
    )


def make_chain() -> QuoteChain:
    return QuoteChain(make_conventions(), make_thresholds())


def make_instrument(
    strike: float = FORWARD,
    expiry: datetime = NEAR,
    kind: OptionKindD = OptionKindD.CALL,
    underlying: str = "BTC",
) -> InstrumentId:
    return InstrumentId(underlying=underlying, expiry=expiry, strike=strike, kind=kind)


def make_observation(
    bid: float | None = 0.050,
    ask: float | None = 0.054,
    bid_size: float = 12.0,
    ask_size: float = 8.0,
    ts_exchange: datetime = NOW,
    exchange_iv: float | None = 0.62,
) -> QuoteObservation:
    """A two-sided, fresh, reasonably tight quote. ``ts_local`` trails the venue by 8 ms."""
    return QuoteObservation(
        bid=bid,
        ask=ask,
        bid_size=bid_size,
        ask_size=ask_size,
        ts_exchange=ts_exchange,
        ts_local=ts_exchange + timedelta(milliseconds=8),
        exchange_iv=exchange_iv,
    )


def make_update(
    strike: float = FORWARD,
    expiry: datetime = NEAR,
    kind: OptionKindD = OptionKindD.CALL,
    bid: float | None = 0.050,
    ask: float | None = 0.054,
    bid_size: float = 12.0,
    ask_size: float = 8.0,
    ts_exchange: datetime = NOW,
    underlying_price: float | None = FORWARD,
    underlying: str = "BTC",
) -> QuoteUpdate:
    return QuoteUpdate(
        instrument=make_instrument(strike, expiry, kind, underlying),
        observation=make_observation(bid, ask, bid_size, ask_size, ts_exchange),
        underlying_price=underlying_price,
    )
