"""A short session against the real venue, run only when somebody asks for one.

Everything else in this package drives the adapter through a scripted socket. This is the one test
that opens a real one, and it is gated twice: on the ``deribit`` extra being installed, and on
``VOLENGINE_LIVE_DERIBIT=1`` in the environment -- CI does not set it, a developer does, when the
question is whether the venue still speaks the grammar the scripted tests encode. It asserts
properties, never values: which strikes are listed and what they cost is the venue's business.
"""

from __future__ import annotations

import asyncio
import os
from datetime import UTC, datetime
from importlib.util import find_spec

import pytest

from tests.market_data.builders import make_conventions
from volengine.market_data.adapters.deribit_ws import DeribitProvider
from volengine.market_data.domain.option_quote import OptionKindD, QuoteUpdate

pytestmark = pytest.mark.skipif(
    os.environ.get("VOLENGINE_LIVE_DERIBIT") != "1"
    or any(find_spec(name) is None for name in ("websockets", "httpx")),
    reason="set VOLENGINE_LIVE_DERIBIT=1 with the deribit extra installed to run against the venue",
)

SESSION_SECONDS = 30.0
UPDATES_WANTED = 50


async def take(provider: DeribitProvider, count: int) -> list[QuoteUpdate]:
    updates: list[QuoteUpdate] = []
    async for update in provider.stream():
        updates.append(update)
        if len(updates) >= count:
            await provider.close()
    return updates


async def test_the_live_chain_is_discovered_and_streams_in_the_chain_own_units() -> None:
    provider = DeribitProvider(make_conventions())
    started = datetime.now(UTC)

    instruments = await asyncio.wait_for(provider.discover(), timeout=SESSION_SECONDS)
    updates = await asyncio.wait_for(take(provider, UPDATES_WANTED), timeout=SESSION_SECONDS)

    assert len(instruments) > 100
    assert {one.underlying for one in instruments} == {"BTC"}
    assert all(one.expiry.hour == 8 and one.expiry.tzinfo is not None for one in instruments)
    assert len(updates) == UPDATES_WANTED
    assert {update.instrument for update in updates} <= set(instruments)
    for update in updates:
        observation = update.observation
        assert update.underlying_price is not None
        # Premiums in the strike's currency: a call is worth less than the forward and a put less
        # than its strike. Not "less than a bitcoin" -- on an inverse market a put struck above
        # twice the spot pays more than one, and the first live run of this test found one.
        ceiling = (
            update.underlying_price
            if update.instrument.kind is OptionKindD.CALL
            else update.instrument.strike
        )
        for side in (observation.bid, observation.ask):
            assert side is None or 0 < side < ceiling
        assert observation.exchange_iv is None or 0.01 < observation.exchange_iv < 5.0
        assert abs((observation.ts_exchange - started).total_seconds()) < 2 * SESSION_SECONDS
        assert observation.ts_local >= started
