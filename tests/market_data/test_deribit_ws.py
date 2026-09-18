"""The Deribit provider, driven through a scripted socket: no network, no extra, every path.

What is asserted is the session as the venue would see it -- which requests went out, in what
order, on which socket -- and what the chain would see, which is a ``QuoteUpdate`` in the chain's
own units. The robustness paths get the most room because they are where these projects die: the
heartbeat answered, the reconnect that resubscribes everything, the backoff that grows and resets,
the silence that is treated as a drop, and the one refusal that must *not* be survived.

``LIVE_TICKER`` is a frame captured from the production feed on 2026-09-18, for the deep wing
where the venue writes an empty bid as ``0.0`` with a zero amount; the decoder is tested against
it and not only against tickers this file invented.
"""

from __future__ import annotations

import json
import math
from collections.abc import Callable
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest

from tests.market_data.builders import (
    FAR,
    FAR_SYMBOL,
    FORWARD,
    NEAR_SYMBOL,
    NOW,
    FakeDiscovery,
    RecordingSleeper,
    ScriptedConnection,
    ScriptedConnector,
    heartbeat,
    make_conventions,
    make_deribit_settings,
    make_instrument,
    make_ticker,
    notification,
    reply,
)
from tests.support import RecordingMetrics, replace_field
from volengine.market_data.adapters.deribit_rest import DiscoveryError
from volengine.market_data.adapters.deribit_ws import (
    SET_HEARTBEAT,
    SUBSCRIBE,
    TEST,
    TICKER_INTERVAL,
    UNSUBSCRIBE,
    ConnectionLost,
    DeribitProvider,
    DeribitSettings,
    ProtocolError,
    channel_of,
    ticker_to_update,
)
from volengine.market_data.domain.errors import MarketDataError
from volengine.market_data.domain.market_conventions import Numeraire
from volengine.market_data.domain.option_quote import OptionKindD, QuoteUpdate
from volengine.market_data.domain.ports import MarketDataProvider

OTHER_SYMBOL = "BTC-27AUG26-70000-C"
"""A third instrument on the near expiry, for the tests about a universe that moves."""

LIVE_TICKER: dict[str, Any] = {
    "timestamp": 1789725528184,
    "state": "open",
    "stats": {"high": 0.0001, "low": 0.0001, "price_change": 0.0, "volume": 0.2},
    "greeks": {"delta": 0.00218, "gamma": 0.0, "vega": 0.73684, "theta": -3.92476},
    "index_price": 78141.3,
    "instrument_name": "BTC-25SEP26-105000-C",
    "last_price": 0.0001,
    "settlement_price": 5.658e-05,
    "min_price": 0.0001,
    "max_price": 0.015,
    "open_interest": 1454.5,
    "mark_price": 0.0001,
    "interest_rate": 0.0,
    "estimated_delivery_price": 78141.3,
    "best_ask_price": 0.0001,
    "best_bid_price": 0.0,
    "mark_iv": 73.69,
    "bid_iv": 0.0,
    "ask_iv": 76.95,
    "underlying_price": 78220.8,
    "underlying_index": "BTC-25SEP26",
    "best_ask_amount": 39.2,
    "best_bid_amount": 0.0,
}


def make_provider(
    connector: ScriptedConnector,
    symbols: tuple[str, ...] = (NEAR_SYMBOL,),
    settings: DeribitSettings | None = None,
    metrics: RecordingMetrics | None = None,
    sleeper: RecordingSleeper | None = None,
    discovery: FakeDiscovery | None = None,
    now: Callable[[], datetime] = lambda: NOW,
) -> DeribitProvider:
    return DeribitProvider(
        make_conventions(),
        settings if settings is not None else make_deribit_settings(),
        connect=connector,
        discovery=discovery if discovery is not None else FakeDiscovery(symbols),
        sleep=sleeper if sleeper is not None else RecordingSleeper(),
        now=now,
        metrics=metrics,
    )


async def take(provider: DeribitProvider, count: int) -> list[QuoteUpdate]:
    """Discover, stream ``count`` updates, then close and let the iterator end on its own.

    Closing rather than breaking out, so the provider's own shutdown runs and the generator is
    finished the way the ingestion loop finishes it.
    """
    await provider.discover()
    updates: list[QuoteUpdate] = []
    async for update in provider.stream():
        updates.append(update)
        if len(updates) >= count:
            await provider.close()
    return updates


def ticker_frame(name: str = NEAR_SYMBOL, bid: float | None = 0.050) -> str:
    return notification(make_ticker(instrument_name=name, bid=bid))


def counted(metrics: RecordingMetrics, name: str) -> int:
    return sum(value for counted, value, _ in metrics.counters if counted == name)


# --- the port


def test_the_provider_satisfies_the_port() -> None:
    provider: MarketDataProvider = DeribitProvider(make_conventions(), connect=ScriptedConnector())

    assert provider is not None


def test_the_settings_in_force_are_readable() -> None:
    settings = make_deribit_settings(heartbeat_seconds=12.0)

    assert make_provider(ScriptedConnector(), settings=settings).settings == settings


def test_a_provider_with_no_settings_uses_the_defaults() -> None:
    assert DeribitProvider(make_conventions(), connect=ScriptedConnector()).settings == (
        DeribitSettings()
    )


# --- discovery


async def test_discover_turns_the_inventory_into_identities_sorted_by_expiry() -> None:
    provider = make_provider(ScriptedConnector(), symbols=(FAR_SYMBOL, NEAR_SYMBOL))

    assert await provider.discover() == (make_instrument(), make_instrument(expiry=FAR))


async def test_discover_skips_what_does_not_parse_and_counts_it() -> None:
    metrics = RecordingMetrics()
    provider = make_provider(
        ScriptedConnector(), symbols=(NEAR_SYMBOL, "BTC-PERPETUAL", "BTC-27MAR26"), metrics=metrics
    )

    assert await provider.discover() == (make_instrument(),)
    assert counted(metrics, "marketdata.deribit.symbol_unparseable") == 2


async def test_discover_skips_another_underlying_and_counts_it() -> None:
    metrics = RecordingMetrics()
    provider = make_provider(
        ScriptedConnector(), symbols=(NEAR_SYMBOL, "ETH-27AUG26-3000-C"), metrics=metrics
    )

    assert await provider.discover() == (make_instrument(),)
    assert counted(metrics, "marketdata.deribit.symbol_foreign") == 1


async def test_an_inventory_with_nothing_on_this_underlying_is_refused() -> None:
    provider = make_provider(ScriptedConnector(), symbols=("ETH-27AUG26-3000-C",))

    with pytest.raises(DiscoveryError, match="lists no option on 'BTC'"):
        await provider.discover()


async def test_a_failed_inventory_call_leaves_as_the_venue_error() -> None:
    provider = make_provider(
        ScriptedConnector(), discovery=FakeDiscovery(DiscoveryError("GET failed"))
    )

    with pytest.raises(MarketDataError):
        await provider.discover()


# --- opening a session


async def test_the_heartbeat_is_the_first_request_on_a_new_socket() -> None:
    connection = ScriptedConnection(ticker_frame())
    provider = make_provider(ScriptedConnector(connection))

    await take(provider, 1)

    assert connection.sent[0]["method"] == SET_HEARTBEAT
    assert connection.sent[0]["params"] == {"interval": 10}


async def test_every_discovered_instrument_is_subscribed_on_its_ticker_channel() -> None:
    connection = ScriptedConnection(ticker_frame())
    provider = make_provider(ScriptedConnector(connection), symbols=(FAR_SYMBOL, NEAR_SYMBOL))

    await take(provider, 1)

    assert connection.channels(SUBSCRIBE) == [channel_of(NEAR_SYMBOL), channel_of(FAR_SYMBOL)]


async def test_nothing_but_ticker_channels_is_ever_subscribed() -> None:
    """Design §4.5 rejects ``book.*``; a subscription to it would be a decision nobody took."""
    connection = ScriptedConnection(ticker_frame())
    provider = make_provider(ScriptedConnector(connection), symbols=(FAR_SYMBOL, NEAR_SYMBOL))

    await take(provider, 1)

    assert connection.channels(SUBSCRIBE)
    assert all(
        channel.startswith("ticker.") and channel.endswith(f".{TICKER_INTERVAL}")
        for channel in connection.channels(SUBSCRIBE)
    )


async def test_subscriptions_go_out_in_batches_of_the_configured_size() -> None:
    connection = ScriptedConnection(ticker_frame())
    provider = make_provider(
        ScriptedConnector(connection),
        symbols=(NEAR_SYMBOL, FAR_SYMBOL, OTHER_SYMBOL),
        settings=make_deribit_settings(subscribe_batch_size=2),
    )

    await take(provider, 1)

    sizes = [len(request["params"]["channels"]) for request in connection.requests(SUBSCRIBE)]
    assert sizes == [2, 1]


async def test_requests_carry_distinct_ids_on_one_socket() -> None:
    connection = ScriptedConnection(ticker_frame())
    provider = make_provider(ScriptedConnector(connection))

    await take(provider, 1)

    ids = [message["id"] for message in connection.sent]
    assert len(set(ids)) == len(ids)


async def test_the_socket_is_opened_at_the_configured_url() -> None:
    connector = ScriptedConnector(ScriptedConnection(ticker_frame()))
    settings = replace(make_deribit_settings(), ws_url="wss://test.deribit.com/ws/api/v2")

    await take(make_provider(connector, settings=settings), 1)

    assert connector.opened == ["wss://test.deribit.com/ws/api/v2"]


# --- the stream


async def test_a_ticker_becomes_an_update_for_the_instrument_it_names() -> None:
    provider = make_provider(ScriptedConnector(ScriptedConnection(ticker_frame())))

    (update,) = await take(provider, 1)

    assert update.instrument == make_instrument()
    assert update.underlying_price == FORWARD


async def test_the_local_stamp_is_the_injected_instant() -> None:
    later = NOW + timedelta(seconds=3)
    provider = make_provider(
        ScriptedConnector(ScriptedConnection(ticker_frame())), now=lambda: later
    )

    (update,) = await take(provider, 1)

    assert update.observation.ts_local == later


async def test_a_ticker_for_an_instrument_not_held_is_ignored() -> None:
    metrics = RecordingMetrics()
    connection = ScriptedConnection(ticker_frame(OTHER_SYMBOL), ticker_frame())
    provider = make_provider(ScriptedConnector(connection), metrics=metrics)

    updates = await take(provider, 1)

    assert [update.instrument for update in updates] == [make_instrument()]
    assert counted(metrics, "marketdata.deribit.ticker_ignored") == 1


async def test_a_notification_on_another_channel_is_ignored() -> None:
    metrics = RecordingMetrics()
    book = notification(make_ticker(), channel=f"book.{NEAR_SYMBOL}.100ms")
    provider = make_provider(
        ScriptedConnector(ScriptedConnection(book, ticker_frame())), metrics=metrics
    )

    updates = await take(provider, 1)

    assert len(updates) == 1
    assert counted(metrics, "marketdata.deribit.message_ignored") == 1


@pytest.mark.parametrize(
    "frame",
    ["not json at all", "[1, 2, 3]", json.dumps({"method": "subscription", "params": 5})],
    ids=["text", "array", "params-not-object"],
)
async def test_a_frame_that_is_not_a_message_is_dropped_and_the_session_goes_on(
    frame: str,
) -> None:
    metrics = RecordingMetrics()
    provider = make_provider(
        ScriptedConnector(ScriptedConnection(frame, ticker_frame())), metrics=metrics
    )

    updates = await take(provider, 1)

    assert len(updates) == 1
    assert counted(metrics, "marketdata.deribit.message_malformed") == 1


async def test_a_malformed_ticker_is_dropped_and_the_session_goes_on() -> None:
    metrics = RecordingMetrics()
    broken = notification(make_ticker(timestamp=None))
    provider = make_provider(
        ScriptedConnector(ScriptedConnection(broken, ticker_frame())), metrics=metrics
    )

    updates = await take(provider, 1)

    assert len(updates) == 1
    assert counted(metrics, "marketdata.deribit.ticker_malformed") == 1


async def test_a_frame_with_an_unknown_method_is_ignored() -> None:
    metrics = RecordingMetrics()
    odd = json.dumps({"jsonrpc": "2.0", "method": "announcement", "params": {}})
    provider = make_provider(
        ScriptedConnector(ScriptedConnection(odd, ticker_frame())), metrics=metrics
    )

    await take(provider, 1)

    assert counted(metrics, "marketdata.deribit.message_ignored") == 1


# --- the heartbeat


async def test_a_test_request_is_answered_with_public_test() -> None:
    connection = ScriptedConnection(heartbeat("test_request"), ticker_frame())
    provider = make_provider(ScriptedConnector(connection))

    await take(provider, 1)

    assert len(connection.requests(TEST)) == 1


async def test_a_plain_heartbeat_is_not_answered() -> None:
    """The guard: answering every heartbeat would double the traffic and prove nothing."""
    connection = ScriptedConnection(heartbeat("heartbeat"), ticker_frame())
    provider = make_provider(ScriptedConnector(connection))

    await take(provider, 1)

    assert connection.requests(TEST) == []


async def test_the_reply_to_public_test_settles_without_complaint() -> None:
    metrics = RecordingMetrics()
    connection = ScriptedConnection(heartbeat("test_request"), ticker_frame())
    provider = make_provider(ScriptedConnector(connection), metrics=metrics)

    await take(provider, 1)

    assert counted(metrics, "marketdata.deribit.test_request") == 1


# --- losing the connection


async def test_a_dropped_connection_is_reopened_and_fully_resubscribed() -> None:
    first = ScriptedConnection(ticker_frame(), ConnectionLost("dropped by the venue"))
    second = ScriptedConnection(ticker_frame(FAR_SYMBOL))
    provider = make_provider(ScriptedConnector(first, second), symbols=(NEAR_SYMBOL, FAR_SYMBOL))

    updates = await take(provider, 2)

    assert [update.instrument.expiry for update in updates] == [NOW.replace(month=8), FAR]
    assert second.sent[0]["method"] == SET_HEARTBEAT
    assert second.channels(SUBSCRIBE) == first.channels(SUBSCRIBE)


async def test_a_drop_is_counted_and_waited_out() -> None:
    metrics = RecordingMetrics()
    sleeper = RecordingSleeper()
    provider = make_provider(
        ScriptedConnector(
            ScriptedConnection(ConnectionLost("dropped")), ScriptedConnection(ticker_frame())
        ),
        metrics=metrics,
        sleeper=sleeper,
    )

    await take(provider, 1)

    assert counted(metrics, "marketdata.deribit.disconnected") == 1
    assert sleeper.waits == [1.0]


async def test_a_refused_connection_is_backed_off_exponentially_up_to_the_ceiling() -> None:
    sleeper = RecordingSleeper()
    refusals = [ConnectionLost("refused") for _ in range(5)]
    provider = make_provider(
        ScriptedConnector(*refusals, ScriptedConnection(ticker_frame())),
        settings=make_deribit_settings(reconnect_initial_seconds=1.0, reconnect_max_seconds=4.0),
        sleeper=sleeper,
    )

    await take(provider, 1)

    assert sleeper.waits == [1.0, 2.0, 4.0, 4.0, 4.0]


async def test_the_backoff_resets_once_a_session_has_opened() -> None:
    """A venue that was down for an hour and is back must not be polled once a minute all day."""
    sleeper = RecordingSleeper()
    provider = make_provider(
        ScriptedConnector(
            ConnectionLost("refused"),
            ConnectionLost("refused"),
            ScriptedConnection(ticker_frame(), ConnectionLost("dropped")),
            ScriptedConnection(ticker_frame()),
        ),
        sleeper=sleeper,
    )

    await take(provider, 2)

    assert sleeper.waits == [1.0, 2.0, 1.0]


async def test_an_os_error_from_the_connector_is_a_failed_attempt_too() -> None:
    sleeper = RecordingSleeper()
    provider = make_provider(
        ScriptedConnector(OSError("name resolution failed"), ScriptedConnection(ticker_frame())),
        sleeper=sleeper,
    )

    updates = await take(provider, 1)

    assert len(updates) == 1
    assert sleeper.waits == [1.0]


async def test_silence_past_the_timeout_is_treated_as_a_drop() -> None:
    """The fake raises what ``asyncio.wait_for`` would; asserted is the provider's answer."""
    metrics = RecordingMetrics()
    provider = make_provider(
        ScriptedConnector(ScriptedConnection(TimeoutError()), ScriptedConnection(ticker_frame())),
        metrics=metrics,
    )

    updates = await take(provider, 1)

    assert len(updates) == 1
    assert counted(metrics, "marketdata.deribit.silent") == 1
    assert counted(metrics, "marketdata.deribit.disconnected") == 1


async def test_a_venue_that_refuses_a_request_ends_the_stream_loudly() -> None:
    """The one failure that is not survived: reconnecting would ask the same thing forever."""
    connection = ScriptedConnection(auto_reply=False)
    connection.feed(reply(2, error={"code": 11050, "message": "bad_request"}))
    provider = make_provider(ScriptedConnector(connection))

    with pytest.raises(ProtocolError, match="11050 bad_request"):
        await take(provider, 1)


async def test_channels_the_venue_silently_dropped_are_counted() -> None:
    """Verified live: an unknown symbol is omitted from the subscribe reply, not refused."""
    metrics = RecordingMetrics()
    connection = ScriptedConnection(ticker_frame(), refuse={channel_of(FAR_SYMBOL)})
    provider = make_provider(
        ScriptedConnector(connection), symbols=(NEAR_SYMBOL, FAR_SYMBOL), metrics=metrics
    )

    await take(provider, 1)

    assert counted(metrics, "marketdata.deribit.subscribe_refused") == 1
    assert counted(metrics, "marketdata.deribit.subscribed") == 1


async def test_a_reply_to_a_request_nobody_made_is_ignored() -> None:
    connection = ScriptedConnection(reply(99, "ok"), ticker_frame())
    provider = make_provider(ScriptedConnector(connection))

    assert len(await take(provider, 1)) == 1


# --- a universe that moves


async def test_rediscovery_subscribes_what_was_born_and_unsubscribes_what_died() -> None:
    connection = ScriptedConnection(ticker_frame())
    discovery = FakeDiscovery((NEAR_SYMBOL, FAR_SYMBOL), (NEAR_SYMBOL, OTHER_SYMBOL))
    provider = make_provider(ScriptedConnector(connection), discovery=discovery)
    await provider.discover()
    stream = provider.stream()
    await anext(stream)

    discovered = await provider.discover()

    assert discovered == (make_instrument(), make_instrument(strike=70_000.0))
    assert connection.channels(SUBSCRIBE)[-1] == channel_of(OTHER_SYMBOL)
    assert connection.channels(UNSUBSCRIBE) == [channel_of(FAR_SYMBOL)]
    await provider.close()
    assert [update async for update in stream] == []


async def test_after_rediscovery_a_ticker_for_the_dead_instrument_is_ignored() -> None:
    connection = ScriptedConnection(ticker_frame())
    discovery = FakeDiscovery((NEAR_SYMBOL, FAR_SYMBOL), (NEAR_SYMBOL,))
    provider = make_provider(ScriptedConnector(connection), discovery=discovery)
    await provider.discover()
    stream = provider.stream()
    await anext(stream)
    await provider.discover()
    connection.feed(ticker_frame(FAR_SYMBOL), ticker_frame(NEAR_SYMBOL))

    update = await anext(stream)

    assert update.instrument == make_instrument()
    await provider.close()
    assert [update async for update in stream] == []


async def test_rediscovery_before_the_stream_opens_touches_no_socket() -> None:
    provider = make_provider(ScriptedConnector(), discovery=FakeDiscovery((NEAR_SYMBOL,)))

    await provider.discover()
    await provider.discover()

    assert provider.settings is not None  # nothing to assert on a socket that was never opened


async def test_the_next_socket_subscribes_the_rediscovered_universe() -> None:
    """Full resubscription means the *current* universe, not the one the first socket had."""
    first = ScriptedConnection(ticker_frame(), ConnectionLost("dropped"))
    second = ScriptedConnection(ticker_frame(OTHER_SYMBOL))
    discovery = FakeDiscovery((NEAR_SYMBOL,), (NEAR_SYMBOL, OTHER_SYMBOL))
    provider = make_provider(ScriptedConnector(first, second), discovery=discovery)
    await provider.discover()
    stream = provider.stream()
    await anext(stream)
    await provider.discover()

    await anext(stream)

    assert second.channels(SUBSCRIBE) == [channel_of(NEAR_SYMBOL), channel_of(OTHER_SYMBOL)]
    await provider.close()
    assert [update async for update in stream] == []


# --- closing


async def test_closing_ends_the_stream_instead_of_reconnecting() -> None:
    connector = ScriptedConnector(ScriptedConnection(ticker_frame()), ScriptedConnection())
    provider = make_provider(connector)

    await take(provider, 1)

    assert connector.opened == [DeribitSettings().ws_url]


async def test_closing_closes_the_socket() -> None:
    connection = ScriptedConnection(ticker_frame())
    provider = make_provider(ScriptedConnector(connection))

    await take(provider, 1)

    assert connection.closed


async def test_closing_twice_is_harmless() -> None:
    provider = make_provider(ScriptedConnector(ScriptedConnection(ticker_frame())))

    await take(provider, 1)
    await provider.close()


async def test_a_provider_closed_before_streaming_yields_nothing() -> None:
    connector = ScriptedConnector(ScriptedConnection(ticker_frame()))
    provider = make_provider(connector)
    await provider.close()

    assert [update async for update in provider.stream()] == []
    assert connector.opened == []


async def test_closing_during_the_backoff_ends_the_loop() -> None:
    async def close_instead_of_waiting(_seconds: float) -> None:
        await provider.close()

    provider = DeribitProvider(
        make_conventions(),
        make_deribit_settings(),
        connect=ScriptedConnector(ConnectionLost("refused")),
        discovery=FakeDiscovery((NEAR_SYMBOL,)),
        sleep=close_instead_of_waiting,
        now=lambda: NOW,
    )
    await provider.discover()

    assert [update async for update in provider.stream()] == []


# --- the decoder


def decode(data: dict[str, Any], numeraire: Numeraire = Numeraire.INVERSE) -> QuoteUpdate:
    return ticker_to_update(data, make_instrument(), numeraire, NOW)


def test_an_inverse_premium_is_multiplied_by_the_forward_of_its_expiry() -> None:
    update = decode(make_ticker(bid=0.05, ask=0.054, underlying_price=FORWARD))

    assert update.observation.bid == pytest.approx(0.05 * FORWARD)
    assert update.observation.ask == pytest.approx(0.054 * FORWARD)


def test_a_quote_currency_premium_is_taken_as_published() -> None:
    update = decode(make_ticker(bid=0.05, ask=0.054), numeraire=Numeraire.QUOTE)

    assert (update.observation.bid, update.observation.ask) == (0.05, 0.054)


def test_the_conversion_really_moves_the_number() -> None:
    """The guard on the two tests above: on a forward of one they could not be told apart."""
    inverse = decode(make_ticker(bid=0.05, underlying_price=FORWARD))
    quoted = decode(make_ticker(bid=0.05, underlying_price=FORWARD), numeraire=Numeraire.QUOTE)

    assert inverse.observation.bid != quoted.observation.bid


def test_the_venue_zero_is_an_empty_side() -> None:
    """The websocket's spelling, verified live: no order can rest at zero on this venue."""
    update = decode(make_ticker(bid=0.0, bid_amount=0.0))

    assert update.observation.bid is None


def test_a_null_is_an_empty_side_too() -> None:
    """REST's spelling of the same absence."""
    update = decode(make_ticker(ask=None, ask_amount=None))

    assert update.observation.ask is None
    assert update.observation.ask_size == 0.0


def test_the_live_deep_wing_ticker_decodes_as_ask_only() -> None:
    update = ticker_to_update(
        LIVE_TICKER, make_instrument(strike=105_000.0), Numeraire.INVERSE, NOW
    )

    assert update.observation.bid is None
    assert update.observation.ask == pytest.approx(0.0001 * 78220.8)
    assert update.observation.ask_size == 39.2
    assert update.underlying_price == 78220.8


def test_the_venue_stamp_is_milliseconds_since_the_epoch_in_utc() -> None:
    update = ticker_to_update(LIVE_TICKER, make_instrument(), Numeraire.INVERSE, NOW)

    assert update.observation.ts_exchange == datetime(2026, 9, 18, 9, 58, 48, 184_000, tzinfo=UTC)


def test_the_venue_iv_arrives_as_a_decimal() -> None:
    update = decode(make_ticker(mark_iv=62.0))

    assert update.observation.exchange_iv == pytest.approx(0.62)


@pytest.mark.parametrize("mark_iv", [None, 0.0, -1.0, math.nan], ids=["null", "zero", "neg", "nan"])
def test_a_venue_iv_that_is_not_a_volatility_is_absent(mark_iv: float | None) -> None:
    assert decode(make_ticker(mark_iv=mark_iv)).observation.exchange_iv is None


def test_a_venue_iv_of_the_wrong_type_is_absent_rather_than_fatal() -> None:
    """Auxiliary data must not cost the quote: a string where a number belongs drops the number."""
    data = make_ticker()
    data["mark_iv"] = "n/a"

    assert decode(data).observation.exchange_iv is None


def test_the_local_stamp_is_the_one_passed_in() -> None:
    assert decode(make_ticker()).observation.ts_local == NOW


def test_the_sizes_are_the_venue_amounts() -> None:
    update = decode(make_ticker(bid_amount=12.0, ask_amount=8.0))

    assert (update.observation.bid_size, update.observation.ask_size) == (12.0, 8.0)


def test_the_kind_is_the_instrument_own() -> None:
    update = ticker_to_update(
        make_ticker(), make_instrument(kind=OptionKindD.PUT), Numeraire.INVERSE, NOW
    )

    assert update.instrument.kind is OptionKindD.PUT


@pytest.mark.parametrize(
    ("field", "value", "fragment"),
    [
        ("timestamp", None, "timestamp"),
        ("timestamp", True, "timestamp"),
        ("timestamp", "1789725528184", "timestamp"),
        ("best_bid_price", -0.01, "negative"),
        ("best_bid_price", "0.05", "number"),
        ("best_bid_price", math.inf, "finite"),
        ("best_bid_amount", -1.0, "negative"),
        ("underlying_price", 0.0, "positive"),
        ("underlying_price", math.nan, "finite"),
    ],
)
def test_a_field_the_domain_would_refuse_is_refused_here_by_name(
    field: str, value: Any, fragment: str
) -> None:
    data = make_ticker()
    data[field] = value

    with pytest.raises(ValueError, match=fragment):
        decode(data)


def test_an_inverse_ticker_without_a_forward_cannot_be_normalised() -> None:
    with pytest.raises(ValueError, match="underlying price"):
        decode(make_ticker(underlying_price=None))


def test_a_quote_currency_ticker_without_a_forward_is_still_a_quote() -> None:
    update = decode(make_ticker(underlying_price=None), numeraire=Numeraire.QUOTE)

    assert update.underlying_price is None
    assert update.observation.bid == 0.05


# --- settings


def test_the_default_settings_construct() -> None:
    settings = DeribitSettings()

    assert settings.ws_url.startswith("wss://")
    assert settings.rest_url.startswith("https://")


@pytest.mark.parametrize(
    ("field", "value", "fragment"),
    [
        ("ws_url", "https://www.deribit.com/ws/api/v2", "ws://"),
        ("ws_url", "", "ws://"),
        ("rest_url", "wss://www.deribit.com/api/v2", "http://"),
        ("heartbeat_seconds", 5.0, "at least"),
        ("heartbeat_seconds", math.nan, "heartbeat_seconds"),
        ("heartbeat_seconds", 0.0, "heartbeat_seconds"),
        ("silence_timeout_seconds", 10.0, "exceed"),
        ("silence_timeout_seconds", math.inf, "silence_timeout_seconds"),
        ("reconnect_initial_seconds", 0.0, "reconnect_initial_seconds"),
        ("reconnect_initial_seconds", math.nan, "reconnect_initial_seconds"),
        ("reconnect_max_seconds", 0.5, "not be below"),
        ("request_timeout_seconds", -1.0, "request_timeout_seconds"),
        ("request_timeout_seconds", math.nan, "request_timeout_seconds"),
        ("subscribe_batch_size", 0, "at least one"),
    ],
)
def test_settings_that_could_not_run_a_session_are_refused(
    field: str, value: Any, fragment: str
) -> None:
    with pytest.raises(ValueError, match=fragment):
        replace_field(make_deribit_settings(), field, value)


def test_the_channel_spelling_is_the_venue_own() -> None:
    assert channel_of("BTC-27MAR26-60000-C") == "ticker.BTC-27MAR26-60000-C.100ms"
