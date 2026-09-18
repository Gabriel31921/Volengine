"""Discovery over REST: the envelope it decodes, the failures it translates, and the call it makes.

No network anywhere. The fetcher is injected, so the client is exercised against canned bodies,
and the bodies are the venue's own: ``LIVE_REPLY`` is a trimmed ``get_book_summary_by_currency``
answer captured from the production API on 2026-09-18, ``null`` bid included.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import pytest

from volengine.market_data.adapters.deribit_rest import (
    BOOK_SUMMARY_METHOD,
    REST_URL,
    DeribitRestClient,
    DiscoveryError,
    instrument_names_from,
)
from volengine.market_data.domain.errors import MarketDataError

LIVE_REPLY: dict[str, Any] = {
    "jsonrpc": "2.0",
    "result": [
        {
            "instrument_name": "BTC-27NOV26-73000-P",
            "bid_price": 0.031,
            "ask_price": 0.032,
            "mark_price": 0.03127502,
            "mark_iv": 36.63,
            "underlying_price": 78877.53,
            "underlying_index": "BTC-27NOV26",
            "base_currency": "BTC",
            "quote_currency": "BTC",
            "creation_timestamp": 1789725494890,
        },
        {
            "instrument_name": "BTC-25SEP26-105000-C",
            "bid_price": None,
            "ask_price": 0.0001,
            "mark_price": 5.658e-05,
            "mark_iv": 73.69,
            "underlying_price": 78217.98,
            "underlying_index": "BTC-25SEP26",
            "base_currency": "BTC",
            "quote_currency": "BTC",
            "creation_timestamp": 1789725494890,
        },
    ],
    "usIn": 1789725494890123,
    "usOut": 1789725494891234,
    "usDiff": 1111,
    "testnet": False,
}


class Fetch:
    """A ``JsonFetcher`` that answers one body, or raises, and records what it was asked."""

    def __init__(self, body: object = None, failure: Exception | None = None) -> None:
        self._body = body
        self._failure = failure
        self.calls: list[tuple[str, dict[str, str], float]] = []

    async def __call__(self, url: str, params: Mapping[str, str], timeout_seconds: float) -> object:
        self.calls.append((url, dict(params), timeout_seconds))
        if self._failure is not None:
            raise self._failure
        return self._body


def make_client(fetch: Fetch, currency: str = "BTC", timeout: float = 5.0) -> DeribitRestClient:
    return DeribitRestClient(REST_URL, currency, timeout, fetch)


# --- the envelope


def test_the_live_reply_yields_its_symbols_in_order() -> None:
    assert instrument_names_from(LIVE_REPLY) == ("BTC-27NOV26-73000-P", "BTC-25SEP26-105000-C")


def test_an_empty_inventory_is_an_empty_tuple() -> None:
    """Empty is the venue's answer, not a decoding failure; refusing it is the provider's call."""
    assert instrument_names_from({"jsonrpc": "2.0", "result": []}) == ()


@pytest.mark.parametrize(
    ("payload", "fragment"),
    [
        ("<html>", "not a JSON object"),
        ([], "not a JSON object"),
        ({"jsonrpc": "2.0"}, "no 'result' list"),
        ({"jsonrpc": "2.0", "result": {"a": 1}}, "no 'result' list"),
        ({"jsonrpc": "2.0", "result": ["BTC-27MAR26-60000-C"]}, "row 0"),
        ({"jsonrpc": "2.0", "result": [{"bid_price": 0.1}]}, "no 'instrument_name'"),
        ({"jsonrpc": "2.0", "result": [{"instrument_name": ""}]}, "no 'instrument_name'"),
    ],
    ids=["html", "array", "no-result", "result-not-a-list", "row-not-object", "no-name", "blank"],
)
def test_a_reply_that_is_not_an_inventory_is_refused_by_shape(
    payload: object, fragment: str
) -> None:
    with pytest.raises(DiscoveryError, match=fragment):
        instrument_names_from(payload)


def test_a_json_rpc_error_carries_the_venue_code_and_message() -> None:
    payload = {"jsonrpc": "2.0", "error": {"code": -32602, "message": "Invalid params"}}

    with pytest.raises(DiscoveryError, match="-32602 Invalid params"):
        instrument_names_from(payload)


def test_the_discovery_error_is_a_market_data_error() -> None:
    """The composition root catches the context's root on a failed poll; this must sit under it."""
    assert issubclass(DiscoveryError, MarketDataError)


# --- the call


async def test_the_client_asks_for_this_currency_options_at_the_summary_method() -> None:
    fetch = Fetch(LIVE_REPLY)

    await make_client(fetch, currency="ETH", timeout=7.5).instrument_names()

    assert fetch.calls == [
        (f"{REST_URL}/{BOOK_SUMMARY_METHOD}", {"currency": "ETH", "kind": "option"}, 7.5)
    ]


async def test_a_trailing_slash_on_the_root_does_not_double_up() -> None:
    fetch = Fetch(LIVE_REPLY)

    await DeribitRestClient(REST_URL + "/", "BTC", 5.0, fetch).instrument_names()

    assert fetch.calls[0][0] == f"{REST_URL}/{BOOK_SUMMARY_METHOD}"


async def test_the_client_hands_back_the_symbols() -> None:
    names = await make_client(Fetch(LIVE_REPLY)).instrument_names()

    assert names == ("BTC-27NOV26-73000-P", "BTC-25SEP26-105000-C")


@pytest.mark.parametrize(
    "failure",
    [OSError("connection refused"), TimeoutError(), ValueError("not json")],
    ids=["os", "timeout", "value"],
)
async def test_a_transport_failure_leaves_as_a_discovery_error(failure: Exception) -> None:
    with pytest.raises(DiscoveryError, match="failed"):
        await make_client(Fetch(failure=failure)).instrument_names()


async def test_a_discovery_error_from_the_fetcher_passes_through_untouched() -> None:
    """The httpx fetcher raises one of its own; wrapping it again would bury the venue's message."""
    original = DiscoveryError("GET failed: 503")

    with pytest.raises(DiscoveryError) as caught:
        await make_client(Fetch(failure=original)).instrument_names()

    assert caught.value is original


async def test_the_original_failure_is_kept_as_the_cause() -> None:
    with pytest.raises(DiscoveryError) as caught:
        await make_client(Fetch(failure=OSError("refused"))).instrument_names()

    assert isinstance(caught.value.__cause__, OSError)


# --- construction


@pytest.mark.parametrize(
    ("url", "currency", "timeout", "fragment"),
    [
        ("  ", "BTC", 5.0, "base URL"),
        (REST_URL, "", 5.0, "currency"),
        (REST_URL, "BTC", 0.0, "timeout"),
        (REST_URL, "BTC", math.nan, "timeout"),
        (REST_URL, "BTC", math.inf, "timeout"),
    ],
)
def test_a_client_that_could_not_make_a_call_is_refused(
    url: str, currency: str, timeout: float, fragment: str
) -> None:
    with pytest.raises(ValueError, match=fragment):
        DeribitRestClient(url, currency, timeout, Fetch())
