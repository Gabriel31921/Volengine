"""Discovery over Deribit's REST API: the pull half of the ``MarketDataProvider`` port.

``public/get_book_summary_by_currency`` answers "which options exist right now" for one currency,
and that is all this module asks of it. The port's ``discover`` wants the **full live set** rather
than a delta (ADR-013), and this endpoint returns exactly that -- one row per listed contract --
so there is nothing to accumulate and one missed poll costs nothing. The rows carry prices too,
and those are deliberately ignored: the book summary is a snapshot of the venue's own mid and
mark, refreshed on its own schedule, while the ticker channel of :mod:`deribit_ws` republishes the
top of book every hundred milliseconds. Reading the same quantity from two sources of different
freshness would put two clocks on one chain.

**The I/O is one function, injected.** :func:`fetch_json_with_httpx` is the production fetcher
and the only place ``httpx`` is imported -- lazily, inside the function, because the library is an
optional extra (``pyproject.toml``'s ``deribit`` group) and the composition root imports this
module on every leg of CI. :class:`DeribitRestClient` takes any callable of the same shape, which
is what lets the envelope decoding below be tested against a canned reply with no socket and no
extra installed. Decoding is separated from fetching for the same reason the recording grammar is
kept apart from the file: the venue's reply is data from outside the process, and every way it can
be wrong deserves a test that does not need the venue.

Every failure leaves as :class:`DiscoveryError`, a ``MarketDataError``: the composition root's
periodic rediscovery (``entrypoints/pipeline.py``) catches the context's root error and counts a
failed poll rather than ending the session, and a REST client raising ``httpx.ConnectError`` from
three layers down would fall through that net.
"""

from __future__ import annotations

import math
from collections.abc import Awaitable, Callable, Mapping
from typing import Any, Final

from volengine.market_data.domain.errors import MarketDataError

REST_URL: Final = "https://www.deribit.com/api/v2"
"""The production API root. The test network lives at ``https://test.deribit.com/api/v2`` and is
selected from the configuration, never from code."""

BOOK_SUMMARY_METHOD: Final = "public/get_book_summary_by_currency"
"""The inventory call Design §4.5 names for discovery."""

OPTION_KIND: Final = "option"
"""The ``kind`` filter, so futures and perpetuals never reach the symbol parser."""

type JsonFetcher = Callable[[str, Mapping[str, str], float], Awaitable[object]]
"""``(url, query parameters, timeout in seconds) -> the decoded JSON body``.

The whole of what the client needs from an HTTP library. Anything raised is translated by the
caller into :class:`DiscoveryError`, so an implementation may raise whatever its library does.
"""


class DeribitError(MarketDataError):
    """Base of every failure the Deribit adapter raises. Not raised directly.

    A sub-hierarchy rather than a flat set of ``MarketDataError`` subclasses so that the two
    modules of this adapter -- discovery and the stream -- share one root a caller can catch as
    *the venue*, while the context's root still catches everything.
    """


class DiscoveryError(DeribitError):
    """The inventory call failed or answered something that is not an inventory.

    Raised for a transport failure, a timeout, a JSON-RPC error object, and an envelope whose
    shape this build cannot read. One class for all four because the caller's response is the
    same: keep the universe it already has, count the failed poll, and try again next cadence.
    """


async def fetch_json_with_httpx(
    url: str, params: Mapping[str, str], timeout_seconds: float
) -> object:
    """The production fetcher: one ``GET``, the body decoded as JSON.

    ``httpx`` is imported here and nowhere else in the engine. A module-level import would make
    this file -- and through ``entrypoints/pipeline.py``, the whole CLI -- unimportable on an
    installation without the ``deribit`` extra, which is every CI leg and every deployment that
    runs the synthetic feed. The composition root checks the extra is present when it *builds* a
    Deribit provider, so an ``ImportError`` from here means that check was bypassed and is left
    to propagate.

    Raises:
        DiscoveryError: On any HTTP failure -- connection refused, DNS, a 5xx, a timeout -- or a
            body that is not JSON. ``httpx.HTTPError`` is not an ``OSError``, so it is translated
            here rather than relied on to be caught downstream.
    """
    import httpx

    try:
        async with httpx.AsyncClient(timeout=timeout_seconds) as client:
            response = await client.get(url, params=dict(params))
            response.raise_for_status()
            return response.json()
    except httpx.HTTPError as failure:
        raise DiscoveryError(f"GET {url} failed: {failure}") from failure
    except ValueError as failure:
        raise DiscoveryError(f"GET {url} did not answer JSON: {failure}") from failure


class DeribitRestClient:
    """The inventory of one currency's options, asked for by name.

    One currency per client because one ``MarketDataProvider`` serves one market, and a market on
    this venue is one currency's chain. The currency is the market's underlying: on the inverse
    options the engine runs today the two are the same word, and the day a linear listing needs
    them to differ is the day this constructor grows an argument.
    """

    def __init__(
        self,
        base_url: str,
        currency: str,
        timeout_seconds: float,
        fetch: JsonFetcher = fetch_json_with_httpx,
    ) -> None:
        """Point the client at an API root.

        Args:
            base_url: The API root, e.g. :data:`REST_URL`. Read from the configuration so that
                the test network is a file change.
            currency: The venue's currency code, ``"BTC"``.
            timeout_seconds: How long one call may take, end to end.
            fetch: The HTTP function. The production one by default; a test passes a callable
                that answers with a canned body.

        Raises:
            ValueError: If the URL or the currency is blank, or the timeout is not positive and
                finite. Wiring mistakes, refused where the composition root can name the table.
        """
        if not base_url.strip():
            raise ValueError("The REST base URL must not be blank")
        if not currency.strip():
            raise ValueError("The currency must not be blank")
        if not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
            raise ValueError(
                f"The request timeout must be positive and finite, got {timeout_seconds}"
            )
        self._url = f"{base_url.rstrip('/')}/{BOOK_SUMMARY_METHOD}"
        self._params: dict[str, str] = {"currency": currency, "kind": OPTION_KIND}
        self._timeout = timeout_seconds
        self._fetch = fetch

    async def instrument_names(self) -> tuple[str, ...]:
        """Every option symbol the venue currently lists for this currency, in the venue's order.

        Symbols rather than ``InstrumentId``: parsing them needs the market's conventions, and
        deciding which of them belong to the configured underlying is the provider's job. This
        client knows the wire and nothing about the market.

        Raises:
            DiscoveryError: If the call fails, times out, or the reply is not an inventory.
        """
        try:
            payload = await self._fetch(self._url, self._params, self._timeout)
        except DiscoveryError:
            raise
        except (OSError, TimeoutError, ValueError) as failure:
            raise DiscoveryError(f"GET {self._url} failed: {failure}") from failure
        return instrument_names_from(payload)


def instrument_names_from(payload: object) -> tuple[str, ...]:
    """Pull the ``instrument_name`` of every row out of a book-summary reply.

    The JSON-RPC envelope is checked piece by piece and every refusal names the piece: an
    operator pointed at the wrong URL gets an HTML page, one on an old API version gets a
    different ``result`` shape, and one asking for a currency the venue does not list gets an
    ``error`` object -- three different fixes behind what would otherwise be one ``KeyError``.

    Raises:
        DiscoveryError: If the payload is not an object, carries an ``error``, lacks a ``result``
            list, or holds a row without a string ``instrument_name``.
    """
    if not isinstance(payload, Mapping):
        raise DiscoveryError(f"The inventory reply is {type(payload).__name__}, not a JSON object")
    error = payload.get("error")
    if error is not None:
        raise DiscoveryError(f"The venue refused the inventory call: {_describe_error(error)}")
    rows = payload.get("result")
    if not isinstance(rows, list):
        raise DiscoveryError(f"The inventory reply has no 'result' list; got {type(rows).__name__}")
    names: list[str] = []
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise DiscoveryError(f"Inventory row {index} is {type(row).__name__}, not an object")
        name = row.get("instrument_name")
        if not isinstance(name, str) or not name:
            raise DiscoveryError(f"Inventory row {index} has no 'instrument_name', got {name!r}")
        names.append(name)
    return tuple(names)


def _describe_error(error: Any) -> str:
    """``code message`` when the venue sent a JSON-RPC error object, ``repr`` otherwise."""
    if isinstance(error, Mapping):
        return f"{error.get('code')} {error.get('message')}"
    return repr(error)
