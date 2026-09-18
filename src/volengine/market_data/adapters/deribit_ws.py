"""The Deribit feed: ``MarketDataProvider`` over the venue's websocket, with its REST discovery.

The first network code in the engine, and the block Design §4.5 calls the highest operational
risk. It arrives on scaffolding already proven by replay (F3-B), which is the point of its place in
the order: everything downstream of the port has been exercised on synthetic and recorded streams,
so what this module has to get right is the venue and nothing else. Four things, each argued where
it is done:

* **The symbol never reaches the domain.** ``BTC-27MAR26-60000-C`` is parsed by
  :mod:`instrument_parser` at discovery, and every ticker is matched back to its ``InstrumentId``
  by the symbol the venue put on it. The published composition event renders the engine's own
  spelling (``acl.instrument_key``), not this one.
* **The numeraire is decided here** (Design §4.5). An inverse option quotes its premium in the
  underlying -- ``0.05`` is 0.05 BTC -- and a Black-76 inversion needs a premium in the currency
  the strike and the forward are in. :func:`ticker_to_update` multiplies by the ticker's own
  ``underlying_price`` when the conventions say ``INVERSE``, which is the exact identity for an
  option settled in the asset: the payoff ``max(S_T - K, 0) / S_T`` valued under the asset as
  numeraire is the forward-measure value divided by ``F``, so ``premium_BTC * F`` is the USD
  premium and nothing about the smile enters the conversion. It is the forward of *that expiry*
  and not the spot index, because that is what makes the venue's own put-call parity,
  ``C - P = (F - K) / F`` in BTC, come out as ``C - P = F - K`` in USD -- the identity the
  forward cross-check downstream is written against.
* **Deribit's IV is auxiliary.** ``mark_iv`` travels as ``exchange_iv``, a percent turned into a
  decimal, and is never a calibration input: calibrating on the venue's number would be
  calibrating on the venue's model.
* **Robustness, which is where these projects die.** The venue's heartbeat is enabled and its
  ``test_request`` answered, or it disconnects the client; silence longer than the configured
  timeout is treated as a dead connection, because a half-open TCP session raises nothing;
  reconnection backs off exponentially from a configured floor to a configured ceiling; and every
  reconnect resubscribes the **whole** current universe, so a message lost in the gap costs
  freshness and never correctness -- the property the port's docstring promises. Per-instrument
  staleness needs nothing from this module beyond stamping ``ts_exchange`` from *each ticker's own*
  ``timestamp``, which is what the admissibility rules measure age against; ``ts_local`` is the
  injected ``now`` at receipt, and the ACL reconciles the two (ADR-021).

**Every collaborator that touches the world is injected.** The socket comes from a
:data:`Connector`, the inventory from a :class:`Discovery`, the wait between attempts from a
:data:`Sleeper`, the local stamp from ``now``. The production ones are the defaults, and
``websockets`` is imported inside :func:`connect_with_websockets` and nowhere else -- it is an
optional extra, and ``entrypoints/pipeline.py`` imports this module on every leg of CI. A test
therefore drives a whole session through a scripted connection: the heartbeat reply, the
resubscription after a drop, the backoff sequence, the decoding of a ticker into the chain's own
object, none of it needing a network or the extra.

What this module does **not** do is subscribe to ``book.*``: the incremental order book is
microstructure v1 has no use for (Design §4.5), and ``ticker.{instrument}.100ms`` republishes the
entire top of book on every change, which is the full-replacement shape ``QuoteUpdate`` is.
"""

from __future__ import annotations

import asyncio
import json
import math
from collections.abc import AsyncIterator, Awaitable, Callable, Iterable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, Final, Protocol

from volengine.market_data.adapters.deribit_rest import (
    REST_URL,
    DeribitError,
    DeribitRestClient,
    DiscoveryError,
)
from volengine.market_data.adapters.instrument_parser import instrument_from_symbol
from volengine.market_data.domain.market_conventions import MarketConventions, Numeraire
from volengine.market_data.domain.option_quote import (
    InstrumentId,
    QuoteObservation,
    QuoteUpdate,
)
from volengine.market_data.domain.ports import MetricsSink

if TYPE_CHECKING:
    from websockets.asyncio.client import ClientConnection

WS_URL: Final = "wss://www.deribit.com/ws/api/v2"
"""The production websocket. The test network is ``wss://test.deribit.com/ws/api/v2``."""

TICKER_INTERVAL: Final = "100ms"
"""The channel cadence Design §4.5 names. ``raw`` would be every change and ``agg2`` slower;
this is a design choice about the feed, not a threshold, which is why it is not configuration."""

TICKER_PREFIX: Final = "ticker."
"""Every channel this provider subscribes to starts with it; anything else is ignored."""

DERIBIT_MIN_HEARTBEAT_SECONDS: Final = 10.0
"""The venue refuses a shorter interval. A fact about Deribit, enforced by ``DeribitSettings``."""

CLOSE_TIMEOUT_SECONDS: Final = 2.0
"""How long the closing handshake may take before the socket is dropped. The library's default
of ten seconds would hold a shutdown for that long on a venue that has stopped answering."""

JSONRPC: Final = "2.0"
SUBSCRIBE: Final = "public/subscribe"
UNSUBSCRIBE: Final = "public/unsubscribe"
SET_HEARTBEAT: Final = "public/set_heartbeat"
TEST: Final = "public/test"
"""The four methods this client sends. ``public/test`` is the reply the venue's ``test_request``
demands; missing it is what gets a client disconnected."""


class ConnectionLost(DeribitError):
    """The socket is gone: closed by the venue, by the network, or by silence.

    The one failure the stream survives. Raised by a :class:`Connection` when a send or receive
    finds the socket closed, and by the provider itself when nothing arrives for longer than the
    configured silence timeout -- both are answered by backing off and reconnecting.
    """


class ProtocolError(DeribitError):
    """The venue answered a request with an error object.

    Not survived. A refused ``public/subscribe`` or ``public/set_heartbeat`` means the client is
    asking for something this venue does not offer, which is a configuration or a version problem
    rather than a transient one, and reconnecting would ask again forever.
    """


class Connection(Protocol):
    """A text websocket, as this provider needs one: send, receive, close.

    The seam the tests drive a session through. ``websockets`` satisfies it through the wrapper
    :func:`connect_with_websockets` returns; a scripted fake satisfies it with a queue.
    """

    async def send(self, text: str) -> None:
        """Send one text frame. Raises :class:`ConnectionLost` if the socket is closed."""
        ...

    async def recv(self) -> str:
        """Wait for the next text frame. Raises :class:`ConnectionLost` if the socket closes."""
        ...

    async def close(self) -> None:
        """Close the socket. Idempotent."""
        ...


type Connector = Callable[[str, float], Awaitable[Connection]]
"""``(url, timeout in seconds) -> an open connection``. Raises ``ConnectionLost``, ``OSError`` or
``TimeoutError`` when the venue cannot be reached; all three are backed off and retried."""

type Sleeper = Callable[[float], Awaitable[None]]
"""How the provider waits between attempts. ``asyncio.sleep`` in production.

Not the engine's ``Clock`` port: the registry hands a provider its ``MarketConfig`` and nothing
else (ADR-022, ``docs/SEAMS.md``), and a reconnection delay is wall-clock time between two real
sockets, which no simulated clock has an opinion about. Injected so a test records the sequence
instead of waiting it out.
"""


class Discovery(Protocol):
    """The inventory call, as the provider needs it: every option symbol the venue lists."""

    async def instrument_names(self) -> tuple[str, ...]:
        """Raises :class:`DiscoveryError` when the venue cannot be asked."""
        ...


@dataclass(frozen=True, slots=True)
class DeribitSettings:
    """Everything about the connection a deployment retunes without touching Python (ADR-012).

    Read from ``[market.deribit]`` by ``entrypoints/config.py`` on the terms ADR-028 set for the
    synthetic feed: the section is optional, and complete when present. Nothing here is a
    threshold about *quotes* -- those are ``AdmissibilityThresholds`` -- and nothing here is about
    the market's conventions; this is the transport.
    """

    ws_url: str = WS_URL
    """Where the stream is. A ``ws://`` or ``wss://`` URL."""

    rest_url: str = REST_URL
    """Where the inventory is. An ``http://`` or ``https://`` URL."""

    heartbeat_seconds: float = 30.0
    """The interval asked of ``public/set_heartbeat``. At least the venue's minimum of ten.

    The venue then sends a ``heartbeat`` at this cadence and a ``test_request`` the client must
    answer; it is what turns a dead connection into a detectable one from the venue's side.
    """

    silence_timeout_seconds: float = 75.0
    """How long the stream may be silent before the connection is declared dead. Longer than one
    heartbeat interval, or a healthy connection would be dropped between two heartbeats; two and a
    half intervals is a margin for a slow hop, not for a dead one."""

    reconnect_initial_seconds: float = 1.0
    """The first wait after a lost connection. Doubles on each consecutive failure."""

    reconnect_max_seconds: float = 60.0
    """The ceiling the doubling stops at. A venue down for an hour is polled once a minute."""

    request_timeout_seconds: float = 10.0
    """How long opening the socket, or one inventory call, may take."""

    subscribe_batch_size: int = 250
    """Channels per ``public/subscribe`` request. A thousand-instrument chain subscribes in four
    messages rather than one the size of the venue's frame limit."""

    def __post_init__(self) -> None:
        if not self.ws_url.startswith(("ws://", "wss://")):
            raise ValueError(
                f"The websocket URL must start with ws:// or wss://, got {self.ws_url!r}"
            )
        if not self.rest_url.startswith(("http://", "https://")):
            raise ValueError(
                f"The REST URL must start with http:// or https://, got {self.rest_url!r}"
            )
        _require_positive_finite(self.heartbeat_seconds, "heartbeat_seconds")
        if self.heartbeat_seconds < DERIBIT_MIN_HEARTBEAT_SECONDS:
            raise ValueError(
                f"The heartbeat_seconds must be at least {DERIBIT_MIN_HEARTBEAT_SECONDS}, the "
                f"venue's minimum, got {self.heartbeat_seconds}"
            )
        _require_positive_finite(self.silence_timeout_seconds, "silence_timeout_seconds")
        if self.silence_timeout_seconds <= self.heartbeat_seconds:
            raise ValueError(
                f"The silence_timeout_seconds must exceed heartbeat_seconds, or a healthy "
                f"connection is dropped between two heartbeats; got "
                f"{self.silence_timeout_seconds} against {self.heartbeat_seconds}"
            )
        _require_positive_finite(self.reconnect_initial_seconds, "reconnect_initial_seconds")
        _require_positive_finite(self.reconnect_max_seconds, "reconnect_max_seconds")
        if self.reconnect_max_seconds < self.reconnect_initial_seconds:
            raise ValueError(
                f"The reconnect_max_seconds must not be below reconnect_initial_seconds, got "
                f"{self.reconnect_max_seconds} against {self.reconnect_initial_seconds}"
            )
        _require_positive_finite(self.request_timeout_seconds, "request_timeout_seconds")
        if self.subscribe_batch_size < 1:
            raise ValueError(
                f"The subscribe_batch_size must be at least one, got {self.subscribe_batch_size}"
            )


# --- the production connector


async def connect_with_websockets(url: str, timeout_seconds: float) -> Connection:
    """Open the venue's websocket with the ``websockets`` library.

    Imported here and nowhere else, for the reason :mod:`deribit_rest` gives for ``httpx``: the
    library is an optional extra and this module is imported by the composition root on every
    installation. The composition root checks the extra is present when it builds the provider,
    so an ``ImportError`` here means that check was bypassed and is left to propagate.

    The library's own ping frames stay on: they are a second, protocol-level detector of a dead
    peer beside the venue's heartbeat, and a ping that times out surfaces as a closed connection
    -- which is the one failure this provider already knows how to survive.

    Raises:
        ConnectionLost: If the venue cannot be reached, refuses the handshake, or does not answer
            within the timeout. All three are transient from the provider's point of view.
    """
    from websockets.asyncio.client import connect
    from websockets.exceptions import ConnectionClosed, InvalidHandshake

    try:
        socket = await connect(
            url, open_timeout=timeout_seconds, close_timeout=CLOSE_TIMEOUT_SECONDS
        )
    except (OSError, InvalidHandshake, TimeoutError) as failure:
        raise ConnectionLost(f"cannot connect to {url}: {failure}") from failure
    return _WebSocketConnection(socket, ConnectionClosed)


class _WebSocketConnection:
    """A ``websockets`` connection behind the :class:`Connection` shape.

    Translates the library's ``ConnectionClosed`` into :class:`ConnectionLost` at both ends, so the
    provider's reconnection logic depends on one exception it owns and not on a library it only
    optionally has. The exception class is handed in rather than imported, because the import
    lives in the one function allowed to make it.
    """

    def __init__(self, socket: ClientConnection, closed: type[Exception]) -> None:
        self._socket = socket
        self._closed = closed

    async def send(self, text: str) -> None:
        try:
            await self._socket.send(text)
        except self._closed as failure:
            raise ConnectionLost(f"the connection closed while sending: {failure}") from failure

    async def recv(self) -> str:
        try:
            message = await self._socket.recv()
        except self._closed as failure:
            raise ConnectionLost(f"the connection closed: {failure}") from failure
        return message.decode("utf-8") if isinstance(message, bytes) else message

    async def close(self) -> None:
        await self._socket.close()


# --- the message that matters


def ticker_to_update(
    data: Mapping[str, Any],
    instrument: InstrumentId,
    numeraire: Numeraire,
    ts_local: datetime,
) -> QuoteUpdate:
    """Turn one ``ticker.*`` notification into the chain's own object.

    A pure function, and the one place the venue's encoding of a top of book is read. The facts
    it encodes, each verified against the live feed:

    * ``timestamp`` is milliseconds since the epoch, UTC, and becomes ``ts_exchange``.
    * An **empty side is a zero price with a zero amount** on the websocket (``null`` on REST).
      No order can rest at zero on this venue -- the tick size forbids it -- so zero means "no
      order" here, and ``QuoteObservation``'s distinction between ``None`` and ``0.0`` is honoured
      by mapping the venue's zero to ``None``. A *negative* price is malformed and refused.
    * ``mark_iv`` is a percent. It becomes ``exchange_iv`` as a decimal, and only when positive:
      the venue publishes ``0.0`` where it has no number, which is an absence and not a
      volatility.
    * ``underlying_price`` is the forward for this expiry -- the synchronous future, or the
      venue's synthetic one where none trades -- and is what the chain keys its forwards by.

    And the one decision that is ours: under ``Numeraire.INVERSE`` both premiums are multiplied
    by that forward, for the reason the module docstring gives. Under ``Numeraire.QUOTE`` they
    are taken as published.

    Raises:
        ValueError: If a field is missing, of the wrong type, non-finite or negative; if an inverse
            ticker carries no ``underlying_price``, since its premiums cannot then be expressed in
            the strike's currency; or if ``QuoteObservation`` or ``QuoteUpdate`` refuse the
            result. The caller counts it and drops the message: one malformed frame is not a
            reason to end a session.
    """
    ts_exchange = _instant(data, "timestamp")
    forward = _optional_positive(data, "underlying_price")
    bid = _side(data, "best_bid_price")
    ask = _side(data, "best_ask_price")
    if numeraire is Numeraire.INVERSE:
        if forward is None:
            raise ValueError(
                "An inverse premium cannot be normalised without an underlying price, and the "
                "ticker carries none"
            )
        bid = None if bid is None else bid * forward
        ask = None if ask is None else ask * forward
    return QuoteUpdate(
        instrument=instrument,
        observation=QuoteObservation(
            bid=bid,
            ask=ask,
            bid_size=_amount(data, "best_bid_amount"),
            ask_size=_amount(data, "best_ask_amount"),
            ts_exchange=ts_exchange,
            ts_local=ts_local,
            exchange_iv=_percent(data, "mark_iv"),
        ),
        underlying_price=forward,
    )


def channel_of(symbol: str) -> str:
    """The ticker channel for one symbol: ``ticker.BTC-27MAR26-60000-C.100ms``."""
    return f"{TICKER_PREFIX}{symbol}.{TICKER_INTERVAL}"


# --- the provider


@dataclass(frozen=True, slots=True)
class _Request:
    """One request in flight, kept until its reply settles it."""

    method: str
    channels: frozenset[str]
    """What a subscribe asked for, so the reply can say what was refused. Empty otherwise."""


class DeribitProvider:
    """A ``MarketDataProvider`` over one Deribit currency's option chain.

    Satisfies the port structurally, like every adapter here. ``discover`` is the REST inventory,
    ``stream`` the websocket with its reconnection loop, ``close`` the end of both. One instance
    is one long-lived stream that outlasts individual sockets, which is the shape the port asks
    for: an adapter that reconnects internally keeps yielding into the same iterator rather than
    ending it.

    **Rediscovery is the caller's cadence, not this class's.** ``discover`` may be called again
    while the stream is open, and then it does the venue-side half of a composition change --
    subscribing what was born, unsubscribing what died -- and answers with the new universe; the
    composition event itself is the use case's to announce. Which is why the periodic poll lives
    in ``entrypoints/pipeline.py`` beside the heartbeat, read from configuration (ADR-012), and
    not on a timer in here.
    """

    def __init__(
        self,
        conventions: MarketConventions,
        settings: DeribitSettings | None = None,
        *,
        connect: Connector = connect_with_websockets,
        discovery: Discovery | None = None,
        sleep: Sleeper = asyncio.sleep,
        now: Callable[[], datetime] = lambda: datetime.now(UTC),
        metrics: MetricsSink | None = None,
    ) -> None:
        """Wire the provider to a market and, optionally, to fakes.

        Args:
            conventions: The market this feed serves. Its ``underlying`` is the venue currency
                asked for and the filter every discovered symbol must pass; its
                ``expiry_time_utc`` places the expiries; its ``numeraire`` decides whether a
                premium is multiplied by the forward.
            settings: The transport knobs, or the defaults the class ships with.
            connect: How a socket is opened. The ``websockets`` connector by default.
            discovery: How the inventory is fetched. A :class:`DeribitRestClient` on
                ``settings.rest_url`` by default.
            sleep: How the provider waits between reconnection attempts.
            now: Where ``ts_local`` comes from. The wall clock by default, for the reason
                ``ConstantProvider`` gives: a venue stamps its own messages and the engine
                reconciles the two (ADR-021); a replay comes from ``RecordedProvider``, not from
                this adapter reading a different clock.
            metrics: Where the connection's observations go -- reconnects, refused channels,
                dropped frames -- or ``None`` to discard them. ``None`` is what the registry can
                supply today (``docs/SEAMS.md``): ``ProviderFactory`` receives a ``MarketConfig``
                and nothing else.
        """
        self._conventions = conventions
        self._settings = settings if settings is not None else DeribitSettings()
        self._connect = connect
        self._discovery: Discovery = (
            discovery
            if discovery is not None
            else DeribitRestClient(
                base_url=self._settings.rest_url,
                currency=conventions.underlying,
                timeout_seconds=self._settings.request_timeout_seconds,
            )
        )
        self._sleep = sleep
        self._now = now
        self._metrics: MetricsSink = metrics if metrics is not None else _NullMetrics()
        self._symbols: dict[str, InstrumentId] = {}
        self._connection: Connection | None = None
        self._closed = False
        self._next_id = 0
        self._pending: dict[int, _Request] = {}
        self._send_lock = asyncio.Lock()

    @property
    def settings(self) -> DeribitSettings:
        """The transport settings in force: what a wiring test asserts a file arrived through."""
        return self._settings

    # --- the port

    async def discover(self) -> tuple[InstrumentId, ...]:
        """The venue's current option chain on this underlying, as identities.

        Every listed symbol is parsed; one that does not parse, or names another underlying, is
        counted and skipped rather than failing the call -- an odd listing is the venue's business
        and one of them must not take the session down. An inventory that leaves *nothing* is
        refused, because a live chain with no options is not a thing that happens and a configured
        underlying the venue does not list is.

        On a call made while the stream is open, the difference against the previous universe is
        applied to the socket: new symbols are subscribed, dead ones unsubscribed. The
        subscription set and the answer are therefore always the same set, which is what makes
        the composition event the use case announces true of the feed as well as of the chain.

        Returns:
            The full live set (ADR-013), sorted by expiry, strike and kind so that two calls on
            one inventory answer identically.

        Raises:
            DiscoveryError: If the inventory cannot be fetched or lists nothing on this
                underlying.
            ConnectionLost: If the socket died while the difference was being applied. The
                stream reconnects on its own with the new universe; the caller counts a failed
                poll.
        """
        names = await self._discovery.instrument_names()
        symbols = self._own(names)
        if not symbols:
            raise DiscoveryError(
                f"The venue lists no option on {self._conventions.underlying!r}; "
                f"{len(names)} symbols were skipped"
            )
        born = symbols.keys() - self._symbols.keys()
        dead = self._symbols.keys() - symbols.keys()
        self._symbols = symbols
        connection = self._connection
        if connection is not None:
            await self._subscribe(connection, born)
            await self._unsubscribe(connection, dead)
        return tuple(
            sorted(symbols.values(), key=lambda one: (one.expiry, one.strike, one.kind.value))
        )

    def stream(self) -> AsyncIterator[QuoteUpdate]:
        """Open the feed. A plain ``def`` returning the iterator, as the port documents."""
        return self._stream()

    async def close(self) -> None:
        """End the stream and drop the socket. Idempotent, as the port requires.

        Closing the socket is what wakes a ``recv`` in progress: it raises ``ConnectionLost``,
        the loop sees the closed flag, and the iterator ends instead of reconnecting.
        """
        self._closed = True
        connection, self._connection = self._connection, None
        if connection is not None:
            await connection.close()

    # --- the session

    async def _stream(self) -> AsyncIterator[QuoteUpdate]:
        """Connect, subscribe, relay; on loss, back off and do it again until closed.

        ``failures`` counts consecutive attempts that did not produce an open session and is what
        the backoff grows on. It resets once a session is open -- a venue that was down for an
        hour and is back should not be polled once a minute for the rest of the day -- which does
        mean a venue that accepts and drops at once is retried at the floor interval every time.
        The floor is configuration, and one second is a rate no venue objects to.
        """
        failures = 0
        while not self._closed:
            try:
                connection = await self._connect(
                    self._settings.ws_url, self._settings.request_timeout_seconds
                )
            except (ConnectionLost, OSError, TimeoutError):
                failures += 1
                self._count("connect_failed")
                await self._backoff(failures)
                continue

            self._connection = connection
            try:
                await self._open_session(connection)
                self._count("connected")
                failures = 0
                while not self._closed:
                    update = await self._handle(connection, await self._receive(connection))
                    if update is not None:
                        yield update
            except (ConnectionLost, OSError, TimeoutError):
                if not self._closed:
                    failures += 1
                    self._count("disconnected")
            finally:
                self._connection = None
                await connection.close()

            if not self._closed:
                await self._backoff(failures)

    async def _open_session(self, connection: Connection) -> None:
        """What every socket is asked first: the heartbeat, then the whole universe.

        The request counter and the pending table are per socket: a reply to a request made on a
        socket that has since died would settle the wrong request on the next one.
        """
        self._next_id = 0
        self._pending = {}
        await self._send(
            connection, SET_HEARTBEAT, {"interval": int(self._settings.heartbeat_seconds)}
        )
        await self._subscribe(connection, self._symbols.keys())

    async def _receive(self, connection: Connection) -> str:
        """The next frame, or ``ConnectionLost`` after the silence timeout.

        The timeout is the detector for a half-open connection, which raises nothing on its own:
        the venue promised a heartbeat every ``heartbeat_seconds`` and has not sent one.
        """
        try:
            return await asyncio.wait_for(
                connection.recv(), timeout=self._settings.silence_timeout_seconds
            )
        except TimeoutError as failure:
            self._count("silent")
            raise ConnectionLost(
                f"nothing received from {self._settings.ws_url} in "
                f"{self._settings.silence_timeout_seconds} s"
            ) from failure

    async def _handle(self, connection: Connection, text: str) -> QuoteUpdate | None:
        """Route one frame: a ticker becomes an update, everything else is bookkeeping.

        Frames that are not JSON objects are counted and dropped rather than ending the session:
        the venue is a third party, and one bad frame is not evidence about the next.
        """
        try:
            message = json.loads(text)
        except ValueError:
            self._count("message_malformed")
            return None
        if not isinstance(message, dict):
            self._count("message_malformed")
            return None

        method = message.get("method")
        if method == "subscription":
            return self._update_from(message.get("params"))
        if method == "heartbeat":
            params = message.get("params")
            if isinstance(params, dict) and params.get("type") == "test_request":
                await self._send(connection, TEST, {})
                self._count("test_request")
            return None
        if "id" in message:
            self._settle(message)
            return None
        self._count("message_ignored")
        return None

    def _update_from(self, params: Any) -> QuoteUpdate | None:
        """A ``subscription`` notification, if it is a ticker for an instrument we hold."""
        if not isinstance(params, dict):
            self._count("message_malformed")
            return None
        channel = params.get("channel")
        if not isinstance(channel, str) or not channel.startswith(TICKER_PREFIX):
            self._count("message_ignored")
            return None
        data = params.get("data")
        if not isinstance(data, dict):
            self._count("ticker_malformed")
            return None
        name = data.get("instrument_name")
        instrument = self._symbols.get(name) if isinstance(name, str) else None
        if instrument is None:
            # Unsubscribed since, or never ours. Not an error: a ticker for a strike that died
            # between the inventory and the unsubscribe reply is a message about nothing.
            self._count("ticker_ignored")
            return None
        try:
            return ticker_to_update(data, instrument, self._conventions.numeraire, self._now())
        except ValueError:
            self._count("ticker_malformed")
            return None

    def _settle(self, message: dict[str, Any]) -> None:
        """Match a reply to the request it answers, and refuse to continue on an error.

        A subscribe reply lists the channels the venue accepted and silently omits the rest --
        verified against the live venue, which drops an unknown symbol without an error -- so
        what was refused is the difference against what was asked, and it is counted rather than
        raised: a symbol that vanished between the inventory and the subscribe is exactly that.
        """
        identifier = message.get("id")
        request = self._pending.pop(identifier, None) if isinstance(identifier, int) else None
        if request is None:
            return
        error = message.get("error")
        if error is not None:
            detail = (
                f"{error.get('code')} {error.get('message')}"
                if isinstance(error, dict)
                else repr(error)
            )
            raise ProtocolError(f"The venue refused {request.method}: {detail}")
        if request.method == SUBSCRIBE:
            result = message.get("result")
            accepted = (
                {one for one in result if isinstance(one, str)}
                if isinstance(result, list)
                else set()
            )
            refused = request.channels - accepted
            self._count("subscribed", len(request.channels) - len(refused))
            if refused:
                self._count("subscribe_refused", len(refused))

    async def _subscribe(self, connection: Connection, symbols: Iterable[str]) -> None:
        await self._request_channels(connection, SUBSCRIBE, symbols)

    async def _unsubscribe(self, connection: Connection, symbols: Iterable[str]) -> None:
        await self._request_channels(connection, UNSUBSCRIBE, symbols)

    async def _request_channels(
        self, connection: Connection, method: str, symbols: Iterable[str]
    ) -> None:
        """Ask for the channels of the given symbols, in batches, sorted for a stable wire."""
        channels = sorted(channel_of(symbol) for symbol in symbols)
        size = self._settings.subscribe_batch_size
        for start in range(0, len(channels), size):
            batch = channels[start : start + size]
            await self._send(connection, method, {"channels": batch}, frozenset(batch))

    async def _send(
        self,
        connection: Connection,
        method: str,
        params: Mapping[str, Any],
        channels: frozenset[str] = frozenset(),
    ) -> None:
        """One JSON-RPC request, remembered until its reply arrives.

        Under a lock because ``discover`` sends from the caller's task while the session loop
        sends from its own; two frames interleaved on one socket is a corrupt frame.
        """
        self._next_id += 1
        self._pending[self._next_id] = _Request(method=method, channels=channels)
        payload = json.dumps(
            {"jsonrpc": JSONRPC, "id": self._next_id, "method": method, "params": dict(params)}
        )
        async with self._send_lock:
            await connection.send(payload)

    async def _backoff(self, failures: int) -> None:
        """Wait ``initial * 2 ** (failures - 1)``, capped at the ceiling. No jitter, on purpose:
        one client per market has no herd to thunder with, and a fixed sequence is assertable."""
        delay = min(
            self._settings.reconnect_initial_seconds * 2 ** (failures - 1),
            self._settings.reconnect_max_seconds,
        )
        self._count("reconnect_wait")
        await self._sleep(delay)

    def _own(self, names: Iterable[str]) -> dict[str, InstrumentId]:
        """Parse every listed symbol and keep the ones on this market's underlying."""
        symbols: dict[str, InstrumentId] = {}
        for name in names:
            try:
                instrument = instrument_from_symbol(name, self._conventions)
            except ValueError:
                self._count("symbol_unparseable")
                continue
            if instrument.underlying != self._conventions.underlying:
                self._count("symbol_foreign")
                continue
            symbols[name] = instrument
        return symbols

    def _count(self, what: str, value: int = 1) -> None:
        self._metrics.counter(
            f"marketdata.deribit.{what}", value, market=self._conventions.market_id
        )


class _NullMetrics:
    """Discards everything: the default while the registry cannot hand a provider a sink."""

    def gauge(self, name: str, value: float, **tags: str) -> None:
        return None

    def counter(self, name: str, value: int = 1, **tags: str) -> None:
        return None

    def timing(self, name: str, ms: float, **tags: str) -> None:
        return None


# --- field readers, each refusing what the domain would refuse one layer later


def _instant(data: Mapping[str, Any], key: str) -> datetime:
    """Milliseconds since the epoch, as an aware UTC instant."""
    value = data.get(key)
    if isinstance(value, bool) or not isinstance(value, int | float) or not math.isfinite(value):
        raise ValueError(
            f"The ticker field {key!r} must be a number of milliseconds, got {value!r}"
        )
    return datetime.fromtimestamp(value / 1000.0, UTC)


def _side(data: Mapping[str, Any], key: str) -> float | None:
    """A price; the venue's two spellings of an empty side, ``null`` and ``0``, both as ``None``."""
    value = data.get(key)
    if value is None:
        return None
    number = _finite(value, key)
    if number < 0:
        raise ValueError(f"The ticker field {key!r} must not be negative, got {number}")
    return None if number == 0 else number


def _amount(data: Mapping[str, Any], key: str) -> float:
    """A size in contracts; absent or ``null`` is zero, which is what an empty side has."""
    value = data.get(key)
    if value is None:
        return 0.0
    number = _finite(value, key)
    if number < 0:
        raise ValueError(f"The ticker field {key!r} must not be negative, got {number}")
    return number


def _optional_positive(data: Mapping[str, Any], key: str) -> float | None:
    """A positive price or ``None``; a zero or negative one is refused, not silenced."""
    value = data.get(key)
    if value is None:
        return None
    number = _finite(value, key)
    if number <= 0:
        raise ValueError(f"The ticker field {key!r} must be positive, got {number}")
    return number


def _percent(data: Mapping[str, Any], key: str) -> float | None:
    """A percent as a decimal, or ``None`` unless it is a positive finite number.

    Lenient where the readers above are strict, deliberately: this is auxiliary data, and a quote
    must not be dropped because the venue's own volatility for it is missing or zero.
    """
    value = data.get(key)
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        return None
    return number / 100.0


def _finite(value: Any, key: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"The ticker field {key!r} must be a number, got {value!r}")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"The ticker field {key!r} must be finite, got {number}")
    return number


def _require_positive_finite(value: float, what: str) -> None:
    """``isfinite`` first and the bad cases joined with ``or``: a NaN walks through ``<= 0``."""
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"The {what} must be positive and finite, got {value}")
