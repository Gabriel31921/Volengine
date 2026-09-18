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

import asyncio
import json
from collections import deque
from collections.abc import AsyncIterator, Iterable, Sequence
from datetime import UTC, datetime, time, timedelta
from typing import Any

from volengine.market_data.adapters.deribit_ws import (
    SET_HEARTBEAT,
    SUBSCRIBE,
    TEST,
    UNSUBSCRIBE,
    ConnectionLost,
    DeribitSettings,
    channel_of,
)
from volengine.market_data.domain.admissibility import AdmissibilityThresholds
from volengine.market_data.domain.errors import MarketDataError
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
from volengine.market_data.domain.snapshot_policy import SnapshotPolicy, SnapshotPolicyConfig

NOW = datetime(2026, 7, 27, 8, 0, tzinfo=UTC)
NAIVE = datetime(2026, 7, 27, 8, 0)
"""Same instant with no zone. Every entry point in this context must refuse it."""

NEAR = datetime(2026, 8, 27, 8, 0, tzinfo=UTC)
FAR = datetime(2026, 10, 27, 8, 0, tzinfo=UTC)
"""Two live expiries at Deribit's 08:00 UTC, roughly one and three months out."""

FORWARD = 60_000.0
"""Forward for both expiries. The default strike sits on it, so k = 0 unless a test moves it."""

NEAR_SYMBOL = "BTC-27AUG26-60000-C"
FAR_SYMBOL = "BTC-27OCT26-60000-C"
"""How the venue spells ``make_instrument()`` and ``make_instrument(expiry=FAR)``."""


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


def make_snapshot_policy(
    cadence_seconds: float = 1.0,
    material_move_threshold: float = 0.0,
    min_coverage_ratio: float = 0.0,
    max_quiet_seconds: float | None = None,
) -> SnapshotPolicy:
    """A policy that emits on every cadence tick and never marks anything degraded.

    The permissive defaults are what make the application tests readable: a test about the
    snapshot *sequence* should not have to reason about whether a movement filter let the second
    snapshot through. Each knob is turned by exactly the test that is about it.
    """
    return SnapshotPolicy(
        SnapshotPolicyConfig(
            cadence_seconds=cadence_seconds,
            material_move_threshold=material_move_threshold,
            min_coverage_ratio=min_coverage_ratio,
            max_quiet_seconds=max_quiet_seconds,
        )
    )


class StubProvider:
    """A ``MarketDataProvider`` that replays a fixed script and then ends.

    Satisfies the port structurally, with no venue, no socket and no asyncio primitive beyond the
    generator itself. ``stream`` is written as ``async def`` returning an ``AsyncIterator`` via
    ``yield``, which is exactly the shape the port's docstring says the annotation was chosen to
    permit.

    ``closed`` is recorded because the use case promises to close the provider however its loop
    ends, and a promise about a ``finally`` block is only worth what a test asserting on it is.
    """

    def __init__(
        self,
        updates: Sequence[QuoteUpdate],
        instruments: Sequence[InstrumentId] = (),
    ) -> None:
        self._updates = tuple(updates)
        self._instruments = tuple(instruments)
        self.closed = False

    async def discover(self) -> tuple[InstrumentId, ...]:
        return self._instruments

    def stream(self) -> AsyncIterator[QuoteUpdate]:
        return self._stream()

    async def _stream(self) -> AsyncIterator[QuoteUpdate]:
        for update in self._updates:
            yield update

    async def close(self) -> None:
        self.closed = True


class ShiftingProvider:
    """A ``MarketDataProvider`` whose inventory changes from one ``discover`` call to the next.

    The shape rediscovery is tested on: each call answers the next scripted universe, the last one
    repeating once the script runs out, and an entry that is an exception is raised instead --
    which is how a venue whose REST is down for one poll is spelled. ``block`` keeps the stream
    open after its script, like ``BlockingProvider`` in the entrypoints builders, so a timer racing
    it has something to race.
    """

    def __init__(
        self,
        updates: Sequence[QuoteUpdate],
        universes: Sequence[Sequence[InstrumentId] | MarketDataError],
        block: bool = False,
    ) -> None:
        self._updates = tuple(updates)
        self._universes = tuple(universes)
        self._block = block
        self.discoveries = 0
        self.closed = False

    async def discover(self) -> tuple[InstrumentId, ...]:
        answer = self._universes[min(self.discoveries, len(self._universes) - 1)]
        self.discoveries += 1
        if isinstance(answer, MarketDataError):
            raise answer
        return tuple(answer)

    def stream(self) -> AsyncIterator[QuoteUpdate]:
        return self._stream()

    async def _stream(self) -> AsyncIterator[QuoteUpdate]:
        for update in self._updates:
            yield update
        if self._block:
            await asyncio.Event().wait()

    async def close(self) -> None:
        self.closed = True


# --- the Deribit adapter's collaborators, scripted


def make_deribit_settings(
    heartbeat_seconds: float = 10.0,
    silence_timeout_seconds: float = 25.0,
    reconnect_initial_seconds: float = 1.0,
    reconnect_max_seconds: float = 4.0,
    subscribe_batch_size: int = 250,
) -> DeribitSettings:
    """Valid transport settings with a short backoff ladder, so a sequence of waits is legible."""
    return DeribitSettings(
        heartbeat_seconds=heartbeat_seconds,
        silence_timeout_seconds=silence_timeout_seconds,
        reconnect_initial_seconds=reconnect_initial_seconds,
        reconnect_max_seconds=reconnect_max_seconds,
        subscribe_batch_size=subscribe_batch_size,
    )


def make_ticker(
    instrument_name: str = NEAR_SYMBOL,
    bid: float | None = 0.050,
    ask: float | None = 0.054,
    bid_amount: float | None = 12.0,
    ask_amount: float | None = 8.0,
    timestamp: int | None = int(NOW.timestamp() * 1000),
    underlying_price: float | None = FORWARD,
    mark_iv: float | None = 62.0,
) -> dict[str, Any]:
    """The ``data`` of one ``ticker.*`` notification, as the live venue spells it.

    Premiums in BTC, the volatility in percent, the stamp in milliseconds: the venue's encoding,
    not the chain's, so that a test of the decoder starts from what actually arrives. ``None`` for a
    key writes a JSON ``null``; the venue's *other* spelling of an empty side, ``0.0``, is passed
    explicitly by the test that is about it.
    """
    return {
        "timestamp": timestamp,
        "state": "open",
        "instrument_name": instrument_name,
        "index_price": FORWARD - 100.0,
        "mark_price": 0.052,
        "mark_iv": mark_iv,
        "bid_iv": 60.0,
        "ask_iv": 64.0,
        "best_bid_price": bid,
        "best_bid_amount": bid_amount,
        "best_ask_price": ask,
        "best_ask_amount": ask_amount,
        "underlying_price": underlying_price,
        "underlying_index": "BTC-27AUG26",
        "open_interest": 100.0,
    }


def notification(data: dict[str, Any], channel: str | None = None) -> str:
    """One ``subscription`` frame carrying ``data`` on its instrument's ticker channel."""
    name = data.get("instrument_name")
    return json.dumps(
        {
            "jsonrpc": "2.0",
            "method": "subscription",
            "params": {
                "channel": channel if channel is not None else channel_of(str(name)),
                "data": data,
            },
        }
    )


def reply(request_id: int, result: Any = None, error: dict[str, Any] | None = None) -> str:
    """A JSON-RPC reply, either a result or an error object, the way the venue writes them."""
    message: dict[str, Any] = {"jsonrpc": "2.0", "id": request_id}
    if error is not None:
        message["error"] = error
    else:
        message["result"] = result
    return json.dumps(message)


def heartbeat(kind: str = "test_request") -> str:
    """A heartbeat frame: ``"heartbeat"`` on the interval, ``"test_request"`` demanding a reply."""
    return json.dumps({"jsonrpc": "2.0", "method": "heartbeat", "params": {"type": kind}})


class ScriptedConnection:
    """A ``Connection`` that plays a script of frames and records every request sent on it.

    Replies itself to the requests the provider makes -- a subscribe is answered with the channels
    it asked for minus ``refuse``, a heartbeat request with ``"ok"``, a ``public/test`` with a
    version -- because that is what the live venue does, and a provider waiting on a reply that
    never came would look like a bug in the test rather than in the code. ``auto_reply=False``
    turns that off for the tests that script the replies themselves.

    Replies are served **before** the scripted frames, as the live venue does -- the answer to a
    subscribe arrives before the tickers it opens -- so a test may script its tickers at
    construction and still have the session's opening exchanges settled first.

    The script ends how the test says: a ``ConnectionLost`` in it is a drop, and ``close`` from
    either side wakes a pending ``recv`` with one, which is what the real socket does.
    """

    def __init__(
        self,
        *frames: str | Exception,
        auto_reply: bool = True,
        refuse: Iterable[str] = (),
    ) -> None:
        self._incoming: asyncio.Queue[str | Exception] = asyncio.Queue()
        self._replies: deque[str] = deque()
        self._auto_reply = auto_reply
        self._refuse = set(refuse)
        self.sent: list[dict[str, Any]] = []
        self.closed = False
        self.feed(*frames)

    def feed(self, *frames: str | Exception) -> None:
        for frame in frames:
            self._incoming.put_nowait(frame)

    def requests(self, method: str) -> list[dict[str, Any]]:
        """Every request of one method, in the order they were sent."""
        return [message for message in self.sent if message.get("method") == method]

    def channels(self, method: str = SUBSCRIBE) -> list[str]:
        """Every channel named across the requests of one method, in wire order."""
        return [
            channel
            for message in self.requests(method)
            for channel in message["params"]["channels"]
        ]

    async def send(self, text: str) -> None:
        if self.closed:
            raise ConnectionLost("sending on a closed scripted connection")
        message = json.loads(text)
        self.sent.append(message)
        if not self._auto_reply:
            return
        method = message.get("method")
        if method in (SUBSCRIBE, UNSUBSCRIBE):
            accepted = [one for one in message["params"]["channels"] if one not in self._refuse]
            self._replies.append(reply(message["id"], accepted))
        elif method == SET_HEARTBEAT:
            self._replies.append(reply(message["id"], "ok"))
        elif method == TEST:
            self._replies.append(reply(message["id"], {"version": "1.2.26"}))

    async def recv(self) -> str:
        if self.closed:
            raise ConnectionLost("receiving on a closed scripted connection")
        if self._replies:
            return self._replies.popleft()
        item = await self._incoming.get()
        if isinstance(item, Exception):
            raise item
        return item

    async def close(self) -> None:
        self.closed = True
        self._incoming.put_nowait(ConnectionLost("closed by the client"))


class ScriptedConnector:
    """A ``Connector`` handing out scripted connections in order; an exception in the list is a
    connection attempt that fails. Running out is a failure too, so a test that expected fewer
    connections than the provider opened finds out."""

    def __init__(self, *outcomes: ScriptedConnection | Exception) -> None:
        self._outcomes = list(outcomes)
        self.opened: list[str] = []

    async def __call__(self, url: str, timeout_seconds: float) -> ScriptedConnection:
        self.opened.append(url)
        if not self._outcomes:
            raise ConnectionLost("the connector's script ran out")
        outcome = self._outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


class FakeDiscovery:
    """A ``Discovery`` answering scripted inventories in order, the last one repeating."""

    def __init__(self, *answers: Sequence[str] | Exception) -> None:
        self._answers = list(answers)
        self.calls = 0

    async def instrument_names(self) -> tuple[str, ...]:
        answer = self._answers[min(self.calls, len(self._answers) - 1)]
        self.calls += 1
        if isinstance(answer, Exception):
            raise answer
        return tuple(answer)


class RecordingSleeper:
    """A ``Sleeper`` that records the waits instead of waiting, and refuses to be asked forever.

    The limit is the guard against a reconnection loop that never opens a session: with a sleep
    that returns at once, such a loop would spin instead of hang, and a test would time out with
    nothing to say. Raising after ``limit`` calls makes it fail with the sequence in hand.
    """

    def __init__(self, limit: int = 32) -> None:
        self.waits: list[float] = []
        self._limit = limit

    async def __call__(self, seconds: float) -> None:
        self.waits.append(seconds)
        if len(self.waits) > self._limit:
            raise AssertionError(
                f"the provider backed off more than {self._limit} times: {self.waits}"
            )
        await asyncio.sleep(0)
