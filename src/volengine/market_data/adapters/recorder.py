"""Writing a session down: the half of ADR-004 that makes a replay possible at all.

A recording is the normalised stream, not the venue's wire traffic. What is written here is a
``QuoteUpdate`` -- the object this context's port already yields -- after the adapter has parsed
the symbol, resolved the expiry instant and decided the numeraire. Two consequences, and both are
the point:

* **Any provider is recordable.** ``RecordingProvider`` wraps *another* ``MarketDataProvider`` and
  satisfies the same port, so the synthetic feed, the constant feed and F3-C's Deribit client are
  all recorded by this one module, with nothing venue-specific in it. Recording the raw websocket
  frames instead would have meant one recorder per venue and a replay that re-ran the parser --
  which is the code most likely to have changed between the recording and the replay.
* **What is replayed is what the chain saw.** A parsing bug is *inside* the recording, not
  reproduced from it. That is a deliberate trade: this file exists to make the engine's decisions
  reproducible, not to keep a forensic copy of the venue's bytes. ``docs/SEAMS.md`` says so.

**The format is JSON Lines**: one self-describing object per line, a header first and one quote per
line after it. Chosen against a binary or pickled format for one reason -- a recording is meant to
be *diffable*. ADR-021 made every identifier in this engine derived rather than random precisely so
that a recording and its replay could be compared line by line, and that argument is worth nothing
if a person cannot read either. It is also append-only and streamable, so a session that is killed
halfway leaves a file that replays up to the point it died.

**Primitives only, and a ``schema_version`` from the first line**, for the same reasons
``contracts/`` carries both (ADR-003): a recording outlives the process that wrote it, so it must
survive a JSON hop and must say which grammar it is written in. This is not a published contract --
it never crosses a context boundary, and the domain types it carries are Market Data's own -- but
it is persisted data, and persisted data with no version is a migration nobody can write.
"""

from __future__ import annotations

import json
import math
from collections.abc import AsyncIterator, Sequence
from datetime import datetime
from pathlib import Path
from types import TracebackType
from typing import Any, Self

from volengine.market_data.domain.option_quote import (
    InstrumentId,
    OptionKindD,
    QuoteObservation,
    QuoteUpdate,
)
from volengine.market_data.domain.ports import MarketDataProvider

RECORDING_SCHEMA_VERSION = 1
"""Grammar of the file below. Checked on the first line when a recording is opened.

Bumped when a field changes meaning or disappears, never when one is added with a default -- the
same rule the contracts follow, and for the same reason: a reader that refuses every unfamiliar
file is a reader nobody can extend.
"""

HEADER_KIND = "header"
"""``kind`` of the first line: what market this is, and what it quoted."""

QUOTE_KIND = "quote"
"""``kind`` of every other line: one instrument's top of book at one instant."""


class RecordingSink:
    """An open recording file, and the two kinds of line that go into it.

    Deliberately not a provider and not a context manager over the whole session: the object that
    owns the *stream* is :class:`RecordingProvider`, and this one owns the *file*. Splitting them
    is what lets a test assert on the bytes without an event loop, and what would let a future
    caller record two markets into two files through one wrapper.

    Not thread-safe, and does not need to be: everything in this engine that touches a provider
    runs on the event loop (``entrypoints/pipeline.py`` says so, and only the calibrations leave
    it).
    """

    def __init__(self, path: Path) -> None:
        """Open the file for writing, truncating whatever was there.

        Truncating rather than appending, unlike ``CsvReportWriter``: two sessions concatenated
        into one file would produce a second header in the middle of a stream and a recording
        whose instants go backwards, which is a file no replay can honestly consume. A report is a
        log; a recording is a session.

        Args:
            path: Where the session is written.

        Raises:
            OSError: If the file cannot be opened. Raised here, at construction, so that a
                directory that does not exist is heard about before a session's quotes are
                already gone.
        """
        self._path = path
        # `newline=""` rather than the default: on Windows the default translates every "\n" into
        # "\r\n", so the same session would record to different bytes on two platforms and the
        # byte-for-byte comparison this whole stage exists for would be a comparison of line
        # endings. UTF-8 is stated for the same reason -- the platform's default encoding is not
        # part of the format.
        self._file = path.open("w", encoding="utf-8", newline="")
        self._closed = False
        self._written = 0

    @property
    def path(self) -> Path:
        """Where this recording is being written."""
        return self._path

    @property
    def written(self) -> int:
        """How many quote lines have been recorded. The header is not counted."""
        return self._written

    def header(self, market_id: str, underlying: str, instruments: Sequence[InstrumentId]) -> None:
        """Write the first line: the schema, the market, and the universe it was discovered with.

        The instrument set is recorded because ``MarketDataProvider.discover`` is a *pull* call
        that a replay has no way to re-issue: the venue is gone. Without it the replayed chain
        would learn about an instrument only when the first quote for it arrived, so every
        coverage ratio in the session would start low and climb -- a difference between a run and
        its replay that has nothing to do with the engine.

        Raises:
            ValueError: If the market id or the underlying is blank. Both end up naming the
                recording, and an unnamed recording cannot be matched to a configuration.
        """
        if not market_id.strip():
            raise ValueError(f"The recorded market id must not be blank, got {market_id!r}")
        if not underlying.strip():
            raise ValueError(f"The recorded underlying must not be blank, got {underlying!r}")
        self._write(
            {
                "schema_version": RECORDING_SCHEMA_VERSION,
                "kind": HEADER_KIND,
                "market_id": market_id,
                "underlying": underlying,
                "instruments": [_instrument_to_dict(one) for one in instruments],
            }
        )

    def quote(self, update: QuoteUpdate) -> None:
        """Write one update, exactly as the chain will be handed it on replay."""
        observation = update.observation
        self._write(
            {
                "kind": QUOTE_KIND,
                "instrument": _instrument_to_dict(update.instrument),
                "bid": observation.bid,
                "ask": observation.ask,
                "bid_size": observation.bid_size,
                "ask_size": observation.ask_size,
                "ts_exchange": observation.ts_exchange.isoformat(),
                "ts_local": observation.ts_local.isoformat(),
                "exchange_iv": observation.exchange_iv,
                "underlying_price": update.underlying_price,
            }
        )
        self._written += 1

    def close(self) -> None:
        """Flush and close. Idempotent, like every ``close`` on this side of the engine."""
        if self._closed:
            return
        self._closed = True
        self._file.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        kind: type[BaseException] | None,
        value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()

    def _write(self, payload: dict[str, Any]) -> None:
        """One compact line, flushed.

        ``allow_nan=False`` is a guard rather than a formality: ``json.dumps`` writes bare ``NaN``
        and ``Infinity`` by default, which is not JSON, which no other reader would accept, and
        which ``float()`` on the way back in would turn into a quote no constructor can build.
        ``QuoteObservation`` already refuses non-finite values, so reaching it means something
        upstream has stopped being true and the recording should say so loudly.

        Flushed per line so that a session killed by a signal leaves a usable file: everything up
        to the last complete line replays, and :mod:`recorded` stops at the first line it cannot
        parse rather than discarding the session.
        """
        if self._closed:
            raise ValueError(f"The recording at {self._path} is closed and cannot be written to")
        self._file.write(json.dumps(payload, allow_nan=False, separators=(",", ":")))
        self._file.write("\n")
        self._file.flush()


class RecordingProvider:
    """Another provider, seen through a tap: every update it yields is written down first.

    A decorator rather than a base class, and it satisfies ``MarketDataProvider`` structurally like
    every other adapter here. The composition root wraps whatever the configuration named
    (``entrypoints/pipeline.with_recording``), so *recording* is a mode of a run rather than a
    provider a file can select -- which is right, because "record the Deribit feed" and "run the
    Deribit feed" must be the same session or the recording is of something else.

    The tap is written **before** the update is yielded, so a consumer that raises leaves the quote
    it choked on in the file. That is the quote a person will want.
    """

    def __init__(
        self,
        inner: MarketDataProvider,
        sink: RecordingSink,
        market_id: str,
        underlying: str,
    ) -> None:
        """Wire the tap.

        Args:
            inner: The provider actually supplying quotes. Its lifetime is owned here: closing
                this one closes that one.
            sink: The open file the session is written to.
            market_id: Which market this is, for the header. Passed rather than inferred, for the
                reason ``IngestStreamUseCase`` takes one: a provider has no ``market_id`` and
                deriving it from the first instrument would guess.
            underlying: What the chain is written on, for the header. It is what a replay checks
                the configured market against before feeding it a single quote.
        """
        self._inner = inner
        self._sink = sink
        self._market_id = market_id
        self._underlying = underlying

    async def discover(self) -> tuple[InstrumentId, ...]:
        """Ask the inner provider, record the answer as the header, and pass it through.

        The header is written here because this is the first call the ingestion loop makes
        (``IngestStreamUseCase.run`` discovers before it streams), which is what puts the header on
        the first line without this class having to sequence anything itself.
        """
        instruments = await self._inner.discover()
        self._sink.header(self._market_id, self._underlying, instruments)
        return instruments

    def stream(self) -> AsyncIterator[QuoteUpdate]:
        """The inner stream, with every update written down on its way past."""
        return self._record(self._inner.stream())

    async def close(self) -> None:
        """Close the inner provider first, then the file. Idempotent, as the port requires.

        That order matters on the way out: the inner provider may still yield on its way to
        finishing, and a sink closed first would raise on the last quote of the session.
        """
        await self._inner.close()
        self._sink.close()

    async def _record(self, updates: AsyncIterator[QuoteUpdate]) -> AsyncIterator[QuoteUpdate]:
        async for update in updates:
            self._sink.quote(update)
            yield update


def _instrument_to_dict(instrument: InstrumentId) -> dict[str, Any]:
    """One contract's identity as primitives: no datetime, no enum object.

    ``kind`` goes out as ``str(...)`` of a ``StrEnum``, so the value written is the member's own
    string and a rename cannot silently change the file's grammar -- the same rule the published
    contracts follow.
    """
    return {
        "underlying": instrument.underlying,
        "expiry": instrument.expiry.isoformat(),
        "strike": instrument.strike,
        "kind": str(instrument.kind),
    }


def instrument_from_dict(payload: dict[str, Any]) -> InstrumentId:
    """Rebuild one contract's identity, rejecting anything that is not the shape written above.

    Lives here rather than in :mod:`recorded` so that the two halves of the grammar sit in one
    file: a field renamed on the way out and not on the way in is the classic serialisation bug,
    and it is much harder to write when both spellings are three lines apart.

    Raises:
        ValueError: If a field is missing, is of the wrong type, or names an ``OptionKindD`` that
            does not exist. The ``InstrumentId`` constructor then applies its own invariants on
            top, so a recording cannot introduce an instrument the domain would have refused.
    """
    return InstrumentId(
        underlying=_text(payload, "underlying"),
        expiry=_instant(payload, "expiry"),
        strike=_number(payload, "strike"),
        kind=_kind(payload, "kind"),
    )


def observation_from_dict(payload: dict[str, Any]) -> QuoteObservation:
    """Rebuild one top of book from a recorded line."""
    return QuoteObservation(
        bid=_optional_number(payload, "bid"),
        ask=_optional_number(payload, "ask"),
        bid_size=_number(payload, "bid_size"),
        ask_size=_number(payload, "ask_size"),
        ts_exchange=_instant(payload, "ts_exchange"),
        ts_local=_instant(payload, "ts_local"),
        exchange_iv=_optional_number(payload, "exchange_iv"),
    )


def update_from_dict(payload: dict[str, Any]) -> QuoteUpdate:
    """Rebuild one recorded quote line into the object the chain consumes."""
    instrument = payload.get("instrument")
    if not isinstance(instrument, dict):
        raise ValueError("A recorded quote needs an 'instrument' table")
    return QuoteUpdate(
        instrument=instrument_from_dict(instrument),
        observation=observation_from_dict(payload),
        underlying_price=_optional_number(payload, "underlying_price"),
    )


def _text(payload: dict[str, Any], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str):
        raise ValueError(f"The recorded field {key!r} must be a string, got {value!r}")
    return value


def _kind(payload: dict[str, Any], key: str) -> OptionKindD:
    value = _text(payload, key)
    try:
        return OptionKindD(value)
    except ValueError as failure:
        known = ", ".join(str(one) for one in OptionKindD)
        raise ValueError(
            f"The recorded field {key!r} is not an option kind: {value!r}; known: {known}"
        ) from failure


def _number(payload: dict[str, Any], key: str) -> float:
    value = _optional_number(payload, key)
    if value is None:
        raise ValueError(f"The recorded field {key!r} must be a number, got null")
    return value


def _optional_number(payload: dict[str, Any], key: str) -> float | None:
    """A number or an explicit ``None``, and never a ``bool``.

    ``isinstance(True, int)`` is ``True`` in Python, so a JSON ``true`` would otherwise arrive as
    the size ``1.0``. Non-finite values are refused here as well as on the way out: a hand-edited
    recording is the one way ``Infinity`` can reach this function.
    """
    value = payload.get(key)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"The recorded field {key!r} must be a number, got {value!r}")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"The recorded field {key!r} must be finite, got {number}")
    return number


def _instant(payload: dict[str, Any], key: str) -> datetime:
    """An ISO-8601 instant, which must be timezone-aware.

    Aware is not a preference: every quantity this engine computes from a recording is a
    subtraction of instants, and mixing a naive one with an aware one raises ``TypeError`` several
    layers away from the line that was missing its offset.
    """
    text = _text(payload, key)
    try:
        moment = datetime.fromisoformat(text)
    except ValueError as failure:
        raise ValueError(f"The recorded field {key!r} is not an instant: {text!r}") from failure
    if moment.tzinfo is None:
        raise ValueError(f"The recorded field {key!r} must be timezone-aware, got naive {text!r}")
    return moment
