"""Reading a session back, and driving the engine's clock from it: the other half of ADR-004.

``RecordingProvider`` writes the normalised stream; this module replays it. What comes out is the
same ``QuoteUpdate`` that went in, handed to the same ingestion loop, so nothing downstream of the
port can tell a replay from a live feed -- which is the whole claim: *the full pipeline reproducible
from recorded messages through to the final report*.

**The recording is the clock.** A replay that read the wall clock would compute a staleness of
weeks for a file recorded last month, and every quote in it would be refused as inadmissible before
the first snapshot. So :class:`RecordedProvider` moves a :class:`ReplayClock` to each update's own
``ts_local`` before yielding it -- the instant we *received* that message during the original
session -- and the engine relives the session's own timeline. ``platform.clock.SimulatedClock`` is
the implementation, satisfied structurally here: this context may not import ``platform`` (rule 6),
and does not need to.

``ts_local`` rather than ``ts_exchange``, deliberately. The two differ by the transport latency the
freshness policy exists to see, and the engine's clock during the original session was, by
definition, at ``ts_local`` when the message landed. Setting it from the venue's stamp instead
would delete that latency from the replay and make every recorded quote look fresher than it was.

**No pacing.** The updates are yielded as fast as the consumer takes them, and simulated time jumps
from stamp to stamp. A replay is not a re-enactment in real time: waiting out the gaps would make a
recorded hour take an hour and would put a wall-clock dependency back into the one path built to
have none.

**Streamed, not loaded.** The file is read line by line as the consumer iterates, so a recording
larger than memory replays fine. Only the header and the first quote's instant are read eagerly, by
:func:`open_recording` -- the composition root needs the second of those to place the clock at the
session's start *before* it builds anything.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Iterator
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Protocol

from volengine.market_data.adapters.recorder import (
    HEADER_KIND,
    QUOTE_KIND,
    RECORDING_SCHEMA_VERSION,
    instrument_from_dict,
    update_from_dict,
)
from volengine.market_data.domain.option_quote import InstrumentId, QuoteUpdate


class ReplayClock(Protocol):
    """A clock somebody else moves: what a replay needs beyond reading the time.

    Declared here rather than imported from ``platform/`` for the same reason
    ``domain/ports.Clock`` is declared in this context -- a port describes a need, and this need is
    the replay driver's. ``platform.clock.SimulatedClock`` satisfies it structurally, with no
    import in either direction, and the connection is made once in the composition root.

    Deliberately *not* the ``Clock`` protocol with a method added. Reading the time and setting it
    are two different permissions: every use case in the engine holds a ``Clock``, and none of them
    may move it. Keeping ``set`` on a separate protocol is what stops that from being an option.

    The argument is positional-only so that an implementation may spell its parameter however it
    likes; ``SimulatedClock.set`` calls it ``time``.
    """

    def set(self, instant: datetime, /) -> None:
        """Place the clock at a recorded instant."""
        ...


@dataclass(frozen=True, slots=True)
class Recording:
    """A recording that has been opened and found readable: its header, and where it starts.

    Everything a composition root needs to *decide* before it builds anything -- which market the
    file holds, what the chain is written on, and the instant the engine's clock must be placed at
    -- without reading the quotes. The quotes stay on disk until somebody iterates them.
    """

    path: Path
    """The file, kept so the provider can open it again for streaming."""

    market_id: str
    """Which market was recorded. A replay runs this market and refuses any other."""

    underlying: str
    """What the recorded chain is written on. Checked against the configured market before a
    single quote is applied: ``QuoteChain.apply`` refuses another underlying's quotes with a
    ``ValueError`` that would surface mid-session as a crashed task."""

    instruments: tuple[InstrumentId, ...]
    """The universe ``discover`` returned during the original session, in its original order."""

    started_at: datetime
    """``ts_local`` of the first recorded quote: where the replay clock is placed.

    The first quote's rather than the header's, because the header carries no instant of its own --
    and because the first thing that must be true of a replay is that its clock is not ahead of the
    first message it will deliver. A ``CalibratedSurface`` refuses a fit stamped before its
    snapshot, and that is exactly what a clock starting late would produce.
    """


def open_recording(path: Path) -> Recording:
    """Read the header and the first quote, and refuse anything that is not a recording.

    Everything this function can refuse, it refuses *now*: a truncated file, a file from another
    schema, a file with a header and no quotes. The alternative is a run that starts, builds four
    contexts, subscribes a bus and then discovers on its first message that it was handed a CSV.

    Raises:
        OSError: If the file cannot be read.
        ValueError: If the file is empty, does not begin with a header, was written under another
            schema version, or contains no quote at all.
    """
    with path.open("r", encoding="utf-8") as handle:
        lines = _lines(handle)
        first = next(lines, None)
        if first is None:
            raise ValueError(f"The recording at {path} is empty")
        header = _decode(first, path)
        _require_header(header, path)
        started_at = _first_instant(lines, path)
    return Recording(
        path=path,
        market_id=_required_text(header, "market_id", path),
        underlying=_required_text(header, "underlying", path),
        instruments=_instruments(header, path),
        started_at=started_at,
    )


class RecordedProvider:
    """A ``MarketDataProvider`` that reads a file and moves the clock as it goes.

    Satisfies the port structurally, like every adapter here: ``discover`` answers with the
    universe the header recorded, ``stream`` yields the quotes in the order they were written, and
    ``close`` ends the stream. A finite source, so a run over it terminates on its own -- which is
    what ``Pipeline.run`` calls a stream that ended, and it drains the work in flight before it
    stops the consumers.

    **Not re-entrant, and re-openable.** Each call to ``stream`` opens the file again from the top,
    so a provider replays the same session twice rather than continuing it -- the same property
    ``SyntheticProvider`` gets by re-seeding. Two iterators at once is not a supported shape and
    never was; the port describes one long-lived stream.
    """

    def __init__(self, recording: Recording, clock: ReplayClock | None = None) -> None:
        """Point the provider at an opened recording.

        Args:
            recording: The file, already validated by :func:`open_recording`.
            clock: The clock to move, or ``None`` to replay the quotes without touching time.
                ``None`` is for a test that only wants the messages; a *run* passes the engine's
                own clock, and it is the only way the recorded timeline reaches the freshness
                policy, the snapshot cadence and the fit's own stamp.
        """
        self._recording = recording
        self._clock = clock
        self._closed = False

    @property
    def recording(self) -> Recording:
        """The session being replayed."""
        return self._recording

    async def discover(self) -> tuple[InstrumentId, ...]:
        """The universe as it was when the session was recorded.

        The full live set and never a delta, like the port asks. Replaying it rather than
        rebuilding it from the quotes is what makes a replayed coverage ratio equal to the
        original one instead of climbing from zero as instruments are seen for the first time.
        """
        return self._recording.instruments

    def stream(self) -> AsyncIterator[QuoteUpdate]:
        """Open the file and replay it. A plain ``def`` returning the iterator, as the port says."""
        return self._stream()

    async def _stream(self) -> AsyncIterator[QuoteUpdate]:
        """Yield every recorded quote, moving the clock to each one's ``ts_local`` first.

        The clock is moved **before** the update is yielded, which is the order the original
        session had: the message had already landed when the engine looked at the time. Moving it
        afterwards would have every quote applied one instant in the chain's past and would make
        the first snapshot of a replay disagree with the first snapshot of the run it came from.

        ``close`` is checked between updates, so closing ends the stream rather than merely
        promising to.
        """
        with self._recording.path.open("r", encoding="utf-8") as handle:
            for index, line in enumerate(_lines(handle)):
                payload = _decode(line, self._recording.path, index)
                if payload.get("kind") == HEADER_KIND:
                    continue
                if self._closed:
                    return
                update = _quote(payload, self._recording.path, index)
                if self._clock is not None:
                    self._clock.set(update.observation.ts_local)
                yield update

    async def close(self) -> None:
        """Stop the replay at the next update. Idempotent, as the port requires."""
        self._closed = True


def _lines(handle: Iterator[str]) -> Iterator[str]:
    """Every non-blank line of the file, stripped.

    Blank lines are skipped rather than refused: a recording is flushed line by line and a file
    that grew a trailing newline through an editor is still the session that was recorded.
    """
    for line in handle:
        stripped = line.strip()
        if stripped:
            yield stripped


def _decode(line: str, path: Path, index: int | None = None) -> dict[str, Any]:
    where = f"{path}" if index is None else f"{path} line {index + 1}"
    try:
        payload = json.loads(line)
    except json.JSONDecodeError as failure:
        raise ValueError(f"The recording at {where} is not readable: {failure}") from failure
    if not isinstance(payload, dict):
        raise ValueError(f"The recording at {where} holds {type(payload).__name__}, not an object")
    return payload


def _require_header(header: dict[str, Any], path: Path) -> None:
    """Refuse a file whose first line is not a header of a schema this build reads."""
    if header.get("kind") != HEADER_KIND:
        raise ValueError(
            f"The recording at {path} does not start with a {HEADER_KIND!r} line; "
            f"its first line is {header.get('kind')!r}"
        )
    version = header.get("schema_version")
    if version != RECORDING_SCHEMA_VERSION:
        raise ValueError(
            f"The recording at {path} is schema version {version!r}, and this build reads "
            f"version {RECORDING_SCHEMA_VERSION}"
        )


def _instruments(header: dict[str, Any], path: Path) -> tuple[InstrumentId, ...]:
    listed = header.get("instruments")
    if not isinstance(listed, list):
        raise ValueError(f"The recording at {path} has no 'instruments' list in its header")
    rebuilt: list[InstrumentId] = []
    for entry in listed:
        if not isinstance(entry, dict):
            raise ValueError(f"The recording at {path} lists an instrument that is not an object")
        rebuilt.append(instrument_from_dict(entry))
    return tuple(rebuilt)


def _first_instant(lines: Iterator[str], path: Path) -> datetime:
    """``ts_local`` of the first quote line, which is where the replay clock starts."""
    for index, line in enumerate(lines):
        payload = _decode(line, path, index + 1)
        if payload.get("kind") == QUOTE_KIND:
            return _quote(payload, path, index + 1).observation.ts_local
    raise ValueError(f"The recording at {path} holds a header and no quote")


def _quote(payload: dict[str, Any], path: Path, index: int) -> QuoteUpdate:
    """Rebuild one line, naming the line when it cannot be rebuilt.

    The line number is the whole value of this wrapper. A recording is a long file, and "the
    recorded field 'bid' must be a number" without a position is a message that sends a person
    reading a hundred thousand lines by hand.
    """
    try:
        return update_from_dict(payload)
    except ValueError as failure:
        raise ValueError(f"The recording at {path} line {index + 1} is broken: {failure}") from (
            failure
        )


def _required_text(header: dict[str, Any], key: str, path: Path) -> str:
    value = header.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"The recording at {path} has no {key!r} in its header")
    return value
