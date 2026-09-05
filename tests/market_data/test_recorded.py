"""Reading a session back: what it refuses, what it hands over, and what it does to the clock.

The clock is the half worth being loud about. A replay that left time alone would compute a
staleness of weeks for a file recorded last month, and every quote in it would be filtered out
before the first snapshot -- a replay that runs, publishes nothing, and looks exactly like a market
that was not moving. So the tests below check not only *that* the clock moves but that it is at
each update's own ``ts_local`` when that update is handed over, which is the property the whole of
ADR-004 rests on.

The refusals get as much room as the happy path on purpose. A recording is data from outside the
process, arriving long after the run that produced it, and every one of these failures is a file
somebody will hand the engine one day: truncated, empty, written by an older build, or a CSV
renamed.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from tests.market_data.builders import FAR, NOW, make_instrument, make_update
from volengine.market_data.adapters.recorded import RecordedProvider, open_recording
from volengine.market_data.adapters.recorder import RecordingSink
from volengine.market_data.domain.option_quote import QuoteUpdate


class SpyClock:
    """A ``ReplayClock`` that records every instant it is placed at, in order.

    The order is the point: the assertion is not that the clock ended up somewhere plausible but
    that it visited each update's own stamp, in the sequence the file holds them.
    """

    def __init__(self) -> None:
        self.instants: list[datetime] = []

    def set(self, instant: datetime, /) -> None:
        self.instants.append(instant)


def write(path: Path, *updates: QuoteUpdate, market_id: str = "BTC-DERIBIT") -> Path:
    """A recording of the given updates, with a header naming a two-instrument universe."""
    with RecordingSink(path) as sink:
        sink.header(market_id, "BTC", (make_instrument(), make_instrument(expiry=FAR)))
        for update in updates:
            sink.quote(update)
    return path


async def drain(updates: AsyncIterator[QuoteUpdate]) -> list[QuoteUpdate]:
    return [update async for update in updates]


def session(tmp_path: Path) -> Path:
    """Three quotes, one second apart, which is enough for an ordering to be visible."""
    return write(
        tmp_path / "session.jsonl",
        make_update(),
        make_update(strike=70_000.0, ts_exchange=NOW + timedelta(seconds=1)),
        make_update(strike=80_000.0, ts_exchange=NOW + timedelta(seconds=2)),
    )


# --- opening


def test_opening_reads_the_market_and_the_universe_without_reading_the_quotes(
    tmp_path: Path,
) -> None:
    """The header is what a composition root needs before it builds anything."""
    recording = open_recording(session(tmp_path))

    assert recording.market_id == "BTC-DERIBIT"
    assert recording.underlying == "BTC"
    assert len(recording.instruments) == 2


def test_the_session_starts_at_the_first_quotes_local_stamp(tmp_path: Path) -> None:
    """Where the replay clock is placed, and it must not be ahead of the first message.

    A clock starting late would stamp the first fit before the snapshot it was computed from, and
    ``CalibratedSurface`` refuses that order outright.
    """
    recording = open_recording(session(tmp_path))

    assert recording.started_at == make_update().observation.ts_local


def test_an_empty_file_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "empty.jsonl"
    path.write_text("", encoding="utf-8")

    with pytest.raises(ValueError, match="empty"):
        open_recording(path)


def test_a_file_that_is_not_json_is_refused_by_name(tmp_path: Path) -> None:
    """A CSV renamed, which is the mistake that actually happens."""
    path = tmp_path / "report.jsonl"
    path.write_text("underlying,strike,vol\nBTC,60000,0.6\n", encoding="utf-8")

    with pytest.raises(ValueError, match="not readable"):
        open_recording(path)


def test_a_file_whose_first_line_is_a_quote_is_refused(tmp_path: Path) -> None:
    """Header first, always: a recording that lost its head has lost its universe too."""
    path = tmp_path / "headless.jsonl"
    with RecordingSink(path) as sink:
        sink.quote(make_update())

    with pytest.raises(ValueError, match="does not start with"):
        open_recording(path)


def test_a_recording_from_another_schema_version_is_refused(tmp_path: Path) -> None:
    """The reason ``schema_version`` is on the first line at all.

    Reading a file of an unknown grammar as though it were this one is how a replay quietly
    reproduces something other than the session.
    """
    path = session(tmp_path)
    original = path.read_text(encoding="utf-8").splitlines()
    header = json.loads(original[0])
    header["schema_version"] = 99
    path.write_text("\n".join([json.dumps(header), *original[1:]]), encoding="utf-8")

    with pytest.raises(ValueError, match="schema version"):
        open_recording(path)


def test_a_header_with_no_quotes_is_refused(tmp_path: Path) -> None:
    """There is no instant to place the clock at, and no session to replay."""
    path = tmp_path / "header-only.jsonl"
    with RecordingSink(path) as sink:
        sink.header("BTC-DERIBIT", "BTC", ())

    with pytest.raises(ValueError, match="no quote"):
        open_recording(path)


def test_a_broken_quote_line_names_its_line_number(tmp_path: Path) -> None:
    """A recording is a long file; a message with no position sends a person reading it by hand."""
    path = session(tmp_path)
    original = path.read_text(encoding="utf-8").splitlines()
    broken = json.loads(original[1])
    broken["bid_size"] = "wide"
    path.write_text("\n".join([original[0], json.dumps(broken), *original[2:]]), encoding="utf-8")

    with pytest.raises(ValueError, match="line 2"):
        open_recording(path)


def test_a_naive_timestamp_is_refused(tmp_path: Path) -> None:
    """Aware or nothing: subtracting a naive instant from an aware one raises far from here."""
    path = session(tmp_path)
    original = path.read_text(encoding="utf-8").splitlines()
    quote = json.loads(original[1])
    quote["ts_local"] = NOW.replace(tzinfo=None).isoformat()
    path.write_text("\n".join([original[0], json.dumps(quote), *original[2:]]), encoding="utf-8")

    with pytest.raises(ValueError, match="timezone-aware"):
        open_recording(path)


# --- replaying


async def test_the_recorded_universe_is_what_discovery_answers(tmp_path: Path) -> None:
    """The venue is gone, so the answer has to come out of the file."""
    provider = RecordedProvider(open_recording(session(tmp_path)))

    assert await provider.discover() == (make_instrument(), make_instrument(expiry=FAR))


async def test_every_recorded_quote_is_replayed_in_order(tmp_path: Path) -> None:
    provider = RecordedProvider(open_recording(session(tmp_path)))

    replayed = await drain(provider.stream())

    assert [update.instrument.strike for update in replayed] == [60_000.0, 70_000.0, 80_000.0]


async def test_the_clock_is_moved_to_each_updates_local_stamp(tmp_path: Path) -> None:
    """The recording *is* the clock. This is the assertion ADR-004 stands on.

    ``ts_local`` and not ``ts_exchange``: the two differ by the transport latency the freshness
    policy exists to see, and the engine's clock during the original session was at ``ts_local``
    when the message landed.
    """
    clock = SpyClock()
    provider = RecordedProvider(open_recording(session(tmp_path)), clock)

    replayed = await drain(provider.stream())

    assert clock.instants == [update.observation.ts_local for update in replayed]


async def test_the_clock_is_moved_before_the_update_is_yielded(tmp_path: Path) -> None:
    """Order, not merely arrival: the message had already landed when the engine read the time.

    Moving the clock afterwards would apply every quote one instant in the chain's past, which is
    enough to make a replay's first snapshot disagree with the original run's.
    """
    clock = SpyClock()
    provider = RecordedProvider(open_recording(session(tmp_path)), clock)

    seen: list[int] = []
    async for _ in provider.stream():
        seen.append(len(clock.instants))

    assert seen == [1, 2, 3]


async def test_a_replay_without_a_clock_still_yields_the_quotes(tmp_path: Path) -> None:
    """The provider is usable without time: a test about messages should not need a clock."""
    provider = RecordedProvider(open_recording(session(tmp_path)))

    assert len(await drain(provider.stream())) == 3


async def test_closing_ends_the_stream(tmp_path: Path) -> None:
    """The port promises that closing ends the iterator, and a promise is worth its test."""
    provider = RecordedProvider(open_recording(session(tmp_path)))

    seen: list[QuoteUpdate] = []
    async for update in provider.stream():
        seen.append(update)
        await provider.close()

    assert len(seen) == 1


async def test_closing_twice_is_not_an_error(tmp_path: Path) -> None:
    provider = RecordedProvider(open_recording(session(tmp_path)))

    await provider.close()
    await provider.close()


async def test_streaming_twice_replays_the_same_session_from_the_top(tmp_path: Path) -> None:
    """Reproducibility is a property of the provider, not of the moment it was built.

    The same claim ``SyntheticProvider`` makes by re-seeding, and a caller comparing two runs needs
    it for the same reason.
    """
    provider = RecordedProvider(open_recording(session(tmp_path)))

    assert await drain(provider.stream()) == await drain(provider.stream())


async def test_a_quote_recorded_at_a_microsecond_replays_at_that_microsecond(
    tmp_path: Path,
) -> None:
    """The guard on the clock assertions: they are not comparing two rounded seconds.

    Every instant in this engine is subtracted from another one, so a replay that quantised its
    stamps would move the transport latency and the staleness by more than the numbers themselves.
    """
    stamped = datetime(2026, 8, 27, 12, 0, 0, 123_456, tzinfo=UTC)
    path = write(tmp_path / "micro.jsonl", make_update(ts_exchange=stamped))
    clock = SpyClock()

    await drain(RecordedProvider(open_recording(path), clock).stream())

    assert clock.instants == [stamped + timedelta(milliseconds=8)]
