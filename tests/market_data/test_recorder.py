"""The tap: what a recorded session looks like on disk, and what survives the round trip.

Two claims are being made here and they are separable, so they are tested separately. One, the
sink writes a file of the shape :mod:`recorded` promises to read -- a header, then one line per
update, JSON, primitives only. Two, the wrapper is invisible: a provider seen through
``RecordingProvider`` discovers, streams and closes exactly as it would have on its own.

The round trip is asserted through a real ``json`` hop and on **equality of the domain objects**,
not field by field. ``QuoteUpdate`` is a frozen dataclass all the way down, so ``==`` compares
every value including the two timestamps -- and a float that did not survive the encoding, or a
timezone dropped on the way back in, fails it. Field-by-field assertions would pass the day a new
field was added and forgotten by the encoder.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from datetime import timedelta
from pathlib import Path

import pytest

from tests.market_data.builders import FAR, NEAR, NOW, StubProvider, make_instrument, make_update
from volengine.market_data.adapters.recorded import RecordedProvider, open_recording
from volengine.market_data.adapters.recorder import (
    RECORDING_SCHEMA_VERSION,
    RecordingProvider,
    RecordingSink,
    update_from_dict,
)
from volengine.market_data.domain.option_quote import OptionKindD, QuoteUpdate


def lines(path: Path) -> list[dict[str, object]]:
    """Every line of the recording, decoded. The format is JSON or the test is meaningless."""
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


async def drain(updates: AsyncIterator[QuoteUpdate]) -> list[QuoteUpdate]:
    return [update async for update in updates]


def recorded(path: Path) -> RecordingProvider:
    """A two-quote session through the tap, on the market the builders describe."""
    inner = StubProvider(
        updates=(make_update(), make_update(strike=70_000.0, kind=OptionKindD.PUT)),
        instruments=(make_instrument(), make_instrument(strike=70_000.0, expiry=FAR)),
    )
    return RecordingProvider(
        inner=inner, sink=RecordingSink(path), market_id="BTC-DERIBIT", underlying="BTC"
    )


# --- the file


def test_the_first_line_is_a_header_naming_the_schema_and_the_market(tmp_path: Path) -> None:
    """A recording says what it is before it says anything else.

    Without this a reader has to guess the grammar from the data, which is the one thing a
    versioned format exists to stop.
    """
    path = tmp_path / "session.jsonl"
    with RecordingSink(path) as sink:
        sink.header("BTC-DERIBIT", "BTC", (make_instrument(),))

    header = lines(path)[0]
    assert header["kind"] == "header"
    assert header["schema_version"] == RECORDING_SCHEMA_VERSION
    assert header["market_id"] == "BTC-DERIBIT"


def test_the_header_carries_the_universe_discovery_returned(tmp_path: Path) -> None:
    """The set ``discover`` answered with, because a replay cannot ask the venue again.

    A replayed chain that learned its instruments from the quotes would start every session with a
    coverage ratio climbing from zero, which is a difference between a run and its replay that has
    nothing to do with the engine.
    """
    path = tmp_path / "session.jsonl"
    with RecordingSink(path) as sink:
        sink.header("BTC-DERIBIT", "BTC", (make_instrument(), make_instrument(expiry=FAR)))

    listed = lines(path)[0]["instruments"]
    assert isinstance(listed, list)
    assert len(listed) == 2


def test_every_update_is_one_line(tmp_path: Path) -> None:
    """One line per update, so the file is streamable and a killed session is still readable."""
    path = tmp_path / "session.jsonl"
    with RecordingSink(path) as sink:
        sink.header("BTC-DERIBIT", "BTC", ())
        sink.quote(make_update())
        sink.quote(make_update(strike=70_000.0))

    assert len(lines(path)) == 3
    assert [one["kind"] for one in lines(path)[1:]] == ["quote", "quote"]


def test_the_line_holds_primitives_and_no_python_objects(tmp_path: Path) -> None:
    """The instants go out as strings and the option kind as its own value.

    A ``StrEnum`` *is* a ``str``, so ``json.dumps`` would have written it either way -- the
    assertion is that the value written is the member's declared string, which is the wire format
    and must not change when a member is renamed.
    """
    path = tmp_path / "session.jsonl"
    with RecordingSink(path) as sink:
        sink.header("BTC-DERIBIT", "BTC", ())
        sink.quote(make_update(kind=OptionKindD.PUT))

    quote = lines(path)[1]
    instrument = quote["instrument"]
    assert isinstance(instrument, dict)
    assert instrument["kind"] == "PUT"
    assert instrument["expiry"] == NEAR.isoformat()
    assert quote["ts_exchange"] == NOW.isoformat()


def test_an_absent_side_is_recorded_as_null_and_not_as_zero(tmp_path: Path) -> None:
    """``None`` and ``0.0`` are different states and the format must keep them apart.

    An empty side is the absence of information; a zero bid is information. Collapsing them is the
    mistake the whole quote model exists to prevent, and a serialisation is where it happens.
    """
    path = tmp_path / "session.jsonl"
    with RecordingSink(path) as sink:
        sink.header("BTC-DERIBIT", "BTC", ())
        sink.quote(make_update(bid=None))
        sink.quote(make_update(bid=0.0))

    assert lines(path)[1]["bid"] is None
    assert lines(path)[2]["bid"] == 0.0


def test_a_recorded_update_survives_the_json_hop_unchanged(tmp_path: Path) -> None:
    """The headline of the format: what goes in comes out equal, timestamps included."""
    path = tmp_path / "session.jsonl"
    update = make_update(bid=0.05123456789012345, ts_exchange=NOW + timedelta(microseconds=1))
    with RecordingSink(path) as sink:
        sink.header("BTC-DERIBIT", "BTC", ())
        sink.quote(update)

    assert update_from_dict(dict(lines(path)[1])) == update


def test_a_recording_is_truncated_rather_than_appended_to(tmp_path: Path) -> None:
    """A report is a log and a recording is a session.

    Two sessions concatenated would put a header in the middle of a stream and send the instants
    backwards -- a file no replay can honestly consume.
    """
    path = tmp_path / "session.jsonl"
    with RecordingSink(path) as first:
        first.header("BTC-DERIBIT", "BTC", ())
        first.quote(make_update())
    with RecordingSink(path) as second:
        second.header("BTC-DERIBIT", "BTC", ())

    assert len(lines(path)) == 1


def test_closing_twice_is_not_an_error(tmp_path: Path) -> None:
    """Shutdown paths overlap, and a close that raises is a close that leaks."""
    sink = RecordingSink(tmp_path / "session.jsonl")

    sink.close()
    sink.close()


def test_writing_after_close_is_refused(tmp_path: Path) -> None:
    """The one state where silence would be worse than an error: a quote nobody recorded."""
    sink = RecordingSink(tmp_path / "session.jsonl")
    sink.close()

    with pytest.raises(ValueError, match="closed"):
        sink.quote(make_update())


def test_a_blank_market_id_is_refused(tmp_path: Path) -> None:
    """An unnamed recording cannot be matched to a configuration on the way back in."""
    with (
        RecordingSink(tmp_path / "session.jsonl") as sink,
        pytest.raises(ValueError, match="blank"),
    ):
        sink.header("   ", "BTC", ())


def test_the_counter_reports_what_was_recorded(tmp_path: Path) -> None:
    """The header is not a quote, and the count is what a caller reports at the end of a session."""
    with RecordingSink(tmp_path / "session.jsonl") as sink:
        sink.header("BTC-DERIBIT", "BTC", ())
        sink.quote(make_update())

        assert sink.written == 1


# --- the wrapper


async def test_the_wrapper_yields_exactly_what_the_inner_provider_yields(tmp_path: Path) -> None:
    """A tap is invisible to the consumer, or the recorded session is not the session that ran."""
    path = tmp_path / "session.jsonl"
    provider = recorded(path)

    await provider.discover()
    updates = await drain(provider.stream())
    await provider.close()

    assert updates == [make_update(), make_update(strike=70_000.0, kind=OptionKindD.PUT)]


async def test_the_wrapper_writes_the_header_when_discovery_is_asked_for(tmp_path: Path) -> None:
    """The header lands on the first line because discovery is the first call ingestion makes.

    That ordering is what the sink leans on instead of sequencing anything itself, so it is
    asserted here rather than assumed.
    """
    path = tmp_path / "session.jsonl"
    provider = recorded(path)

    await provider.discover()
    await provider.close()

    assert lines(path)[0]["kind"] == "header"


async def test_the_recorded_file_replays_into_the_same_updates(tmp_path: Path) -> None:
    """The two halves meet: what the tap wrote is what the replay hands back.

    The one test that would fail if the encoder and the decoder drifted apart, which is the failure
    mode a format split across two modules actually has.
    """
    path = tmp_path / "session.jsonl"
    provider = recorded(path)
    await provider.discover()
    original = await drain(provider.stream())
    await provider.close()

    replayed = await drain(RecordedProvider(open_recording(path)).stream())

    assert replayed == original


async def test_closing_the_wrapper_closes_the_inner_provider(tmp_path: Path) -> None:
    """The wrapper owns the lifetime it was handed; a socket left open is a leak per session."""
    inner = StubProvider(updates=(make_update(),))
    provider = RecordingProvider(
        inner=inner,
        sink=RecordingSink(tmp_path / "session.jsonl"),
        market_id="BTC-DERIBIT",
        underlying="BTC",
    )

    await provider.close()

    assert inner.closed
