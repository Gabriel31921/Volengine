"""The one implementation of ``ReportWriter``, and the end of the whole pipeline.

What is asserted here is what a person would see. The formatting itself is deliberately not
pinned character by character -- a test that did that would break on every improvement to the
layout and would be measuring nothing -- but four things are: that every number a reader needs is
present, that a rejected report is printed rather than swallowed, that the stream is the one it
was given, and that nothing here decides *whether* to write.
"""

from __future__ import annotations

import io
import sys
from datetime import timedelta

import pytest

from tests.risk.builders import NOW, make_position, make_position_risk, make_report
from volengine.risk.adapters.console_report_writer import ConsoleReportWriter
from volengine.risk.domain.freshness_policy import FreshnessDecision
from volengine.risk.domain.ports import ReportWriter
from volengine.risk.domain.risk_report import RiskReport


def written(report: RiskReport) -> str:
    """The exact text one report produces, captured off a stream of our own."""
    stream = io.StringIO()
    writer: ReportWriter = ConsoleReportWriter(stream)
    writer.write(report)
    return stream.getvalue()


def test_a_healthy_report_names_the_market_the_producer_and_the_verdict() -> None:
    """Which numbers these are and where they came from, on the first line."""
    text = written(make_report())

    assert "BTC-DERIBIT" in text
    assert "svi-scipy" in text
    assert "NORMAL" in text


def test_both_instants_are_printed() -> None:
    """A fresh report of a stale market is a state the two timestamps together make readable, and
    either one alone hides."""
    text = written(make_report())

    assert NOW.isoformat() in text
    assert (NOW + timedelta(seconds=2)).isoformat() in text


def test_a_report_resting_on_no_surface_at_all_says_so() -> None:
    """``ts_snapshot`` is ``None`` before the first calibration lands, and inventing an instant
    there would print the start of the session as if it were market data."""
    text = written(
        make_report(
            ts_snapshot=None,
            freshness=FreshnessDecision.REJECT,
            positions=(),
            message="no surface for BTC-DERIBIT yet",
        )
    )

    assert "none" in text


def test_every_valued_position_gets_a_line_of_its_own() -> None:
    """Not netted and not aggregated: two lots are two lines, as they are two positions."""
    positions = (
        make_position_risk(position=make_position(strike=60_000.0)),
        make_position_risk(position=make_position(strike=70_000.0), value=1_000.0),
    )

    lines = written(make_report(positions=positions)).splitlines()

    assert len(lines) == 1 + len(positions) + 1  # header, one per position, total
    assert "60,000.00" in lines[1]
    assert "70,000.00" in lines[2]


def test_a_line_carries_the_volatility_it_was_valued_at() -> None:
    """What makes the report auditable: with this number the value can be recomputed by hand."""
    text = written(make_report(positions=(make_position_risk(vol=0.6375),)))

    assert "0.6375" in text


def test_the_total_is_printed_and_is_the_sum_of_the_lines() -> None:
    positions = (
        make_position_risk(value=61_000.0),
        make_position_risk(value=-1_000.0),
    )

    text = written(make_report(positions=positions))

    assert "total" in text
    assert "60,000.00" in text.splitlines()[-1]


def test_a_rejected_report_is_written_with_its_reason_and_no_numbers() -> None:
    """The loudest thing this context says (Design 7.2). A writer that skipped it would turn
    "there is no valid surface" into silence, which is what a dead process also looks like."""
    text = written(
        make_report(
            freshness=FreshnessDecision.REJECT,
            positions=(),
            message="no valid surface for BTC-DERIBIT: the last snapshot is 94 seconds old",
        )
    )

    assert "REJECT" in text
    assert "94 seconds old" in text
    assert "total" not in text


def test_a_degraded_report_still_carries_its_numbers() -> None:
    """Degraded means "read these with suspicion", never "there are none"."""
    text = written(make_report(freshness=FreshnessDecision.DEGRADED, message="42 seconds old"))

    assert "DEGRADED" in text
    assert "42 seconds old" in text
    assert "total" in text


def test_one_report_is_one_write() -> None:
    """Two producers reporting on one market must not interleave halfway through a block."""

    class CountingStream(io.StringIO):
        def __init__(self) -> None:
            super().__init__()
            self.writes = 0

        def write(self, text: str) -> int:
            self.writes += 1
            return super().write(text)

    stream = CountingStream()
    ConsoleReportWriter(stream).write(make_report())

    assert stream.writes == 1


def test_the_output_ends_with_a_newline() -> None:
    """So that a run of reports reads as a list rather than as one long line."""
    assert written(make_report()).endswith("\n")


def test_the_stream_is_resolved_at_write_time(monkeypatch: pytest.MonkeyPatch) -> None:
    """The composition root builds this writer once, long before anything redirects stdout.

    The replacement is installed *after* the writer exists, which is the whole assertion: a
    writer that had bound ``sys.stdout`` in its constructor would write past it -- and the same
    bug hides every report inside anything that redirects a stream around a call.
    """
    writer = ConsoleReportWriter()
    replacement = io.StringIO()
    monkeypatch.setattr(sys, "stdout", replacement)

    writer.write(make_report())

    assert "BTC-DERIBIT" in replacement.getvalue()
