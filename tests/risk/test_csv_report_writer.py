"""The second implementation of ``ReportWriter``, and the one whose output is read by a machine.

The console writer's tests deliberately do not pin its layout character by character. This one's
do the opposite, because here the layout *is* the contract: a column order, a header written once,
a row per position, and floats that survive the trip. What is asserted is what a reader of the
file gets -- parsed back with :mod:`csv`, never by string matching, so that a test cannot pass on
a file no parser would accept.
"""

from __future__ import annotations

import csv
from datetime import timedelta
from pathlib import Path

import pytest

from tests.risk.builders import NOW, make_position, make_position_risk, make_report
from volengine.risk.adapters.csv_report_writer import COLUMNS, CsvReportWriter
from volengine.risk.domain.freshness_policy import FreshnessDecision
from volengine.risk.domain.ports import ReportWriter
from volengine.risk.domain.pricing import OptionKindR
from volengine.risk.domain.risk_report import RiskReport


def rows_of(path: Path) -> list[dict[str, str]]:
    """Every row of the file, parsed by the same library that wrote it."""
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def written(path: Path, *reports: RiskReport) -> list[dict[str, str]]:
    """The rows a fresh writer produces for these reports, in order."""
    writer: ReportWriter = CsvReportWriter(path)
    for report in reports:
        writer.write(report)
    return rows_of(path)


def test_the_file_exists_with_its_header_before_any_report_arrives(tmp_path: Path) -> None:
    """Constructing the writer is where an unwritable path must fail: thirty seconds later, on
    the first report, the session's quotes are already gone."""
    path = tmp_path / "reports.csv"

    CsvReportWriter(path)

    assert path.read_text(encoding="utf-8") == ",".join(COLUMNS) + "\n"


def test_a_path_that_cannot_be_written_fails_at_construction(tmp_path: Path) -> None:
    """The other half of the same claim: the failure lands while the pipeline is still wiring."""
    missing = tmp_path / "no-such-directory" / "reports.csv"

    with pytest.raises(OSError):
        CsvReportWriter(missing)


def test_every_valued_position_becomes_a_row(tmp_path: Path) -> None:
    """Not netted and not aggregated: two lots are two rows, as they are two positions."""
    positions = (
        make_position_risk(position=make_position(strike=60_000.0)),
        make_position_risk(position=make_position(strike=70_000.0)),
    )

    rows = written(tmp_path / "reports.csv", make_report(positions=positions))

    assert [row["strike"] for row in rows] == ["60000.0", "70000.0"]


def test_the_report_fields_are_repeated_on_every_row(tmp_path: Path) -> None:
    """The denormalisation that makes one row readable on its own: a reader filtering on
    ``freshness`` or grouping on ``ts_report`` needs no second table to join against."""
    positions = (make_position_risk(), make_position_risk(value=-1_000.0))

    rows = written(tmp_path / "reports.csv", make_report(positions=positions))

    assert [row["market_id"] for row in rows] == ["BTC-DERIBIT", "BTC-DERIBIT"]
    assert [row["ts_report"] for row in rows] == [(NOW + timedelta(seconds=2)).isoformat()] * 2


def test_both_instants_are_written_as_iso_8601_with_their_offset(tmp_path: Path) -> None:
    """A fresh report of a stale market is a state only the two instants together make readable,
    and a stamp without an offset is a stamp nobody can compare across machines."""
    (row,) = written(tmp_path / "reports.csv", make_report())

    assert row["ts_snapshot"] == NOW.isoformat()
    assert row["ts_report"] == (NOW + timedelta(seconds=2)).isoformat()
    assert row["ts_snapshot"].endswith("+00:00")


def test_a_report_resting_on_no_surface_at_all_leaves_the_instant_empty(tmp_path: Path) -> None:
    """Empty rather than the word "none": a reader parsing the column as a date takes an empty
    cell as "no value" and a word as a broken column."""
    (row,) = written(
        tmp_path / "reports.csv",
        make_report(
            ts_snapshot=None,
            freshness=FreshnessDecision.REJECT,
            positions=(),
            message="no surface for BTC-DERIBIT yet",
        ),
    )

    assert row["ts_snapshot"] == ""


def test_a_rejected_report_is_still_a_row_carrying_its_reason(tmp_path: Path) -> None:
    """The loudest thing this context says (Design 7.2). A gap in the file for the interval an
    operator most needs to look at is indistinguishable from a process that died."""
    (row,) = written(
        tmp_path / "reports.csv",
        make_report(
            freshness=FreshnessDecision.REJECT,
            positions=(),
            message="no valid surface for BTC-DERIBIT: the last snapshot is 94 seconds old",
        ),
    )

    assert row["freshness"] == "REJECT"
    assert "94 seconds old" in row["message"]


def test_a_rejected_report_leaves_its_numeric_columns_empty(tmp_path: Path) -> None:
    """Empty and not zero. ``0.0`` is a legitimate delta, and a reader summing the column would
    otherwise fold a refusal into a number."""
    (row,) = written(
        tmp_path / "reports.csv",
        make_report(freshness=FreshnessDecision.REJECT, positions=(), message="no valid surface"),
    )

    assert [row[name] for name in ("strike", "vol", "value", "delta", "gamma", "vega")] == [""] * 6


def test_a_degraded_report_carries_both_its_numbers_and_its_warning(tmp_path: Path) -> None:
    """Degraded means "read these with suspicion", never "there are none" -- so the message has to
    reach the valued rows too, not only the rows that have nothing else on them."""
    (row,) = written(
        tmp_path / "reports.csv",
        make_report(freshness=FreshnessDecision.DEGRADED, message="42 seconds old"),
    )

    assert row["freshness"] == "DEGRADED"
    assert row["message"] == "42 seconds old"
    assert row["value"] != ""


def test_a_row_carries_the_volatility_it_was_valued_at(tmp_path: Path) -> None:
    """What makes the row auditable: with this number the value can be recomputed by hand."""
    (row,) = written(tmp_path / "reports.csv", make_report(positions=(make_position_risk(),)))

    assert float(row["vol"]) == 0.6375


def test_the_numbers_are_written_at_full_precision(tmp_path: Path) -> None:
    """The difference from the console writer, and the reason this file is worth having: an
    analysis needs the float that was actually priced with, not the four decimals a person reads.

    Asserted as exact equality after a round trip through ``float``, which is what "shortest form
    that round-trips" means and what any fixed number of decimals would break.
    """
    vol = 0.6375123456789
    gamma = 1.7654321e-08

    (row,) = written(
        tmp_path / "reports.csv",
        make_report(positions=(make_position_risk(vol=vol, gamma=gamma),)),
    )

    assert float(row["vol"]) == vol
    assert float(row["gamma"]) == gamma


def test_a_message_containing_a_comma_survives_the_round_trip(tmp_path: Path) -> None:
    """Quoting is :mod:`csv`'s job, which is the reason this module never joins its own lines.
    A message is free text written by a use case, and the failure mode of building the line by
    hand is a report that silently shifts every column after it."""
    message = 'no valid surface, "BTC-DERIBIT" is 94s old'

    (row,) = written(
        tmp_path / "reports.csv",
        make_report(freshness=FreshnessDecision.REJECT, positions=(), message=message),
    )

    assert row["message"] == message


def test_the_option_kind_is_written_as_its_wire_value(tmp_path: Path) -> None:
    """A ``StrEnum`` member's explicit value is the string that goes on the wire, and the file is
    a wire. Writing the member would tie the column to how ``str()`` happens to render it."""
    position = make_position(kind=OptionKindR.PUT, quantity=-4.0)

    (row,) = written(
        tmp_path / "reports.csv",
        make_report(positions=(make_position_risk(position=position),)),
    )

    assert row["kind"] == OptionKindR.PUT.value


def test_the_header_is_written_once_however_many_reports_arrive(tmp_path: Path) -> None:
    """A repeated header mid-file is not a CSV any reader parses."""
    path = tmp_path / "reports.csv"

    written(path, make_report(), make_report(), make_report())

    lines = path.read_text(encoding="utf-8").splitlines()
    assert lines.count(",".join(COLUMNS)) == 1
    assert len(lines) == 4


def test_a_second_writer_appends_rather_than_destroying_the_first_session(tmp_path: Path) -> None:
    """Truncation is destructive, and only the caller knows whether the operator asked for it.
    A restarted engine pointed at the same file continues it, header and all."""
    path = tmp_path / "reports.csv"
    written(path, make_report())

    CsvReportWriter(path).write(make_report(market_id="ETH-DERIBIT"))

    assert [row["market_id"] for row in rows_of(path)] == ["BTC-DERIBIT", "ETH-DERIBIT"]


def test_the_file_is_complete_after_every_report(tmp_path: Path) -> None:
    """No handle is held between calls, so a reader that catches the file mid-run sees whole rows
    -- not a report buffered inside a writer nobody can flush, since the port has no ``close``."""
    path = tmp_path / "reports.csv"
    writer = CsvReportWriter(path)

    writer.write(make_report())
    after_one = rows_of(path)
    writer.write(make_report())

    assert len(after_one) == 1
    assert len(rows_of(path)) == 2


def test_the_columns_are_exactly_the_declared_header(tmp_path: Path) -> None:
    """The file's shape is decided in one place. A row built with a key that is not a column
    raises rather than shifting the ones after it."""
    (row,) = written(tmp_path / "reports.csv", make_report())

    assert tuple(row) == COLUMNS


def test_the_lines_end_with_a_bare_newline(tmp_path: Path) -> None:
    """Set against :mod:`csv`'s ``\\r\\n`` default: a line ending that varies with the dialect is
    one more difference two supposedly identical runs can show."""
    path = tmp_path / "reports.csv"

    written(path, make_report())

    assert "\r" not in path.read_bytes().decode("utf-8")


def test_two_runs_of_the_same_reports_produce_identical_bytes(tmp_path: Path) -> None:
    """The determinism claim of Design 8.2, at the only place it is observable: nothing in this
    writer reads a clock, a locale or a set's iteration order."""
    first = tmp_path / "a.csv"
    second = tmp_path / "b.csv"
    reports = (
        make_report(),
        make_report(freshness=FreshnessDecision.REJECT, positions=(), message="gone"),
    )

    written(first, *reports)
    written(second, *reports)

    assert first.read_bytes() == second.read_bytes()
