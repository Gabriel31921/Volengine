"""The same finished report, written for a machine instead of for a person.

The second implementation of ``ReportWriter`` and the sibling the console writer's docstring
promised: everything that differs between the two is presentation, and none of it reached the
domain. A run pointed at this adapter produces a file that opens in a spreadsheet, loads with
``pandas.read_csv`` and diffs byte for byte against another run -- which is what makes the
determinism claim of Design 8.2 checkable rather than asserted.

**One row per valued position, never one row per report.** A report is a header and a list of
lines; flattening it into a single row would need either a column per position (a schema that
changes with the book) or a nested cell (a CSV that is not tabular). The report-level fields --
the two instants, the market, the producer, the verdict, the message -- are repeated on each of
its rows. That denormalisation is deliberate: it is what makes any single row self-describing and
lets a reader filter on ``freshness`` or group on ``ts_report`` without carrying a second table.

**A report with no positions still gets a row.** ``FreshnessDecision.REJECT`` carries no lines and
a message saying why, and it is the loudest thing this context ever says (Design 7.2). Skipping it
would leave the file silent for exactly the interval an operator most needs to see, and a gap in a
file is indistinguishable from a process that died. Its position and greek columns are empty --
empty, not zero, because ``0.0`` is a legitimate delta and a reader summing the column would fold
a refusal into a number.

**There is no total row.** The console writer prints one because a person reads down a block and
wants the sum; a file does not, and a totals row inside the data is the classic way to make every
subsequent ``sum()`` double count. ``RiskReport.total_value`` is the sum of this file's ``value``
column grouped by report, and it stays a derived quantity in both places.

**Numbers are written at full precision**, through ``repr``'s shortest round-tripping form rather
than the console's fixed decimals. A person needs a gamma of ``1.7e-08`` rounded to something
readable; an analysis needs the float that was actually priced with, and a file that had lost four
digits could not be compared against the run that produced it.

**Not registered in ``default_adapters()`` yet**, and the seam is in ``docs/SEAMS.md``:
``RiskConfig`` names its writer but carries no path, and a writer factory receives the risk
configuration and nothing else. F2-07 owns the TOML, the same way it owns ``FitSettings`` and
``SyntheticConfig``; a field added here would be configuration nothing reads, which is what
ADR-012 asks configuration not to be. Until then this adapter is constructed by tests only.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Final, TextIO

from volengine.risk.domain.risk_report import PositionRisk, RiskReport

COLUMNS: Final[tuple[str, ...]] = (
    "ts_report",
    "ts_snapshot",
    "market_id",
    "producer_id",
    "freshness",
    "underlying",
    "expiry",
    "kind",
    "strike",
    "quantity",
    "vol",
    "value",
    "delta",
    "gamma",
    "vega",
    "message",
)
"""The header, and the only place the field order is decided.

Report-level fields first, then the position, then what came out of valuing it, and the free text
last -- so that the numeric block is contiguous and a reader loading a subset of columns names a
range rather than a scatter. The order is part of the file format: appending a column is
compatible with anything reading by name, reordering these is not.
"""

LINE_TERMINATOR: Final = "\n"
"""Set explicitly because :mod:`csv` defaults to ``\\r\\n``.

RFC 4180 asks for CRLF and every reader in practice accepts either, while a file whose line
endings depend on the writer's dialect is one more difference two supposedly identical runs can
show. One byte per line, the same on every platform.
"""

EMPTY: Final = ""
"""What a column a given row does not fill in is written as.

A rejected report simply leaves its ten position columns out of the mapping and
:class:`csv.DictWriter` fills them with this. Empty and not zero: ``0.0`` is a legitimate delta,
and a reader summing the column would otherwise fold a refusal into a number.
"""


class CsvReportWriter:
    """A ``ReportWriter`` that appends each report to a file as one row per position.

    Structural conformance like every adapter here: one method, no base class, and no ``close``.
    The port has none, so this writer owns no open handle between calls -- it opens, appends,
    and closes on each report. At a report per second that is a syscall nobody will measure, and
    it buys three things a long-lived handle does not: the file is complete and readable while the
    engine is still running, an external rotation or truncation cannot leave the writer scribbling
    into a deleted inode, and there is no resource whose release depends on a method the port
    never declared.
    """

    def __init__(self, path: Path) -> None:
        """Create the file if it is not there yet, and write the header if it is empty.

        The header is written **here, not on the first report**, so that a path that cannot be
        written to fails while the composition root is still wiring -- not thirty seconds into a
        run, on the first report, with the session's quotes already gone. It is the same
        fail-fast argument the configuration loader makes about a bad threshold.

        Appended to, never truncated. A second run against the same path continues the file
        rather than destroying the evidence of the first, and the header is skipped because the
        file is no longer empty -- a repeated header mid-file is not a CSV any reader parses. A
        deployment that wants a fresh file each time deletes it first: truncation is a
        destructive act, and the caller is the only one that knows whether the operator asked for
        one.

        Args:
            path: Where the reports go. Its parent directory must already exist -- creating a
                path silently is how a typo becomes a file nobody looks in again.

        Raises:
            OSError: If the file cannot be created or written to.
        """
        self._path = path
        with path.open("a", encoding="utf-8", newline="") as handle:
            if handle.tell() == 0:
                _writer(handle).writeheader()

    def write(self, report: RiskReport) -> None:
        """Append one report: one row per valued position, or one row saying there are none.

        Every row of a report is written inside a single ``open`` and the file is closed on the
        way out, so a reader that catches the file between two reports sees whole rows and never
        half a report.
        """
        with self._path.open("a", encoding="utf-8", newline="") as handle:
            _writer(handle).writerows(_rows(report))


def _writer(handle: TextIO) -> csv.DictWriter[str]:
    """One place where the file's shape is decided, for the header and for every row alike.

    A mapping per row rather than a tuple, so :data:`COLUMNS` is the single source of the field
    order: a positional row would have to be padded by hand wherever a report has no positions,
    and a padding count that drifted from the header would shift every later column by one
    without failing anything. Here a key that does not name a column raises instead.
    """
    return csv.DictWriter(handle, fieldnames=COLUMNS, restval=EMPTY, lineterminator=LINE_TERMINATOR)


def _rows(report: RiskReport) -> list[dict[str, str]]:
    """The rows one report becomes: one per position, or exactly one when it carries none."""
    head = _report_fields(report)
    if not report.positions:
        return [head]
    return [head | _position_fields(one) for one in report.positions]


def _report_fields(report: RiskReport) -> dict[str, str]:
    """The six fields every row of a report repeats.

    Both instants, as ISO 8601 with the offset the domain guarantees is there: the report's own
    stamp says when this was computed, the snapshot's says what it was computed from, and it is
    the gap between them that the ``freshness`` verdict beside them was derived from. A missing
    snapshot is an empty cell, not the string ``"none"`` -- a reader parsing the column as a date
    takes the empty cell as "no value" and the word as a broken column.

    The message travels here rather than with the position fields because it belongs to the
    report, not to a line of it: a degraded report carries both its numbers and the reason to read
    them with suspicion, and attaching the message to the valued path only would make the warning
    visible exactly when there was nothing left to warn about.
    """
    return {
        "ts_report": report.ts_report.isoformat(),
        "ts_snapshot": EMPTY if report.ts_snapshot is None else report.ts_snapshot.isoformat(),
        "market_id": report.market_id,
        "producer_id": report.producer_id,
        "freshness": report.freshness.value,
        "message": EMPTY if report.message is None else report.message,
    }


def _position_fields(risk: PositionRisk) -> dict[str, str]:
    """What the position is, and what valuing it produced.

    ``vol`` sits between the two halves on purpose: it is the number that makes the row auditable,
    since the value can be recomputed from the strike, the expiry and it alone. ``.value`` on the
    enum rather than the member, because a ``StrEnum`` written straight out would depend on
    ``str()`` returning the wire string and that is exactly the coupling the explicit member
    values exist to avoid.
    """
    position = risk.position
    return {
        "underlying": position.underlying,
        "expiry": position.expiry.isoformat(),
        "kind": position.kind.value,
        "strike": repr(position.strike),
        "quantity": repr(position.quantity),
        "vol": repr(risk.vol),
        "value": repr(risk.value),
        "delta": repr(risk.delta),
        "gamma": repr(risk.gamma),
        "vega": repr(risk.vega),
    }
