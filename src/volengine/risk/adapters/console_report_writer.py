"""A finished report, rendered for a person watching a terminal.

The first implementation of ``ReportWriter``, and the one the walking skeleton ends at: without
it a run crosses all four contexts and leaves no trace anyone can see. Everything the port keeps
out of the domain lives here and only here -- column widths, decimal places, the order of the
fields, what a rejection looks like -- so that the CSV writer of F3-E is a sibling of this module
rather than an edit to it.

**A rejected report is printed like any other.** ``FreshnessDecision.REJECT`` carries no valued
lines and a message saying why, and that message is the loudest thing this context ever says
(Design 7.2). A writer that skipped it would turn "there is no valid surface" into silence, which
is the one state an operator cannot tell apart from a dead process.

**The stream is resolved at write time, not at construction.** Binding ``sys.stdout`` in
``__init__`` captures whatever object happened to be installed when the pipeline was composed,
which is a different one under ``pytest``'s capture, under a redirected shell and inside anything
that swaps the stream around a call. Reading it per write costs an attribute lookup and makes the
writer behave the way every other console-writing tool does.
"""

from __future__ import annotations

import sys
from typing import TextIO

from volengine.risk.domain.risk_report import PositionRisk, RiskReport

INDENT = "  "
"""What sets a report's lines apart from its header. Two spaces, so a run of reports reads as a
list rather than as one paragraph."""


class ConsoleReportWriter:
    """A ``ReportWriter`` that prints one block of text per report.

    Structural conformance, like every adapter here: one method, no base class, and no ``close``
    -- the port deliberately has none, and a stream this object did not open is not a stream it
    may close.
    """

    def __init__(self, stream: TextIO | None = None) -> None:
        """Args:
        stream: Where the text goes, or ``None`` for whatever ``sys.stdout`` is at the moment
            of each write. A test passes a ``StringIO`` and asserts on the exact characters;
            the composition root passes nothing.
        """
        self._stream = stream

    def write(self, report: RiskReport) -> None:
        """Render one report and flush it.

        Flushed because the alternative is invisible. ``sys.stdout`` is block-buffered whenever it
        is not a terminal -- a pipe, a log file, a container's captured output -- so an engine
        reporting once a second would appear to produce nothing for minutes at a time, which is
        indistinguishable from the failure this whole context exists to report.

        One ``write`` of one string rather than a line at a time, so that two producers reporting
        on the same market cannot interleave halfway through a block.
        """
        stream = sys.stdout if self._stream is None else self._stream
        lines = [_header(report)]
        if report.message is not None:
            lines.append(f"{INDENT}{report.message}")
        lines.extend(_position_line(one) for one in report.positions)
        if report.positions:
            lines.append(f"{INDENT}total {report.total_value:>18,.2f}")
        stream.write("\n".join(lines) + "\n")
        stream.flush()


def _header(report: RiskReport) -> str:
    """Which market, which producer, how fresh, and the two instants the verdict rests on.

    Both timestamps, never one. A report produced a moment ago from data that is ten minutes old
    is a fresh report of a stale market, and printing only one of the two would make that state
    unreadable -- it is the gap between them that the freshness decision beside them was computed
    from.
    """
    snapshot = "none" if report.ts_snapshot is None else report.ts_snapshot.isoformat()
    return (
        f"[{report.market_id} / {report.producer_id}] {report.freshness.value} "
        f"snapshot={snapshot} report={report.ts_report.isoformat()}"
    )


def _position_line(risk: PositionRisk) -> str:
    """One valued position: what it is, what it was valued at, and what came out.

    The volatility is printed next to the value because it is what makes the line auditable --
    given the strike, the expiry and this one number, the value can be recomputed by hand.

    Three different precisions, one per magnitude rather than one for the table. A value in the
    thousands and a gamma around ``1e-8`` cannot share a format: two decimals would print the
    gamma as ``0.00`` and eight would drown the value. ``g`` is used for the two greeks whose
    scale depends entirely on the position's size.
    """
    position = risk.position
    return (
        f"{INDENT}{position.underlying} {position.expiry:%Y-%m-%d} "
        f"{position.kind.value:<4} K={position.strike:>12,.2f} qty={position.quantity:>10,.4f} "
        f"vol={risk.vol:>7.4f} value={risk.value:>16,.2f} "
        f"delta={risk.delta:>12.6g} gamma={risk.gamma:>12.6g} vega={risk.vega:>14,.2f}"
    )
