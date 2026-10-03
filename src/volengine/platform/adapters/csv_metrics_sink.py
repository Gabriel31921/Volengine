"""Every measurement the engine takes, one row each, in a file a chart can be drawn from.

Design 8.3 asks for a ``MetricsSink`` adapter to "structured log or CSV -- no Prometheus", and
names the measurements as "the direct basis of the charts for write-ups". ``LoggingMetricsSink``
is the structured log; this is the CSV. The difference that matters is not the format but the
reader: a log line is for a person scrolling, a row is for ``pandas.read_csv`` grouping by
``name`` and plotting ``value`` against ``ts``.

**Where each metric of Design 8.3 comes from**, because none of them is computed here -- a sink
records what it is told and has no opinion -- and the list is what a chart-maker needs first:

* *ws -> snapshot latency*: ``marketdata.clock_skew_seconds``, the freshest venue stamp minus the
  instant the snapshot was frozen. Signed, and its negation is the latency; it was already emitted
  on every snapshot and a second name for the same subtraction would be a second series to keep in
  agreement with the first.
* *snapshot -> surface latency, per producer*: ``risk.surface.snapshot_to_surface_ms``, measured
  where every producer's surface arrives (``risk/application/surface_cache.py``), tagged by
  producer. ``pricing.cycle_ms`` beside it is the fit alone, so the difference between the two is
  the time a snapshot waited before a calibrator took it.
* *RMSE per calibration*: ``pricing.rmse_vol_bp`` and ``neural.rmse_vol_bp``.
* *Failure rate*: ``risk.surface.received`` with its ``status`` tag -- the share of
  ``STALE_REPUBLISH`` is the failure rate as Risk lives it -- beside the producers' own
  ``pricing.calibration.refused`` and ``neural.publication.refused``.
* *Conflation lag*: ``risk.surface.delivery_lag_ms``, from publication to the consumer taking the
  surface, with ``bus.dropped`` counting what conflation threw away on the way.
* *Staleness observed by Risk*: ``risk.surface_age_seconds``.
* *Distance between producers*: ``risk.comparison.distance_rms_vol_bp`` and its ``_max_``
  sibling, from the comparative report (``risk/application/compare_producers.py``).

**The timestamp comes from the engine's clock, not from the wall.** Under a ``SimulatedClock`` a
replay relives the recorded instants (ADR-004), and a metrics file stamped by ``datetime.now``
would plot an hour of recorded market against the four seconds it took to replay it. With the
injected clock the two files of two replays differ only where the measurements themselves do.

**Tags are one JSON object per row, keys sorted.** The tag set varies by metric -- ``market`` and
``producer`` on one, ``topic`` and ``subscriber`` on another -- so a column per tag would be a
schema that grows with the code, and a ``key=value;key=value`` cell is a grammar of its own that a
subscriber label containing either delimiter would break. JSON is unambiguous, quoted by
:mod:`csv` like any other cell, and one ``json.loads`` away from a dict.

**Not selectable from a configuration file yet.** ``--metrics`` still chooses between the logging
and the null sink; picking this one needs a path in the TOML and a composition root that owns
:meth:`CsvMetricsSink.close`. ``docs/SEAMS.md`` (Entrypoints) records it.
"""

from __future__ import annotations

import csv
import json
import threading
from pathlib import Path
from types import TracebackType
from typing import Final, Self

from volengine.platform.clock import Clock

COLUMNS: Final[tuple[str, ...]] = ("ts", "kind", "name", "value", "tags")
"""The header. Appending a column is compatible with anything reading by name; reordering is not."""

LINE_TERMINATOR: Final = "\n"
"""Set explicitly because :mod:`csv` defaults to ``\\r\\n``; the same choice, and the same reason,
as ``risk/adapters/csv_report_writer.py``: one byte per line on every platform, so two runs that
should agree cannot differ in their line endings."""


class CsvMetricsSink:
    """A ``MetricsSink`` appending one row per measurement to a file it holds open.

    Satisfies every context's ``MetricsSink`` structurally, like ``LoggingMetricsSink``.

    **It holds the file open, unlike ``CsvReportWriter``**, and the difference is volume. A report
    arrives about once a second and that writer can afford to open and close on each one; a
    metric is emitted several times per *event* -- every publish, every handled message, every
    snapshot -- and a syscall pair per row would be the most expensive thing in the ingestion
    loop. The cost is a :meth:`close` the port does not declare, which is why this class is also a
    context manager: the composition root that opens it is the one that knows when the run ends.

    **Rows are buffered, not flushed one by one.** A run killed by a signal loses the tail of its
    metrics, which is the opposite of the choice ``RecordingSink`` makes for a recording. The
    asymmetry is what each file is for: a recording that ends early is a replay that cannot reach
    the moment something went wrong, while a metrics file that ends early is a chart missing its
    last second. :meth:`flush` exists for a caller that wants a consistent file mid-run.

    **Thread-safe, which the bus and the cache are not.** The calibrations run on per-producer
    pools (ADR-005) and emit their RMSE and timings from the worker thread, while the event loop
    emits everything else; two threads interleaving inside one ``writerow`` would tear a row in
    half. A lock per row is cheap next to the fit that produced the number.
    """

    def __init__(self, path: Path, clock: Clock) -> None:
        """Open the file for appending and write the header if it is empty.

        Appended to, never truncated, for the reason ``CsvReportWriter`` gives: a second run
        continues the file rather than destroying the evidence of the first, and an operator who
        wants a fresh file deletes the old one -- truncation is a destructive act the caller has to
        ask for. The header is written here, so a path that cannot be written fails while the
        composition root is still wiring.

        Args:
            path: Where the rows go. The parent directory must exist.
            clock: What stamps each row. The engine's own, so a replay is stamped with recorded
                time (ADR-004).

        Raises:
            OSError: If the file cannot be opened for writing.
        """
        self._clock = clock
        self._lock = threading.Lock()
        self._handle = path.open("a", encoding="utf-8", newline="")
        self._writer = csv.writer(self._handle, lineterminator=LINE_TERMINATOR)
        if self._handle.tell() == 0:
            self._writer.writerow(COLUMNS)

    def gauge(self, name: str, value: float, **tags: str) -> None:
        self._emit("gauge", name, repr(float(value)), tags)

    def counter(self, name: str, value: int = 1, **tags: str) -> None:
        self._emit("counter", name, str(int(value)), tags)

    def timing(self, name: str, ms: float, **tags: str) -> None:
        self._emit("timing", name, repr(float(ms)), tags)

    def flush(self) -> None:
        """Push buffered rows to the operating system. Not part of any ``MetricsSink`` port."""
        with self._lock:
            self._handle.flush()

    def close(self) -> None:
        """Flush and release the file. Idempotent, so a ``finally`` and an ``__exit__`` may both
        call it."""
        with self._lock:
            if not self._handle.closed:
                self._handle.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()

    def _emit(self, kind: str, name: str, value: str, tags: dict[str, str]) -> None:
        """One row: when, what kind, which metric, how much, and the dimensions to filter by.

        **A value is never refused.** ``nan`` and ``inf`` are written as ``repr`` writes them, and
        that is deliberate: a gauge that came out non-finite is a finding about the code that
        emitted it, and a sink that raised would turn an observation into an outage, while one that
        dropped the row would hide exactly the measurement somebody needs to see. ``float`` is
        applied first so that an ``int`` passed as a gauge -- legal under the protocol's numeric
        tower -- is written as ``3.0`` and the column parses as one type.

        Writing after :meth:`close` raises ``ValueError`` from the file object, unchanged: a metric
        emitted after the run was declared over is a shutdown-ordering bug in the composition root,
        and silence would hide it.
        """
        stamp = self._clock.now().isoformat()
        rendered = json.dumps(tags, sort_keys=True, separators=(",", ":"))
        with self._lock:
            self._writer.writerow((stamp, kind, name, value, rendered))
