"""Observability as a port, so the domain never chooses a destination.

``logging`` is banned in the domain layers, and this is what replaces it. The difference is
not cosmetic: a logger makes a module decide on a format, a level and a sink, three things
that have nothing to do with volatility surfaces and that a test then has to work around.
A ``MetricsSink`` lets a module state *what* it observed and stop there.

The other half of the reason is that these measurements are the experiment. Whether the
neural calibrator is competitive with SVI is answered by RMSE, latency and failure counts
over a long run -- so the numbers have to be structured and taggable from day one, not parsed
back out of log prose later.

Each context declares its own ``MetricsSink`` protocol; the implementations here satisfy it
structurally, with no import in either direction.
"""

from __future__ import annotations

import logging
from typing import Protocol

_LOGGER = logging.getLogger("volengine.metrics")


class MetricsSink(Protocol):
    """Where observability goes.

    A port rather than a logger because the domain must not choose a destination, a format
    or a level. It states *what* it observes; the composition root decides where that ends
    up — console while developing, CSV once the plots of F3-E need feeding.

    Tags are the dimensions a measurement can later be filtered by: market_id, producer_id,
    topic. They are strings so any backend can carry them.
    """

    def gauge(self, name: str, value: float, **tags: str) -> None:
        """A value that goes up and down: the RMSE of the last calibration."""
        ...

    def counter(self, name: str, value: int = 1, **tags: str) -> None:
        """A value that only grows: how many calibrations have failed."""
        ...

    def timing(self, name: str, ms: float, **tags: str) -> None:
        """A duration in milliseconds: snapshot to published surface."""
        ...


class NullMetricsSink:
    """Discards everything. The default wherever metrics are irrelevant, mostly in tests."""

    def gauge(self, name: str, value: float, **tags: str) -> None:
        return None

    def counter(self, name: str, value: int = 1, **tags: str) -> None:
        return None

    def timing(self, name: str, ms: float, **tags: str) -> None:
        return None


class LoggingMetricsSink:
    """Emits one structured line per measurement. No Prometheus, by design."""

    def __init__(self, logger: logging.Logger | None = None) -> None:
        self._logger = logger if logger is not None else _LOGGER

    def gauge(self, name: str, value: float, **tags: str) -> None:
        self._emit("gauge", name, value, tags)

    def counter(self, name: str, value: int = 1, **tags: str) -> None:
        self._emit("counter", name, value, tags)

    def timing(self, name: str, ms: float, **tags: str) -> None:
        self._emit("timing", name, ms, tags)

    def _emit(self, kind: str, name: str, value: float, tags: dict[str, str]) -> None:
        # Tags are sorted so the same measurement always renders identically, which makes
        # the output diffable and greppable.
        rendered = " ".join(f"{key}={value}" for key, value in sorted(tags.items()))
        self._logger.info("%s %s=%s %s", kind, name, value, rendered)
