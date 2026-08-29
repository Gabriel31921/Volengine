"""Helpers shared by every test package.

Deliberately not ``conftest.py``: conftest is where pytest looks for fixtures and hooks it
*injects*, and importing from it is discouraged precisely because it is loaded by collection
magic rather than by an import statement. This repo's builders are plain functions that tests
call directly, so they live in ordinary modules that say where they came from.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from dataclasses import field as dc_field
from typing import Any


def replace_field[T](instance: T, field: str, value: Any) -> T:
    """``dataclasses.replace`` with the field name chosen at runtime.

    The parametrised rejection tests pick which field to poison from the parameter list, so the
    keyword is a string known only while the test runs. Splatting that dict into ``replace``
    defeats mypy: it sees ``**dict[str, float]`` and reports every *other* field of the
    dataclass as receiving a ``float``. Scattering ``# type: ignore`` across twenty tests to
    silence that would also silence the real mistakes those tests exist to catch, so the
    dynamism is confined to this one call instead.
    """
    return replace(instance, **{field: value})  # type: ignore[type-var]


@dataclass(slots=True)
class RecordingMetrics:
    """A ``MetricsSink`` that keeps what it was told, so a test can assert on observability.

    Every context declares its own ``MetricsSink`` Protocol and every one of them has the same
    three methods, so this single class satisfies all four structurally -- which is itself worth
    a test: if two contexts' sinks ever diverge, this object stops fitting one of them and mypy
    says so at the assignment.

    Observability is behaviour here, not decoration. ADR-010's refusal rate, ADR-003's dropped
    events and the clock skew Market Data reconciles against are all invisible except through a
    sink, so a use case that computed the right answer and reported nothing would be half built.
    """

    gauges: list[tuple[str, float, dict[str, str]]] = dc_field(default_factory=list)
    counters: list[tuple[str, int, dict[str, str]]] = dc_field(default_factory=list)
    timings: list[tuple[str, float, dict[str, str]]] = dc_field(default_factory=list)

    def gauge(self, name: str, value: float, **tags: str) -> None:
        self.gauges.append((name, value, dict(tags)))

    def counter(self, name: str, value: int = 1, **tags: str) -> None:
        self.counters.append((name, value, dict(tags)))

    def timing(self, name: str, ms: float, **tags: str) -> None:
        self.timings.append((name, ms, dict(tags)))

    def counter_names(self) -> list[str]:
        """Just the names, for the common assertion that something was counted at all."""
        return [name for name, _, _ in self.counters]

    def gauge_value(self, name: str) -> float:
        """The last value gauged under ``name``.

        Raises:
            KeyError: If nothing was ever gauged under that name. A ``KeyError`` rather than a
                default, because a test asserting on a metric that was never emitted has found a
                bug and must not be able to pass by comparing against a zero this helper invented.
        """
        values = [value for gauged, value, _ in self.gauges if gauged == name]
        if not values:
            emitted = sorted({gauged for gauged, _, _ in self.gauges})
            raise KeyError(f"no gauge named {name!r} was emitted, only {emitted}")
        return values[-1]
