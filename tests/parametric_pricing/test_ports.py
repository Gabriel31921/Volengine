"""Architectural tests for the driven ports of the Parametric Pricing context.

Nothing here asserts business behaviour -- the ports have no behaviour. What these tests keep
standing is a decision: the domain declares what it needs, and the outside world satisfies it
*structurally*, with no inheritance and no import from ``parametric_pricing/domain/`` towards
``platform/``. Because nothing inherits from a ``Protocol``, the only places that shape is ever
checked are (a) an assignment of a concrete object to a variable annotated with the port and
(b) a call made through such a variable. This file is both, deliberately.

One decision here is load-bearing beyond the wiring, and several tests below exist only to
protect it: ``Calibrator.calibrate`` is a **pure function**. The warm start arrives as an
argument and the result leaves as a return value, so a recorded session replays exactly and two
implementations can be compared on identical inputs (Design 5.6, 5.7). Purity cannot be checked
by a type, so it is checked by a signature and by calling twice.

Test modules are free to import ``platform/`` -- the import rules constrain production code.
"""

from __future__ import annotations

import inspect
import logging
import re
from collections.abc import Mapping
from datetime import datetime

import pytest

from tests.parametric_pricing.builders import (
    FAR,
    NEAR,
    NOW,
    make_calibration_task,
    make_params,
    make_slice_result,
)
from volengine.parametric_pricing.domain import ports
from volengine.parametric_pricing.domain.calibration import CalibrationResult, CalibrationTask
from volengine.parametric_pricing.domain.ports import Calibrator, Clock, MetricsSink
from volengine.parametric_pricing.domain.svi_slice import SVIParams
from volengine.platform.clock import ManualClock, SimulatedClock, SystemClock
from volengine.platform.metrics import LoggingMetricsSink, NullMetricsSink


class FakeCalibrator:
    """A calibrator with no optimiser in it, written against the port and nothing else.

    It inherits from nothing, which is the point -- if this satisfies ``Calibrator``, so will
    the scipy adapter of F2-05 and the JAX one of F3-A, and none of the three had to know the
    protocol object exists.

    It records the warm start it was handed so the tests can assert what reached it, and that
    recording is the only state it keeps: a real implementation keeps none at all, and the
    ``seen_previous`` list is a test instrument rather than an example to copy.
    """

    def __init__(self, producer_id: str = "svi-fake") -> None:
        self._producer_id = producer_id
        self.seen_previous: list[Mapping[datetime, SVIParams] | None] = []

    @property
    def producer_id(self) -> str:
        return self._producer_id

    def calibrate(
        self,
        previous: Mapping[datetime, SVIParams] | None,
        task: CalibrationTask,
    ) -> CalibrationResult:
        self.seen_previous.append(previous)
        return CalibrationResult(
            slices=tuple(
                make_slice_result(
                    expiry=one.expiry,
                    tenor_years=one.tenor_years,
                    # A warm start is used when there is one for this expiry, and the cold
                    # default is used when there is not -- which is what makes "keyed by expiry"
                    # observable from the outside.
                    params=(previous or {}).get(one.expiry, make_params()),
                    n_quotes_used=len(one.log_moneyness),
                )
                for one in task.slices
            ),
            n_iterations=0 if previous is not None else 7,
            duration_ms=1.5,
        )


# --- calls made through the ports
# Annotated with the port, never with the concrete class. These are stand-ins for the use case
# of F1-06, and the reason a drift in a signature fails here.


def run_one_cycle(
    calibrator: Calibrator,
    previous: Mapping[datetime, SVIParams] | None,
    task: CalibrationTask,
) -> CalibrationResult:
    return calibrator.calibrate(previous, task)


def stamp(clock: Clock) -> datetime:
    return clock.now()


def record_calibration(metrics: MetricsSink) -> None:
    metrics.gauge("pricing.rmse_vol_bp", 12.4, market="BTC-DERIBIT", producer="svi-fake")
    metrics.counter("pricing.fits_accepted", 2, producer="svi-fake")
    metrics.counter("pricing.fits_at_bound", producer="svi-fake")  # relies on the default value=1
    metrics.timing("pricing.calibrate_ms", 1.5, producer="svi-fake")


# --- Calibrator


def test_a_calibrator_fits_every_slice_of_the_task() -> None:
    calibrator: Calibrator = FakeCalibrator()
    task = make_calibration_task()

    result = run_one_cycle(calibrator, None, task)

    assert tuple(one.expiry for one in result.slices) == (NEAR, FAR)


def test_a_calibrator_accepts_a_cold_start() -> None:
    """``None`` is the first cycle of a market and the recovery path after a failure, so an
    implementation may not assume a warm start is always there.
    """
    calibrator: Calibrator = FakeCalibrator()

    result = run_one_cycle(calibrator, None, make_calibration_task())

    assert result.n_iterations == 7


def test_a_warm_start_reaches_the_calibrator_keyed_by_expiry() -> None:
    """The whole reason ``previous`` is a mapping and not a sequence: a chain's composition
    changes between snapshots, so only the expiry identifies which history belongs where.
    """
    fake = FakeCalibrator()
    calibrator: Calibrator = fake
    warm = {NEAR: make_params(b=0.42)}

    result = run_one_cycle(calibrator, warm, make_calibration_task())

    near, far = result.slices
    assert near.params.b == 0.42
    assert far.params == make_params()  # no history for that expiry, so the cold default


def test_calibrating_twice_from_the_same_inputs_returns_the_same_result() -> None:
    """Purity, made observable. A calibrator that cached, counted its own calls or read a clock
    would still type-check; this is what would fail. It is also exactly what a replay does.
    """
    calibrator: Calibrator = FakeCalibrator()
    task = make_calibration_task()
    warm = {NEAR: make_params(b=0.42)}

    first = run_one_cycle(calibrator, warm, task)
    second = run_one_cycle(calibrator, warm, task)

    assert first == second


def test_two_calibrators_are_told_apart_by_their_producer_id() -> None:
    """Two implementations on one market is the normal case here, not an exception: the id names
    the bus topic and travels inside the published surface.
    """
    scipy_like: Calibrator = FakeCalibrator("svi-scipy")
    jax_like: Calibrator = FakeCalibrator("svi-jax")

    assert scipy_like.producer_id != jax_like.producer_id


def test_a_poor_fit_is_returned_rather_than_raised() -> None:
    """ADR-006 is decided by the use case, so the port must be able to carry a bad fit home. A
    calibrator that raised, or that dropped the slice, would take that decision away from the
    layer that owns it and leave nothing to report.
    """
    calibrator: Calibrator = FakeCalibrator()
    task = make_calibration_task()

    result = calibrator.calibrate(None, task)

    assert len(result.slices) == len(task.slices)


def test_calibrate_takes_only_a_warm_start_and_a_task() -> None:
    # The signature *is* the purity guarantee: no clock, no metrics sink, no snapshot, no
    # configuration. Anything else in this list would be a door for state or I/O to enter the
    # one function two implementations are compared on.
    parameters = list(inspect.signature(Calibrator.calibrate).parameters)

    assert parameters == ["self", "previous", "task"]


def test_calibrate_is_synchronous() -> None:
    # Deliberate: calibration is CPU-bound and the use case hands it to the thread pool of
    # ADR-005. An `async def` here would invite an implementation to do I/O inside the one
    # function that must not.
    assert not inspect.iscoroutinefunction(Calibrator.calibrate)


# --- Clock


@pytest.mark.parametrize(
    "clock",
    [SystemClock(), ManualClock(NOW), SimulatedClock(NOW)],
    ids=["system", "manual", "simulated"],
)
def test_every_platform_clock_satisfies_the_pricing_clock_port(clock: Clock) -> None:
    now = stamp(clock)

    assert now.tzinfo is not None


def test_the_pricing_clock_port_does_not_ask_for_sleep() -> None:
    # The concrete argument for declaring a port per context instead of sharing one: this
    # context reacts to snapshots on the bus and never waits, so `sleep` -- which Market Data's
    # Clock does declare -- would be a method no code path here ever calls.
    assert not hasattr(Clock, "sleep")


# --- MetricsSink


def test_null_metrics_sink_satisfies_the_pricing_metrics_port() -> None:
    metrics: MetricsSink = NullMetricsSink()

    # A null sink records nothing anywhere; not raising is the behaviour.
    record_calibration(metrics)


def test_logging_metrics_sink_emits_every_measurement_made_through_the_port(
    caplog: pytest.LogCaptureFixture,
) -> None:
    metrics: MetricsSink = LoggingMetricsSink(logging.getLogger("test.pricing.metrics"))

    with caplog.at_level(logging.INFO, logger="test.pricing.metrics"):
        record_calibration(metrics)

    assert len(caplog.records) == 4


# --- the module itself


def test_the_ports_module_does_not_import_asyncio() -> None:
    # Import rule 3: */domain/** never imports asyncio. Nothing here needs a runtime, which is
    # what keeps the domain independent of how the composition root schedules anything.
    # import-linter enforces this repo-wide in F1-09.
    source = inspect.getsource(ports)

    assert re.search(r"^\s*(import asyncio|from asyncio\b)", source, re.MULTILINE) is None


def test_the_ports_module_does_not_import_the_contracts() -> None:
    # Import rule 3 again, and the stronger half of it: the domain does not know the published
    # language. A port typed in DTOs would put `contracts/` in the signature every adapter has
    # to implement, and the ACL would have nothing left to translate.
    source = inspect.getsource(ports)

    assert re.search(r"^\s*from volengine\.contracts\b", source, re.MULTILINE) is None


@pytest.mark.parametrize("port", [Calibrator, Clock, MetricsSink])
def test_the_ports_are_not_runtime_checkable(port: type) -> None:
    # Deliberate: isinstance against a runtime-checkable Protocol compares member *names* only,
    # so it would bless an adapter whose signatures are wrong while reading like a real check.
    # Conformance is verified statically, and by the calls made through the ports above.
    with pytest.raises(TypeError, match="runtime_checkable"):
        isinstance(object(), port)
