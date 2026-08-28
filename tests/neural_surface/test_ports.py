"""Architectural tests for the driven ports of the Neural Surface context.

Nothing here asserts business behaviour -- the ports have no behaviour. What these tests keep
standing is a decision: the domain declares what it needs, and the outside world satisfies it
*structurally*, with no inheritance and no import from ``neural_surface/domain/`` towards
``platform/`` or towards torch. Because nothing inherits from a ``Protocol``, the only places that
shape is ever checked are (a) an assignment of a concrete object to a variable annotated with the
port and (b) a call made through such a variable. This file is both, deliberately.

Two decisions are load-bearing beyond the wiring, and most of the tests below exist only to
protect them. First, ``SurfaceLearner.update`` is a **pure function**: the surface to continue from
arrives as an argument and the new one leaves as a return value, so a recorded session replays
exactly and the neural producer can be compared with the parametric one on identical terms
(ADR-004, Design 6.4, 6.5). Purity cannot be checked by a type, so it is checked by a signature and
by calling twice. Second, the port deliberately contradicts ``Implementation.md``'s sketch in two
places -- no ``evaluate``, and a ``previous`` -- and a test guards each, because a contradiction
nobody records is a contradiction somebody later "fixes".

Test modules are free to import ``platform/`` -- the import rules constrain production code.
"""

from __future__ import annotations

import inspect
import logging
import math
import re
from datetime import datetime

import numpy as np
import pytest

from tests.neural_surface.builders import (
    NEAR_TENOR,
    NOW,
    CallableSurface,
    FlatVolSurface,
    make_batch,
)
from volengine.neural_surface.domain import ports
from volengine.neural_surface.domain.learned_surface import LearnedSurface
from volengine.neural_surface.domain.ports import Clock, MetricsSink, SurfaceLearner
from volengine.neural_surface.domain.training_batch import TrainingBatch
from volengine.platform.clock import ManualClock, SimulatedClock, SystemClock
from volengine.platform.metrics import LoggingMetricsSink, NullMetricsSink

ATM = np.array([0.0])
"""One point at the forward, for reading a surface's level back out of it."""

NEAR = np.array([NEAR_TENOR])
"""One tenor, matched to ``ATM`` so the pair addresses a single cell of the mesh."""


class FakeLearner:
    """A learner with no network in it, written against the port and nothing else.

    It inherits from nothing, which is the point -- if this satisfies ``SurfaceLearner``, so will
    the torch adapter of F3-D, and neither had to know the protocol object exists.

    Its one interesting move is that it reads ``previous`` **through the port**: it evaluates the
    surface it was handed to recover the level to continue from, exactly as a real fine-tune reads
    weights it did not produce. That is the deleted ``evaluate`` argued in code -- a learner never
    needs the port to grow an evaluation method, because a ``LearnedSurface`` already is one.

    It records what reached it so the tests can assert on it, and that recording is the only state
    it keeps: a real implementation keeps none at all, and the ``seen_*`` lists are test
    instruments rather than an example to copy.
    """

    def __init__(self, producer_id: str = "mlp-fake") -> None:
        self._producer_id = producer_id
        self.seen_previous: list[LearnedSurface | None] = []
        self.seen_split: list[tuple[int, int]] = []

    @property
    def producer_id(self) -> str:
        return self._producer_id

    def update(self, previous: LearnedSurface | None, batch: TrainingBatch) -> LearnedSurface:
        self.seen_previous.append(previous)
        self.seen_split.append((len(batch.fresh), len(batch.replayed)))
        if previous is None:
            # The cold start, which is also the scheduled restart: no history to read, so the
            # level comes from the batch instead.
            return FlatVolSurface(vol=batch.samples[0].implied_vol, version=1)
        total_variance = float(previous.total_variance(ATM, NEAR)[0, 0])
        return FlatVolSurface(
            vol=math.sqrt(total_variance / NEAR_TENOR),
            version=previous.version + 1,
        )


class ArbitrageableLearner:
    """A learner whose surface dives negative in the wings, and that returns it anyway.

    ADR-010 is decided by the use case, after the gate has run. This class is what "the port must
    be able to carry an unpublishable surface home" looks like from the inside.
    """

    producer_id = "mlp-broken"

    def update(self, previous: LearnedSurface | None, batch: TrainingBatch) -> LearnedSurface:
        return CallableSurface(fn=lambda k, tenors: 0.4225 * tenors - 0.5 * k * k)


class LearnerWithoutProducerId:
    """A learner that forgot its identity. Structurally *not* a ``SurfaceLearner``."""

    def update(self, previous: LearnedSurface | None, batch: TrainingBatch) -> LearnedSurface:
        return FlatVolSurface()


# --- calls made through the ports
# Annotated with the port, never with the concrete class. These are stand-ins for the use case of
# F1-06, and the reason a drift in a signature fails here.


def run_one_update(
    learner: SurfaceLearner,
    previous: LearnedSurface | None,
    batch: TrainingBatch,
) -> LearnedSurface:
    return learner.update(previous, batch)


def level_at_the_money(surface: LearnedSurface) -> float:
    return float(surface.total_variance(ATM, NEAR)[0, 0])


def stamp(clock: Clock) -> datetime:
    return clock.now()


def record_gate(metrics: MetricsSink) -> None:
    metrics.gauge("neural.butterfly_violation", 0.004, market="BTC-DERIBIT", producer="mlp-fake")
    metrics.gauge("neural.buffer_cell_coverage", 0.83, producer="mlp-fake")
    metrics.counter("neural.publications_refused", producer="mlp-fake")  # default value=1
    metrics.timing("neural.update_ms", 42.0, producer="mlp-fake")


# --- SurfaceLearner


def test_a_learner_returns_a_surface_that_can_be_evaluated() -> None:
    """The return value is the whole product of the port, and it is evaluable on arrival --
    nothing has to be asked of the learner to make it usable.
    """
    learner: SurfaceLearner = FakeLearner()

    surface = run_one_update(learner, None, make_batch())

    assert level_at_the_money(surface) == pytest.approx(0.65**2 * NEAR_TENOR)


def test_a_learner_accepts_a_cold_start() -> None:
    """``None`` is the first snapshot of a market and the recovery path after a failure, so an
    implementation may not assume there is always a surface to continue from.
    """
    learner: SurfaceLearner = FakeLearner()

    surface = run_one_update(learner, None, make_batch())

    assert surface.version == 1


def test_a_scheduled_restart_is_a_cold_start_and_not_a_second_method() -> None:
    """Design 6.4's restart every M hours is ``update(None, batch)`` over the whole buffer: a batch
    with no fresh quotes in it at all. A dedicated ``restart`` would be a second code path firing
    once every few hours -- the rarest path in the system, and therefore the least exercised.
    """
    learner: SurfaceLearner = FakeLearner()
    whole_buffer = make_batch(n_fresh=0)

    surface = run_one_update(learner, None, whole_buffer)

    assert surface.version == 1
    assert not hasattr(SurfaceLearner, "restart")


def test_the_previous_surface_reaches_the_learner_and_is_evaluable_there() -> None:
    """The warm start is not an opaque token: a fine-tune has to be able to read what it is
    continuing from, which is why ``LearnedSurface`` is evaluable and the learner is not.
    """
    fake = FakeLearner()
    learner: SurfaceLearner = fake
    previous = FlatVolSurface(vol=0.80, version=3)

    surface = run_one_update(learner, previous, make_batch())

    assert fake.seen_previous == [previous]
    assert level_at_the_money(surface) == pytest.approx(0.80**2 * NEAR_TENOR)


def test_an_update_leaves_the_previous_surface_usable() -> None:
    """Design 6.4 measures drift by comparing the surface before a restart with the one after, so
    both must exist at once. A learner that retrained itself in place would destroy the comparison.
    """
    learner: SurfaceLearner = FakeLearner()
    previous = FlatVolSurface(vol=0.80, version=3)

    surface = run_one_update(learner, previous, make_batch())

    assert surface is not previous
    assert surface.version == 4
    assert level_at_the_money(previous) == pytest.approx(0.80**2 * NEAR_TENOR)


def test_updating_twice_from_the_same_inputs_returns_the_same_surface() -> None:
    """Purity, made observable. A learner that stashed the last surface, counted its own calls or
    carried an optimiser's moment estimates between invocations would still type-check; this is
    what would fail. It is also exactly what a replay does.
    """
    learner: SurfaceLearner = FakeLearner()
    previous = FlatVolSurface(vol=0.80, version=3)
    batch = make_batch()

    first = run_one_update(learner, previous, batch)
    second = run_one_update(learner, previous, batch)

    assert first == second


def test_the_fresh_and_replayed_split_survives_the_trip_through_the_port() -> None:
    """``n_fresh`` is the fine-tuning regime of Design 6.4 made visible to the learner: this
    snapshot's quotes first, the stratified draw from the buffer behind them.
    """
    fake = FakeLearner()
    learner: SurfaceLearner = fake

    run_one_update(learner, None, make_batch())

    assert fake.seen_split == [(2, 2)]


def test_two_producers_are_told_apart_by_their_producer_id() -> None:
    """This context runs beside the parametric one on the same market by design, so the id names
    the bus topic, travels inside the published surface, and keeps the two metric series apart.
    """
    torch_like: SurfaceLearner = FakeLearner("mlp-torch")
    other: SurfaceLearner = FakeLearner("mlp-wide")

    assert torch_like.producer_id != other.producer_id


def test_an_arbitrageable_surface_is_returned_rather_than_raised() -> None:
    """ADR-010's gate runs after the learner, in the use case, and republishes the last good
    surface when it refuses. A learner that raised on a surface it suspected would take that
    decision away from the layer that owns it and leave the gate nothing to measure.
    """
    learner: SurfaceLearner = ArbitrageableLearner()

    surface = run_one_update(learner, None, make_batch())

    # The guard against a vacuous assertion above: this really is a surface the gate would refuse,
    # not merely one that happened not to raise.
    wing = float(surface.total_variance(np.array([-0.60]), NEAR)[0, 0])
    assert wing < 0.0


def test_update_takes_only_a_previous_surface_and_a_batch() -> None:
    # The signature *is* the purity guarantee: no clock, no metrics sink, no replay buffer, no
    # configuration, and no flag saying whether this is a fine-tune or a restart. Anything else in
    # this list would be a door for state or I/O to enter the one function the two producers are
    # compared on.
    parameters = list(inspect.signature(SurfaceLearner.update).parameters)

    assert parameters == ["self", "previous", "batch"]


def test_update_is_synchronous() -> None:
    # Deliberate: K gradient steps are compute-bound and the use case hands them to the thread pool
    # of ADR-005. An `async def` here would invite an implementation to do I/O inside the one
    # function that must not.
    assert not inspect.iscoroutinefunction(SurfaceLearner.update)


def test_the_learner_does_not_evaluate_on_its_surfaces_behalf() -> None:
    # Departure from Implementation.md's sketch, recorded so nobody "restores" it. Evaluation lives
    # on LearnedSurface, which is what lets the hard gate of ADR-010 judge a surface written by
    # hand in numpy, with no learner and no torch anywhere near it.
    assert not hasattr(SurfaceLearner, "evaluate")
    assert hasattr(LearnedSurface, "total_variance")


def test_a_learner_without_a_producer_id_does_not_satisfy_the_port() -> None:
    # How non-conformance is caught here: the annotated assignment is the check, and mypy is what
    # runs it. The `type: ignore` is therefore the assertion -- without it this line fails the type
    # check that gates this repo. The runtime assertion below guards it from being vacuous.
    broken: SurfaceLearner = LearnerWithoutProducerId()  # type: ignore[assignment]

    assert not hasattr(broken, "producer_id")


# --- Clock


@pytest.mark.parametrize(
    "clock",
    [SystemClock(), ManualClock(NOW), SimulatedClock(NOW)],
    ids=["system", "manual", "simulated"],
)
def test_every_platform_clock_satisfies_the_neural_clock_port(clock: Clock) -> None:
    now = stamp(clock)

    assert now.tzinfo is not None


def test_the_neural_clock_port_does_not_ask_for_sleep() -> None:
    # The restart every M hours sounds like a timer, and a timer would need `sleep` -- which Market
    # Data's Clock does declare. It is not one: the restart is evaluated when a snapshot arrives,
    # by comparing instants, which keeps this context reactive and keeps the restart from firing
    # when there is nothing to train on.
    assert not hasattr(Clock, "sleep")


# --- MetricsSink


def test_null_metrics_sink_satisfies_the_neural_metrics_port() -> None:
    metrics: MetricsSink = NullMetricsSink()

    # A null sink records nothing anywhere; not raising is the behaviour.
    record_gate(metrics)


def test_logging_metrics_sink_emits_every_measurement_made_through_the_port(
    caplog: pytest.LogCaptureFixture,
) -> None:
    metrics: MetricsSink = LoggingMetricsSink(logging.getLogger("test.neural.metrics"))

    with caplog.at_level(logging.INFO, logger="test.neural.metrics"):
        record_gate(metrics)

    assert len(caplog.records) == 4


# --- the module itself


def test_the_ports_module_does_not_import_torch() -> None:
    # Import rule 3, and the one this context exists to keep: the hard gate of ADR-010 is only
    # worth something while the domain can be evaluated without a model. A port typed in tensors
    # would put torch in the signature of everything that touches it.
    source = inspect.getsource(ports)

    assert re.search(r"^\s*(import torch|from torch\b)", source, re.MULTILINE) is None


def test_the_ports_module_does_not_import_asyncio() -> None:
    # Import rule 3: */domain/** never imports asyncio. Nothing here needs a runtime, which is what
    # keeps the domain independent of how the composition root schedules anything.
    # import-linter enforces this repo-wide in F1-09.
    source = inspect.getsource(ports)

    assert re.search(r"^\s*(import asyncio|from asyncio\b)", source, re.MULTILINE) is None


def test_the_ports_module_does_not_import_the_contracts() -> None:
    # Import rule 3 again, and the stronger half of it: the domain does not know the published
    # language. A port typed in DTOs would put `contracts/` in the signature every adapter has to
    # implement, and the ACL would have nothing left to translate.
    source = inspect.getsource(ports)

    assert re.search(r"^\s*from volengine\.contracts\b", source, re.MULTILINE) is None


@pytest.mark.parametrize("port", [SurfaceLearner, Clock, MetricsSink])
def test_the_ports_are_not_runtime_checkable(port: type) -> None:
    # Deliberate: isinstance against a runtime-checkable Protocol compares member *names* only, so
    # it would bless a learner that accepts `previous` and ignores it -- continuous fine-tuning
    # silently degraded into a cold restart every snapshot. Conformance is verified statically, and
    # by the calls made through the ports above.
    with pytest.raises(TypeError, match="runtime_checkable"):
        isinstance(object(), port)
