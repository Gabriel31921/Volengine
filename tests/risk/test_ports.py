"""Architectural tests for the driven ports of the Risk context.

Nothing here asserts business behaviour -- the ports have no behaviour. What these tests keep
standing is a decision: the domain declares what it needs, and the outside world satisfies it
*structurally*, with no inheritance and no import from ``risk/domain/`` towards ``platform/``,
towards the bus, or towards the producers. Because nothing inherits from a ``Protocol``, the only
places that shape is ever checked are (a) an assignment of a concrete object to a variable
annotated with the port and (b) a call made through such a variable. This file is both,
deliberately.

One decision is load-bearing far beyond the wiring, and most of the tests below exist to protect
it: ``SurfaceProvider`` is **defined by the reader**. Risk is the only context in this engine that
purely consumes, so the port belongs to it, and the consequence is that a single call site can be
served by the parametric producer, the neural one, or the composite last-value cache of Design 7.3
without Risk knowing that any of them exist. The tests below exercise exactly that: the same
``fetch`` function, three different providers, and no branch anywhere.

The second decision is quieter and is guarded here too: ``latest`` returning ``None`` is an
ordinary answer. At start-up nobody has published yet, and that is not an error -- it is one of
the two ways a report says there is nothing to value, the other being ``REJECT``.

Test modules are free to import ``platform/`` -- the import rules constrain production code.
"""

from __future__ import annotations

import inspect
import logging
import re
import typing
from dataclasses import dataclass
from datetime import datetime, timedelta

import pytest

from tests.risk.builders import NOW, make_report, make_view
from volengine.platform.clock import ManualClock, SimulatedClock, SystemClock
from volengine.platform.metrics import LoggingMetricsSink, NullMetricsSink
from volengine.risk.domain import ports
from volengine.risk.domain.freshness_policy import FreshnessDecision
from volengine.risk.domain.ports import Clock, MetricsSink, ReportWriter, SurfaceProvider
from volengine.risk.domain.risk_report import RiskReport
from volengine.risk.domain.surface_view import SurfaceView

OTHER_MARKET = "ETH-DERIBIT"
"""A second market, so "keyed by market_id" is observable rather than merely declared."""


class FakeSurfaceProvider:
    """A last-value cache with no bus behind it, written against the port and nothing else.

    It inherits from nothing, which is the point -- if this satisfies ``SurfaceProvider``, so will
    the subscription-fed cache of F2-06 and the composite of Design 7.3, and none of them had to
    know the protocol object exists.

    It records the keys it was asked for so a test can assert what reached it; that recording is
    the only state it keeps beyond the surfaces themselves.
    """

    def __init__(self, surfaces: dict[str, SurfaceView] | None = None) -> None:
        self._surfaces = {} if surfaces is None else surfaces
        self.seen_market_ids: list[str] = []

    def latest(self, market_id: str) -> SurfaceView | None:
        self.seen_market_ids.append(market_id)
        return self._surfaces.get(market_id)


@dataclass(frozen=True)
class PublishedSurface:
    """Stand-in for ``contracts.CalibratedSurface``: a DTO, and not this context's view.

    Built here rather than imported so the test states the shape of the mistake it is guarding
    against without depending on the real DTO's fields. What matters is that it is *not* a
    ``SurfaceView`` while still being a perfectly plausible thing for an adapter to hand back.
    """

    market_id: str


class LeakyProvider:
    """A provider that returns the published contract instead of Risk's own view.

    The exact failure ``@runtime_checkable`` would bless: the member is there, it is spelled
    ``latest``, it takes a ``market_id``, and it hands the domain the language the ACL exists to
    translate. Structurally *not* a ``SurfaceProvider``.
    """

    def latest(self, market_id: str) -> PublishedSurface:
        return PublishedSurface(market_id=market_id)


class RecordingWriter:
    """A writer that keeps what it was given, so a test can look at it. Formats nothing."""

    def __init__(self) -> None:
        self.written: list[RiskReport] = []

    def write(self, report: RiskReport) -> None:
        self.written.append(report)


# --- calls made through the ports
# Annotated with the port, never with the concrete class. These are stand-ins for the use case of
# F1-06, and the reason a drift in a signature fails here.


def fetch(provider: SurfaceProvider, market_id: str) -> SurfaceView | None:
    return provider.latest(market_id)


def publish(writer: ReportWriter, report: RiskReport) -> None:
    writer.write(report)


def stamp(clock: Clock) -> datetime:
    return clock.now()


def record_report(metrics: MetricsSink) -> None:
    metrics.gauge("risk.surface_staleness_s", 3.2, market="BTC-DERIBIT", producer="svi-scipy")
    metrics.gauge("risk.producer_value_gap", 118.5, market="BTC-DERIBIT")
    metrics.counter("risk.reports_rejected", producer="svi-scipy")  # relies on the default value=1
    metrics.timing("risk.report_ms", 4.0, producer="svi-scipy")


# --- SurfaceProvider


def test_a_provider_hands_back_the_surface_it_holds_for_the_market() -> None:
    view = make_view()
    provider: SurfaceProvider = FakeSurfaceProvider({"BTC-DERIBIT": view})

    assert fetch(provider, "BTC-DERIBIT") is view


def test_a_provider_with_nothing_published_yet_answers_none() -> None:
    """``None`` is an ordinary answer, not an error: at start-up no producer has published. It is
    one of the two ways a report says there is nothing to value, ``REJECT`` being the other.
    """
    provider: SurfaceProvider = FakeSurfaceProvider()

    assert fetch(provider, "BTC-DERIBIT") is None


def test_a_provider_answers_none_for_a_market_it_does_not_serve() -> None:
    """Same answer, a different situation: the provider is warm, this market simply has no
    producer wired to it. Neither case is exceptional, which is why one return type covers both.
    """
    provider: SurfaceProvider = FakeSurfaceProvider({"BTC-DERIBIT": make_view()})

    assert fetch(provider, OTHER_MARKET) is None


def test_one_provider_serves_several_markets_keyed_by_market_id() -> None:
    """Why ``latest`` takes a key at all: the multi-market run of F3-F wires a single composite
    and lets the key do the routing, so Risk grows no registry and no per-market instance.
    """
    btc = make_view()
    eth = make_view(market_id=OTHER_MARKET)
    fake = FakeSurfaceProvider({"BTC-DERIBIT": btc, OTHER_MARKET: eth})
    provider: SurfaceProvider = fake

    assert fetch(provider, "BTC-DERIBIT") is btc
    assert fetch(provider, OTHER_MARKET) is eth
    assert fake.seen_market_ids == ["BTC-DERIBIT", OTHER_MARKET]


def test_the_same_call_site_is_served_by_either_producer() -> None:
    """Design 7.1's inversion, made observable: two providers, one ``fetch``, no branch. This is
    what lets the comparative report of Design 7.3 value one portfolio twice.
    """
    parametric: SurfaceProvider = FakeSurfaceProvider({"BTC-DERIBIT": make_view()})
    neural: SurfaceProvider = FakeSurfaceProvider(
        {"BTC-DERIBIT": make_view(producer_id="mlp-torch")}
    )

    from_parametric = fetch(parametric, "BTC-DERIBIT")
    from_neural = fetch(neural, "BTC-DERIBIT")

    assert from_parametric is not None
    assert from_neural is not None
    # The guard against a vacuous assertion: the two really are different surfaces, so the equal
    # treatment above is substitutability and not a coincidence of identical fixtures.
    assert from_parametric.producer_id != from_neural.producer_id


def test_the_producer_is_named_by_the_surface_and_not_by_the_provider() -> None:
    """Where Risk parts company with its two sibling contexts. ``Calibrator`` and
    ``SurfaceLearner`` each carry a ``producer_id``, because each *is* one producer; a composite
    provider serves several, so the question has no single answer and is answered by the surface
    actually handed back.
    """
    provider: SurfaceProvider = FakeSurfaceProvider({"BTC-DERIBIT": make_view()})

    view = fetch(provider, "BTC-DERIBIT")

    assert not hasattr(SurfaceProvider, "producer_id")
    assert view is not None
    assert view.producer_id == "svi-scipy"


def test_a_provider_returns_this_contexts_view_and_not_the_published_contract() -> None:
    """Rule 3 in the one place it is hardest to see. The ACL builds the view; the port's return
    type is what makes that unavoidable rather than merely intended.
    """
    provider: SurfaceProvider = FakeSurfaceProvider({"BTC-DERIBIT": make_view()})

    assert isinstance(fetch(provider, "BTC-DERIBIT"), SurfaceView)


def test_a_provider_that_returns_the_contract_does_not_satisfy_the_port() -> None:
    # How non-conformance is caught here: the annotated assignment is the check, and mypy is what
    # runs it. The `type: ignore` is therefore the assertion -- without it this line fails the type
    # check that gates this repo. The runtime assertions below guard it from being vacuous, and
    # spell out why a runtime-checkable Protocol would have blessed this class instead.
    leaky: SurfaceProvider = LeakyProvider()  # type: ignore[assignment]

    assert hasattr(leaky, "latest")  # the member name is right, which is all isinstance would see
    assert not isinstance(leaky.latest("BTC-DERIBIT"), SurfaceView)


def test_latest_takes_only_a_market_id() -> None:
    # No clock, no freshness policy, no portfolio, no producer to choose from. Anything else in
    # this list would make the provider judge the surface it is handing over, and Design 7.2's
    # decision would leave the layer that owns it.
    parameters = list(inspect.signature(SurfaceProvider.latest).parameters)

    assert parameters == ["self", "market_id"]


def test_latest_is_synchronous() -> None:
    # Deliberate: a provider is a cache, and reading it is a dictionary lookup. An `async def`
    # would invite an implementation to fetch on demand, which would make a report's latency
    # depend on a producer's availability.
    assert not inspect.iscoroutinefunction(SurfaceProvider.latest)


# --- ReportWriter


def test_a_writer_receives_the_finished_report_through_the_port() -> None:
    recorder = RecordingWriter()
    writer: ReportWriter = recorder
    report = make_report()

    publish(writer, report)

    assert recorder.written == [report]


def test_a_rejected_report_reaches_the_writer_like_any_other() -> None:
    """The loudest thing this context ever says (Design 7.2). A writer that skipped it would turn
    "there is no valid surface" into silence, which an operator cannot tell from a dead process.
    """
    recorder = RecordingWriter()
    writer: ReportWriter = recorder
    refusal = make_report(
        ts_snapshot=None,
        freshness=FreshnessDecision.REJECT,
        positions=(),
        message="no valid surface",
    )

    publish(writer, refusal)

    written = recorder.written[0]
    assert written.freshness is FreshnessDecision.REJECT
    assert written.positions == ()


def test_writing_a_report_returns_nothing() -> None:
    # A writer that returned rendered text, a row count or a status would tempt a use case into
    # inspecting it, and the first thing anyone would inspect is the formatting this port exists
    # to stay ignorant of. Asserted on the declared type rather than on a call, because mypy
    # refuses to let a `-> None` result be compared at all -- which is the guarantee itself.
    assert typing.get_type_hints(ReportWriter.write)["return"] is type(None)


def test_the_writer_port_declares_one_method_and_no_formatting_vocabulary() -> None:
    # Every method this protocol does not have is a formatting decision that stayed outside the
    # domain, which is what makes a console writer, a CSV writer and the comparative report of
    # F3-E substitutable without the domain learning what a column is.
    members = sorted(name for name in vars(ReportWriter) if not name.startswith("_"))

    assert members == ["write"]


def test_the_writer_port_does_not_manage_its_own_lifetime() -> None:
    # Market Data's `MarketDataProvider` does declare a `close`; this one deliberately does not. A
    # writer holding a file handle has a lifetime, and it is managed where the handle is opened.
    assert not hasattr(ReportWriter, "close")


def test_write_is_synchronous() -> None:
    # The one port here that really does I/O, and still synchronous: a report is written once per
    # snapshot, and a writer that blocks is handed to the thread pool of ADR-005. `async` would put
    # asyncio in the signature of every implementation and pull a runtime into the domain.
    assert not inspect.iscoroutinefunction(ReportWriter.write)


# --- Clock


@pytest.mark.parametrize(
    "clock",
    [SystemClock(), ManualClock(NOW), SimulatedClock(NOW)],
    ids=["system", "manual", "simulated"],
)
def test_every_platform_clock_satisfies_the_risk_clock_port(clock: Clock) -> None:
    now = stamp(clock)

    assert now.tzinfo is not None


def test_time_moves_for_the_freshness_comparison_only_through_the_port() -> None:
    """ADR-004's promise ends here: a recorded session must produce the same final risk report. A
    module reading ``datetime.now()`` would make a ``DEGRADED`` label depend on when the replay was
    run rather than on what was recorded. Driving the clock by hand is what replaces the wait.
    """
    manual = ManualClock(NOW)
    clock: Clock = manual

    before = stamp(clock)
    manual.advance(45.0)
    after = stamp(clock)

    assert before == NOW
    assert after - before == timedelta(seconds=45)


def test_the_risk_clock_port_does_not_ask_for_sleep() -> None:
    # A report is computed on demand or when a surface arrives; nothing here polls. `sleep` --
    # which Market Data's Clock does declare -- would be a method no code path in this context
    # ever calls, and every implementation wired in would have to grow it for nobody.
    assert not hasattr(Clock, "sleep")


# --- MetricsSink


def test_null_metrics_sink_satisfies_the_risk_metrics_port() -> None:
    metrics: MetricsSink = NullMetricsSink()

    # A null sink records nothing anywhere; not raising is the behaviour.
    record_report(metrics)


def test_logging_metrics_sink_emits_every_measurement_made_through_the_port(
    caplog: pytest.LogCaptureFixture,
) -> None:
    metrics: MetricsSink = LoggingMetricsSink(logging.getLogger("test.risk.metrics"))

    with caplog.at_level(logging.INFO, logger="test.risk.metrics"):
        record_report(metrics)

    assert len(caplog.records) == 4


# --- the module itself


def test_the_ports_module_does_not_import_asyncio() -> None:
    # Import rule 3: */domain/** never imports asyncio. Nothing here needs a runtime, which is
    # what keeps the domain independent of how the composition root schedules anything.
    # import-linter enforces this repo-wide in F1-09.
    source = inspect.getsource(ports)

    assert re.search(r"^\s*(import asyncio|from asyncio\b)", source, re.MULTILINE) is None


def test_the_ports_module_does_not_import_the_contracts() -> None:
    # Import rule 3 again, and the half this context is most exposed to: Risk is the only pure
    # consumer in the engine, so it is the context most tempted to read the published language
    # directly. A `latest` typed in DTOs would leave the ACL with nothing to translate.
    source = inspect.getsource(ports)

    assert re.search(r"^\s*from volengine\.contracts\b", source, re.MULTILINE) is None


def test_the_ports_module_does_not_import_another_context() -> None:
    # Import rule 6. The producers are what this file most obviously *could* have imported, and
    # the fact that it does not is Design 7.1's inversion in one line of grep.
    source = inspect.getsource(ports)

    assert (
        re.search(
            r"^\s*from volengine\.(market_data|parametric_pricing|neural_surface)\b",
            source,
            re.MULTILINE,
        )
        is None
    )


@pytest.mark.parametrize("port", [SurfaceProvider, ReportWriter, Clock, MetricsSink])
def test_the_ports_are_not_runtime_checkable(port: type) -> None:
    # Deliberate: isinstance against a runtime-checkable Protocol compares member *names* only, so
    # it would bless the LeakyProvider above -- a `latest` handing the domain the published
    # contract, which is the exact leak the ACL exists to prevent. Conformance is verified
    # statically, and by the calls made through the ports above.
    with pytest.raises(TypeError, match="runtime_checkable"):
        isinstance(object(), port)
