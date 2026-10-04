"""The two engines on one market, from the file F3-W2 ships: ``svi-scipy-vs-mlp-torch.toml``.

The configuration is the deliverable, as it was for ``svi-jax`` in F3-W1: ``mlp-torch`` reachable
from TOML, every collaborator ``TrainOnSnapshot`` needs under ``[neural]``, and the two producers
compared by Risk while they run. So the file itself is loaded, and the runs below swap only its far
ends -- a writer the test can read and a sink it can query.

Two runs are about real data rather than the invented market. The golden fixture is replayed with
the network beside the fit, which answers the stage's open question -- whether ``TorchFitSettings``'
defaults, argued on synthetic data only, publish on a real chain -- and it is replayed twice, which
answers the other: whether a seeded network training through the whole engine reproduces its
report byte for byte (ADR-004).

Everything that trains needs the ``neural`` extra and is skipped without it, except the refusal
that only exists without it, observed on every installation by hiding the extra from the lookup.
"""

from __future__ import annotations

import asyncio
from dataclasses import replace
from datetime import UTC, datetime
from importlib.util import find_spec
from pathlib import Path

import pytest

from tests.entrypoints.builders import (
    RecordingBus,
    RecordingWriter,
    SteadyClock,
    make_risk_config,
)
from tests.support import RecordingMetrics
from volengine.contracts.calibrated_surface import CalibratedSurface, SurfaceStatus
from volengine.contracts.events import CalibrationFailed, SurfaceCalibrated
from volengine.entrypoints import config as config_module
from volengine.entrypoints.config import (
    TORCH_LEARNER,
    AppConfig,
    ConfigError,
    MetricsSinkKind,
    load_config,
)
from volengine.entrypoints.pipeline import (
    Adapters,
    build_pipeline,
    default_adapters,
    with_replay,
)
from volengine.market_data.adapters.recorded import open_recording
from volengine.market_data.adapters.synthetic import SyntheticProvider
from volengine.neural_surface.domain.invariants import ArbitrageMesh
from volengine.parametric_pricing.adapters.scipy_calibrator import PRODUCER_ID as SCIPY_ID
from volengine.platform.bus import InProcessConflatingBus
from volengine.platform.clock import SimulatedClock, SystemClock
from volengine.risk.domain.portfolio import Position
from volengine.risk.domain.pricing import OptionKindR
from volengine.risk.domain.risk_report import RiskReport

ROOT = Path(__file__).resolve().parents[2]
EXAMPLE = ROOT / "examples" / "svi-scipy-vs-mlp-torch.toml"
"""The configuration shipped for F3-W2. Its own header holds the commands."""

VENUE = ROOT / "examples" / "deribit-live.toml"
FIXTURE = ROOT / "tests" / "fixtures" / "deribit-btc-2026-09-18.jsonl"

HAS_TORCH_EXTRA = find_spec("torch") is not None

needs_torch = pytest.mark.skipif(not HAS_TORCH_EXTRA, reason="needs the neural extra")

FIXTURE_MESH_TENORS = (0.019, 0.05, 0.1, 0.27)
"""The gate's tenors for the fixture: its two expiries, a week and fourteen weeks out on the day,
and two between. The example's own axis -- a month to a year -- is the synthetic market's; on the
fixture it trains the soft tier on tenors nobody quoted, and measured, the fit to what was quoted
goes from 64 bp to 180 bp. ``[neural]`` is one table for the engine, so the mesh is a statement
about the market the file runs (``docs/SEAMS.md``), and this replay states it for the venue's."""

FIXTURE_MAX_RMSE_VOL_BP = 200.0
"""The bound the network's fit to the fixture is held to: ``deribit-live.toml``'s own acceptance
RMSE for ``svi-scipy``, so both engines meet one standard on real data. Measured: 64 bp, against
the fit's 52 bp on the same snapshot.

**The gate's margin is the thinner one.** On this mesh the published surface measured a butterfly
depth of 6e-6 and a calendar crossing of 7e-5 against the example's 1e-4 lines -- inside, and by a
factor of 1.4 on the calendar side. A host whose float arithmetic trains the network along a
different trajectory could land outside it, and the failure would then be a refusal of this test,
not a crash; that is the soft tier being a preference (ADR-010), measured rather than hidden."""


def test_the_shipped_example_names_only_adapters_this_build_registers() -> None:
    """Read on every installation: it states no table whose type lives beside torch."""
    config = load_config(EXAMPLE)
    registry = default_adapters()

    assert list(config.calibration.calibrators) == [SCIPY_ID, TORCH_LEARNER]
    assert SCIPY_ID in registry.calibrators
    assert TORCH_LEARNER in registry.learners
    assert config.markets[0].provider in registry.providers
    assert config.risk.writer in registry.writers
    assert config.neural is not None
    assert config.metrics.sink is MetricsSinkKind.CSV


def test_the_shipped_example_without_the_extra_is_refused_with_the_remedy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """What a reader without ``uv sync --extra neural`` meets: the command, at start-up."""
    monkeypatch.setattr(
        config_module, "find_spec", lambda name: None if name == "torch" else find_spec(name)
    )
    writer = RecordingWriter()
    config, adapters = runnable(load_config(EXAMPLE), writer)

    with pytest.raises(ConfigError, match="uv sync --extra neural"):
        build_pipeline(
            config,
            adapters,
            SystemClock(),
            InProcessConflatingBus(RecordingMetrics()),
            RecordingMetrics(),
        )


def runnable(config: AppConfig, writer: RecordingWriter) -> tuple[AppConfig, Adapters]:
    """The example over eight seconds rather than twenty, with a book the feed really quotes.

    Eight half-second cycles leave room for the network's cold start -- seconds, on one CPU --
    and for a fine-tuning step after it. The writer is the only adapter swapped, so both producers
    come out of the real registry.
    """
    market = config.markets[0]
    assert market.synthetic is not None
    start = datetime.now(UTC)
    synthetic = replace(
        market.synthetic,
        config=replace(market.synthetic.config, interval_seconds=0.5, cycles=8),
        start=start,
    )
    market = replace(market, synthetic=synthetic)
    expiry = SyntheticProvider(market.conventions, synthetic.config, start).instruments[0].expiry
    position = Position(
        underlying=market.underlying,
        expiry=expiry,
        strike=synthetic.config.forward0,
        kind=OptionKindR.CALL,
        quantity=1.0,
    )
    adapters = replace(default_adapters(), writers={"recording": lambda _risk: writer})
    risk = make_risk_config(positions=(position,))
    return replace(config, markets=(market,), risk=risk), adapters


@needs_torch
@pytest.mark.e2e
async def test_both_engines_publish_and_are_compared_on_one_market() -> None:
    """Surfaces from the fit and the network reach Risk, and each arrival measures the distance."""
    writer, metrics = RecordingWriter(), RecordingMetrics()
    config, adapters = runnable(load_config(EXAMPLE), writer)
    synthetic = config.markets[0].synthetic
    assert synthetic is not None and synthetic.start is not None

    # Steady rather than `SystemClock`, so a host stepping its wall clock back mid-session cannot
    # silence the snapshots the network trains on (`SteadyClock` has the measurement).
    pipeline = build_pipeline(
        config,
        adapters,
        SteadyClock(synthetic.start),
        InProcessConflatingBus(metrics),
        metrics,
    )
    await pipeline.run()

    assert {report.producer_id for report in writer.reports} == {SCIPY_ID, TORCH_LEARNER}
    compared = [tags for name, _, tags in metrics.gauges if name.startswith("risk.comparison.")]
    assert compared
    assert {tags.get("challenger") for tags in compared} == {TORCH_LEARNER}


# --- the golden fixture, with the network beside the fit


def fixture_configuration() -> AppConfig:
    """``deribit-live.toml`` as ``volengine replay`` runs it, plus the network.

    The venue's file, with ``mlp-torch`` listed after the fit and the example's ``[neural]`` given
    a mesh over the fixture's own tenors (:data:`FIXTURE_MESH_TENORS`). Everything else -- the
    buffer, the gate, the schedule, the seed, and the learner's shipped defaults -- is the
    example's, unbent.
    """
    venue = load_config(VENUE)
    neural = load_config(EXAMPLE).neural
    assert neural is not None
    mesh = ArbitrageMesh(log_moneyness=neural.mesh.log_moneyness, tenors=FIXTURE_MESH_TENORS)
    return replace(
        venue,
        calibration=replace(venue.calibration, calibrators=(SCIPY_ID, TORCH_LEARNER)),
        neural=replace(neural, mesh=mesh),
    )


def replay_fixture() -> tuple[RecordingBus, RecordingWriter]:
    """One ``volengine replay`` of the fixture: the recorded clock, no timers, both producers."""
    recording = open_recording(FIXTURE)
    clock = SimulatedClock(recording.started_at)
    metrics = RecordingMetrics()
    bus = RecordingBus(metrics)
    writer = RecordingWriter()
    adapters = with_replay(default_adapters(), recording, clock)
    adapters = replace(adapters, writers={**adapters.writers, "console": lambda _risk: writer})
    pipeline = build_pipeline(fixture_configuration(), adapters, clock, bus, metrics, timers=False)
    asyncio.run(pipeline.run())
    return bus, writer


@pytest.fixture(scope="module")
def fixture_session() -> tuple[RecordingBus, RecordingWriter]:
    """One replay shared by the assertions below; the determinism test makes its own second."""
    return replay_fixture()


def network_surfaces(bus: RecordingBus) -> list[CalibratedSurface]:
    return [
        event.surface
        for event in bus.events_of(SurfaceCalibrated)
        if event.surface.producer_id == TORCH_LEARNER
    ]


@needs_torch
@pytest.mark.e2e
def test_the_network_publishes_an_accepted_surface_on_the_golden_fixture(
    fixture_session: tuple[RecordingBus, RecordingWriter],
) -> None:
    """The defaults argued on synthetic data hold on a real chain: the gate lets the fit out.

    Accepted means published as a fresh fit -- not refused, not a stale republish -- and within
    the venue file's acceptance RMSE (:data:`FIXTURE_MAX_RMSE_VOL_BP`).
    """
    bus, _ = fixture_session
    refused = [
        event for event in bus.events_of(CalibrationFailed) if event.producer_id == TORCH_LEARNER
    ]
    published = network_surfaces(bus)

    assert not refused
    assert published
    assert published[-1].status is SurfaceStatus.OK
    assert published[-1].fit.rmse_vol_bp < FIXTURE_MAX_RMSE_VOL_BP


@needs_torch
@pytest.mark.e2e
def test_the_network_and_the_fit_both_value_the_book_on_the_golden_fixture(
    fixture_session: tuple[RecordingBus, RecordingWriter],
) -> None:
    _, writer = fixture_session

    assert {report.producer_id for report in writer.reports} == {SCIPY_ID, TORCH_LEARNER}
    assert all(report.positions for report in writer.reports)


@needs_torch
@pytest.mark.e2e
def test_two_replays_of_the_fixture_with_one_seed_write_identical_reports(
    fixture_session: tuple[RecordingBus, RecordingWriter],
) -> None:
    """ADR-004 through the network: the same recording and seeds give the same bytes.

    Compared on the reports and on the network's published grids, the two ends that would show a
    nondeterministic training step. Both producers' reports are compared, so the fit's
    reproducibility under the replay is re-asserted beside the network's rather than assumed.
    """
    first_bus, first_writer = fixture_session
    second_bus, second_writer = replay_fixture()

    def reports(writer: RecordingWriter) -> list[RiskReport]:
        return sorted(writer.reports, key=lambda report: report.producer_id)

    assert reports(first_writer)
    assert reports(first_writer) == reports(second_writer)
    assert [s.grid for s in network_surfaces(first_bus)] == [
        s.grid for s in network_surfaces(second_bus)
    ]
