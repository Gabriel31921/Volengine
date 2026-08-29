"""A whole engine's configuration, and the fakes the composition root is exercised against.

The composition root is the one module whose subject *is* the other four contexts, so these
builders import from all of them -- which nothing in ``src/`` may do and ``tests/`` may, because
``tests/`` is subject to none of the import rules. The stubs come from the contexts that own the
ports: ``StubProvider`` from Market Data, ``StubCalibrator`` from Parametric Pricing. Writing a
second pair here would be two more objects to keep in step with two protocols.

The one fake that is genuinely this package's is :class:`RecordingWriter`: ``ReportWriter`` has no
implementation anywhere yet -- ``ConsoleReportWriter`` is F1-08 -- and a report that was written
is the only observable end of the whole pipeline.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Mapping, Sequence
from datetime import datetime, time
from pathlib import Path
from time import sleep as thread_sleep

from tests.market_data.builders import FORWARD, NEAR, make_instrument, make_update
from tests.parametric_pricing.builders import StubCalibrator, make_grid_spec, make_weighting
from volengine.entrypoints.config import (
    AppConfig,
    CalibrationConfig,
    MarketConfig,
    RiskConfig,
)
from volengine.entrypoints.pipeline import Adapters, CalibratorFactory
from volengine.market_data.domain.admissibility import AdmissibilityThresholds
from volengine.market_data.domain.market_conventions import (
    DayCount,
    ForwardMethod,
    MarketConventions,
    Numeraire,
)
from volengine.market_data.domain.option_quote import InstrumentId, OptionKindD, QuoteUpdate
from volengine.market_data.domain.ports import MarketDataProvider
from volengine.market_data.domain.snapshot_policy import SnapshotPolicyConfig
from volengine.parametric_pricing.application.calibrate_on_snapshot import Acceptance
from volengine.parametric_pricing.domain.calibration import CalibrationResult, CalibrationTask
from volengine.parametric_pricing.domain.ports import Calibrator
from volengine.parametric_pricing.domain.svi_slice import SVIParams
from volengine.risk.application.compute_report import ReportSettings
from volengine.risk.domain.freshness_policy import FreshnessPolicy
from volengine.risk.domain.portfolio import Portfolio, Position
from volengine.risk.domain.ports import ReportWriter
from volengine.risk.domain.pricing import OptionKindR
from volengine.risk.domain.risk_report import RiskReport
from volengine.risk.domain.valuation import BumpSpec

PROVIDER_NAME = "stub"
CALIBRATOR_NAME = "svi-stub"
WRITER_NAME = "recording"
MARKET_ID = "BTC-DERIBIT"
UNDERLYING = "BTC"
"""What the default market quotes, and what the default book is written on. The composition root
splits one by the other, so a test about that split changes exactly one of them."""


def make_market_config(
    market_id: str = MARKET_ID,
    provider: str = PROVIDER_NAME,
    cadence_seconds: float = 1.0,
    material_move_threshold: float = 0.0,
    max_quiet_seconds: float | None = None,
    underlying: str = UNDERLYING,
) -> MarketConfig:
    """One market whose policy publishes on every cadence tick and never marks anything degraded.

    Permissive on purpose: a test about the *graph* should not have to reason about whether a
    movement filter let the second snapshot through. The heartbeat is off unless a test asks for
    it, because under a ``ManualClock`` the timer is what makes time move at all.
    """
    return MarketConfig(
        conventions=MarketConventions(
            market_id=market_id,
            underlying=underlying,
            day_count=DayCount.ACT_365F,
            expiry_time_utc=time(8, 0),
            numeraire=Numeraire.INVERSE,
            forward_method=ForwardMethod.PROVIDER_UNDERLYING,
        ),
        admissibility=AdmissibilityThresholds(
            max_spread_rel=0.5,
            max_age_seconds=5.0,
            moneyness_range=(-1.5, 1.5),
            max_iv_divergence_bp=500.0,
            convexity_tolerance=0.0005,
            min_size=1.0,
        ),
        snapshot=SnapshotPolicyConfig(
            cadence_seconds=cadence_seconds,
            material_move_threshold=material_move_threshold,
            min_coverage_ratio=0.0,
            max_quiet_seconds=max_quiet_seconds,
        ),
        provider=provider,
        max_skew_seconds=30.0,
    )


def make_calibration_config(
    calibrators: tuple[str, ...] = (CALIBRATOR_NAME,),
    max_rmse_vol_bp: float = 50.0,
) -> CalibrationConfig:
    """A mesh wider than the quoted band and a threshold the stub's fits comfortably clear."""
    return CalibrationConfig(
        calibrators=calibrators,
        grid=make_grid_spec(),
        weighting=make_weighting(),
        acceptance=Acceptance(max_rmse_vol_bp=max_rmse_vol_bp),
    )


def make_position(underlying: str = UNDERLYING) -> Position:
    """One at-the-money call on the near expiry, which the stub surface can always value."""
    return Position(
        underlying=underlying,
        expiry=NEAR,
        strike=FORWARD,
        kind=OptionKindR.CALL,
        quantity=1.0,
    )


def make_risk_config(
    writer: str = WRITER_NAME,
    positions: tuple[Position, ...] | None = None,
) -> RiskConfig:
    """One at-the-money call on the near expiry, valued under a policy nothing here trips."""
    return RiskConfig(
        portfolio=Portfolio(
            positions=(make_position(),) if positions is None else positions,
        ),
        freshness=FreshnessPolicy(warn_seconds=30.0, reject_seconds=120.0),
        settings=ReportSettings(bumps=BumpSpec(forward_rel=0.01, vol_abs=0.01)),
        writer=writer,
    )


def make_app_config(
    markets: tuple[MarketConfig, ...] | None = None,
    calibration: CalibrationConfig | None = None,
    risk: RiskConfig | None = None,
) -> AppConfig:
    return AppConfig(
        markets=(make_market_config(),) if markets is None else markets,
        calibration=make_calibration_config() if calibration is None else calibration,
        risk=make_risk_config() if risk is None else risk,
    )


class SlowCalibrator(StubCalibrator):
    """A ``StubCalibrator`` that takes longer than one hop of the event loop to answer.

    The fake that keeps the end-to-end tests honest. Every use case in this engine is synchronous
    and the composition root pushes the fit onto a thread pool (ADR-005), so a stub that returns
    inside the same event-loop iteration lets a run finish *before* the pipeline has to decide
    whether to wait for work in flight -- and a test built on one asserts that the scheduler was
    lucky rather than that the graph is right. A real fit takes milliseconds to seconds; this
    takes a real millisecond, on the pool, which is enough for the loop to come back and find the
    stream already exhausted.

    ``time.sleep`` and not ``asyncio.sleep``: this runs on the executor thread, where there is no
    event loop to await, which is exactly the situation a real calibrator is in.
    """

    def __init__(self, seconds: float = 0.01, producer_id: str = "svi-stub") -> None:
        super().__init__(producer_id=producer_id)
        self._seconds = seconds

    def calibrate(
        self,
        previous: Mapping[datetime, SVIParams] | None,
        task: CalibrationTask,
    ) -> CalibrationResult:
        thread_sleep(self._seconds)
        return super().calibrate(previous, task)


class RecordingWriter:
    """A ``ReportWriter`` that keeps every report it was handed, rejected ones included."""

    def __init__(self) -> None:
        self.reports: list[RiskReport] = []

    def write(self, report: RiskReport) -> None:
        self.reports.append(report)


class BlockingProvider:
    """A provider that replays a script and then goes quiet **without ending its stream**.

    The difference from ``StubProvider`` is the whole point: a stream that ends lets the pipeline
    finish, while a feed that has stopped keeps the task alive with nothing coming out of it. That
    second shape is the one ``max_quiet_seconds`` exists for, so it is the only shape in which the
    heartbeat can be observed.
    """

    def __init__(
        self,
        updates: Sequence[QuoteUpdate],
        instruments: Sequence[InstrumentId] = (),
    ) -> None:
        self._updates = tuple(updates)
        self._instruments = tuple(instruments)
        self.closed = False

    async def discover(self) -> tuple[InstrumentId, ...]:
        return self._instruments

    def stream(self) -> AsyncIterator[QuoteUpdate]:
        return self._stream()

    async def _stream(self) -> AsyncIterator[QuoteUpdate]:
        for update in self._updates:
            yield update
        # Never resolves: only cancellation ends this task, which is what a live feed's silence
        # looks like from inside the loop.
        await asyncio.Event().wait()

    async def close(self) -> None:
        self.closed = True


def two_sided(strike: float = FORWARD) -> list[QuoteUpdate]:
    """One two-sided strike, which is the least that produces a publishable slice."""
    return [make_update(strike=strike, kind=kind) for kind in (OptionKindD.CALL, OptionKindD.PUT)]


def one_instrument() -> list[InstrumentId]:
    return [make_instrument()]


def make_adapters(
    provider: MarketDataProvider,
    calibrators: dict[str, Calibrator],
    writer: ReportWriter,
    provider_name: str = PROVIDER_NAME,
    writer_name: str = WRITER_NAME,
) -> Adapters:
    """The registry, filled with objects the test already holds.

    The factories ignore their argument because the objects are built by the test rather than from
    the configuration -- which is exactly the substitution ``build_pipeline`` taking a registry
    was for.
    """
    return Adapters(
        providers={provider_name: lambda _config: provider},
        calibrators={name: _always(held) for name, held in calibrators.items()},
        writers={writer_name: lambda _config: writer},
    )


def _always(calibrator: Calibrator) -> CalibratorFactory:
    """A factory over an object the test already built.

    A named closure rather than a lambda with a default argument, because a lambda capturing the
    loop variable would hand every registered name the *last* calibrator -- which is exactly the
    bug the two-producer test exists to catch, and it would catch it in the builder instead.
    """

    def build(_config: CalibrationConfig) -> Calibrator:
        return calibrator

    return build


CONFIG_TOML = """
[[market]]
id = "BTC-DERIBIT"
underlying = "BTC"
provider = "constant"
max_skew_seconds = 30.0

[market.conventions]
day_count = "ACT/365F"
expiry_time_utc = 08:00:00
numeraire = "INVERSE"
forward_method = "PROVIDER_UNDERLYING"

[market.admissibility]
max_spread_rel = 0.5
max_age_seconds = 5.0
moneyness_range = [-1.5, 1.5]
max_iv_divergence_bp = 500.0
convexity_tolerance = 0.0005
min_size = 1.0

[market.snapshot]
cadence_seconds = 1.0
material_move_threshold = 0.002
min_coverage_ratio = 0.6
max_quiet_seconds = 30.0

[calibration]
calibrators = ["svi-scipy"]

[calibration.grid]
k_min = -0.4
k_max = 0.4
n_nodes = 9

[calibration.weighting]
spread_scale = 0.05
flagged_factor = 0.25
unpaired_itm_factor = 0.1

[calibration.acceptance]
max_rmse_vol_bp = 50.0

[risk]
writer = "console"

[risk.freshness]
warn_seconds = 30.0
reject_seconds = 120.0

[risk.report]
discount = 1.0

[risk.report.bumps]
forward_rel = 0.01
vol_abs = 0.01

[[risk.position]]
underlying = "BTC"
expiry = 2026-08-27T08:00:00Z
strike = 60000.0
kind = "CALL"
quantity = 1.0
"""
"""A complete, valid file. Every rejection test starts from this text and breaks one line."""


def write_config(directory: Path, text: str = CONFIG_TOML) -> Path:
    path = directory / "volengine.toml"
    path.write_text(text, encoding="utf-8")
    return path


def without(text: str, line_start: str) -> str:
    """The same file with one key removed, so a test names the key it is about and nothing else."""
    kept = [line for line in text.splitlines() if not line.startswith(line_start)]
    return "\n".join(kept)


def replacing(text: str, line_start: str, replacement: str) -> str:
    """The same file with one key given a different value."""
    return "\n".join(
        replacement if line.startswith(line_start) else line for line in text.splitlines()
    )
