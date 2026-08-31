"""The known-truth test of F2: a surface invented, quoted badly, and recovered through the engine.

Every other test of the calibrator hands it volatilities. This one hands the *engine* premiums --
Black-76 prices of a surface whose five parameters were chosen before the run started -- and asks
whether the five come back out the far end. In between sit the whole of Market Data and the ACL
that inverts each mid against our own forward and our own tenor: a sign slip in the moneyness axis,
a tenor read off the wrong calendar or a forward taken from the wrong side of the book all change
the answer, and none of them is visible to a calibrator test that starts from volatilities somebody
already computed.

**The parameters are asserted, not only the residual.** A fit can have a small RMSE and still be
the wrong surface -- that is the trap the plan names for this stage, and it is not hypothetical:
an optimiser is free to trade level against skew, reproduce the quotes it was given, and
extrapolate nowhere near the truth. So there is an assertion per parameter, and a guard showing
that the tolerances are narrow enough to reject a neighbouring slice of the same generated surface.

Those parameters do not survive publication: ``CalibratedSurface`` carries a table of volatilities
and no parameters at all (ADR-001). ``RecordingCalibrator`` is therefore wrapped around the real
``ScipyCalibrator`` *inside* the running pipeline, where they still exist -- the fit is the
engine's own, not one this test performed on the side.

**The feed is spoiled on purpose**, which is what separates this from
``test_synthetic_vertical.py``. There the quotes are exact and the tolerance is about the wiring;
here there is volatility noise on every mid and one quote in twenty comes out as junk of a named
kind, so the recovery has to happen *through* the conditions the admissibility rules exist for.
"""

from __future__ import annotations

import math
from datetime import UTC, datetime, timedelta

import pytest

from tests.entrypoints.builders import (
    UNDERLYING,
    WRITER_NAME,
    RecordingCalibrator,
    RecordingWriter,
    make_app_config,
    make_calibration_config,
    make_market_config,
    make_risk_config,
)
from tests.support import RecordingMetrics
from volengine.entrypoints.config import SYNTHETIC_PROVIDER, SyntheticSettings
from volengine.entrypoints.pipeline import Adapters, Pipeline, build_pipeline, default_adapters
from volengine.market_data.adapters.synthetic import (
    SVIParamsSpec,
    SyntheticConfig,
    SyntheticProvider,
)
from volengine.parametric_pricing.adapters.scipy_calibrator import PRODUCER_ID, ScipyCalibrator
from volengine.parametric_pricing.domain.calibration import SliceResult
from volengine.platform.bus import InProcessConflatingBus
from volengine.platform.clock import SystemClock
from volengine.risk.domain.portfolio import Position
from volengine.risk.domain.pricing import OptionKindR
from volengine.risk.domain.risk_report import RiskReport

pytestmark = pytest.mark.e2e

TENOR = timedelta(days=30)
TRUE = SVIParamsSpec(a=0.020, b=0.050, rho=-0.30, m=0.0, sigma=0.20)
"""The generating slice: the near expiry of the feed's own default surface."""

NEIGHBOUR = SVIParamsSpec(a=0.055, b=0.090, rho=-0.25, m=0.0, sigma=0.25)
"""Its three-month sibling on the same surface -- the guard's counter-example, and a real one:
recovering *a* plausible crypto slice is not the same as recovering the one that was quoted."""

FORWARD = 60_000.0
"""The feed's own initial forward, restated because the book below is struck on it: with no
forward walk configured, ``k = ln(K / F)`` is exactly zero for the whole session."""

VOL_NOISE_BP = 20.0
"""Two tenths of a volatility point on every mid, the order of a real bid-ask on a liquid crypto
wing. The number the recovery has to survive, and the one no tolerance here may be tuned against."""

JUNK_RATE = 0.05
"""One quote in twenty replaced by junk of a named kind: crossed, absurdly wide, one-sided,
sizeless or an hour old. There to make the admissibility rules of Design 4.3 actually fire."""

TOLERANCES = {"a": 4e-3, "b": 6e-3, "rho": 0.08, "m": 0.02, "sigma": 0.04}
"""How far each recovered parameter may sit from the truth.

Roughly four times what the fit actually misses by, which leaves room for a machine and a scipy
release to differ without leaving room for a *different surface* -- the guard below misses three
of the five by an order of magnitude.
"""


def spoiled_market() -> SyntheticConfig:
    """One expiry, twenty-one strikes, noise on every mid and junk scattered through it.

    Twenty-one because five parameters over a ladder that size are over-determined rather than
    interpolated. The forward is pinned: the subject here is the recovery, and a walking forward
    would add a second reason for the last cycle to differ from the first.
    """
    return SyntheticConfig(
        expiries=(TENOR,),
        true_params={TENOR: TRUE},
        strikes_per_expiry=21,
        log_moneyness_range=(-0.6, 0.6),
        vol_noise_bp=VOL_NOISE_BP,
        junk_quote_rate=JUNK_RATE,
        forward_move_rel=0.0,
        cycles=3,
        interval_seconds=0.05,
    )


def quoted_expiry(start: datetime) -> datetime:
    """The instant the feed will place its only expiry on, asked of the feed itself.

    Built from the same conventions and the same pinned origin as the provider the pipeline
    builds, so the book below names the expiry the market actually quotes rather than a date
    restated here -- which would be this test computing the venue's expiry hour on its own.
    """
    provider = SyntheticProvider(make_market_config().conventions, spoiled_market(), start)
    return provider.instruments[0].expiry


def session(start: datetime) -> tuple[RecordingCalibrator, RecordingWriter, Pipeline]:
    """One session of the real pipeline over the spoiled feed, from quotes to a written report.

    ``start`` is pinned by the caller so that the expiry the feed lists is knowable before the
    pipeline builds its own provider: the book has to name that same instant, and a feed reading
    the wall clock twice would place it twice.
    """
    calibrator = RecordingCalibrator(ScipyCalibrator())
    writer = RecordingWriter()
    metrics = RecordingMetrics()
    config = make_app_config(
        markets=(
            make_market_config(
                provider=SYNTHETIC_PROVIDER,
                cadence_seconds=0.02,
                synthetic=SyntheticSettings(config=spoiled_market(), start=start),
            ),
        ),
        calibration=make_calibration_config(calibrators=(PRODUCER_ID,)),
        risk=make_risk_config(
            positions=(
                Position(
                    underlying=UNDERLYING,
                    expiry=quoted_expiry(start),
                    strike=FORWARD,
                    kind=OptionKindR.CALL,
                    quantity=1.0,
                ),
            )
        ),
    )
    adapters = Adapters(
        providers=default_adapters().providers,
        calibrators={PRODUCER_ID: lambda _config: calibrator},
        writers={WRITER_NAME: lambda _config: writer},
    )
    pipeline = build_pipeline(
        config, adapters, SystemClock(), InProcessConflatingBus(metrics), metrics
    )
    return calibrator, writer, pipeline


async def run_a_session() -> tuple[SliceResult, RiskReport]:
    """Run to the end of the feed and return the last fit and the last report.

    The *last* of each, because the first snapshot of a session goes out on the first quote and
    rests on a single instrument; only once the whole ladder has been through the chain does a
    five-parameter fit have a surface to find.
    """
    start = datetime.now(UTC)
    calibrator, writer, pipeline = session(start)

    await pipeline.run()

    return calibrator.results[-1].slices[0], writer.reports[-1]


async def test_the_engine_recovers_the_slice_the_feed_was_generated_from() -> None:
    """Five parameters in, five parameters out, across every hop that could have bent them.

    One assertion per parameter rather than one over the tuple: the level, the wings, the skew,
    the position of the minimum and the curvature fail for five different reasons, and a single
    comparison would report "the surface is wrong" and leave the reader to work out which.
    """
    fitted, _ = await run_a_session()

    assert fitted.params.a == pytest.approx(TRUE.a, abs=TOLERANCES["a"])
    assert fitted.params.b == pytest.approx(TRUE.b, abs=TOLERANCES["b"])
    assert fitted.params.rho == pytest.approx(TRUE.rho, abs=TOLERANCES["rho"])
    assert fitted.params.m == pytest.approx(TRUE.m, abs=TOLERANCES["m"])
    assert fitted.params.sigma == pytest.approx(TRUE.sigma, abs=TOLERANCES["sigma"])


def test_the_neighbouring_slice_of_the_same_surface_would_have_missed_those_bands() -> None:
    """The guard on the tolerances: they admit one surface, not any plausible one.

    The three-month slice of the generator's own default surface is a perfectly reasonable crypto
    smile, and it is not this one. If the bands above were loose enough to accept it, the test
    would be asserting that a fit converged rather than that it converged on the truth.
    """
    missed = [
        name
        for name in TOLERANCES
        if abs(getattr(NEIGHBOUR, name) - getattr(TRUE, name)) > TOLERANCES[name]
    ]

    assert sorted(missed) == ["a", "b", "sigma"]


async def test_the_recovery_happened_through_the_noise_rather_than_around_it() -> None:
    """The guard on the run itself: the quotes really were spoiled.

    A residual of the order of the noise is what a fit of noisy data looks like. A number near
    zero would mean the feed had quietly gone clean -- a configuration mistake that would leave
    every assertion above passing for a reason nobody intended.
    """
    fitted, _ = await run_a_session()

    assert VOL_NOISE_BP / 4.0 < fitted.rmse_vol_bp < VOL_NOISE_BP * 3.0


async def test_the_report_at_the_far_end_carries_the_volatility_of_that_surface() -> None:
    """And the recovered surface survives publication, translation and interpolation.

    The parameters above are read between two ACLs, where they still exist. This is the assertion
    that what came out the other side -- a table of volatilities, a ``SurfaceView``, a bilinear
    lookup at one strike -- is still the same market. At ``k = 0`` and ``m = 0`` the generating
    slice's total variance is ``a + b * sigma``, which is the one closed form this needs and the
    reason the position is struck on the forward.
    """
    _, report = await run_a_session()

    tenor_years = (report.positions[0].position.expiry - report.ts_report).total_seconds() / (
        365 * 86_400
    )
    assert report.positions[0].vol == pytest.approx(
        math.sqrt(TRUE.total_variance(0.0) / tenor_years), abs=5e-3
    )


def test_a_position_off_the_generating_surface_would_have_moved_that_number() -> None:
    """The guard on the assertion above: five thousandths of a vol point is a real bound.

    The same closed form on the tenor the generator does *not* use has to disagree by more than
    the tolerance, or the report assertion would be satisfied by any surface of roughly the right
    shape.
    """
    thirty_days = math.sqrt(TRUE.total_variance(0.0) / (30 / 365))
    ninety_days = math.sqrt(NEIGHBOUR.total_variance(0.0) / (90 / 365))

    assert abs(thirty_days - ninety_days) > 5e-3
