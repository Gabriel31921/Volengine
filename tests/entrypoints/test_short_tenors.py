"""The F3-F audit of risk 2 of Design §11: daycount and the 08:00 UTC expiry on short tenors.

"Tenor errors that are small but visible on short tenors if neglected." An audit, so this module is
organised by what was checked rather than by component, and each section says what it found.

1. **The convention reaches the calibrator intact.** A chain priced twenty hours before a Deribit
   08:00 expiry is inverted, at the calibrator's door, to the volatility it was generated at --
   which needs ACT/365F in seconds and the venue's hour of day to agree at every hop between the
   generator and the ACL. They do. The guard says how visible a mistake would be: the same
   premiums read against a midnight expiry miss the volatility by about a fifth.
2. **Risk reads the producing market's daycount without owning one.** ``SurfaceView.tenor_of``
   interpolates the grid's own (expiry, tenor) pairs, so at and between the nodes it reproduces
   ``MarketConventions.tenor_years`` exactly.
3. **Below the first node it does not, by the venue's clock skew.** Tenors are measured from the
   snapshot's ``ts_local``; Risk anchors tenor zero at ``ts_snapshot``, which is the venue's stamp
   (ADR-021). A position expiring before the nearest grid expiry -- the Deribit daily left over
   after the quoted one -- is therefore priced at a tenor off by up to the skew itself. Bounded,
   measured here, and left open: closing it moves either Market Data's tenor origin or Risk's
   extrapolation, and F3-F was the stage that proves neither context needs to change for a second
   market (``docs/SEAMS.md``).
4. **The scipy calibrator cannot fit a slice under about a day.** Not a daycount fault -- item 1
   shows the volatilities arriving right -- but where the audit led: the cold start and the
   ridge are in absolute total variance, sized for tenors of a week and more, and a one-day slice
   has a hundredth of that. The fit comes back "converged" tens of vol points off, the acceptance
   rule refuses the slice, and the surface goes out ``DEGRADED`` without it -- which on a live
   Deribit chain, where the nearest expiry is a daily nearly all the time, would be every surface.
   Pinned as a strict ``xfail``, so the fix has a test waiting for it.
"""

from __future__ import annotations

import asyncio
import math
from datetime import UTC, datetime, timedelta

import pytest

from tests.entrypoints.builders import make_calibration_config
from tests.market_data.builders import (
    FAR,
    FORWARD,
    NEAR,
    NOW,
    make_conventions,
    make_thresholds,
    make_update,
)
from tests.parametric_pricing.builders import StubCalibrator
from tests.support import RecordingMetrics
from volengine.contracts.events import SurfaceCalibrated
from volengine.contracts.market_snapshot import MarketSnapshot
from volengine.market_data.adapters.synthetic import (
    SVIParamsSpec,
    SyntheticConfig,
    SyntheticProvider,
)
from volengine.market_data.application.acl import build_snapshot_id, to_market_snapshot
from volengine.market_data.domain.option_quote import OptionKindD
from volengine.market_data.domain.quote_chain import QuoteChain
from volengine.market_data.domain.snapshot_policy import QualityAssessment
from volengine.parametric_pricing.adapters.scipy_calibrator import ScipyCalibrator
from volengine.parametric_pricing.application.acl import to_calibration_task
from volengine.parametric_pricing.application.calibrate_on_snapshot import CalibrateOnSnapshot
from volengine.parametric_pricing.application.calibration_state import CalibrationState
from volengine.platform.clock import ManualClock
from volengine.risk.application.acl import to_surface_view
from volengine.risk.domain.surface_view import SurfaceView
from volengine.shared_kernel.domain import black76

SECONDS_PER_YEAR = 365.0 * 86_400.0
"""ACT/365F, restated on purpose: the audit checks the code against the convention, not against
itself."""

EXPIRY = datetime(2026, 9, 19, 8, 0, tzinfo=UTC)
"""A Deribit daily: 08:00 UTC."""

SESSION_START = EXPIRY - timedelta(hours=20)
"""Twenty hours before it, the shortest tenor the venue lists for most of every day."""

FLAT_VOL = 0.5

NOISE_BP = 50.0
"""The synthetic feed's default volatility noise, for the two fits in section 4."""


def tenor_of(expiry: datetime, origin: datetime) -> float:
    return (expiry - origin).total_seconds() / SECONDS_PER_YEAR


# --- 1. the convention reaches the calibrator intact


def short_dated_snapshot(vol_noise_bp: float = 0.0) -> MarketSnapshot:
    """A synthetic chain twenty hours from a daily expiry, beside a monthly, frozen one second in.

    The daily's SVI is flat at ``FLAT_VOL`` in total variance at the session's start, so the
    volatility it implies at any later instant is ``sqrt(w / T(now))`` -- known exactly.
    """
    daily = EXPIRY - SESSION_START
    monthly = timedelta(days=30)
    w = FLAT_VOL * FLAT_VOL * tenor_of(EXPIRY, SESSION_START)
    config = SyntheticConfig(
        expiries=(daily, monthly),
        true_params={
            daily: SVIParamsSpec(a=w, b=0.0, rho=0.0, m=0.0, sigma=0.1),
            monthly: SVIParamsSpec(a=0.02, b=0.05, rho=-0.3, m=0.0, sigma=0.2),
        },
        strikes_per_expiry=11,
        log_moneyness_range=(-0.05, 0.05),
        forward0=FORWARD,
        cycles=1,
        interval_seconds=0.0,
        junk_quote_rate=0.0,
        vol_noise_bp=vol_noise_bp,
    )
    conventions = make_conventions()
    provider = SyntheticProvider(conventions, config, SESSION_START)
    chain = QuoteChain(conventions, make_thresholds())

    async def feed() -> None:
        chain.set_live_instruments(await provider.discover())
        async for update in provider.stream():
            chain.apply(update)

    asyncio.run(feed())
    frozen = chain.snapshot(SESSION_START + timedelta(seconds=1))
    return to_market_snapshot(
        snapshot=frozen,
        snapshot_id=build_snapshot_id(frozen.market_id, 0),
        quality=QualityAssessment(reasons=()),
        max_skew_seconds=30.0,
    )


def test_the_daily_is_published_at_its_0800_tenor_in_seconds() -> None:
    snapshot = short_dated_snapshot()
    daily = snapshot.slices[0]

    assert daily.expiry == EXPIRY
    assert daily.tenor_years == tenor_of(EXPIRY, snapshot.ts_local)


def test_a_one_day_slice_reaches_the_calibrator_at_the_volatility_it_was_priced_at() -> None:
    """Twenty hours out, every inverted volatility within a basis point of the generator's.

    One basis point is the second between the generator's stamp and the snapshot's instant, at a
    twenty-hour tenor, plus the inversion's own tolerance -- a daycount or hour-of-day mistake is
    thousands of times larger (the guard below).
    """
    snapshot = short_dated_snapshot()
    task = to_calibration_task(snapshot, make_calibration_config().weighting)
    assert task is not None
    daily = task.slices[0]

    assert daily.expiry == EXPIRY
    assert len(daily.implied_vol) >= 10
    for vol in daily.implied_vol:
        assert vol == pytest.approx(FLAT_VOL, abs=1e-4)


def test_a_midnight_expiry_would_miss_the_daily_by_a_fifth() -> None:
    """The guard on the test above: how visible the 08:00 is, at this tenor.

    The at-the-money premium the chain published, inverted against a tenor that ends at midnight
    instead of 08:00 -- twelve hours instead of twenty -- reads a volatility about 29% too high. So
    the one-basis-point agreement above could not survive an expiry hour applied wrongly anywhere
    between the symbol and the ACL.
    """
    snapshot = short_dated_snapshot()
    daily = snapshot.slices[0]
    at_the_money = min(daily.quotes, key=lambda quote: abs(math.log(quote.strike / daily.forward)))
    midnight = EXPIRY - timedelta(hours=8)

    misread = black76.implied_vol(
        at_the_money.mid,
        daily.forward,
        at_the_money.strike,
        tenor_of(midnight, snapshot.ts_local),
        at_the_money.kind.value == "C",
    )

    assert misread / FLAT_VOL - 1.0 > 0.2


# --- 2 and 3. Risk's tenor, read off the grid


SKEW = timedelta(seconds=1.5)
"""The venue's stamp trailing our receipt -- the median measured on the golden fixture."""


def published_view(skew: timedelta) -> tuple[SurfaceView, datetime]:
    """One surface through the real ACLs and use case, with the venue's stamp ``skew`` behind us.

    Two expiries, each quoted on both legs at the money, so the grid has two nodes. The snapshot is
    frozen at ``received`` -- its ``ts_local``, where its tenors are measured from -- while every
    quote carries ``ts_exchange = NOW``, which the snapshot publishes as its own instant and the
    surface carries on as ``ts_snapshot``. A stub fit, because what is audited is the tenor axis
    and not the smile.
    """
    chain = QuoteChain(make_conventions(), make_thresholds())
    for expiry in (NEAR, FAR):
        for kind in (OptionKindD.CALL, OptionKindD.PUT):
            chain.apply(make_update(expiry=expiry, kind=kind, ts_exchange=NOW))
    received = NOW + skew
    frozen = chain.snapshot(received)
    snapshot = to_market_snapshot(
        snapshot=frozen,
        snapshot_id=build_snapshot_id(frozen.market_id, 0),
        quality=QualityAssessment(reasons=()),
        max_skew_seconds=30.0,
    )
    calibration = make_calibration_config()
    fit = CalibrateOnSnapshot(
        calibrator=StubCalibrator(),
        state=CalibrationState(),
        clock=ManualClock(received),
        metrics=RecordingMetrics(),
        weighting=calibration.weighting,
        grid=calibration.grid,
        acceptance=calibration.acceptance,
    )
    (published,) = fit.handle(snapshot)
    assert isinstance(published, SurfaceCalibrated)
    surface = published.surface
    assert surface.grid.expiries == (NEAR, FAR)
    assert surface.ts_snapshot == NOW
    return to_surface_view(surface), received


def test_risk_reads_the_market_daycount_exactly_at_the_nodes() -> None:
    view, received = published_view(SKEW)

    assert view.tenor_of(NEAR) == make_conventions().tenor_years(NEAR, received)
    assert view.tenor_of(FAR) == make_conventions().tenor_years(FAR, received)


def test_risk_reads_the_market_daycount_between_the_nodes() -> None:
    """Exact up to rounding: ACT/365F is linear in calendar time, so is the interpolation."""
    view, received = published_view(SKEW)
    between = NEAR + timedelta(days=17, hours=3)

    assert view.tenor_of(between) == pytest.approx(
        make_conventions().tenor_years(between, received), rel=1e-12
    )


def test_below_the_first_node_the_tenor_is_off_by_at_most_the_venue_skew() -> None:
    """Finding 3, measured: the anchor is the venue's stamp, the tenors are from our receipt.

    For an expiry ``E`` before the first node ``E1``, Risk extrapolates the line through
    ``(ts_snapshot, 0)`` and ``(E1, T1)``; the market's own tenor is ``(E - ts_local) / year``. The
    difference is exactly ``skew * (E1 - E) / (E1 - ts_snapshot) / year`` -- positive, never more
    than the skew in years, and largest for the shortest expiries, where it matters most.
    """
    view, received = published_view(SKEW)
    daily = NOW + timedelta(days=1)
    assert daily < NEAR

    error = view.tenor_of(daily) - make_conventions().tenor_years(daily, received)
    expected = (
        SKEW.total_seconds()
        * (NEAR - daily).total_seconds()
        / (NEAR - NOW).total_seconds()
        / SECONDS_PER_YEAR
    )

    assert error == pytest.approx(expected, rel=1e-6, abs=1e-15)
    assert 0.0 < error <= SKEW.total_seconds() / SECONDS_PER_YEAR
    # In relative terms on a one-day option: about seventeen parts per million at the measured
    # median skew; at the sixty seconds `deribit-live.toml` tolerates it would be two-thirds of a
    # per mille, and on an hour-dated option the same skew is forty times worse.
    assert error / tenor_of(daily, received) < 2e-5


def test_with_no_skew_the_extrapolation_is_exact() -> None:
    """The guard on the test above: the error is the skew's and nothing else's."""
    view, received = published_view(timedelta(0))
    daily = NOW + timedelta(days=1)

    assert view.tenor_of(daily) == pytest.approx(
        make_conventions().tenor_years(daily, received), rel=1e-12
    )


# --- 4. what the audit found one context over


@pytest.mark.xfail(
    strict=True,
    reason=(
        "F3-F audit finding: ScipyCalibrator's cold start (B_START_MIN, SIGMA_START) and ridge are "
        "in absolute total variance and swamp a slice under about a day; docs/SEAMS.md"
    ),
)
def test_the_scipy_calibrator_fits_a_one_day_slice() -> None:
    """Fifty basis points of noise, the synthetic feed's default. Measured: the daily comes back
    "converged" at about 8,200 bp RMSE, from twenty hours out down to ten minutes; a slice three
    days out with any skew fails the same way, and a week out every shape tried fits."""
    task = to_calibration_task(
        short_dated_snapshot(vol_noise_bp=NOISE_BP), make_calibration_config().weighting
    )
    assert task is not None

    result = ScipyCalibrator().calibrate(None, task)

    daily = result.slices[0]
    assert daily.expiry == EXPIRY
    assert daily.converged
    assert daily.rmse_vol_bp < 50.0


def test_the_scipy_calibrator_does_fit_the_monthly_beside_it() -> None:
    """The guard on the xfail above: the same task's thirty-day slice fits, so what fails is the
    tenor and not the chain, the task or the calibrator wholesale."""
    task = to_calibration_task(
        short_dated_snapshot(vol_noise_bp=NOISE_BP), make_calibration_config().weighting
    )
    assert task is not None

    result = ScipyCalibrator().calibrate(None, task)

    monthly = result.slices[1]
    assert monthly.converged
    assert monthly.rmse_vol_bp < 50.0
