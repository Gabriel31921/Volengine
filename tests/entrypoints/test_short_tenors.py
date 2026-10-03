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
4. **Both SVI calibrators fit short slices** -- since F3-W1. F3-F found the scipy one could not:
   not a daycount fault -- item 1 shows the volatilities arriving right -- but its cold start and
   ridge were in absolute total variance, sized for a week and more, and a one-day slice has a
   ninth of a week's. The fit came back "converged" thousands of basis points off, the acceptance
   rule refused the slice, and the surface went out ``DEGRADED`` without it -- which on a live
   Deribit chain, where the nearest expiry is a daily nearly all the time, would have been every
   surface. Both calibrators now shrink their start and ridge with the slice's own variance below
   a week's (``scipy_calibrator.REFERENCE_TOTAL_VARIANCE``), and the frontier F3-F measured --
   twenty hours, three days with skew, a week -- is asserted here on both, with a guard showing
   the twenty-hour slice still fails without the shrink.
"""

from __future__ import annotations

import asyncio
import math
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from importlib.util import find_spec

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
from volengine.parametric_pricing.adapters import scipy_calibrator
from volengine.parametric_pricing.adapters.scipy_calibrator import ScipyCalibrator
from volengine.parametric_pricing.application.acl import to_calibration_task
from volengine.parametric_pricing.application.calibrate_on_snapshot import CalibrateOnSnapshot
from volengine.parametric_pricing.application.calibration_state import CalibrationState
from volengine.parametric_pricing.domain.calibration import CalibrationTask
from volengine.parametric_pricing.domain.ports import Calibrator
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


def short_dated_snapshot(
    vol_noise_bp: float = 0.0,
    before: timedelta = EXPIRY - SESSION_START,
    shape: SVIParamsSpec | None = None,
    half_band: float = 0.05,
) -> MarketSnapshot:
    """A synthetic chain ``before`` a daily expiry -- twenty hours unless told otherwise -- beside a
    monthly, frozen one second in.

    The near slice's SVI is flat at ``FLAT_VOL`` in total variance at the session's start unless a
    ``shape`` is given, so by default the volatility it implies at any later instant is
    ``sqrt(w / T(now))`` -- known exactly. Its strikes span ``half_band`` either side of the money.
    """
    start = EXPIRY - before
    monthly = timedelta(days=30)
    w = FLAT_VOL * FLAT_VOL * tenor_of(EXPIRY, start)
    config = SyntheticConfig(
        expiries=(before, monthly),
        true_params={
            before: SVIParamsSpec(a=w, b=0.0, rho=0.0, m=0.0, sigma=0.1)
            if shape is None
            else shape,
            monthly: SVIParamsSpec(a=0.02, b=0.05, rho=-0.3, m=0.0, sigma=0.2),
        },
        strikes_per_expiry=11,
        log_moneyness_range=(-half_band, half_band),
        forward0=FORWARD,
        cycles=1,
        interval_seconds=0.0,
        junk_quote_rate=0.0,
        vol_noise_bp=vol_noise_bp,
    )
    conventions = make_conventions()
    provider = SyntheticProvider(conventions, config, start)
    chain = QuoteChain(conventions, make_thresholds())

    async def feed() -> None:
        chain.set_live_instruments(await provider.discover())
        async for update in provider.stream():
            chain.apply(update)

    asyncio.run(feed())
    frozen = chain.snapshot(start + timedelta(seconds=1))
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


# --- 4. what the audit found one context over, fixed in F3-W1


HAS_JAX_EXTRA = all(find_spec(name) is not None for name in ("jax", "optax"))


def scipy() -> Calibrator:
    return ScipyCalibrator()


def jax() -> Calibrator:
    """The JAX calibrator on the test suite's shared shape and settings, and so its compilation.

    Imported here rather than at the top: the module imports ``jax``, an optional extra, and every
    other test in this file runs without it.
    """
    from tests.parametric_pricing.jax_builders import make_jax_calibrator

    return make_jax_calibrator()


CALIBRATORS = [
    pytest.param(scipy, id="svi-scipy"),
    pytest.param(
        jax,
        id="svi-jax",
        marks=pytest.mark.skipif(not HAS_JAX_EXTRA, reason="needs the jax extra"),
    ),
]
"""Both SVI producers, the second skipped where the extra is not installed."""


def task_of(snapshot: MarketSnapshot) -> CalibrationTask:
    task = to_calibration_task(snapshot, make_calibration_config().weighting)
    assert task is not None
    return task


def skewed(before: timedelta) -> SVIParamsSpec:
    """A 50% at-the-money smile with a lifted downside, its width scaled to the tenor.

    Half of the at-the-money variance in the level and half in the wings, ``rho = -0.5``, and a
    curvature that is ``0.2`` at a 50%-vol month and narrows like ``sqrt(w)`` below it -- the shape
    F3-F's audit found three-day slices failing on.
    """
    w = FLAT_VOL * FLAT_VOL * before.total_seconds() / SECONDS_PER_YEAR
    width = math.sqrt(w / 0.02)
    return SVIParamsSpec(a=0.5 * w, b=0.5 * w / (0.2 * width), rho=-0.5, m=0.0, sigma=0.2 * width)


def assert_fits_the_near_slice(calibrator: Calibrator, snapshot: MarketSnapshot) -> None:
    """Converged, unpinned and under the acceptance RMSE: what the use case would publish."""
    near = calibrator.calibrate(None, task_of(snapshot)).slices[0]

    assert near.expiry == EXPIRY
    assert near.converged
    assert not near.at_bound
    assert near.rmse_vol_bp < make_calibration_config().acceptance.max_rmse_vol_bp


@pytest.mark.parametrize("make", CALIBRATORS)
def test_the_calibrator_fits_a_one_day_slice(make: Callable[[], Calibrator]) -> None:
    """Twenty hours out, flat, at fifty basis points of noise -- the synthetic feed's default.

    F3-F's strict ``xfail`` until F3-W1: measured then at about 8,200 bp RMSE on the scipy
    calibrator, from twenty hours out down to ten minutes.
    """
    assert_fits_the_near_slice(make(), short_dated_snapshot(vol_noise_bp=NOISE_BP))


@pytest.mark.parametrize("make", CALIBRATORS)
def test_the_calibrator_fits_a_skewed_three_day_slice(make: Callable[[], Calibrator]) -> None:
    """Three days out, skewed, quoted across a band that widens with the tenor."""
    before = timedelta(days=3)
    snapshot = short_dated_snapshot(
        vol_noise_bp=NOISE_BP, before=before, shape=skewed(before), half_band=0.095
    )

    assert_fits_the_near_slice(make(), snapshot)


@pytest.mark.parametrize("make", CALIBRATORS)
def test_the_calibrator_fits_a_skewed_one_week_slice(make: Callable[[], Calibrator]) -> None:
    """A week out: the shortest tenor the absolute constants fitted, and the reference the shrink
    is measured from. Asserted so the change of units below it cannot cost the tenor above."""
    before = timedelta(days=7)
    snapshot = short_dated_snapshot(
        vol_noise_bp=NOISE_BP, before=before, shape=skewed(before), half_band=0.145
    )

    assert_fits_the_near_slice(make(), snapshot)


def test_without_the_shrink_the_one_day_slice_is_still_not_fitted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The guard on the tests above: the fix is the change of units, and nothing else.

    A reference variance far below any slice makes the shrink one everywhere, which is the
    calibrator as F3-F found it -- absolute start, absolute ridge -- and the same twenty-hour
    slice comes back thousands of basis points off again.
    """
    monkeypatch.setattr(scipy_calibrator, "REFERENCE_TOTAL_VARIANCE", 1e-12)

    near = ScipyCalibrator().calibrate(None, task_of(short_dated_snapshot(NOISE_BP))).slices[0]

    assert near.rmse_vol_bp > 1_000.0


@pytest.mark.parametrize("make", CALIBRATORS)
def test_the_calibrator_does_fit_the_monthly_beside_it(make: Callable[[], Calibrator]) -> None:
    """The guard on the tests above from the other side: the same task's thirty-day slice fits,
    so what is being measured is the tenor and not the chain, the task or the calibrator."""
    result = make().calibrate(None, task_of(short_dated_snapshot(vol_noise_bp=NOISE_BP)))

    monthly = result.slices[1]
    assert monthly.converged
    assert monthly.rmse_vol_bp < 50.0
