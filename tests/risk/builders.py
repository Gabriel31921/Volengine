"""Valid Risk objects, one per type, with a knob for every field a test bends.

The convention this repo tests by: a builder returns **one valid object**, and a test changes the
minimum needed to make its point -- either through a keyword here or through
``dataclasses.replace``, which re-runs ``__post_init__``. What a test says is then exactly what it
is probing, instead of five parameters of noise around one poisoned value.

Shared from a module rather than from ``conftest.py``: conftest is where pytest looks for fixtures
and hooks it *injects*, and importing from it is discouraged because it is loaded by collection
magic rather than by an import anyone can follow. See ``tests/support.py``.

The numbers below are chosen so the algebra is doable by hand. The forward at the three-month node
is exactly the default position's strike, so its log-moneyness is exactly ``0.0``; the expiries sit
at 30, 90 and 365 days after the snapshot, so under ACT/365 the tenor axis is exactly
``30/365, 90/365, 1.0`` and ``SurfaceView.tenor_of`` interpolates over a straight line a test can
verify with a division.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from volengine.contracts.calibrated_surface import (
    CalibratedSurface,
    FitMetrics,
    SurfaceStatus,
    VolGrid,
)
from volengine.risk.domain.freshness_policy import FreshnessDecision, FreshnessPolicy
from volengine.risk.domain.portfolio import Portfolio, Position
from volengine.risk.domain.pricing import OptionKindR
from volengine.risk.domain.risk_report import PositionRisk, RiskReport
from volengine.risk.domain.surface_view import SurfaceView
from volengine.risk.domain.valuation import BumpSpec

NAIVE = datetime(2026, 7, 27, 12, 0)
"""An instant with no zone. Every entry point in this context must refuse it."""

NOW = datetime(2026, 7, 27, 12, 0, tzinfo=UTC)
"""The instant the surface describes, and the origin of every tenor below."""

EXPIRIES = (NOW + timedelta(days=30), NOW + timedelta(days=90), NOW + timedelta(days=365))
TENORS = (30 / 365.0, 90 / 365.0, 1.0)
"""Three expiries and their year fractions under ACT/365F, exactly consistent with each other so
that a test can assert on ``tenor_of`` without importing a daycount."""

FORWARDS = (60_000.0, 60_400.0, 61_000.0)
"""A gently rising forward curve. The middle one is the default position's strike, which puts that
position exactly at the money forward and its log-moneyness at exactly zero."""

K_AXIS = (-0.20, -0.10, 0.0, 0.10, 0.20)
"""Roughly +-20% in log-forward-moneyness: wide enough that a test can step outside it."""


def smile_vol(tenor_years: float, k: float) -> float:
    """A plausible crypto smile: 65% at the money, decaying with tenor, wings curving up.

    Not what any test is about, but plausible numbers make a failure readable -- 0.65 is a
    believable BTC vol and 4.2 would send you hunting the wrong bug. The ``k**2`` term is what
    makes the surface genuinely skewed, which is what the vacuity guards in the greek tests lean
    on: against a flat surface the smile cannot enter a delta, so a test that only ever used one
    would prove nothing.
    """
    return 0.65 - 0.05 * tenor_years + 0.35 * k * k


def total_variance_grid(
    tenors: tuple[float, ...] = TENORS,
    log_moneyness: tuple[float, ...] = K_AXIS,
) -> tuple[tuple[float, ...], ...]:
    """``w = vol**2 * T`` at every node, indexed ``[tenor][k]`` like the published grid."""
    return tuple(tuple(smile_vol(tenor, k) ** 2 * tenor for k in log_moneyness) for tenor in tenors)


def make_view(
    market_id: str = "BTC-DERIBIT",
    producer_id: str = "svi-scipy",
    surface_id: str = "01JZQ0T4M2",
    ts_snapshot: datetime = NOW,
    log_moneyness: tuple[float, ...] = K_AXIS,
    tenors: tuple[float, ...] = TENORS,
    expiries: tuple[datetime, ...] = EXPIRIES,
    forwards: tuple[float, ...] = FORWARDS,
    total_variance: tuple[tuple[float, ...], ...] | None = None,
) -> SurfaceView:
    """The surface as Risk holds it: three tenors, five moneyness nodes, in total variance.

    ``total_variance`` defaults to ``None`` rather than to the grid itself so that a test bending
    an axis gets a grid that matches it, and so that no mutable-looking default is evaluated at
    import time.
    """
    return SurfaceView(
        market_id=market_id,
        producer_id=producer_id,
        surface_id=surface_id,
        ts_snapshot=ts_snapshot,
        log_moneyness=log_moneyness,
        tenors=tenors,
        expiries=expiries,
        forwards=forwards,
        total_variance=(
            total_variance_grid(tenors, log_moneyness) if total_variance is None else total_variance
        ),
    )


def make_flat_view(vol: float = 0.65) -> SurfaceView:
    """The same axes with no smile and no term structure: one volatility everywhere.

    The control case. A surface with no skew has a delta that must equal the analytic Black-76
    one, so it is what pins the bump-and-revalue machinery against a closed form; and comparing a
    greek computed on it against the same greek on :func:`make_view` is what proves the smile
    really does enter the number.
    """
    return make_view(
        total_variance=tuple(tuple(vol * vol * tenor for _ in K_AXIS) for tenor in TENORS)
    )


def make_position(
    underlying: str = "BTC",
    expiry: datetime = EXPIRIES[1],
    strike: float = 60_400.0,
    kind: OptionKindR = OptionKindR.CALL,
    quantity: float = 10.0,
) -> Position:
    """Ten at-the-money three-month calls: exactly on a grid node, so no interpolation hides."""
    return Position(
        underlying=underlying, expiry=expiry, strike=strike, kind=kind, quantity=quantity
    )


def make_portfolio(positions: tuple[Position, ...] | None = None) -> Portfolio:
    """A long call and a short put at different expiries: the smallest book with two signs."""
    if positions is None:
        positions = (
            make_position(),
            make_position(expiry=EXPIRIES[0], strike=58_000.0, kind=OptionKindR.PUT, quantity=-4.0),
        )
    return Portfolio(positions=positions)


def make_policy(warn_seconds: float = 5.0, reject_seconds: float = 30.0) -> FreshnessPolicy:
    """Five seconds to a warning, thirty to a refusal: a streaming engine's timescales."""
    return FreshnessPolicy(warn_seconds=warn_seconds, reject_seconds=reject_seconds)


def make_bumps(forward_rel: float = 0.01, vol_abs: float = 0.01) -> BumpSpec:
    """One percent on the forward, one vol point on the volatility. Both are configuration."""
    return BumpSpec(forward_rel=forward_rel, vol_abs=vol_abs)


def make_position_risk(
    position: Position | None = None,
    vol: float = 0.6375,
    value: float = 61_000.0,
    delta: float = 5.4,
    gamma: float = 0.0002,
    vega: float = 1_180.0,
) -> PositionRisk:
    """One valued line of the report, with numbers of the right order for ten ATM BTC calls.

    ``position`` defaults to ``None`` rather than to ``make_position()`` because a default
    argument is evaluated once at import time, and this repo avoids that on principle even where
    the object is frozen.
    """
    return PositionRisk(
        position=make_position() if position is None else position,
        vol=vol,
        value=value,
        delta=delta,
        gamma=gamma,
        vega=vega,
    )


def make_report(
    market_id: str = "BTC-DERIBIT",
    producer_id: str = "svi-scipy",
    ts_snapshot: datetime | None = NOW,
    ts_report: datetime = NOW + timedelta(seconds=2),
    freshness: FreshnessDecision = FreshnessDecision.NORMAL,
    positions: tuple[PositionRisk, ...] | None = None,
    message: str | None = None,
) -> RiskReport:
    """A healthy report: one valued position, two seconds after the snapshot it describes."""
    return RiskReport(
        market_id=market_id,
        producer_id=producer_id,
        ts_snapshot=ts_snapshot,
        ts_report=ts_report,
        freshness=freshness,
        positions=(make_position_risk(),) if positions is None else positions,
        message=message,
    )


def vol_grid(
    tenors: tuple[float, ...] = TENORS,
    log_moneyness: tuple[float, ...] = K_AXIS,
) -> tuple[tuple[float, ...], ...]:
    """The same surface as :func:`total_variance_grid`, in the volatilities the contract carries."""
    return tuple(tuple(smile_vol(tenor, k) for k in log_moneyness) for tenor in tenors)


def make_calibrated_surface(
    surface_id: str = "01JZQ0T4M2",
    market_id: str = "BTC-DERIBIT",
    producer_id: str = "svi-scipy",
    ts_snapshot: datetime = NOW,
    ts_calibrated: datetime | None = None,
    status: SurfaceStatus = SurfaceStatus.OK,
    tenors: tuple[float, ...] = TENORS,
    expiries: tuple[datetime, ...] = EXPIRIES,
    forwards: tuple[float, ...] = FORWARDS,
    log_moneyness: tuple[float, ...] = K_AXIS,
) -> CalibratedSurface:
    """The published surface the ACL translates, on the same axes as :func:`make_view`.

    Deliberately built from the same ``smile_vol`` the view builder uses, so a test can assert
    that translating this one produces that one -- which is the whole content of the ACL.
    """
    return CalibratedSurface(
        surface_id=surface_id,
        source_snapshot_id="BTC-DERIBIT:00000000",
        market_id=market_id,
        ts_snapshot=ts_snapshot,
        ts_calibrated=ts_snapshot if ts_calibrated is None else ts_calibrated,
        producer_id=producer_id,
        grid=VolGrid(
            log_moneyness=log_moneyness,
            tenors=tenors,
            expiries=expiries,
            forwards=forwards,
            vols=vol_grid(tenors, log_moneyness),
        ),
        fit=FitMetrics(
            rmse_vol_bp=12.0,
            max_err_vol_bp=31.0,
            n_quotes_used=40,
            n_iterations=17,
            duration_ms=8.4,
        ),
        status=status,
        producer_meta=None,
    )


class StubSurfaceProvider:
    """A ``SurfaceProvider`` holding whatever the test put in it, or nothing at all.

    ``None`` is the interesting default: no surface is the ordinary state at start-up and one of
    the two honest ways a report has nothing to say, so the empty provider is what a test of that
    path needs and it should take no arguments to build.
    """

    def __init__(self, view: SurfaceView | None = None) -> None:
        self._view = view

    def latest(self, market_id: str) -> SurfaceView | None:
        return self._view
