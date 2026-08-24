"""Valid Parametric Pricing objects, one per type, with a knob for every field a test bends.

The convention this repo tests by: a builder returns **one valid object**, and a test changes
the minimum needed to make its point -- either through a keyword here or through
``dataclasses.replace``, which re-runs ``__post_init__``. What a test says is then exactly what
it is probing, instead of five parameters of noise around one poisoned value.

Shared from a module rather than from ``conftest.py``: conftest is where pytest looks for
fixtures and hooks it *injects*, and importing from it is discouraged because it is loaded by
collection magic rather than by an import anyone can follow. See ``tests/support.py``.
"""

from __future__ import annotations

from datetime import UTC, datetime

from volengine.parametric_pricing.domain.calibration import (
    CalibrationResult,
    CalibrationTask,
    SliceResult,
    SliceTask,
)
from volengine.parametric_pricing.domain.svi_slice import (
    FreeParams,
    SVIParams,
    SVISlice,
    SVISurface,
)

NAIVE = datetime(2026, 7, 27, 8, 0)
"""An expiry with no zone. Every entry point in this context must refuse it."""

NEAR = datetime(2026, 8, 27, 8, 0, tzinfo=UTC)
FAR = datetime(2026, 10, 27, 8, 0, tzinfo=UTC)
"""Two live expiries at Deribit's 08:00 UTC, roughly one and three months out."""

NEAR_TENOR = 1.0 / 12.0
FAR_TENOR = 3.0 / 12.0
"""Their year fractions under ACT/365F, rounded to clean months so a test can do the algebra."""


def make_params(
    a: float = 0.04,
    b: float = 0.10,
    rho: float = -0.35,
    m: float = -0.02,
    sigma: float = 0.20,
) -> SVIParams:
    """A plausible one-month crypto slice: 20% at-the-money vol, negative skew, rounded bottom.

    ``a = 0.04`` is a total variance of 0.04 over the tenor, ``rho < 0`` lifts the downside
    wing, and the minimum sits just below the forward. Comfortably inside every invariant, so a
    single knob is what breaks it.
    """
    return SVIParams(a=a, b=b, rho=rho, m=m, sigma=sigma)


def make_free_params(
    a: float = 0.04,
    b_raw: float = -2.20,
    rho_raw: float = -0.37,
    m: float = -0.02,
    sigma_raw: float = -1.50,
) -> FreeParams:
    """The same slice in the optimiser's coordinates, near enough for the shapes to match.

    The raw values are not the exact pre-images of ``make_params`` -- deriving those would make
    this builder depend on the mapping it is used to test -- only ordinary points of R^5 that
    map back to an admissible slice.
    """
    return FreeParams(a=a, b_raw=b_raw, rho_raw=rho_raw, m=m, sigma_raw=sigma_raw)


def make_slice(
    expiry: datetime = NEAR,
    tenor_years: float = NEAR_TENOR,
    params: SVIParams | None = None,
    k_min: float = -0.60,
    k_max: float = 0.60,
) -> SVISlice:
    """One fitted expiry over a band of roughly +-60% in log-forward-moneyness.

    ``params`` defaults to ``None`` rather than to ``make_params()`` because a mutable-looking
    default evaluated at import time is the kind of shared state this repo avoids on principle,
    even where the object happens to be frozen.
    """
    return SVISlice(
        expiry=expiry,
        tenor_years=tenor_years,
        params=make_params() if params is None else params,
        k_min=k_min,
        k_max=k_max,
    )


def make_surface(slices: tuple[SVISlice, ...] | None = None) -> SVISurface:
    """Two slices, correctly ordered: the smallest surface where ordering can be broken."""
    if slices is None:
        slices = (
            make_slice(expiry=NEAR, tenor_years=NEAR_TENOR),
            make_slice(expiry=FAR, tenor_years=FAR_TENOR),
        )
    return SVISurface(slices=slices)


NOW = datetime(2026, 7, 27, 12, 0, tzinfo=UTC)
"""The snapshot instant, comfortably before both expiries above."""

FORWARD = 60_000.0
"""A round crypto forward, so a strike can be read off a log-moneyness by eye."""

LOG_MONEYNESS = (-0.30, -0.10, 0.05, 0.25)
IMPLIED_VOL = (0.72, 0.65, 0.62, 0.66)
WEIGHTS = (0.15, 0.35, 0.30, 0.20)
"""Four quotes across the smile: a lifted downside wing, the minimum just above the forward,
weights normalised to one and heaviest where the quotes are tightest. Four is the smallest
count where the ordering of the axis can be broken in the middle rather than at an end."""


def make_slice_task(
    expiry: datetime = NEAR,
    tenor_years: float = NEAR_TENOR,
    forward: float = FORWARD,
    log_moneyness: tuple[float, ...] = LOG_MONEYNESS,
    implied_vol: tuple[float, ...] = IMPLIED_VOL,
    weights: tuple[float, ...] = WEIGHTS,
) -> SliceTask:
    """One expiry ready to fit: an ordered axis, our own volatilities, normalised weights.

    The three tuples are parallel by construction, so a test that shortens or poisons exactly
    one of them is testing that the constructor notices.
    """
    return SliceTask(
        expiry=expiry,
        tenor_years=tenor_years,
        forward=forward,
        log_moneyness=log_moneyness,
        implied_vol=implied_vol,
        weights=weights,
    )


def make_calibration_task(
    market_id: str = "BTC-DERIBIT",
    snapshot_id: str = "snap-000001",
    ts_snapshot: datetime = NOW,
    slices: tuple[SliceTask, ...] | None = None,
) -> CalibrationTask:
    """Two correctly ordered slices: the smallest task where ordering can be broken."""
    if slices is None:
        slices = (
            make_slice_task(expiry=NEAR, tenor_years=NEAR_TENOR),
            make_slice_task(expiry=FAR, tenor_years=FAR_TENOR),
        )
    return CalibrationTask(
        market_id=market_id,
        snapshot_id=snapshot_id,
        ts_snapshot=ts_snapshot,
        slices=slices,
    )


def make_slice_result(
    expiry: datetime = NEAR,
    tenor_years: float = NEAR_TENOR,
    params: SVIParams | None = None,
    rmse_vol_bp: float = 12.0,
    max_err_vol_bp: float = 31.0,
    n_quotes_used: int = 4,
    converged: bool = True,
    at_bound: bool = False,
) -> SliceResult:
    """A healthy fit: 12 bp of vol RMSE, worst quote missed by 31 bp, nothing pinned.

    Comfortably inside any plausible acceptance threshold, so a single knob is what makes it
    unpublishable. ``params`` defaults to ``None`` rather than to ``make_params()`` for the
    reason ``make_slice`` gives.
    """
    return SliceResult(
        expiry=expiry,
        tenor_years=tenor_years,
        params=make_params() if params is None else params,
        rmse_vol_bp=rmse_vol_bp,
        max_err_vol_bp=max_err_vol_bp,
        n_quotes_used=n_quotes_used,
        converged=converged,
        at_bound=at_bound,
    )


def make_calibration_result(
    slices: tuple[SliceResult, ...] | None = None,
    n_iterations: int = 17,
    duration_ms: float = 8.4,
) -> CalibrationResult:
    """Two fitted slices, ordered, in single-digit milliseconds: a warm-started cycle."""
    if slices is None:
        slices = (
            make_slice_result(expiry=NEAR, tenor_years=NEAR_TENOR),
            make_slice_result(expiry=FAR, tenor_years=FAR_TENOR),
        )
    return CalibrationResult(slices=slices, n_iterations=n_iterations, duration_ms=duration_ms)
