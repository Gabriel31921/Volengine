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

import math
from collections.abc import Mapping
from datetime import UTC, datetime

from volengine.contracts.market_snapshot import (
    MarketSnapshot,
    OptionKind,
    QualityBlock,
    QuoteData,
    QuoteFlag,
    SliceData,
)
from volengine.parametric_pricing.application.acl import Weighting
from volengine.parametric_pricing.application.grid_spec import GridSpec
from volengine.parametric_pricing.domain.black76 import OptionKindP, price
from volengine.parametric_pricing.domain.calibration import (
    CalibrationResult,
    CalibrationTask,
    SliceResult,
    SliceTask,
)
from volengine.parametric_pricing.domain.errors import CalibrationError
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


SNAPSHOT_MONEYNESS = (-0.30, -0.15, 0.0, 0.15, 0.30)
"""Five strikes spanning the quoted band, symmetric about the forward.

Symmetric on purpose: the at-the-money strike sits exactly on the forward, which is the boundary
case the out-of-the-money rule has to have an answer for, and it is in the middle rather than at
an end so a rule that got the comparison backwards fails visibly on both wings.
"""


def synthetic_quotes(
    tenor_years: float,
    forward: float = FORWARD,
    params: SVIParams | None = None,
    log_moneyness: tuple[float, ...] = SNAPSHOT_MONEYNESS,
    both_sides: bool = True,
    spread_rel: float = 0.02,
    flags: tuple[QuoteFlag, ...] = (),
) -> tuple[QuoteData, ...]:
    """Quotes priced from a known SVI slice, so an inversion has a right answer to recover.

    This is D-7's synthetic chain: prices are generated *forwards* through Black-76 from
    parameters the test holds, so a test can assert that the ACL's inversion returns those exact
    volatilities. A builder that invented plausible premiums instead would let an inversion that
    was quietly wrong by a vol point pass every test in the suite.

    ``both_sides`` puts a call and a put at every strike, which is what a liquid chain looks like
    and what lets the out-of-the-money rule actually choose. Turning it off leaves only the leg
    that is *in* the money on each side of the forward, which is the case the unpaired
    down-weighting exists for.
    """
    fitted = make_params() if params is None else params
    quotes: list[QuoteData] = []
    for k in log_moneyness:
        strike = forward * math.exp(k)
        vol = fitted.implied_vol(k, tenor_years)
        kinds = (
            (OptionKind.CALL, OptionKind.PUT)
            if both_sides
            else ((OptionKind.PUT,) if k >= 0 else (OptionKind.CALL,))
        )
        for kind in kinds:
            side = OptionKindP.CALL if kind is OptionKind.CALL else OptionKindP.PUT
            quotes.append(
                QuoteData(
                    strike=strike,
                    kind=kind,
                    mid=price(forward, strike, tenor_years, vol, side),
                    spread_rel=spread_rel,
                    age_seconds=0.4,
                    flags=flags,
                    exchange_iv=vol,
                )
            )
    return tuple(quotes)


def make_market_snapshot(
    snapshot_id: str = "BTC-DERIBIT:00000000",
    market_id: str = "BTC-DERIBIT",
    ts_exchange: datetime = NOW,
    forward: float = FORWARD,
    params: SVIParams | None = None,
    log_moneyness: tuple[float, ...] = SNAPSHOT_MONEYNESS,
    both_sides: bool = True,
    spread_rel: float = 0.02,
    flags: tuple[QuoteFlag, ...] = (),
    degraded: bool = False,
    tenors: tuple[tuple[datetime, float], ...] = ((NEAR, NEAR_TENOR), (FAR, FAR_TENOR)),
) -> MarketSnapshot:
    """A published snapshot whose premiums come from a known surface.

    Two expiries, because one is the smallest number where a term structure exists and a slice
    can be dropped without emptying the snapshot.
    """
    return MarketSnapshot(
        snapshot_id=snapshot_id,
        market_id=market_id,
        ts_exchange=ts_exchange,
        ts_local=ts_exchange,
        underlying="BTC",
        slices=tuple(
            SliceData(
                expiry=expiry,
                tenor_years=tenor_years,
                forward=forward,
                quotes=synthetic_quotes(
                    tenor_years=tenor_years,
                    forward=forward,
                    params=params,
                    log_moneyness=log_moneyness,
                    both_sides=both_sides,
                    spread_rel=spread_rel,
                    flags=flags,
                ),
                flags=(),
            )
            for expiry, tenor_years in tenors
        ),
        quality=QualityBlock(
            coverage_ratio=1.0,
            max_age_seconds=0.4,
            n_quotes_admissible=0 if degraded else 20,
            n_quotes_total=20,
            forward_crosscheck_error=0.0,
            degraded=degraded,
        ),
    )


def make_weighting(
    spread_scale: float = 0.05,
    flagged_factor: float = 0.25,
    unpaired_itm_factor: float = 0.10,
) -> Weighting:
    """Discounts that are visible without being extreme, so a test can tell them apart."""
    return Weighting(
        spread_scale=spread_scale,
        flagged_factor=flagged_factor,
        unpaired_itm_factor=unpaired_itm_factor,
    )


def make_grid_spec(k_min: float = -0.4, k_max: float = 0.4, n_nodes: int = 9) -> GridSpec:
    """A mesh wider than the quoted band, so the published wings are extrapolation."""
    return GridSpec(k_min=k_min, k_max=k_max, n_nodes=n_nodes)


class StubCalibrator:
    """A ``Calibrator`` that answers with whatever the test put in it.

    Satisfies the port structurally, with no optimiser anywhere. It records the ``previous`` it
    was handed, which is the only way to assert that the warm start is actually chained: a
    learner or calibrator that accepted a history and ignored it would pass every other test.
    """

    def __init__(
        self,
        result: CalibrationResult | None = None,
        producer_id: str = "svi-stub",
        failure: CalibrationError | None = None,
    ) -> None:
        self._result = result
        self._producer_id = producer_id
        self._failure = failure
        self.calls: list[Mapping[datetime, SVIParams] | None] = []

    @property
    def producer_id(self) -> str:
        return self._producer_id

    def answer_with(self, result: CalibrationResult) -> None:
        """Change what the next call returns, so one test can drive two consecutive cycles.

        The stale republish of ADR-006 is a statement about *sequence* -- a good cycle, then a bad
        one -- so a test of it needs the same use case to be handed two different answers.
        """
        self._result = result

    def calibrate(
        self,
        previous: Mapping[datetime, SVIParams] | None,
        task: CalibrationTask,
    ) -> CalibrationResult:
        self.calls.append(previous)
        if self._failure is not None:
            raise self._failure
        if self._result is not None:
            return self._result
        return CalibrationResult(
            slices=tuple(
                make_slice_result(expiry=one.expiry, tenor_years=one.tenor_years)
                for one in task.slices
            ),
            n_iterations=3,
            duration_ms=1.0,
        )
