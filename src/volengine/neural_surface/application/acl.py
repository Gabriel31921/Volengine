"""The one module in Neural Surface that speaks both languages (rule 5).

The same two translations its parametric sibling performs, and the differences between the two are
exactly the modelling difference between the contexts:

* **Inbound, a snapshot becomes a flat cloud of points rather than a stack of slices.** SVI fits
  one smile at a time, so ``SliceTask`` groups quotes by expiry and the grouping *is* the problem
  statement. A network sees the whole surface at once, with parameters shared across every tenor,
  so what it needs is ``(k, T, w)`` triples with nothing grouping them -- which is why
  ``TrainingSample`` carries its own tenor and why nothing here sorts anything.
* **Each point keeps its own observation instant.** ``QuoteData.age_seconds`` is measured at the
  snapshot, so subtracting it recovers when the venue last touched that quote. That instant is
  what the replay buffer's retention policy is measured on, and it is the reason a batch can mix
  points seen minutes apart and still say how old any one of them is.

Everything else is deliberately identical to the parametric ACL: the same out-of-the-money twin
rule, the same weighting formula, the same down-weighting of a flagged or unpaired quote. That is
not laziness about duplication, it is the precondition for the whole project -- Design 6.5
compares two producers fitted to one market, and a comparison in which the two engines weighted
that market differently would measure the weighting rather than the models.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import datetime, timedelta

import numpy as np

from volengine.contracts.calibrated_surface import (
    CalibratedSurface,
    FitMetrics,
    SurfaceStatus,
    VolGrid,
)
from volengine.contracts.events import CalibrationFailed, SurfaceCalibrated
from volengine.contracts.market_snapshot import MarketSnapshot, OptionKind, QuoteData, SliceData
from volengine.neural_surface.application.grid_spec import GridSpec
from volengine.neural_surface.domain.errors import NoImpliedVolError
from volengine.neural_surface.domain.learned_surface import LearnedSurface, implied_vol_grid
from volengine.neural_surface.domain.pricing import OptionKindN, implied_vol, vega
from volengine.neural_surface.domain.training_batch import TrainingBatch, TrainingSample

BASIS_POINTS_PER_UNIT = 10_000.0
"""One basis point of volatility is ``0.0001``. The unit the published fit metrics are stated in."""


@dataclass(frozen=True, slots=True)
class Weighting:
    """How much each observed point counts in the loss. Configuration, never constants (ADR-012).

    **The same three numbers the parametric ACL takes, and they have to stay the same numbers.**
    They are duplicated here because rule 6 forbids importing that context's type, not because the
    two producers are entitled to different answers: the weights are what Design 6.5's comparison
    holds constant while the models vary. A deployment that configured these two differently would
    be running an experiment nobody designed.
    """

    spread_scale: float
    """Relative spread at which a point's weight is halved. Positive and finite.

    The discount is ``1 / (1 + spread_rel / spread_scale)``. Deliberately not inverse-variance
    weighting: that makes influence scale like the fourth power of vega, and one tight
    at-the-money quote then carries a whole region of the surface while the wings stop
    constraining anything.
    """

    flagged_factor: float
    """What a point carrying any ingestion flag is multiplied by. In ``[0, 1]``.

    Zero keeps the point in the batch with no influence at all, which is what
    ``TrainingSample.weight`` documents a zero weight is for: it stays visible in the reported
    residuals and in the buffer's cell occupancy without steering a single network weight.
    """

    unpaired_itm_factor: float
    """What an in-the-money quote with no out-of-the-money twin is multiplied by. In ``[0, 1]``.

    Small rather than zero: the inversion is badly conditioned there, so the point deserves little
    influence, but it is the only observation of that strike and dropping it would leave a hole in
    the wing -- and a hole in a network's training data is worse than a noisy point, because
    nothing constrains what it interpolates across the gap.
    """

    def __post_init__(self) -> None:
        if not math.isfinite(self.spread_scale) or self.spread_scale <= 0:
            raise ValueError(
                f"The spread scale must be positive and finite, got {self.spread_scale}"
            )
        # Zero is legitimate for both factors -- it is how a deployment excludes a class of quote
        # without removing it -- so only negatives and values above one are refused. Finiteness
        # first, because `float("nan") < 0` is `False`.
        for name, value in (
            ("flagged factor", self.flagged_factor),
            ("unpaired in-the-money factor", self.unpaired_itm_factor),
        ):
            if not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"The {name} must lie in [0, 1], got {value}")


def to_training_samples(
    snapshot: MarketSnapshot, weighting: Weighting
) -> tuple[TrainingSample, ...]:
    """Turn a published snapshot into observed points of the surface.

    One point per strike, inverted from the out-of-the-money leg with our own forward and our own
    tenor. See the module docstring on why the venue's published IV is never used, and
    ``parametric_pricing``'s ACL for the measurements behind the out-of-the-money rule.

    Returns:
        The points, in no particular order and with no grouping. Possibly empty, when nothing in
        the snapshot admitted an implied volatility -- which the caller turns into a refusal,
        because ``TrainingBatch`` requires at least one sample and a gradient step over no data is
        a no-op that would still be counted, timed and published as an update.

        **Weights are normalised across the whole snapshot**, not per expiry, and that is the one
        place this differs from the parametric weighting rather than merely reorganising it. SVI
        fits each slice independently, so a per-slice normalisation is the only one that means
        anything there; a network takes one gradient step over every point at once, so the
        normalisation has to span the same set the step does. Per-slice weights here would give a
        thin far expiry the same total pull as a hundred-quote front month.
    """
    weights: list[float] = []
    points: list[tuple[float, float, float, datetime]] = []

    for slice_data in snapshot.slices:
        for strike, quote, paired in _out_of_the_money_quotes(slice_data):
            vol = _invert(quote, slice_data)
            if vol is None:
                continue
            points.append(
                (
                    math.log(strike / slice_data.forward),
                    slice_data.tenor_years,
                    vol,
                    _observed_at(snapshot, quote),
                )
            )
            weights.append(_weight(quote, slice_data, vol, paired, weighting))

    total = sum(weights)
    if not points or total <= 0:
        return ()

    return tuple(
        TrainingSample(
            log_moneyness=k,
            tenor_years=tenor,
            implied_vol=vol,
            weight=weight / total,
            ts_observed=observed,
        )
        for (k, tenor, vol, observed), weight in zip(points, weights, strict=True)
    )


def _observed_at(snapshot: MarketSnapshot, quote: QuoteData) -> datetime:
    """When the venue last touched this quote, recovered from its published age.

    ``QuoteData.age_seconds`` is measured from ``ts_exchange`` at the snapshot instant, so this
    subtraction inverts the same arithmetic that produced it. It is the one field of the published
    quote that this context needs and the parametric one does not: the replay buffer ages its
    contents out, and a batch that mixes points seen minutes apart needs a per-sample instant
    rather than the snapshot's.

    The age is non-negative by the contract's own invariant, so the result never lands after the
    snapshot. ``TrainingBatch`` would accept it if it did -- it deliberately does not constrain
    its samples' timestamps against its own -- but nothing here relies on that latitude.
    """
    return snapshot.ts_exchange - timedelta(seconds=quote.age_seconds)


def _out_of_the_money_quotes(slice_data: SliceData) -> list[tuple[float, QuoteData, bool]]:
    """Pick one quote per strike: the out-of-the-money leg where there is one.

    A call is out of the money above the forward and a put below it; at the forward exactly the
    call is chosen, arbitrarily and deterministically -- at ``k = 0`` the two legs have identical
    time value and identical vega, so what matters is only that the rule is fixed.

    Returns ``(strike, quote, paired)``, where ``paired`` is ``False`` only when the sole quote
    available was the in-the-money one. A lone out-of-the-money quote is the well-conditioned side
    and keeps its full weight.
    """
    by_strike: dict[float, dict[OptionKind, QuoteData]] = {}
    for quote in slice_data.quotes:
        by_strike.setdefault(quote.strike, {})[quote.kind] = quote

    chosen: list[tuple[float, QuoteData, bool]] = []
    for strike in sorted(by_strike):
        legs = by_strike[strike]
        wanted = OptionKind.CALL if strike >= slice_data.forward else OptionKind.PUT
        preferred = legs.get(wanted)
        if preferred is not None:
            chosen.append((strike, preferred, True))
            continue
        fallback = next(iter(legs.values()), None)
        if fallback is not None:
            chosen.append((strike, fallback, False))
    return chosen


_KINDS: Mapping[OptionKind, OptionKindN] = {
    OptionKind.CALL: OptionKindN.CALL,
    OptionKind.PUT: OptionKindN.PUT,
}
"""Wire side to this context's own spelling. The fifth enum for one idea, and this table is where
the published one meets it -- rule 3 keeps the domain from importing the contract, rule 6 keeps
this context from borrowing anyone else's."""


def _invert(quote: QuoteData, slice_data: SliceData) -> float | None:
    """Our own Black-76 volatility for one quote, or ``None`` when the price admits none.

    ``NoImpliedVolError`` is swallowed on purpose: a mid from a crossed or stale book falls below
    intrinsic regularly, and dropping the point is the honest answer. A NaN premium would be a
    different matter, but ``QuoteData`` refuses one at construction.
    """
    try:
        return implied_vol(
            target_price=quote.mid,
            forward=slice_data.forward,
            strike=quote.strike,
            tenor_years=slice_data.tenor_years,
            kind=_KINDS[quote.kind],
        )
    except NoImpliedVolError:
        return None


def _weight(
    quote: QuoteData,
    slice_data: SliceData,
    vol: float,
    paired: bool,
    weighting: Weighting,
) -> float:
    """How much this point counts, before the snapshot-wide normalisation.

    ``vega / (1 + spread_rel / spread_scale)``, cut by a factor for each thing wrong with the
    quote. Vega is evaluated at the volatility just inverted, which is where the model reproduces
    the observed price exactly.
    """
    sensitivity = vega(
        forward=slice_data.forward,
        strike=quote.strike,
        tenor_years=slice_data.tenor_years,
        vol=vol,
    )
    weight = sensitivity / (1.0 + quote.spread_rel / weighting.spread_scale)
    if quote.flags or slice_data.flags:
        weight *= weighting.flagged_factor
    if not paired:
        weight *= weighting.unpaired_itm_factor
    return weight


def to_training_batch(
    snapshot: MarketSnapshot,
    fresh: Sequence[TrainingSample],
    replayed: Sequence[TrainingSample],
) -> TrainingBatch | None:
    """Assemble one update step's data: this snapshot's points, then the buffer's.

    Fresh first and ``n_fresh`` marking the split, which is the convention ``TrainingBatch``
    documents and this module is one of the two places responsible for honouring it.

    Returns ``None`` when the combined weights sum to zero or there is nothing to train on --
    every point flagged under a zero factor, say. That is a legal configuration meeting a legal
    market, and the caller waits for the next snapshot rather than constructing a placeholder.
    """
    samples = tuple(fresh) + tuple(replayed)
    if not samples or sum(sample.weight for sample in samples) <= 0:
        return None
    return TrainingBatch(
        market_id=snapshot.market_id,
        snapshot_id=snapshot.snapshot_id,
        ts_snapshot=snapshot.ts_exchange,
        samples=samples,
        n_fresh=len(fresh),
    )


def to_calibrated_surface(
    surface: LearnedSurface,
    snapshot: MarketSnapshot,
    grid: GridSpec,
    producer_id: str,
    surface_id: str,
    ts_calibrated: datetime,
    status: SurfaceStatus,
    fit: FitMetrics,
) -> CalibratedSurface | None:
    """Evaluate the learned surface on the published mesh and wrap it in the contract.

    The network is evaluated at the snapshot's own tenors rather than at an axis of this module's
    choosing. It could answer anywhere -- that is the whole difference from a slice-wise
    parameterisation -- and publishing rows no market data stands behind would make the
    producer-to-producer comparison measure an interpolation choice instead of two models.

    Args:
        surface: The trained model, evaluated through ``implied_vol_grid`` so that a diverged one
            raises rather than serialising a row of NaN.
        snapshot: The market this update was occasioned by. Read for the tenor axis, the expiry
            instants and the forwards -- the three things the network does not know and cannot
            invent, because they are convention-dependent and were resolved upstream (ADR-002).
        grid: Where the published moneyness nodes sit.
        producer_id: Which producer this is, for attribution only.
        surface_id: Identity of this result.
        ts_calibrated: When the update finished, from the injected clock.
        status: Trust level, decided by the use case.
        fit: What the step cost and how well it fits, measured by the use case -- see
            :func:`fit_metrics`.

    Returns:
        The published surface, or ``None`` when the snapshot carried no usable tenor at all.

    Raises:
        SurfaceEvaluationError: If the model answered with the wrong shape, a non-finite value, or
            a non-positive total variance. Deliberately not caught here: that is a diverged model
            rather than an arbitrageable one, and the two must never arrive at the operator
            wearing the same face.
    """
    tenors = tuple(slice_data.tenor_years for slice_data in snapshot.slices)
    if not tenors:
        return None

    nodes = grid.nodes()
    vols = implied_vol_grid(
        surface,
        np.asarray(nodes, dtype=np.float64),
        np.asarray(tenors, dtype=np.float64),
    )

    return CalibratedSurface(
        surface_id=surface_id,
        source_snapshot_id=snapshot.snapshot_id,
        market_id=snapshot.market_id,
        ts_snapshot=snapshot.ts_exchange,
        ts_calibrated=ts_calibrated,
        producer_id=producer_id,
        grid=VolGrid(
            log_moneyness=nodes,
            tenors=tenors,
            expiries=tuple(slice_data.expiry for slice_data in snapshot.slices),
            forwards=tuple(slice_data.forward for slice_data in snapshot.slices),
            # `float(...)` rather than the numpy scalar: `VolGrid` promises primitives only
            # (ADR-011), and a `np.float64` would survive every invariant here and then need a
            # codec at the first `json.dumps`.
            vols=tuple(tuple(float(vol) for vol in smile) for smile in vols),
        ),
        fit=fit,
        status=status,
        producer_meta={"weights_version": float(surface.version)},
    )


def fit_metrics(
    surface: LearnedSurface,
    fresh: Sequence[TrainingSample],
    n_iterations: int,
    duration_ms: float,
) -> FitMetrics:
    """Measure the trained surface against the quotes that were just observed.

    **This exists because ``SurfaceLearner.update`` returns a bare surface** where
    ``Calibrator.calibrate`` returns a metrics-carrying result. That asymmetry is a deliberate
    seam in the port -- a learner that reported on its own fit would be a second way to evaluate a
    surface, and the gate of ADR-010 would then need a live learner to judge anything -- and its
    cost is paid right here: the residual has to be recomputed by evaluating the surface again.

    Measured against the **fresh** points only, not the whole batch. Design 6.5's question is "does
    the model fit the market as it is now", and a residual over the replayed points as well is
    diluted by however much history the buffer happened to contribute. When the batch has no fresh
    points at all -- the scheduled restart -- there is nothing current to measure and the caller
    supplies the whole batch instead.

    Errors are stated in basis points of volatility rather than of total variance, even though
    variance is what the network outputs, because that is the unit the contract publishes and the
    unit the parametric producer reports in. A comparison between two producers whose error
    columns meant different things would not be a comparison.
    """
    k = np.asarray([sample.log_moneyness for sample in fresh], dtype=np.float64)
    tenors = np.asarray([sample.tenor_years for sample in fresh], dtype=np.float64)

    # One evaluation per sample, read off the diagonal of the mesh: `implied_vol_grid` answers on
    # the full outer product of the two axes, and what is wanted here is the pairing. That costs
    # n^2 evaluations to use n of them, which is the price of the port answering on a mesh -- and
    # the mesh is right for its two other callers, the gate and the published grid, both of which
    # genuinely want every combination. At a few hundred fresh quotes per snapshot the waste is a
    # forward pass over a matrix a network handles in one batch; if it ever stops being, the fix
    # is a paired-evaluation method on `LearnedSurface`, not a reshaping trick here.
    grid = implied_vol_grid(surface, k, tenors)
    predicted = np.diagonal(grid)
    observed = np.asarray([sample.implied_vol for sample in fresh], dtype=np.float64)
    weights = np.asarray([sample.weight for sample in fresh], dtype=np.float64)

    errors = np.abs(predicted - observed) * BASIS_POINTS_PER_UNIT
    total = float(np.sum(weights))
    rmse = (
        math.sqrt(float(np.sum(weights * errors**2)) / total)
        if total > 0
        else float(np.sqrt(np.mean(errors**2)))
    )
    worst = float(np.max(errors))
    return FitMetrics(
        # The weighted RMSE cannot exceed the unweighted maximum in exact arithmetic; the clamp is
        # against the ulp that a weighted sum of equal errors can land above it, which `FitMetrics`
        # refuses outright.
        rmse_vol_bp=min(rmse, worst),
        max_err_vol_bp=worst,
        n_quotes_used=len(fresh),
        n_iterations=n_iterations,
        duration_ms=duration_ms,
    )


def to_surface_calibrated(surface: CalibratedSurface) -> SurfaceCalibrated:
    """Wrap a published surface in the event that carries it. See rule 5."""
    return SurfaceCalibrated(surface=surface)


def to_calibration_failed(
    market_id: str,
    source_snapshot_id: str,
    producer_id: str,
    reason: str,
    ts: datetime,
) -> CalibrationFailed:
    """Announce that this producer could not publish a surface for this snapshot.

    The same event the parametric context publishes, and that is the point: a consumer counting
    failures per producer sees both engines through one channel, which is what makes ADR-010's
    refusal rate comparable against ADR-006's.
    """
    return CalibrationFailed(
        market_id=market_id,
        source_snapshot_id=source_snapshot_id,
        producer_id=producer_id,
        reason=reason,
        ts=ts,
    )


def as_stale_republish(surface: CalibratedSurface) -> CalibratedSurface:
    """The last good surface, relabelled for republication (ADR-006).

    Everything else kept: the original ``ts_snapshot``, because it is what staleness is measured
    against; the original ``ts_calibrated``, which under this status refers to the earlier fit;
    and the original ``surface_id``, because this *is* that surface and a new id would suggest a
    new one exists.
    """
    return replace(surface, status=SurfaceStatus.STALE_REPUBLISH)
