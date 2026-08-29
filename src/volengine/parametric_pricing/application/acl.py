"""The one module in Parametric Pricing that speaks both languages (rule 5).

Two translations, and they are not symmetric. Inbound, a ``MarketSnapshot`` of premiums becomes a
``CalibrationTask`` of volatilities -- which means *inverting Black-76 once per strike*, and that
inversion is where almost every decision in this module lives. Outbound, a ``CalibrationResult``
of fitted parameters becomes a ``CalibratedSurface`` of evaluated numbers, which is arithmetic.

**The out-of-the-money twin is the rule this module exists to enforce.**
``black76.implied_vol`` is well conditioned only out of the money. Deep in the money the option's
time value -- the only part a volatility can move -- falls below the last bits of its intrinsic
value, so the price stops carrying information about vol: the inversion returns a plausible
number that is simply wrong, and nothing about it looks wrong. A 200k-case sweep at realistic
crypto ranges gave up to **0.17 of silent vol error** on in-the-money quotes against 0.003 out of
the money, and a 0.1% error in the forward becomes 4.2 vol points through an in-the-money
inversion versus 0.08 through its out-of-the-money twin.

Nothing is lost by preferring the twin. Put-call parity makes the in-the-money option's *time
value* equal to the out-of-the-money option's whole price, and vega is identical on both sides,
so the two quotes carry the same information about the smile and one of them carries it in a
form arithmetic can read. What the discarded quote carried in addition was a parity residual,
which is a statement about the forward and the discount and never about the smile -- and it is
already measured upstream, as ``QualityBlock.forward_crosscheck_error``. Inverting the
in-the-money side would not surface that anomaly, it would launder it into the wing of the fit.

**When a strike has no out-of-the-money twin the in-the-money leg is inverted anyway, and
down-weighted.** A one-sided book is ordinary in the wings, and dropping the strike outright
would leave the fit blind exactly where the market is thinnest. So the quote enters, with its
weight cut by a configured factor: "ingestion flags, the calibrator weights or excludes" is the
rule, and this is the calibrator exercising the *weights* half of it. The weight is also the only
marking channel available here -- ``SliceTask`` carries no flags by design, because a loss
function consumes three parallel arrays of numbers -- so the mark and the consequence are the
same number, which is at least impossible to leave inconsistent.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import datetime

from volengine.contracts.calibrated_surface import (
    CalibratedSurface,
    FitMetrics,
    SurfaceStatus,
    VolGrid,
)
from volengine.contracts.events import CalibrationFailed, SurfaceCalibrated
from volengine.contracts.market_snapshot import MarketSnapshot, OptionKind, QuoteData, SliceData
from volengine.parametric_pricing.application.grid_spec import GridSpec
from volengine.parametric_pricing.domain.black76 import OptionKindP, implied_vol, vega
from volengine.parametric_pricing.domain.calibration import (
    CalibrationTask,
    SliceResult,
    SliceTask,
)
from volengine.parametric_pricing.domain.errors import NoImpliedVolError

BASIS_POINTS_PER_UNIT = 10_000.0
"""One basis point of volatility is ``0.0001``. The unit the published fit metrics are stated in.

A module constant and explicitly not TOML configuration: ADR-012 governs business thresholds, and
this is the definition of a unit. An operator who changed it would only be relabelling every
number in the contract.
"""


@dataclass(frozen=True, slots=True)
class Weighting:
    """How much each quote counts in the loss. Configuration, never constants (ADR-012).

    Design 5.3 says the weights are built from vega and the relative spread, and this is where
    that sentence becomes numbers. Both halves are needed and they say different things: vega is
    how much the quote *knows* about volatility, and the spread is how much of what it says is
    noise. A far wing quote has almost no vega, so a vol error there barely moves its price and
    its price barely constrains the vol; a wide quote has a mid that is a guess between two
    numbers a long way apart.
    """

    spread_scale: float
    """Relative spread at which a quote's weight is halved. Positive and finite.

    The discount is ``1 / (1 + spread_rel / spread_scale)``, so a quote exactly this wide counts
    half as much as a locked one and the falloff is gentle in both directions.

    **Deliberately not inverse-variance weighting**, which is the textbook answer and is wrong
    here. The vol error implied by a spread is roughly ``mid * spread_rel / (2 * vega)``, so
    weighting by one over its square makes the weight scale like the *fourth* power of vega and
    the inverse square of the spread: a tenfold difference in spread becomes a hundredfold
    difference in influence, one tight at-the-money quote ends up carrying a whole slice, and the
    wings -- where the smile's shape is actually decided -- stop constraining anything. The gentle
    form keeps the ordering that matters (tight beats wide, high vega beats low) without letting
    one quote win outright.
    """

    flagged_factor: float
    """What a quote carrying any ingestion flag is multiplied by. In ``[0, 1]``.

    Zero excludes flagged quotes from the fit while keeping them in the slice, which is what
    ``SliceTask.weights`` documents a zero weight is for: the point stays visible in the reported
    residuals without steering a single parameter. One ignores the flags entirely. Neither
    extreme is wrong, which is exactly why the number is configuration rather than a constant --
    a deployment whose venue flags half the chain as ``WIDE_SPREAD`` wants a different answer
    from one whose venue is tight.

    A single factor for every flag, rather than one per flag. The flags say different things and
    a per-flag table would be defensible, but it would be six more numbers to tune against a
    market nobody has measured yet; the honest v1 answer is one knob, and the flags remain
    individually visible upstream for whoever wants to justify splitting it.
    """

    unpaired_itm_factor: float
    """What an in-the-money quote with no out-of-the-money twin is multiplied by. In ``[0, 1]``.

    The number that implements the decision in the module docstring. Small rather than zero in
    any sensible deployment: the inversion is badly conditioned, so the quote deserves little
    influence, but it is the only observation of that strike and excluding it would leave a hole
    in the wing rather than a noisy point in it.
    """

    def __post_init__(self) -> None:
        if not math.isfinite(self.spread_scale) or self.spread_scale <= 0:
            raise ValueError(
                f"The spread scale must be positive and finite, got {self.spread_scale}"
            )
        # Zero is a legitimate setting for both factors -- it is how a deployment excludes a class
        # of quote without removing it from the slice -- so only negatives and values above one
        # are refused. Finiteness first, because `float("nan") < 0` is `False`.
        for name, value in (
            ("flagged factor", self.flagged_factor),
            ("unpaired in-the-money factor", self.unpaired_itm_factor),
        ):
            if not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"The {name} must lie in [0, 1], got {value}")


def to_calibration_task(snapshot: MarketSnapshot, weighting: Weighting) -> CalibrationTask | None:
    """Turn a published snapshot into the problem the optimiser is handed.

    One volatility per strike, inverted from the out-of-the-money leg with our own forward and our
    own tenor -- never the venue's published IV, which is the output of the venue's model and
    would enter the surface disguised as a feature of the smile.

    Args:
        snapshot: The market, as published. Its ``quality`` block is not read here: whether a
            degraded snapshot is worth fitting is the use case's decision, and this module's job
            is to express it as numbers either way.
        weighting: How much each quote counts. See :class:`Weighting`.

    Returns:
        The task, or ``None`` when not one usable quote survived the inversion across the whole
        snapshot. ``None`` rather than an empty task because ``CalibrationTask`` refuses to hold
        no slices, and rightly: an empty problem wearing the shape of a full one would be handed
        to a calibrator that could only fail at it.

        Slices are dropped individually for the same reason -- a whole expiry whose quotes are all
        uninvertible is a market condition, not a failure of the rest of the snapshot.
    """
    tasks = [
        slice_task
        for slice_data in snapshot.slices
        if (slice_task := _to_slice_task(slice_data, weighting)) is not None
    ]
    if not tasks:
        return None
    return CalibrationTask(
        market_id=snapshot.market_id,
        snapshot_id=snapshot.snapshot_id,
        ts_snapshot=snapshot.ts_exchange,
        slices=tuple(tasks),
    )


def _to_slice_task(slice_data: SliceData, weighting: Weighting) -> SliceTask | None:
    """One expiry, inverted and weighted, or ``None`` if nothing in it survived.

    Ordered by strike, which is also ascending in ``k = ln(K / F)``, because ``SliceTask`` requires
    a strictly increasing moneyness axis and refuses to sort on arrival: its three tuples are
    addressable by index, and a sort touching one of them and not the others would pair a
    volatility with someone else's weight.

    A slice whose weights sum to zero is dropped here rather than allowed to raise. That happens
    when every quote in it is flagged and ``flagged_factor`` is zero, or when every vega underflows
    -- both are legal configurations meeting a legal market, and neither is a reason to fail the
    whole snapshot.
    """
    quotes = _out_of_the_money_quotes(slice_data)
    moneyness: list[float] = []
    vols: list[float] = []
    weights: list[float] = []

    for strike, quote, paired in quotes:
        vol = _invert(quote, slice_data)
        if vol is None:
            continue
        moneyness.append(math.log(strike / slice_data.forward))
        vols.append(vol)
        weights.append(_weight(quote, slice_data, vol, paired, weighting))

    total = sum(weights)
    if not vols or total <= 0:
        return None

    return SliceTask(
        expiry=slice_data.expiry,
        tenor_years=slice_data.tenor_years,
        forward=slice_data.forward,
        log_moneyness=tuple(moneyness),
        implied_vol=tuple(vols),
        weights=tuple(weight / total for weight in weights),
    )


def _out_of_the_money_quotes(slice_data: SliceData) -> list[tuple[float, QuoteData, bool]]:
    """Pick one quote per strike: the out-of-the-money leg where there is one.

    A call is out of the money above the forward and a put below it. **At the forward exactly the
    call is chosen**, arbitrarily and deliberately: at ``k = 0`` the two legs have identical time
    value and identical vega, so the choice cannot matter, and having a rule at all is what stops
    the boundary case from depending on dictionary order.

    Returns:
        ``(strike, quote, paired)`` ascending by strike, where ``paired`` says whether the strike
        also had its in-the-money leg quoted. ``paired`` is ``False`` in exactly two situations,
        and they are not the same: the twin was genuinely absent, or the only leg present was the
        in-the-money one -- which is the case the down-weighting is for. The first costs nothing:
        a lone out-of-the-money quote is the well-conditioned side and deserves full weight.
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
            # Present but alone is still the good side of the strike, so it is "paired" for
            # weighting purposes: the flag below means "we had to use the bad side", not "the
            # book was two-sided".
            chosen.append((strike, preferred, True))
            continue
        fallback = next(iter(legs.values()), None)
        if fallback is not None:
            chosen.append((strike, fallback, False))
    return chosen


def _invert(quote: QuoteData, slice_data: SliceData) -> float | None:
    """Our own Black-76 volatility for one quote, or ``None`` when the price admits none.

    ``NoImpliedVolError`` is routine rather than exceptional and is swallowed here on purpose: a
    mid from a crossed or stale book falls below intrinsic regularly, and a deep in-the-money
    quote sits close enough to a bound that one tick of noise crosses it. The quote is dropped
    from the slice, which is the outcome ``SliceResult.n_quotes_used`` exists to make visible.

    Undiscounted, matching ``black76``'s own default: the options this engine was built for settle
    in the same numeraire as their premium, so there is no cash leg to discount, and a discount
    invented here would be a rate assumption nobody chose. Note that it must be *the same* factor
    the target was quoted under, which is why it is not a knob on this function.
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


_KINDS: Mapping[OptionKind, OptionKindP] = {
    OptionKind.CALL: OptionKindP.CALL,
    OptionKind.PUT: OptionKindP.PUT,
}
"""Wire side to this context's own spelling. Two enums for one idea, again -- rule 6 forbids
importing Market Data's ``OptionKindD`` and rule 3 forbids the domain importing the contract's
``OptionKind``, so ``OptionKindP`` exists and this table is where the two meet."""


def _weight(
    quote: QuoteData,
    slice_data: SliceData,
    vol: float,
    paired: bool,
    weighting: Weighting,
) -> float:
    """How much this quote counts, before the slice is normalised.

    ``vega / (1 + spread_rel / spread_scale)``, cut by a factor for each thing wrong with the
    quote. The factors multiply rather than taking a minimum, so a flagged quote inverted from its
    in-the-money leg is penalised for both -- which is the honest reading, since the two problems
    are independent and compound.

    Vega is evaluated at the volatility just inverted, which is the volatility the quote actually
    implies rather than an approximation to it: at that point the model reproduces the observed
    price exactly, so the derivative is taken where the fit is anchored.
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


def to_calibrated_surface(
    task: CalibrationTask,
    accepted: Sequence[SliceResult],
    n_iterations: int,
    duration_ms: float,
    grid: GridSpec,
    producer_id: str,
    surface_id: str,
    ts_calibrated: datetime,
    status: SurfaceStatus,
) -> CalibratedSurface | None:
    """Evaluate the accepted slices on the published mesh and wrap them in the contract.

    ADR-001 in one function: the fitted parameters are an evaluable object and what crosses the
    boundary is a table of numbers, because an object cannot be serialised, cannot cross a process
    and cannot be read back out of a recording tomorrow.

    Args:
        task: The problem that was fitted. Read for two things a ``SliceResult`` does not carry:
            the identity of the snapshot, and the **forward** each expiry was fitted against. The
            forward has to travel with the published grid or its moneyness axis is a coordinate
            system with no origin, and recovering it later would pair this smile with a forward
            that has since moved.
        accepted: The slices the use case decided to publish, ascending in tenor. A subset of the
            result's own slices: acceptance is per slice (ADR-006), so a surface may go out
            describing the expiries that fitted while omitting one that did not.
        n_iterations: Optimiser iterations across the whole calibration. Zero is legitimate and
            is the warm-start case, so it must never be tested for truthiness.
        duration_ms: Wall-clock cost of the fit.
        grid: Where the published moneyness nodes sit.
        producer_id: Which calibrator produced this, for attribution only.
        surface_id: Identity of this result.
        ts_calibrated: When the fit finished, from the injected clock (ADR-004).
        status: Trust level, decided by the use case from the input's quality and the acceptance.

    Returns:
        The published surface, or ``None`` when not one accepted slice could be evaluated into a
        publishable row.

        That second case is narrow and real. ``SVIParams`` guarantees a non-negative minimum total
        variance and deliberately not a strictly positive one -- the constructor admits a
        collapsed slice so that the arbitrage metrics can measure it rather than being denied the
        object -- while ``VolGrid`` requires every published volatility to be strictly positive and
        finite. A slice that collapsed to zero variance somewhere on the mesh therefore cannot be
        published, and dropping that one row is better than either failing the whole surface or
        publishing a zero volatility that every consumer would divide by.
    """
    forwards = {slice_task.expiry: slice_task.forward for slice_task in task.slices}
    nodes = grid.nodes()

    rows: list[tuple[SliceResult, tuple[float, ...]]] = []
    for fitted in accepted:
        smile = tuple(fitted.params.implied_vol(k, fitted.tenor_years) for k in nodes)
        if all(math.isfinite(vol) and vol > 0 for vol in smile):
            rows.append((fitted, smile))

    if not rows:
        return None

    return CalibratedSurface(
        surface_id=surface_id,
        source_snapshot_id=task.snapshot_id,
        market_id=task.market_id,
        ts_snapshot=task.ts_snapshot,
        ts_calibrated=ts_calibrated,
        producer_id=producer_id,
        grid=VolGrid(
            log_moneyness=nodes,
            tenors=tuple(fitted.tenor_years for fitted, _ in rows),
            expiries=tuple(fitted.expiry for fitted, _ in rows),
            forwards=tuple(forwards[fitted.expiry] for fitted, _ in rows),
            vols=tuple(smile for _, smile in rows),
        ),
        fit=_to_fit_metrics([fitted for fitted, _ in rows], n_iterations, duration_ms),
        status=status,
        producer_meta=None,
    )


def _to_fit_metrics(
    published: Sequence[SliceResult], n_iterations: int, duration_ms: float
) -> FitMetrics:
    """Aggregate the per-slice diagnostics into the one block the contract publishes.

    The RMSE is pooled by quote count -- ``sqrt(sum(n_i * rmse_i^2) / sum(n_i))`` -- which is the
    root-mean-square of every residual in the surface and not the mean of the slices' own RMSEs. A
    plain average would give a thin front-week slice the same say as a hundred-quote one, and the
    published number is meant to answer "how far is this surface from the market", not "how did the
    expiries do on average".

    The maximum is the maximum, which is the whole reason it is published beside the RMSE: a mean
    absorbs one badly missed quote across the surface, and the wings are where a calibrator fails
    first and where a consumer's risk is most sensitive.
    """
    n_quotes = sum(fitted.n_quotes_used for fitted in published)
    pooled = math.sqrt(
        sum(fitted.n_quotes_used * fitted.rmse_vol_bp**2 for fitted in published) / n_quotes
    )
    worst = max(fitted.max_err_vol_bp for fitted in published)
    return FitMetrics(
        # Pooling is exact arithmetic on numbers already bounded by their own maxima, but floating
        # point can still land the pooled RMSE an ulp above a maximum that every slice met exactly
        # -- a synthetic chain fitted perfectly gives 0.0 against 0.0 -- and `FitMetrics` refuses
        # an RMSE above its maximum. The clamp is against that ulp and nothing else.
        rmse_vol_bp=min(pooled, worst),
        max_err_vol_bp=worst,
        n_quotes_used=n_quotes,
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
    """Announce that this producer could not fit this snapshot.

    A first-class outcome rather than an error channel: a calibrator that fails on hard chains is
    worse than one that does not, even when its successful fits are more accurate, so the failure
    is published and counted rather than logged and forgotten.
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

    Everything else is kept exactly as it was, and each field is deliberate. ``ts_snapshot`` stays
    old because it is what staleness is measured against downstream -- restamping it would make a
    surface from ten minutes ago look current, which is the single most dangerous thing this
    engine could do. ``ts_calibrated`` stays old for the same reason the contract documents: under
    this status it refers to the earlier fit, not to now. And ``surface_id`` stays the same,
    because this *is* that surface: a new id would suggest a new fit exists, and a consumer
    correlating results would count one calibration twice.

    Republishing beats silence. A consumer cannot distinguish a missing message from a quiet
    market, but it can act on a surface that says out loud that it is old.
    """
    return replace(surface, status=SurfaceStatus.STALE_REPUBLISH)
