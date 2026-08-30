"""The baseline fit: ``scipy.optimize.least_squares`` over SVI's free variables, one slice at a
time.

This is the first calibrator in the engine that is actually a calibrator, and its job is as much
to *validate the mathematics* as to produce surfaces: it is the reference the JAX implementation
of F3-A is measured against, on the same tasks, behind the same port (Design 5.7). Everything the
``Calibrator`` port promises holds here -- no I/O, no clock, no state between calls -- so the same
task fed twice gives the same answer bit for bit, and a recorded session replays without a reset
ritual.

**The five decisions that make up the fit** (Design 5.3, 5.4):

* **The residual is a volatility error in basis points, weighted by the task's own weights.** Not
  a price error: a whole currency unit is noise on a fat at-the-money premium and absurd in a wing
  worth a fraction of a tick, so a price-space fit is dominated by wherever the premiums happen to
  be largest. The weights arrive already built and already normalised (ADR-018) and are consumed
  exactly as given -- no calibrator re-derives them, which is what stops this implementation and
  the JAX one from silently weighting the same market differently.
* **Huber, not plain least squares.** Ingestion flags but does not filter (that is the standing
  rule), so junk reaches the loss by design. Squared error lets one absurd quote drag a whole
  slice; Huber charges it linearly past :attr:`FitSettings.huber_scale_bp` and the smile stays
  where the other twenty quotes say it is.
* **A soft butterfly penalty on a dense mesh**, from ``durrleman.durrleman_g``, one hinge residual
  per mesh point. Soft rather than hard, because the constraint is what the *fit* should be pushed
  towards, not what the *type* should refuse -- ``SVIParams`` deliberately admits an arbitrageable
  slice so that the violation stays measurable and the optimiser keeps its search path. Per point
  rather than one scalar depth, because ``max`` over the mesh has a moving argmax and a
  finite-difference Jacobian reads that as noise.
* **A ridge towards the starting point.** Five basis points per unit of free coordinate, which
  is nothing against a residual that means anything -- and everything on a market with no smile,
  where ``b -> 0`` leaves ``rho``, ``m`` and ``sigma`` unidentified, the cost surface flat in
  those directions, and the optimiser free to drift into a corner of the box that ``at_bound``
  then reports as a broken fit. See :attr:`FitSettings.ridge_bp`.
* **Unconstrained variables plus a practical box.** ``FreeParams`` already makes ``b, sigma > 0``
  and ``|rho| < 1`` structural, so no optimiser can propose ``rho = 1.4``. The box on top of it is
  the "practical bounds" of Design 5.4 -- a slice whose ``b`` has run to 4 is not a smile, it is a
  fit escaping -- and hitting one is what ``SliceResult.at_bound`` reports.

**Every point of R^5 must produce finite residuals**, and two do not on their own: an ``a`` deep
enough to make the minimum total variance negative (which ``SVIParams`` refuses outright) and a
slice whose total variance touches zero somewhere on the mesh (which ``durrleman_g`` refuses,
because ``g`` divides by ``w`` -- the seam recorded in ``docs/SEAMS.md``). Both are answered with
:data:`BARRIER_BP`, a residual so large that a trust-region step into that region is always
rejected, rather than by letting the exception escape and kill the whole surface for one bad
iterate. The barrier grows as ``a`` falls, so it carries a gradient back towards the admissible
set instead of being a flat wall the optimiser cannot see the far side of.

**What is not here.** No multi-start: Design 5.6 gives that to the cold JAX cycle, and what this
adapter does instead is retry a *failed warm start* from its own cold guess, which is the same
idea at the scale a scipy baseline can afford. No cross-slice regularisation either: ADR-008 fits
each expiry independently, so a thin slice is regularised by **pinning ``m`` and ``sigma``** at the
start values rather than by pulling on its neighbour -- the alternative ADR-008 offers, and the one
that keeps ``calibrate`` a per-slice function with no coupling to invent.

**Tuning is injected, not buried** (:class:`FitSettings`). The Huber scale, the penalty weight and
the mesh are empirical and every one of them will move against real Deribit data; they arrive
through the constructor so that F2-07 can feed them from TOML without reopening this file. The
*bounds*, by contrast, are module constants on purpose: ``at_bound`` is half of the acceptance
rule and ADR-021 says that half is not configurable, so a deployment that could widen a bound
until nothing was ever pinned is a deployment that eventually does.

Every mechanism above that ``Plan.md`` and ``Implementation.md`` do not name -- the ridge, the
residual scaling, the cold retry, the barrier, the non-configurable box, and ``n_iterations``
carrying an evaluation count -- is collected in **ADR-027**, which is where the reasoning is
recorded once rather than inferred from five docstrings.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, replace
from datetime import datetime
from typing import Final

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import least_squares

from volengine.parametric_pricing.domain.calibration import (
    CalibrationResult,
    CalibrationTask,
    SliceResult,
    SliceTask,
)
from volengine.parametric_pricing.domain.durrleman import durrleman_g
from volengine.parametric_pricing.domain.errors import CalibrationError
from volengine.parametric_pricing.domain.svi_slice import FreeParams, SVIParams

PRODUCER_ID: Final[str] = "svi-scipy"
"""How this producer names itself everywhere downstream: the topic its surfaces are published on,
the tag on its metrics, and the ``producer_id`` a risk report says it trusted."""

BASIS_POINTS_PER_VOL: Final[float] = 10_000.0
"""One basis point of volatility is ``0.0001`` in decimal terms, which is the unit every fit
metric in this context -- and every residual below -- is stated in."""

BARRIER_BP: Final[float] = 1.0e6
"""Residual charged at a point where the model is not evaluable at all.

A hundred vol *units* against fit errors measured in basis points, so the cost of an inadmissible
iterate is beyond anything a real slice can produce even after Huber flattens it to linear growth.
It has to be finite: ``least_squares`` refuses a non-finite residual outright, and a NaN returned
from the residual function is exactly the failure this repository keeps rediscovering -- it does
not raise, it makes the fit stop somewhere arbitrary.
"""

# --- practical parameter bounds (Design 5.4). Not configurable, by ADR-021.

A_LIMIT: Final[float] = 4.0
"""Largest total-variance level the fit may reach, either sign.

Four is a 200% volatility at a one-year tenor and a 630% one at a month -- far past any market
this engine is meant for, which is what a *practical* bound should be: never binding on a real
slice, and unmistakable when it binds. Negative values are legal for ``a`` on its own (with
``b > 0`` the square-root term lifts the curve back above zero), so the box is symmetric.
"""

B_MAX: Final[float] = 4.0
"""Largest wing steepness. ``b`` is half the difference of the two asymptotic slopes of ``w``, and
a slice claiming four units of total variance per unit of log-moneyness has stopped describing a
smile. This is the bound Design 5.4 names explicitly: ``b`` is what explodes on short chains,
where a handful of near-expiry quotes can be interpolated by an arbitrarily sharp V."""

RHO_MAX: Final[float] = 0.999
"""Largest skew magnitude. The reparameterisation already keeps ``|rho| < 1``, but it approaches
the bound so slowly that an optimiser can spend its whole budget walking out to ``rho_raw = 30``
without the fit improving. Stopping at 0.999 turns that walk into a reportable ``at_bound``."""

M_LIMIT: Final[float] = 2.0
"""Furthest the smile's minimum may sit from the forward, in log-moneyness. A crypto chain quotes
roughly ``|k| < 1``; a minimum at ``k = 2`` is outside the data and the curve there is fitting the
shape of its own asymptote."""

SIGMA_MIN: Final[float] = 1.0e-3
"""Sharpest admissible bottom. As ``sigma`` falls the curve tends to a kink at ``m``, whose second
derivative -- the quantity the butterfly condition is written in -- blows up; the fit becomes
ill-conditioned long before that. A thousandth of a log-moneyness unit is already far sharper than
any quoted smile."""

SIGMA_MAX: Final[float] = 5.0
"""Widest admissible bottom. Beyond this the curvature term is flat across the whole quoted band,
``sigma`` stops being identifiable and the optimiser is free to wander along a valley of equal
cost."""

BOUND_RTOL: Final[float] = 1.0e-4
"""How close to a bound counts as pinned, relatively.

Generous on purpose. ``trf`` keeps its iterates strictly feasible and returns a point that
approaches an active bound rather than one that sits exactly on it, so an exact comparison would
report ``at_bound = False`` for precisely the fits the rule exists to catch. A tenth of a
per mille of a bound that is itself an order of magnitude past any real market is not a value a
healthy slice reaches by accident.
"""

BOUND_ATOL: Final[float] = 1.0e-8
"""Absolute floor on the same test, so that a bound near zero is still comparable."""


def _free_of(b: float = 1.0, rho: float = 0.0, sigma: float = 1.0) -> FreeParams:
    """The free image of a parameter triple, through the domain's own mapping.

    The bounds above are stated where they are meaningful -- in ``b``, ``rho`` and ``sigma`` -- but
    the optimiser searches the free variables, so they have to be carried across. ``softplus`` and
    ``tanh`` are strictly increasing, so a box maps to a box and the two statements are the same
    constraint. Going through ``SVIParams.to_free`` rather than writing ``log(expm1(x))`` here is
    the point: this module owns no copy of the reparameterisation, and cannot drift from it.

    ``a = 0`` in the probe because the minimum total variance is then
    ``b * sigma * sqrt(1 - rho^2)``, which is non-negative for every admissible triple, so the
    constructor can never refuse the probe itself.
    """
    return SVIParams(a=0.0, b=b, rho=rho, m=0.0, sigma=sigma).to_free()


LOWER: Final[NDArray[np.float64]] = np.array(
    [
        -A_LIMIT,
        -np.inf,
        -_free_of(rho=RHO_MAX).rho_raw,
        -M_LIMIT,
        _free_of(sigma=SIGMA_MIN).sigma_raw,
    ]
)
"""Lower box in free coordinates, ordered as :func:`_to_vector` orders them.

``b_raw`` is unbounded below, and that absence is deliberate: ``b = 0`` is the legal flat slice a
market with no smile really produces, and a finite floor would report it as a pinned parameter and
have ADR-006 refuse a perfectly honest fit.
"""

UPPER: Final[NDArray[np.float64]] = np.array(
    [
        A_LIMIT,
        _free_of(b=B_MAX).b_raw,
        _free_of(rho=RHO_MAX).rho_raw,
        M_LIMIT,
        _free_of(sigma=SIGMA_MAX).sigma_raw,
    ]
)
"""Upper box in free coordinates."""

B_START_MIN: Final[float] = 0.05
"""Floor applied to ``b`` in the *starting point* only, never as a bound.

The one trap of an otherwise structural reparameterisation: ``softplus'`` underflows to zero for a
very negative ``b_raw``, so a fit started at ``b = 0`` has an exactly zero Jacobian column for the
wings and can never leave the flat slice, whatever the market says. It would converge, report a
respectable-looking flat fit and never be noticed. Starting a little way up the curve costs
nothing -- the optimiser is free to walk straight back down, and does on a genuinely flat chain.
"""

SIGMA_START: Final[float] = 0.10
"""Curvature the cold start assumes: a moderately rounded bottom, roughly a tenth of the quoted
band of a crypto chain. Only a starting point; nothing downstream depends on the value."""

RHO_START: Final[float] = -0.30
"""Skew the cold start assumes. Negative, because a lifted downside wing is the shape of every
equity and crypto smile; starting at zero makes the first steps fight the data on both wings at
once."""


@dataclass(frozen=True, slots=True)
class FitSettings:
    """The empirical half of the fit: everything Design 5.3 leaves to tuning.

    Every field has a default, so ``FitSettings()`` is a working calibrator and a test bends the
    one knob it is about. They are constructor arguments rather than module constants because they
    will move against real data (that is what makes them empirical) and because ADR-012 wants a
    number a deployment may want to change to arrive from outside the code. They are *not* in
    ``entrypoints/config.py`` yet: F2-07 owns the TOML, and inventing a section here that no file
    reads would be the guesswork ADR-012 exists to prevent. Recorded in ``docs/SEAMS.md``.
    """

    huber_scale_bp: float = 100.0
    """Vol error, in basis points, at which a residual stops being treated as ordinary noise.

    Below it Huber is exactly least squares; above it the cost grows linearly, so one absurd quote
    can no longer buy the optimiser more than a bounded amount of movement. A hundred basis points
    is a full vol point of error -- several times a good fit and several times a crypto bid-ask --
    which puts the transition where genuine misfit ends and junk begins. Positive and finite:
    ``least_squares`` divides by it.
    """

    durrleman_penalty_bp: float = 10_000.0
    """Vol basis points charged per unit depth of butterfly violation.

    The exchange rate between two quantities that have no natural common unit: ``g`` is in the
    units of a density and the residuals are in basis points of volatility. At the default a
    violation of 0.01 costs the same as roughly a hundred and forty basis points spread over the
    mesh, which is heavy enough that a fit will give up real accuracy to leave the arbitrageable
    region and light enough that it does not simply refuse to bend into a steep wing.

    Zero is legal and turns the penalty off, which is how a test isolates the fit term.
    """

    durrleman_mesh_nodes: int = 51
    """Points on the penalty mesh. At least two.

    ``butterfly_violation`` and this penalty are both grid measures and can only see what they
    sample: a dip between two nodes is invisible. Fifty-one across a band a little over two units
    wide puts a node every four percent of log-moneyness, which is finer than any smile feature a
    five-parameter form can produce.
    """

    durrleman_mesh_margin: float = 0.5
    """How far past the outermost quote the mesh reaches, in log-moneyness. Non-negative, finite.

    Not zero, and this is the whole reason the mesh is not simply the quoted strikes. The
    published grid is deliberately wider than the quoted band (ADR-001), so the wings a consumer
    prices against are extrapolation -- and extrapolated wings are exactly where raw SVI produces
    a negative density. Penalising only where quotes exist would leave the fit free to be
    arbitrage-free precisely where nobody looks.
    """

    min_quotes_for_free_shape: int = 5
    """Below this many quotes, ``m`` and ``sigma`` are pinned at their starting values.

    Five parameters through four points is not a fit, it is an interpolation with a spare degree
    of freedom, and the shape parameters are the two that absorb it -- the bottom slides sideways
    and flattens to pass through whatever is there. ADR-008 offers two answers to risk 3 of
    Design 11 and this is the one that keeps slices independent: pinning, rather than a pull
    towards a neighbour that ``calibrate`` would have to invent a coupling for. At five the
    problem is exactly determined and the level, wings and skew are still fitted, which is what a
    thin slice can honestly support.
    """

    ridge_bp: float = 5.0
    """Vol basis points charged per unit of movement away from the starting point, measured in
    free coordinates. Non-negative and finite.

    A very small Tikhonov pull, and the answer to a failure that is invisible until it happens:
    **a market with no smile has no ``rho``, no ``m`` and no ``sigma``.** As ``b`` goes to zero
    the curve is ``w(k) = a`` whatever the other three say, the cost surface becomes a plateau in
    those directions, and the optimiser drifts along it until it hits the side of the box --
    which ``at_bound`` then reports, and ADR-006 refuses a fit whose RMSE was perfectly healthy.
    On a flat chain with twenty basis points of noise that happens on most seeds without a ridge,
    on one in twenty at ``1.0``, and on none at the default -- while the parameters recovered from
    a clean smile move in the sixth decimal and the fitted RMSE stays under a thousandth of a
    basis point.

    It is deliberately negligible against a real residual -- a whole unit of free coordinate costs
    what five basis points of vol error costs, and a fit that is genuinely determined moves as far
    as it likes -- which is what keeps this a tie-breaker rather than a prior. Its second
    effect is the one a streaming engine wants anyway: under a warm start the tie breaks
    towards *yesterday's* parameters, so consecutive surfaces do not jump between equally
    good answers.

    Zero is legal and turns it off, which is how a test shows what it prevents.
    """

    max_nfev: int = 500
    """Residual evaluations allowed per attempt. At least one.

    The budget whose exhaustion is what ``SliceResult.converged`` reports as ``False``. With a
    finite-difference Jacobian each iteration costs about six evaluations, so five hundred is
    around eighty iterations -- far past what a warm-started slice needs and enough that a cold
    start on a difficult chain is not cut off while still improving.
    """

    def __post_init__(self) -> None:
        if not math.isfinite(self.huber_scale_bp) or self.huber_scale_bp <= 0:
            raise ValueError(
                f"The Huber scale must be positive and finite, got {self.huber_scale_bp}"
            )
        if not math.isfinite(self.durrleman_penalty_bp) or self.durrleman_penalty_bp < 0:
            raise ValueError(
                "The Durrleman penalty must be non-negative and finite, got "
                f"{self.durrleman_penalty_bp}"
            )
        if self.durrleman_mesh_nodes < 2:
            raise ValueError(
                f"The penalty mesh needs at least two nodes, got {self.durrleman_mesh_nodes}"
            )
        if not math.isfinite(self.durrleman_mesh_margin) or self.durrleman_mesh_margin < 0:
            raise ValueError(
                "The penalty mesh margin must be non-negative and finite, got "
                f"{self.durrleman_mesh_margin}"
            )
        if not math.isfinite(self.ridge_bp) or self.ridge_bp < 0:
            raise ValueError(f"The ridge must be non-negative and finite, got {self.ridge_bp}")
        if self.min_quotes_for_free_shape < 0:
            raise ValueError(
                f"The pinning threshold cannot be negative, got {self.min_quotes_for_free_shape}"
            )
        if self.max_nfev < 1:
            raise ValueError(f"The evaluation budget must be at least one, got {self.max_nfev}")


@dataclass(frozen=True, slots=True)
class _Attempt:
    """One run of the optimiser over one slice, plus what it cost.

    Separate from ``SliceResult`` because the retry below has to *compare* two runs and then
    report the cost of both, and a result carries no evaluation count of its own.
    """

    result: SliceResult
    nfev: int


def _to_vector(free: FreeParams) -> NDArray[np.float64]:
    """Free parameters as the optimiser's vector, in the order the boxes above are written in."""
    return np.array([free.a, free.b_raw, free.rho_raw, free.m, free.sigma_raw], dtype=np.float64)


def _from_vector(x: NDArray[np.float64]) -> FreeParams:
    """The inverse of :func:`_to_vector`. Raises ``ValueError`` on a non-finite component, which
    is why every caller checks finiteness first and answers with the barrier instead."""
    return FreeParams(
        a=float(x[0]), b_raw=float(x[1]), rho_raw=float(x[2]), m=float(x[3]), sigma_raw=float(x[4])
    )


def _clipped(free: FreeParams) -> NDArray[np.float64]:
    """A starting point inside the box, with ``b`` lifted off the flat floor.

    A warm start is the previous cycle's *accepted* parameters, so it is admissible -- but the box
    is this adapter's own and nothing guarantees the previous fit respected it (a different
    calibrator's result, or these constants changed between releases). ``least_squares`` refuses an
    infeasible ``x0`` outright, so it is clipped rather than rejected: an out-of-box warm start is
    still far better information than a cold guess.

    The ``b`` floor is the underflow trap :data:`B_START_MIN` documents, and it applies to a warm
    start for exactly the same reason it applies to a cold one -- a slice that fitted flat
    yesterday would otherwise be unable to grow wings today.
    """
    x = np.clip(_to_vector(free), LOWER, UPPER)
    x[1] = max(x[1], _free_of(b=B_START_MIN).b_raw)
    return x


def _cold_start(task: SliceTask) -> NDArray[np.float64]:
    """A first guess read off the quotes themselves: level, span and the position of the minimum.

    Not a constant, and not a multi-start. The cheapest reliable information about a slice is in
    the data it is going to be fitted to: the lowest observed total variance is roughly the bottom
    of the curve, the spread of total variance across the quoted band is roughly what the wings
    have to cover, and the strike where the minimum sits is roughly ``m``. Everything else --
    curvature and skew -- starts at a typical crypto shape, because nothing in the data pins them
    without already solving the problem.

    The level is set so that the *minimum* of the curve lands on the lowest observed total
    variance rather than ``a`` itself, since ``a`` is the intercept and sits roughly
    ``b * sigma`` below the bottom -- exactly that for a symmetric smile, and near enough for a
    starting point on a skewed one. Floored at zero, which keeps ``SVIParams`` constructible
    whatever the arithmetic produces.
    """
    k = np.asarray(task.log_moneyness, dtype=np.float64)
    vol = np.asarray(task.implied_vol, dtype=np.float64)
    w = vol * vol * task.tenor_years

    span_k = float(k.max() - k.min())
    span_w = float(w.max() - w.min())
    b = min(max(span_w / span_k if span_k > 0 else 0.0, B_START_MIN), B_MAX)
    m = min(max(float(k[int(np.argmin(w))]), -M_LIMIT), M_LIMIT)
    a = min(max(float(w.min()) - b * SIGMA_START, 0.0), A_LIMIT)

    return _clipped(SVIParams(a=a, b=b, rho=RHO_START, m=m, sigma=SIGMA_START).to_free())


def _mesh(task: SliceTask, settings: FitSettings) -> NDArray[np.float64]:
    """The moneyness mesh the butterfly penalty is evaluated on: the quoted band plus a margin."""
    k = np.asarray(task.log_moneyness, dtype=np.float64)
    return np.linspace(
        float(k.min()) - settings.durrleman_mesh_margin,
        float(k.max()) + settings.durrleman_mesh_margin,
        settings.durrleman_mesh_nodes,
    )


def _errors_bp(params: SVIParams, task: SliceTask) -> NDArray[np.float64]:
    """Signed volatility error of the model against every quote, in basis points."""
    k = np.asarray(task.log_moneyness, dtype=np.float64)
    vol = np.asarray(task.implied_vol, dtype=np.float64)
    return (params.implied_vol(k, task.tenor_years) - vol) * BASIS_POINTS_PER_VOL


def _residuals(
    x: NDArray[np.float64], task: SliceTask, mesh: NDArray[np.float64], settings: FitSettings
) -> NDArray[np.float64]:
    """The vector ``least_squares`` minimises the (Huber-transformed) sum of squares of.

    Two blocks, concatenated: one residual per quote and one per mesh point.

    The fit block is ``sqrt(n * weight_i) * error_i`` in basis points. The weights are already
    normalised to one, so ``sum(w_i * e_i^2)`` is the weighted mean square and the ``n`` puts each
    residual back on the scale of an *individual* error -- which is what makes
    :attr:`FitSettings.huber_scale_bp` mean "an error of this many basis points" regardless of how
    many strikes the chain happens to quote today. Without it the same setting would be a
    different robustness rule on a twelve-strike slice and on a forty-strike one.

    The penalty block is ``penalty * max(0, -g(k_j)) / sqrt(nodes)``: zero wherever the slice is
    arbitrage-free, growing with the depth of the violation, and divided so that refining the mesh
    measures the same violation more finely instead of charging more for it.

    Every escape from this function is a finite array of the same length. That is not defensive
    noise: ``least_squares`` raises on a non-finite residual, so an unguarded ``inf`` at one
    iterate destroys a whole surface, and a NaN would quietly pass the ordering tests inside the
    optimiser rather than raising at all.
    """
    n_total = len(task.log_moneyness) + mesh.size
    if not np.all(np.isfinite(x)):
        return np.full(n_total, BARRIER_BP)

    try:
        params = SVIParams.from_free(_from_vector(x))
    except ValueError:
        # The one constraint the reparameterisation cannot make structural: a minimum total
        # variance below zero, which couples `a` with the other four. The barrier rises as `a`
        # falls, so the Jacobian at an inadmissible point still points back towards the surface.
        return np.full(n_total, BARRIER_BP * (1.0 + max(0.0, -float(x[0]))))

    weights = np.asarray(task.weights, dtype=np.float64)
    fit = np.sqrt(weights.size * weights) * _errors_bp(params, task)

    try:
        violation = np.maximum(0.0, -durrleman_g(params, mesh))
    except ValueError:
        # A slice whose total variance touches zero somewhere on the mesh: `g` divides by `w`
        # twice and there is no number to return (docs/SEAMS.md). Legal for `SVIParams`, useless
        # to a fit, and pushed away from rather than raised through.
        penalty = np.full(mesh.size, BARRIER_BP)
    else:
        penalty = settings.durrleman_penalty_bp * violation / math.sqrt(mesh.size)

    out: NDArray[np.float64] = np.concatenate([fit, penalty])
    if not np.all(np.isfinite(out)):
        return np.full(n_total, BARRIER_BP)
    return out


def _at_bound(x: NDArray[np.float64], searched: tuple[int, ...]) -> bool:
    """Whether any *searched* coordinate finished against a finite side of the box.

    Only the searched ones. A pinned ``m`` or ``sigma`` never moved, so reporting it as pinned by
    the optimiser would mean something the acceptance rule does not: ADR-006 refuses a fit that
    was *stopped* at the edge of the admissible region, not one that was held at a value chosen
    before the search began.
    """
    return any(
        bool(np.isclose(x[i], bound, rtol=BOUND_RTOL, atol=BOUND_ATOL))
        for i in searched
        for bound in (LOWER[i], UPPER[i])
        if np.isfinite(bound)
    )


def _fit_metrics(params: SVIParams, task: SliceTask) -> tuple[float, float, int]:
    """``(rmse_vol_bp, max_err_vol_bp, n_quotes_used)`` for a fitted slice.

    Both errors are computed over the quotes that **entered the fit**, meaning those with a
    strictly positive weight. A zero weight is how the ACL keeps a flagged quote visible without
    letting it steer the parameters (``SliceTask.weights``); reporting the model's error against
    one would be reporting how badly we missed something we deliberately ignored, and on a slice
    where every wing quote was flagged that number would dominate the maximum and mean nothing.

    The RMSE is weighted -- it is the loss's own view of the fit, which is what an acceptance
    threshold should be compared against -- and the maximum is not, because its whole job is to
    expose the single quote the mean absorbed. The invariant ``max >= rmse`` survives that, since
    a weighted mean of squares over a set cannot exceed the largest square in it.
    """
    weights = np.asarray(task.weights, dtype=np.float64)
    used = weights > 0.0
    errors_bp = np.abs(_errors_bp(params, task))

    rmse = float(np.sqrt(np.average(np.square(errors_bp), weights=weights)))
    return rmse, float(errors_bp[used].max()), int(used.sum())


def _run(task: SliceTask, start: NDArray[np.float64], settings: FitSettings) -> _Attempt:
    """One optimisation of one slice from one starting point.

    Pinning, when the slice is too thin for five free parameters, is done by searching a subvector
    and writing it back into a fixed template: ``m`` and ``sigma`` keep whatever the start gave
    them -- the previous cycle's values under a warm start, the cold heuristic otherwise -- and
    the level, the wings and the skew are fitted around them.

    Raises:
        CalibrationError: If the point the optimiser returns cannot be turned back into
            ``SVIParams``. Unreachable in principle -- the barrier makes any inadmissible point
            cost more than the admissible ``x0`` the search began at, and ``trf`` never returns a
            point worse than the one it started from -- and reported rather than papered over,
            because a silent fallback here would publish a slice nobody fitted.
    """
    weights = np.asarray(task.weights, dtype=np.float64)
    searched: tuple[int, ...] = (
        (0, 1, 2)
        if int((weights > 0.0).sum()) < settings.min_quotes_for_free_shape
        else (0, 1, 2, 3, 4)
    )
    mesh = _mesh(task, settings)
    template = start.copy()
    index = list(searched)

    def residuals(sub: NDArray[np.float64]) -> NDArray[np.float64]:
        x = template.copy()
        x[index] = sub
        # The ridge lives here rather than in `_residuals` because it is the only term that
        # depends on where the search began, and because it is only ever applied to the
        # coordinates actually being searched: a pinned parameter cannot drift, so charging it
        # for a distance it never travelled would be arithmetic with no meaning.
        ridge = settings.ridge_bp * (sub - start[index])
        return np.concatenate([_residuals(x, task, mesh, settings), ridge])

    solution = least_squares(
        residuals,
        start[index],
        bounds=(LOWER[index], UPPER[index]),
        loss="huber",
        f_scale=settings.huber_scale_bp,
        max_nfev=settings.max_nfev,
    )

    fitted = template.copy()
    fitted[index] = solution.x
    try:
        params = SVIParams.from_free(_from_vector(fitted))
    except ValueError as bad:  # pragma: no cover - see the docstring
        raise CalibrationError(
            f"The optimiser returned an inadmissible slice for expiry {task.expiry}: {bad}"
        ) from bad

    rmse_vol_bp, max_err_vol_bp, n_quotes_used = _fit_metrics(params, task)
    return _Attempt(
        result=SliceResult(
            expiry=task.expiry,
            tenor_years=task.tenor_years,
            params=params,
            rmse_vol_bp=rmse_vol_bp,
            max_err_vol_bp=max_err_vol_bp,
            n_quotes_used=n_quotes_used,
            # `status` is 0 when the evaluation budget ran out and positive when a tolerance was
            # met; -1 is improper input, which would be a bug here rather than a market condition.
            converged=int(solution.status) > 0,
            at_bound=_at_bound(fitted, searched),
        ),
        nfev=int(solution.nfev),
    )


def _healthy(result: SliceResult) -> bool:
    """The two structural halves of ADR-006's acceptance rule, without the configured threshold.

    The RMSE half is deliberately absent: the threshold is TOML (ADR-012) and belongs to the use
    case, and a calibrator that read it would take the publication decision away from the layer
    that owns it. What is left is enough to answer the only question asked here -- is this attempt
    worth retrying from somewhere else?
    """
    return result.converged and not result.at_bound


def _preferred(first: SliceResult, second: SliceResult) -> SliceResult:
    """Which of two attempts on the same slice to keep.

    A converged, unpinned fit beats one that is not, whatever the residuals say: a fit that
    stopped against a bound reports the residual of a projection onto that bound, and comparing it
    against an honest one on RMSE alone would prefer the projection whenever the bound happened to
    sit near the data. Among two of the same kind, the lower RMSE wins, and ties go to the first
    -- which is the warm start, so a calm market keeps returning the same parameters instead of
    oscillating between two equally good answers on floating-point noise.
    """
    if _healthy(first) != _healthy(second):
        return first if _healthy(first) else second
    return first if first.rmse_vol_bp <= second.rmse_vol_bp else second


class ScipyCalibrator:
    """A ``Calibrator`` backed by ``scipy.optimize.least_squares``, one independent fit per slice.

    Structural conformance, like every adapter in this engine: a ``producer_id`` property and a
    ``calibrate`` method, no base class, and nothing here the domain could import back. Stateless
    between calls -- the settings are configuration, not memory -- so two runs over the same
    recording produce the same surfaces.
    """

    def __init__(self, settings: FitSettings | None = None, producer_id: str = PRODUCER_ID) -> None:
        """Args:
        settings: The empirical knobs of the loss. Defaults to :class:`FitSettings`'s own
            defaults, which are a working crypto configuration.
        producer_id: What this instance calls itself. A parameter rather than a constant so
            that two of them can run side by side under different names -- which is the
            arrangement Design 5.7 exists to compare, and which the composition root keys its
            topics and its thread pools by.
        """
        if not producer_id.strip():
            raise ValueError(f"The producer id must not be empty, got {producer_id!r}")
        self._producer_id = producer_id
        self._settings = FitSettings() if settings is None else settings

    @property
    def producer_id(self) -> str:
        return self._producer_id

    @property
    def settings(self) -> FitSettings:
        """The knobs this instance was built with. Read-only, and read by the tests that assert
        what a setting does; nothing in the pipeline needs it."""
        return self._settings

    def calibrate(
        self,
        previous: Mapping[datetime, SVIParams] | None,
        task: CalibrationTask,
    ) -> CalibrationResult:
        """Fit every slice of the task, warm-starting each expiry that has a history.

        Args:
            previous: The parameters each expiry was last fitted to, keyed by expiry. ``None`` is
                a cold start, and so is a key this task does not carry -- a tenor that was born
                since the last cycle (ADR-013), which is an ordinary event rather than an error.
            task: The slices to fit, already inverted, weighted and ordered by tenor.

        Returns:
            One ``SliceResult`` per slice, in the task's own order, which is what keeps the result
            ordered by tenor as its constructor requires. **Including the slices that fitted
            badly**: the acceptance rule of ADR-006 is the use case's, and it needs to see the
            rejects to report them.

            ``duration_ms`` is zero and deliberately not measured, exactly as in
            ``adapters/flat_vol.py``: reading a clock is the one thing this port forbids outright
            and it would make two runs over the same recording differ. The number a consumer sees
            is the use case's own measurement of the call, taken from the injected ``Clock``.

            ``n_iterations`` is the total residual evaluation count across every attempt, retries
            included. ``least_squares`` reports no iteration count of its own, and the evaluation
            count is the honest measure of what a fit cost -- it is also the quantity the JAX
            comparison of Design 5.7 will be read against. The field's unit is the producer's own
            for exactly this reason, and the change of meaning is recorded in ADR-027: unlike the
            flat calibrator's, this producer's count is never zero, since the residual is
            evaluated at ``x0`` before anything is decided.

        Raises:
            CalibrationError: Only if the optimiser returns a point that is not a surface. A poor
                fit is a return value, never an exception.
        """
        attempts = [self._fit(one, (previous or {}).get(one.expiry)) for one in task.slices]
        return CalibrationResult(
            slices=tuple(one.result for one in attempts),
            n_iterations=sum(one.nfev for one in attempts),
            duration_ms=0.0,
        )

    def _fit(self, task: SliceTask, warm: SVIParams | None) -> _Attempt:
        """One expiry, warm-started if it has a history, retried cold if that warm start failed.

        Design 5.6's cold cycle is "periodic or after a failure", and this is the second half of
        it at the scale a scipy baseline can afford. A warm start is the right default -- it is
        what makes a calm market cost almost nothing, and it is what keeps consecutive surfaces
        from jumping between equally good optima -- but it is also how an optimiser gets stuck: the
        previous cycle's parameters can sit in a basin the market has since left, and from there
        the search runs out of budget or walks into a bound and reports exactly that. Retrying from
        the data's own geometry costs one more fit on the cycles that were going to be rejected
        anyway, and nothing at all on the cycles that were not.

        The evaluation counts of both attempts are added, whichever result wins: what the cycle
        cost is not the same question as which answer it kept.
        """
        cold = _cold_start(task)
        if warm is None:
            return _run(task, cold, self._settings)

        attempt = _run(task, _clipped(warm.to_free()), self._settings)
        if _healthy(attempt.result):
            return attempt

        retry = _run(task, cold, self._settings)
        return replace(
            attempt,
            result=_preferred(attempt.result, retry.result),
            nfev=attempt.nfev + retry.nfev,
        )
