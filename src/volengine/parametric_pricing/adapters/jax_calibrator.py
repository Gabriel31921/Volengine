"""The fast fit: one JIT-compiled function, every slice of the surface at once (Design 5.5-5.7).

The second implementer of the ``Calibrator`` port, behind the same signature as the scipy baseline
and handed byte-identical inputs, which is what makes the comparison of Design 5.7 a comparison of
mathematics rather than of plumbing. Everything the port promises holds here too: no I/O, no clock,
no state between calls, so a recorded session replays bit for bit.

**Compiled once, executed forever.** ``jax.jit`` specialises on input *shape*, and a recompilation
costs hundreds of milliseconds to seconds -- while an option chain changes composition on nearly
every snapshot. ADR-009's answer is ``adapters/padding.py``: every task, whatever it holds, becomes
the same rectangle plus a boolean mask, so the compiled signature never moves. This module closes
that loop by compiling in the **constructor**, against dummy arrays of exactly the padded shape, so
the cost lands at start-up where it belongs instead of inside the first snapshot's latency budget.
The permanent guard on it is ``test_jax_calibrator.py``'s recompilation counter: the failure is
silent -- every number stays correct, it just gets slow enough to defeat the entire reason for
using JAX -- and risk 5 of Design 11 is that failure.

**Two cycles, as Design 5.6 sets them out.**

* **Hot**, per snapshot: start from the previous cycle's parameters, Adam, a small step budget, and
  an early stop when the RMSE stops moving by more than a configured number of basis points of
  volatility. Interpretable units, so the tolerance can be argued against a bid-ask spread rather
  than against a floating-point epsilon.
* **Cold**, on the first cycle of a market and after a warm start fails: a deterministic
  multi-start with L-BFGS, which is what gets a slice out of the basin its previous parameters are
  stuck in. Deterministic and not seeded: a multi-start optimiser is welcome, a multi-start
  optimiser seeded from a random draw would break ADR-004's replay outright, so the extra starts
  are fixed offsets in the free coordinates rather than samples.

**Where this implementation genuinely differs from the baseline**, and why:

* **The free space makes admissibility structural.** ``FreeParams`` leaves ``a`` unconstrained, so
  the one SVI condition that couples the parameters -- a non-negative minimum total variance --
  has to be enforced from outside, and the scipy adapter does it with a finite barrier. A barrier
  is a wall to a trust region and a cliff to a gradient method: Adam would step off it and
  ``jnp.sqrt`` would answer with a NaN that propagates through the whole batch. So this adapter
  searches ``(w_min, b, rho, m, sigma)`` through softplus and tanh, where ``w_min`` is the minimum
  total variance itself and ``a`` is recovered from it. **Every point of R^5 is a valid slice**,
  there is no inadmissible region to guard, and no gradient is ever taken at a point the model
  cannot be evaluated at. The parametrisation is an implementation's business -- the port says so
  in as many words -- and what crosses the boundary is ``SVIParams``, identically for both
  producers.
* **The whole surface is one batched call.** ``vmap`` over the padded slice axis, so sixteen
  expiries cost what one costs; the slices remain mathematically independent (ADR-008), because
  Adam is elementwise and each slice's L-BFGS is its own vmapped instance rather than a corner of
  a joint Hessian.
* **Single precision.** JAX's default, unchanged, because ``jax_enable_x64`` is a process-wide
  mutation triggered by an import. The consequence is confined by computing every *reported*
  number -- the RMSE, the maximum error, the bound test -- on the host in ``float64`` through the
  domain's own ``SVIParams.implied_vol``. So the two calibrators' metrics are produced by the same
  code at the same precision and the comparison is not measuring a dtype.

**What is shared with the baseline on purpose: the practical bounds.** They are imported from
``scipy_calibrator`` rather than restated. ``at_bound`` is half of ADR-006's acceptance rule and
ADR-027 argues that a bound is the *ruler* the rule is read against, not a threshold a deployment
tunes -- two producers judged by two rulers would be two acceptance rules wearing one name. The
search here is unconstrained, so the test is "at or past the bound" rather than "stopped at it",
which is the same statement about the same region: the optimiser wanted to leave the region a fit
is allowed to live in.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import datetime
from functools import cache
from typing import Final, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import optax
from numpy.typing import NDArray
from optax import tree_utils as otu

from volengine.parametric_pricing.adapters.jax_black76 import DTYPE, TINY
from volengine.parametric_pricing.adapters.padding import PaddedTask, PadShape, pad
from volengine.parametric_pricing.adapters.scipy_calibrator import (
    A_LIMIT,
    B_MAX,
    BOUND_ATOL,
    BOUND_RTOL,
    M_LIMIT,
    RHO_MAX,
    SIGMA_MAX,
    SIGMA_MIN,
)
from volengine.parametric_pricing.domain.calibration import (
    CalibrationResult,
    CalibrationTask,
    SliceResult,
)
from volengine.parametric_pricing.domain.svi_slice import SVIParams

PRODUCER_ID: Final[str] = "svi-jax"
"""How this producer names itself everywhere downstream: the topic its surfaces are published on,
the tag on its metrics, and the ``producer_id`` a risk report says it trusted."""

BASIS_POINTS_PER_VOL: Final[float] = 10_000.0
"""One basis point of volatility is ``0.0001`` in decimal terms, which is the unit every residual
and every fit metric in this context is stated in."""

W_FLOOR: Final[float] = 1.0e-8
"""Total variance floor applied wherever the model is divided by or rooted.

A legal slice may touch ``w = 0`` -- ``SVIParams`` requires a non-negative minimum, not a positive
one -- and both ``sqrt(w)`` and Durrleman's ``g`` are singular there. In a search that is a NaN
gradient, and a NaN gradient does not stop anything: it spreads to every parameter of every slice
in the batch and the fit returns garbage that looks like arithmetic. The floor is eight orders of
magnitude below the total variance of a one-week 20% option, so it binds only where the curve is
degenerate anyway.
"""

SIGMA_FLOOR: Final[float] = 1.0e-6
"""Floor added to the decoded ``sigma``.

``softplus`` is positive everywhere in exact arithmetic and underflows to exactly zero in float32
past about -104, at which point ``SVIParams`` would refuse the slice the optimiser just produced.
A millionth of a log-moneyness unit is three orders of magnitude below :data:`SIGMA_MIN`, the
bound at which a fit is already reported as pinned, so nothing a healthy slice does can feel it.
"""

RHO_TANH_LIMIT: Final[float] = 0.9999
"""``tanh`` is scaled by this so that ``|rho| < 1`` survives single precision.

``jnp.tanh`` saturates to exactly ``1.0`` well before ``math.tanh`` does, and an ``SVIParams`` with
``|rho| == 1`` is refused by its own constructor. The scaling keeps ``1 - rho^2`` above 2e-4, which
also keeps the square root in the minimum-variance term differentiable.
"""

B_START_MIN: Final[float] = 0.05
"""Floor on ``b`` in a *starting point*, never a bound.

``softplus'`` underflows for a very negative preimage, so a fit started at exactly ``b = 0`` has a
vanishing gradient for the wings and can never grow them, whatever the market says. It would
converge, report a respectable flat fit, and never be noticed. Starting a little way up costs
nothing: the optimiser walks straight back down on a genuinely flat chain.
"""

SIGMA_START: Final[float] = 0.10
"""Curvature a cold start assumes: a moderately rounded bottom, roughly a tenth of the quoted band
of a crypto chain."""

RHO_START: Final[float] = -0.30
"""Skew a cold start assumes. Negative, because a lifted downside wing is the shape of every equity
and crypto smile; starting at zero makes the first steps fight the data on both wings at once."""

PATIENCE: Final[int] = 3
"""Consecutive quiet steps that count as convergence.

One is not enough, and this is the second thing that cost this stage a tuning cycle. A quasi-Newton
method takes small steps while it is still building curvature information, and a gradient method
takes small steps whenever it crosses a narrow valley; either produces a single step whose change
in RMSE is below any sensible tolerance while the fit is still tens of basis points away. Requiring
the tolerance to hold three times in a row costs two extra evaluations on a genuinely converged
slice and is the difference between a cold cycle that lands on the market and one that stops in the
first ditch.
"""

RESTART_OFFSETS: Final[tuple[tuple[float, float, float, float, float], ...]] = (
    (0.0, 0.0, 0.0, 0.0, 0.0),
    (0.0, 1.0, -1.5, 0.0, 0.7),
    (0.0, -1.0, 1.5, 0.0, -0.7),
)
"""The multi-start of the cold cycle, as fixed offsets in the free coordinates.

Three starts: the heuristic read off the quotes, then one with markedly steeper wings and a
lifted downside, and one with flatter wings and the opposite skew. They bracket the two shapes a
raw SVI slice actually gets stuck between, which is what a multi-start is for.

**Offsets, not samples.** ADR-004 requires a recorded session to replay bit for bit, and a random
restart would make two runs over one recording disagree while proving nothing a fixed spread does
not. The count is part of the compiled shape, so it is a constant here rather than a setting: a
deployment that could change it would recompile on start-up without meaning to.
"""


@dataclass(frozen=True, slots=True)
class JaxFitSettings:
    """The empirical half of the fit: what Design 5.3 and 5.6 leave to tuning.

    Every field has a default, so ``JaxFitSettings()`` is a working calibrator and a test bends the
    one knob it is about. Deliberately **not** ``FitSettings``: the two adapters share the loss's
    *definition* and share nothing about how it is searched, and a single settings type would have
    to carry ``max_nfev`` for one and a learning rate for the other, with each producer silently
    ignoring half the file. The numbers that must agree -- the Huber scale, the penalty, the ridge,
    the pinning threshold -- carry the same names and the same defaults on both sides, which is what
    makes a comparison read against one loss.

    Nothing here changes the compiled shape. Everything that does is in :class:`PadShape`.
    """

    huber_scale_bp: float = 100.0
    """Vol error, in basis points, at which a residual stops being ordinary noise. Positive.

    Below it the loss is exactly least squares; above it the cost grows linearly, so one absurd
    quote can no longer buy the optimiser more than a bounded amount of movement.
    """

    durrleman_penalty_bp: float = 10_000.0
    """Vol basis points charged per unit depth of butterfly violation, on the padded mesh.
    Non-negative. Zero turns the penalty off, which is how a test isolates the fit term."""

    durrleman_mesh_margin: float = 0.5
    """How far past the outermost quote the penalty mesh reaches, in log-moneyness. Non-negative.

    A threshold rather than a shape -- it moves where the nodes sit, not how many there are -- so
    it lives here and the node count lives in :class:`PadShape`. Not zero: the published grid is
    wider than the quoted band (ADR-001), so the wings a consumer prices against are extrapolation,
    and that is exactly where raw SVI produces a negative density.
    """

    ridge_bp: float = 5.0
    """Vol basis points charged per unit of movement away from the starting point, in free
    coordinates. Non-negative.

    A tie-breaker and not a prior, for the reason ADR-027 sets out: a market with no smile has no
    ``rho``, no ``m`` and no ``sigma``, the cost surface is a plateau in those directions, and
    without a ridge the search drifts across it until a parameter reports itself pinned and ADR-006
    refuses a fit whose RMSE was perfectly healthy. Under a warm start the tie breaks towards
    yesterday's parameters, so consecutive surfaces do not jump between equally good answers.
    """

    min_quotes_for_free_shape: int = 5
    """Below this many quotes, ``m`` and ``sigma`` are pinned at their starting values.

    ADR-008's answer to risk 3 of Design 11, and the same one the baseline takes: five parameters
    through four points is an interpolation with a spare degree of freedom, and the shape
    parameters are the two that absorb it. Pinning keeps slices independent, where a pull towards a
    neighbour would invent a coupling ADR-008 decided against.
    """

    learning_rate: float = 0.05
    """Adam's step size in the free coordinates. Positive.

    Free coordinates are order one by construction -- they are preimages under softplus and tanh --
    so this is a step of a few percent of the parameter range, small enough not to jump a warm
    start out of its basin and large enough to cross one in the hot budget.

    Note what it is *not*: a guarantee that the fit lands where the gradient is smallest. Adam's
    step is set by its own moment ratio and not by the size of the gradient, so it keeps moving
    about this far at an optimum -- which is why a search here reports the best point it visited
    (:func:`_keep_best`) rather than its last iterate.
    """

    hot_steps: int = 400
    """Ceiling on Adam steps in the hot cycle. At least one.

    A ceiling, not a plan: on an unchanged market the early stop below ends a warm-started fit in
    tens of steps, and the budget is only reached when the market has genuinely moved away from
    yesterday's parameters. Exhausting it is what ``SliceResult.converged`` reports as ``False``,
    which is also what sends the cycle to the cold path.
    """

    cold_steps: int = 300
    """Ceiling on L-BFGS steps per restart in the cold cycle. At least one.

    Fewer than the hot cycle's and worth more each: a linesearch step costs several objective
    evaluations and moves like a Newton step rather than like a gradient one. Three hundred is
    where a cold fit on a twenty-strike crypto slice stops improving; a hundred leaves it about ten
    basis points short, which was measured rather than guessed.
    """

    tolerance_bp: float = 0.001
    """Early stop: the change in the root objective, in basis points of volatility, below which a
    step counts as quiet. Positive, and a run of :data:`PATIENCE` quiet steps is convergence.

    Interpretable units, which is the whole of Design 5.6's phrasing: a thousandth of a basis point
    is "another step would move this fit by a thousandth of a basis point of volatility", four
    orders of magnitude inside a crypto bid-ask.

    Tighter than it looks like it needs to be, and deliberately. At a twentieth of a basis point
    both searches stop while still crawling -- L-BFGS through the flat stretch before it has
    learned the curvature, Adam wherever its own step size happens to leave it -- and the fit lands
    ten to twenty basis points away, which the acceptance rule then rejects for a reason that has
    nothing to do with the market. Convergence and cost trade against each other here, and the
    default buys convergence.
    """

    linesearch_steps: int = 15
    """Zoom linesearch evaluations L-BFGS may spend on one step. At least one."""

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
        if not math.isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ValueError(
                f"The learning rate must be positive and finite, got {self.learning_rate}"
            )
        if self.hot_steps < 1:
            raise ValueError(f"The hot step budget must be at least one, got {self.hot_steps}")
        if self.cold_steps < 1:
            raise ValueError(f"The cold step budget must be at least one, got {self.cold_steps}")
        if not math.isfinite(self.tolerance_bp) or self.tolerance_bp <= 0:
            raise ValueError(
                f"The early-stop tolerance must be positive and finite, got {self.tolerance_bp}"
            )
        if self.linesearch_steps < 1:
            raise ValueError(
                f"The linesearch budget must be at least one, got {self.linesearch_steps}"
            )


class _Batch(NamedTuple):
    """The padded task as device arrays, ready to be ``vmap``ped over its leading axis.

    A ``NamedTuple`` rather than the frozen dataclass this repository uses everywhere else, and the
    exception is technical: JAX walks its arguments as *pytrees*, and a ``NamedTuple`` is one for
    free while a ``dataclass`` has to be registered. The alternative was seven positional array
    arguments through four levels of ``vmap``, where an accidental transposition would be a silent
    wrong answer rather than a type error.
    """

    log_moneyness: jax.Array
    implied_vol: jax.Array
    weights: jax.Array
    quote_mask: jax.Array
    tenor_years: jax.Array
    mesh: jax.Array


class _Fit(NamedTuple):
    """What one optimisation of one slice produced: the point, what it cost, and how it ended."""

    x: jax.Array
    """Free coordinates of the **best point the search visited**, shape ``(5,)``. Not its last
    iterate -- see :func:`_keep_best` for why the distinction is load-bearing here."""

    cost: jax.Array
    """Objective at that point, in squared basis points of volatility. The quantity a multi-start
    picks its winner by."""

    evaluations: jax.Array
    """Objective evaluations this slice's search spent. The unit ``n_iterations`` is reported in --
    see :meth:`JaxCalibrator.calibrate`."""

    converged: jax.Array
    """Whether the search stopped because it settled -- :data:`PATIENCE` consecutive steps inside
    the tolerance -- rather than because the step budget ran out."""


type _HotCycle = Callable[[jax.Array, jax.Array, _Batch], _Fit]
"""The compiled warm cycle: previous coordinates, which rows carry one, and the padded task."""

type _ColdCycle = Callable[[_Batch], _Fit]
"""The compiled cold cycle, which needs no history at all."""


def _softplus(x: jax.Array) -> jax.Array:
    """``log(1 + exp(x))``, in the stable form JAX ships."""
    return jax.nn.softplus(x)


def _inverse_softplus(y: float) -> float:
    """``log(expm1(y))``, evaluated so that neither tail breaks. Host side, on a handful of numbers.

    A deliberate twin of the domain's private helper rather than an import of it: a leading
    underscore says that name is not part of that module's surface. Above 20, ``softplus(x)`` and
    ``x`` agree to a rounding error and the equivalent ``y + log1p(-exp(-y))`` avoids overflowing
    ``expm1``; the caller floors the argument above zero, because ``log(0)`` raises.
    """
    if y > 20.0:
        return y + math.log1p(-math.exp(-y))
    return math.log(math.expm1(y))


def _decode(x: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    """Free coordinates to ``(a, b, rho, m, sigma)``. **Total, by construction.**

    The first coordinate is the *minimum total variance* rather than ``a``, which is what makes
    every point of R^5 a valid slice: ``w_min = softplus(x0) >= 0`` holds structurally, and ``a``
    is recovered as ``w_min - b * sigma * sqrt(1 - rho^2)``. That inverts the closed form the
    shared kernel states for the minimum of the curve, so the two agree by algebra rather than by
    coincidence, and the round trip through ``SVIParams`` is exact.

    The alternative -- ``FreeParams``'s own layout, with ``a`` free -- leaves a region where the
    slice is not a surface, which a trust region can be walled out of with a barrier and a gradient
    method cannot: Adam steps into it, ``sqrt`` of a negative variance is a NaN, and one NaN in a
    batched fit is every parameter of every slice gone.
    """
    w_min = _softplus(x[0])
    b = _softplus(x[1])
    rho = RHO_TANH_LIMIT * jnp.tanh(x[2])
    m = x[3]
    sigma = _softplus(x[4]) + SIGMA_FLOOR
    a = w_min - b * sigma * jnp.sqrt(1.0 - rho * rho)
    return a, b, rho, m, sigma


def _encode(params: SVIParams) -> tuple[float, float, float, float, float]:
    """``SVIParams`` to free coordinates, the inverse of :func:`_decode`. Host side.

    Where a warm start crosses into this adapter's search space. Exact wherever the mapping is
    invertible; the two floors are the same ones every softplus reparametrisation needs, since
    ``softplus`` has no zero in its image and a legal slice may have ``b = 0`` or sit exactly on
    ``w_min = 0``.
    """
    w_min = max(params.min_total_variance, W_FLOOR)
    b = max(params.b, W_FLOOR)
    sigma = max(params.sigma - SIGMA_FLOOR, W_FLOOR)
    rho = min(max(params.rho / RHO_TANH_LIMIT, -0.999999), 0.999999)
    return (
        _inverse_softplus(w_min),
        _inverse_softplus(b),
        math.atanh(rho),
        params.m,
        _inverse_softplus(sigma),
    )


def _to_params(x: NDArray[np.float64]) -> SVIParams:
    """One slice's free coordinates, off the device, as the domain's own value object.

    Repeats :func:`_decode` in float64 on the host rather than reading the device's float32
    parameters, so that the object the rest of the engine sees is built at the precision everything
    else in it runs at. The two agree to single precision by construction -- it is the same
    algebra -- and the test that says so is the reason this is safe to state.
    """
    w_min = _softplus_host(float(x[0]))
    b = _softplus_host(float(x[1]))
    rho = RHO_TANH_LIMIT * math.tanh(float(x[2]))
    sigma = _softplus_host(float(x[4])) + SIGMA_FLOOR
    return SVIParams(
        a=w_min - b * sigma * math.sqrt(1.0 - rho * rho),
        b=b,
        rho=rho,
        m=float(x[3]),
        sigma=sigma,
    )


def _softplus_host(x: float) -> float:
    """``log(1 + exp(x))`` without overflowing either tail, on a scalar."""
    if x > 0:
        return x + math.log1p(math.exp(-x))
    return math.log1p(math.exp(x))


def _huber(error: jax.Array, scale: float) -> jax.Array:
    """Huber's loss, normalised so that it *is* the squared error below ``scale``.

    ``e**2`` while ``|e| <= scale`` and ``scale * (2|e| - scale)`` past it: continuous, with a
    continuous first derivative, and equal to the plain square in the regime a healthy fit lives
    in. That normalisation is what lets the root of the whole objective be read as an RMSE in basis
    points, which is the unit the early stop of Design 5.6 is stated in.

    Ingestion flags but does not filter, so junk reaches the loss by design. Squared error lets one
    absurd quote drag a whole slice; past the scale this charges it linearly and the smile stays
    where the other twenty quotes say it is.
    """
    magnitude = jnp.abs(error)
    return jnp.where(magnitude <= scale, error * error, scale * (2.0 * magnitude - scale))


def _durrleman_g(
    a: jax.Array,
    b: jax.Array,
    rho: jax.Array,
    m: jax.Array,
    sigma: jax.Array,
    k: jax.Array,
) -> jax.Array:
    """Durrleman's function on one slice, elementwise over the mesh.

    ``g(k) >= 0`` everywhere is butterfly arbitrage freedom -- a non-negative implied density -- and
    where it is negative the slice prices a butterfly spread at a negative cost. The closed form is
    ``domain/durrleman.py``'s, restated in ``jnp`` for the same reason Black-76 is: rule 3 forbids
    JAX in the domain, and a penalty an optimiser cannot differentiate is not a penalty. That module
    is this one's oracle, and the test asserts the two agree.

    The one behavioural difference is at the singularity. ``durrleman_g`` *raises* when the total
    variance touches zero, because there is no honest number to return and an infinity would survive
    a ``max(0, -g)`` clamp as a clean zero. Inside a search there is nothing to raise to, so the
    variance is floored at :data:`W_FLOOR` instead: the value stays finite, the gradient keeps
    pointing away from the degenerate curve, and the slice is judged on the metrics computed
    afterwards on the host.
    """
    y = k - m
    root = jnp.sqrt(y * y + sigma * sigma)
    w = jnp.maximum(a + b * (rho * y + root), W_FLOOR)
    w_prime = b * (rho + y / root)
    w_second = b * sigma * sigma / (root * root * root)

    return (
        (1.0 - k * w_prime / (2.0 * w)) ** 2
        - (w_prime * w_prime / 4.0) * (1.0 / w + 0.25)
        + w_second / 2.0
    )


def _objective(
    x: jax.Array,
    start: jax.Array,
    pinned: jax.Array,
    data: _Batch,
    settings: JaxFitSettings,
) -> jax.Array:
    """The cost of one slice at one point, in squared basis points of volatility.

    **The same objective the baseline minimises, residual for residual**, which is the condition
    under which Design 5.7 compares two *methods* rather than two problems. Both build one residual
    vector and both charge it through the same Huber transform, so a difference in the answers is a
    difference in the search. Three blocks:

    * **The fit**, one residual per quote: ``sqrt(n * weight_i) * error_i`` in basis points of
      volatility, summed under ADR-009's masked form ``where(mask, ..., 0)``. The weights do the
      masking's job a second time -- a padded cell weighs zero *and* is masked -- and neither of
      them is what makes the padding safe: the gradient of ``where`` still touches the untaken
      branch, and ``0 * nan`` is ``nan``. What makes it safe is that the fill values are finite,
      which ``padding.py`` owns.

      The ``sqrt(n)`` is not decoration. ADR-018's weights arrive normalised to one, so the sum is
      a weighted *mean* square and the individual residuals would shrink as the chain widened --
      which would silently make one ``huber_scale_bp``, one ``ridge_bp`` and one
      ``durrleman_penalty_bp`` mean different trade-offs on a twelve-strike slice and on a
      forty-strike one, and different trade-offs again between this producer and the baseline.
    * **The butterfly penalty**, one hinge residual per mesh node, divided by ``sqrt(nodes)`` so
      that refining the mesh measures a violation more finely instead of charging more for it.
      Huber applies here too, and that is load-bearing rather than incidental: squared, a penalty
      of ten thousand basis points per unit of violation dominates every quote on the slice as soon
      as the true smile has any curvature the wings cannot hold, and the fit walks away from the
      market to buy a fraction of a percent of density. Charged linearly past the Huber scale it
      stays what it is meant to be -- a strong preference, not a constraint.
    * **The ridge**, one residual per free coordinate, ``ridge_bp`` per unit of movement away from
      the starting point.

    ``pinned`` substitutes the start's own value for a coordinate the slice is too thin to fit,
    before anything else happens. Doing it inside the objective rather than by searching a
    subvector is what keeps the shape static: the gradient of a substituted coordinate is exactly
    zero, so the optimiser leaves it alone without the code ever branching on how many quotes
    arrived.
    """
    x = jnp.where(pinned, start, x)
    a, b, rho, m, sigma = _decode(x)
    huber = settings.huber_scale_bp

    y = data.log_moneyness - m
    w = a + b * (rho * y + jnp.sqrt(y * y + sigma * sigma))
    model_vol = jnp.sqrt(jnp.maximum(w, W_FLOOR) / data.tenor_years)
    error_bp = (model_vol - data.implied_vol) * BASIS_POINTS_PER_VOL
    scaled = jnp.sqrt(_quote_count(data) * data.weights) * error_bp
    fit = jnp.sum(jnp.where(data.quote_mask, _huber(scaled, huber), 0.0))

    violation = jnp.maximum(0.0, -_durrleman_g(a, b, rho, m, sigma, data.mesh))
    penalty = jnp.sum(
        _huber(
            settings.durrleman_penalty_bp * violation / jnp.sqrt(data.mesh.shape[-1]),
            huber,
        )
    )

    ridge = jnp.sum(_huber(settings.ridge_bp * _drift(x, start), huber))
    return fit + penalty + ridge


def _drift(x: jax.Array, start: jax.Array) -> jax.Array:
    """How far a search has travelled from its starting point, **in the baseline's coordinates**.

    Four of the five coordinates are the baseline's already -- ``b``, ``rho`` and ``sigma`` through
    the same softplus and tanh preimages, ``m`` untouched -- and the level is not: this adapter
    searches the minimum total variance through a softplus, where the baseline searches ``a``
    directly. So the level's drift is measured *decoded*, in total-variance units, and the other
    four are measured as they are.

    Without that, :attr:`JaxFitSettings.ridge_bp` would not mean here what it means there. A
    softplus preimage compresses small values logarithmically -- at a total variance of 0.03 its
    derivative is about thirty -- so a ridge that is a tie-breaker in the baseline's coordinates
    becomes a hundred and fifty basis points per unit of variance in this adapter's, and drags
    every fit back towards its own starting level. Two producers can share a tuning constant only
    if it is measured against the same ruler, which is the same argument the practical bounds are
    imported under.

    Rescaling the *search* coordinate instead was tried and rejected: it fixes the ridge and
    wrecks the conditioning, because the level's gradient then dwarfs the other four and L-BFGS
    spends its linesearch on one direction.
    """
    return jnp.concatenate([(_softplus(x[0]) - _softplus(start[0]))[None], x[1:] - start[1:]])


def _quote_count(data: _Batch) -> jax.Array:
    """How many quotes of one slice can steer the fit: masked in and weighing more than zero.

    The same population ``n_quotes_used`` reports and the same one the pinning threshold is read
    against, so the three cannot drift apart. Floored at one, because it divides.
    """
    return jnp.maximum(jnp.sum(jnp.where(data.quote_mask & (data.weights > 0.0), 1.0, 0.0)), 1.0)


def _root_bp(cost: jax.Array, quotes: jax.Array) -> jax.Array:
    """The objective back in basis points of volatility: ``sqrt(cost / n)``.

    What makes the early stop of Design 5.6 readable. The fit term is ``n`` times a weighted mean
    square of vol errors in basis points, so dividing by ``n`` and taking the root returns the
    RMSE of the slice exactly whenever the penalty and the ridge are quiet -- which is the state a
    converging fit is in. A tolerance can therefore be argued against a bid-ask spread rather than
    against a floating-point epsilon, which is the entire point of stating it in interpretable
    units.
    """
    return jnp.sqrt(cost / quotes)


def _cold_start(data: _Batch) -> jax.Array:
    """A first guess for one slice, read off its own quotes. Shape ``(5,)``, free coordinates.

    The cheapest reliable information about a slice is in the data it is about to be fitted to: the
    lowest observed total variance is roughly the bottom of the curve, the spread of total variance
    across the quoted band is roughly what the wings have to cover, and the strike where the
    minimum sits is roughly ``m``. Curvature and skew start at a typical crypto shape, because
    nothing in the data pins them without already solving the problem.

    This adapter's parametrisation makes the level part exact rather than approximate: the first
    coordinate *is* the minimum of the curve, so the lowest observed total variance goes straight
    in, where a parametrisation in ``a`` has to subtract an estimate of ``b * sigma`` first.

    Every reduction is masked with a finite sentinel drawn from the data itself rather than with an
    infinity. A fully padded row then produces zeros, which clamp to the floors below and are
    discarded by the slice mask -- where an ``inf`` sentinel would have produced NaNs that no
    later mask can remove.
    """
    total_variance = data.implied_vol * data.implied_vol * data.tenor_years
    positive = jnp.where(data.quote_mask, total_variance, 0.0)
    w_high = jnp.max(positive)
    w_low = jnp.min(jnp.where(data.quote_mask, total_variance, w_high))

    moneyness = jnp.where(data.quote_mask, data.log_moneyness, 0.0)
    k_high = jnp.max(moneyness)
    k_low = jnp.min(jnp.where(data.quote_mask, data.log_moneyness, k_high))
    span = k_high - k_low

    b = jnp.clip(
        jnp.where(span > 0.0, (w_high - w_low) / jnp.where(span > 0.0, span, 1.0), 0.0),
        B_START_MIN,
        B_MAX,
    )
    at_minimum = jnp.argmin(jnp.where(data.quote_mask, total_variance, w_high))
    m = jnp.clip(data.log_moneyness[at_minimum], -M_LIMIT, M_LIMIT)
    w_min = jnp.clip(w_low, W_FLOOR, A_LIMIT)

    return jnp.stack(
        [
            _inverse_softplus_jnp(w_min),
            _inverse_softplus_jnp(b),
            jnp.arctanh(jnp.asarray(RHO_START / RHO_TANH_LIMIT, dtype=DTYPE)),
            m,
            _inverse_softplus_jnp(jnp.asarray(SIGMA_START, dtype=DTYPE)),
        ]
    )


def _inverse_softplus_jnp(y: jax.Array) -> jax.Array:
    """``log(expm1(y))`` on the device, both tails handled, argument assumed positive.

    Both branches are evaluated -- that is what ``where`` does -- so neither may produce a NaN even
    where it is discarded. ``expm1`` of a large argument overflows to ``inf`` and ``log(inf)`` is
    ``inf``, which is discarded harmlessly; the argument of the linear branch is clamped so that
    ``log1p(-exp(-y))`` never sees ``-exp(0) = -1`` and answers with a NaN that the other branch
    could not undo.
    """
    safe = jnp.maximum(y, TINY)
    linear = safe + jnp.log1p(-jnp.exp(-jnp.maximum(safe, 1.0)))
    return jnp.where(safe > 20.0, linear, jnp.log(jnp.expm1(safe)))


def _pinned_mask(data: _Batch, settings: JaxFitSettings) -> jax.Array:
    """Which coordinates of one slice are held at their starting value. Shape ``(5,)``.

    ``m`` and ``sigma`` on a slice with fewer quotes than
    :attr:`JaxFitSettings.min_quotes_for_free_shape`, counted over the quotes that can actually
    steer the fit -- masked in and weighing more than zero, which is the same population the
    baseline pins on and the same one ``n_quotes_used`` reports.
    """
    used = jnp.sum(jnp.where(data.quote_mask & (data.weights > 0.0), 1, 0))
    thin = used < settings.min_quotes_for_free_shape
    return jnp.array([False, False, False, True, True]) & thin


type _Search = tuple[
    jax.Array, optax.OptState, jax.Array, jax.Array, jax.Array, tuple[jax.Array, jax.Array]
]
"""What a search carries between iterations: the iterate, the optimiser's state, the evaluations
spent, the previous root objective, how many consecutive steps have been quiet, and the best point
seen so far."""


def _quiet_after(
    previous: jax.Array, root: jax.Array, quiet: jax.Array, tolerance: float
) -> jax.Array:
    """Consecutive steps whose change in root objective stayed inside the tolerance.

    Reset to zero the moment a step moves the fit, so that :data:`PATIENCE` counts a *run* of quiet
    steps rather than a total.
    """
    return jnp.where(jnp.abs(previous - root) <= tolerance, quiet + 1, 0)


def _keep_best(
    best: tuple[jax.Array, jax.Array], x: jax.Array, cost: jax.Array
) -> tuple[jax.Array, jax.Array]:
    """The cheaper of the incumbent and the point just evaluated.

    **A search reports the best point it visited, not the last one**, and for a gradient method
    that is not bookkeeping. Adam's step size is set by its own moment ratio rather than by the
    magnitude of the gradient, so at an optimum -- which is exactly where a warm start begins in a
    calm market -- it keeps stepping about a learning rate away and settles wherever the tolerance
    happens to catch it. Without this, a hot cycle handed yesterday's converged parameters would
    return a *worse* fit than the one it was given, every time, and the surface would drift a
    basis point per snapshot with no market behind the movement.

    It also makes the two cycles comparable: L-BFGS with a linesearch is monotone by construction,
    so tracking the best changes nothing there, and both then answer the same question.
    """
    better = cost < best[1]
    return jnp.where(better, x, best[0]), jnp.where(better, cost, best[1])


def _initial_search(start: jax.Array, state: optax.OptState, cost: jax.Array) -> _Search:
    """The carry a search begins with, the starting point already counted as the incumbent."""
    return (
        start,
        state,
        jnp.asarray(0, dtype=jnp.int32),
        jnp.asarray(jnp.inf, dtype=DTYPE),
        jnp.asarray(0, dtype=jnp.int32),
        (start, cost),
    )


def _adam_fit(start: jax.Array, data: _Batch, settings: JaxFitSettings) -> _Fit:
    """The hot cycle on one slice: Adam from ``start``, stopped early when the RMSE settles.

    The loop is ``lax.while_loop`` rather than a Python one, so the whole budget lives inside a
    single compiled function and no iteration count crosses back to the host. Under ``vmap`` each
    slice keeps its own stopping decision: the batched loop runs while *any* lane is still moving
    and the settled lanes stop being updated, which is what makes the reported step count per slice
    a real number rather than a batch maximum.
    """
    pinned = _pinned_mask(data, settings)
    quotes = _quote_count(data)
    optimiser = optax.adam(settings.learning_rate)

    def objective(x: jax.Array) -> jax.Array:
        return _objective(x, start, pinned, data, settings)

    value_and_grad = jax.value_and_grad(objective)

    def step(carry: _Search) -> _Search:
        x, state, count, previous, quiet, best = carry
        cost, gradient = value_and_grad(x)
        updates, state = optimiser.update(gradient, state, x)
        root = _root_bp(cost, quotes)
        return (
            optax.apply_updates(x, updates),
            state,
            count + 1,
            root,
            _quiet_after(previous, root, quiet, settings.tolerance_bp),
            _keep_best(best, x, cost),
        )

    def running(carry: _Search) -> jax.Array:
        _, _, count, _, quiet, _ = carry
        return jnp.logical_and(count < settings.hot_steps, quiet < PATIENCE)

    initial = _initial_search(start, optimiser.init(start), objective(start))
    _, _, count, _, quiet, best = jax.lax.while_loop(running, step, initial)
    return _Fit(
        x=jnp.where(pinned, start, best[0]),
        cost=best[1],
        evaluations=count + 1,
        converged=quiet >= PATIENCE,
    )


def _lbfgs_fit(start: jax.Array, data: _Batch, settings: JaxFitSettings) -> _Fit:
    """One start of the cold cycle on one slice: L-BFGS with a zoom linesearch.

    The curvature-aware half of Design 5.6. Adam takes the same small step whatever the local
    geometry, which is exactly what a warm start wants and exactly what a cold one does not: from a
    guess read off the quotes the fit has to cross the valley, and a quasi-Newton step with a
    linesearch does that in tens of iterations where Adam needs thousands.

    ``optax.lbfgs`` rather than ``jaxopt``, which Design 5.6 names: jaxopt is deprecated and its
    solvers now live in optax, so it is not in this project's optional extra at all. The behaviour
    named in the design -- multi-start L-BFGS for the cold cycle -- is unchanged.

    The evaluation count is accumulated rather than assumed: a linesearch step evaluates the
    objective several times, and reporting one per iteration would make this producer's
    ``n_iterations`` an undercount of exactly the quantity the baseline reports honestly.
    """
    pinned = _pinned_mask(data, settings)
    quotes = _quote_count(data)

    def objective(x: jax.Array) -> jax.Array:
        return _objective(x, start, pinned, data, settings)

    optimiser = optax.lbfgs(
        linesearch=optax.scale_by_zoom_linesearch(max_linesearch_steps=settings.linesearch_steps)
    )
    value_and_grad = optax.value_and_grad_from_state(objective)

    def step(carry: _Search) -> _Search:
        x, state, evaluations, previous, quiet, best = carry
        cost, gradient = value_and_grad(x, state=state)
        updates, state = optimiser.update(
            gradient, state, x, value=cost, grad=gradient, value_fn=objective
        )
        root = _root_bp(cost, quotes)
        spent = evaluations + 1 + otu.tree_get(state, "num_linesearch_steps")
        return (
            optax.apply_updates(x, updates),
            state,
            spent,
            root,
            _quiet_after(previous, root, quiet, settings.tolerance_bp),
            _keep_best(best, x, cost),
        )

    def running(carry: _Search) -> jax.Array:
        _, state, _, _, quiet, _ = carry
        iterations = otu.tree_get(state, "count")
        return jnp.logical_and(iterations < settings.cold_steps, quiet < PATIENCE)

    initial = _initial_search(start, optimiser.init(start), objective(start))
    _, _, evaluations, _, quiet, best = jax.lax.while_loop(running, step, initial)
    return _Fit(
        x=jnp.where(pinned, start, best[0]),
        cost=best[1],
        evaluations=evaluations + 1,
        converged=quiet >= PATIENCE,
    )


def _better(first: _Fit, second: _Fit) -> _Fit:
    """Pick the cheaper of two fits of the same slice, keeping the cost of both.

    Ties go to the first, which is the warm-started one, so a calm market keeps returning the same
    parameters instead of oscillating between two equally good answers on floating-point noise.
    """
    keep_first = first.cost <= second.cost
    return _Fit(
        x=jnp.where(keep_first, first.x, second.x),
        cost=jnp.where(keep_first, first.cost, second.cost),
        evaluations=first.evaluations + second.evaluations,
        converged=jnp.where(keep_first, first.converged, second.converged),
    )


def _hot_cycle(
    warm: jax.Array, has_warm: jax.Array, batch: _Batch, settings: JaxFitSettings
) -> _Fit:
    """Every slice of the surface, Adam, warm-started where there is a history to start from."""
    starts = jnp.where(has_warm[:, None], warm, jax.vmap(_cold_start)(batch))
    # Bound rather than returned directly: `vmap` is typed as returning `Any`, and a bare `return`
    # of it would be a silent hole in the one boundary where the two cycles agree on a shape.
    fitted: _Fit = jax.vmap(lambda start, slice_data: _adam_fit(start, slice_data, settings))(
        starts, batch
    )
    return fitted


def _cold_cycle(batch: _Batch, settings: JaxFitSettings) -> _Fit:
    """Every slice of the surface, multi-start L-BFGS, keeping each slice's own best restart.

    The reduction is per slice and not per surface: expiry three may prefer the steep-wing restart
    while expiry four prefers the flat one, and collapsing that choice to a single winner for the
    whole surface would throw away the independence ADR-008 fits each slice with.

    The evaluation counts of *every* restart are summed into the winner. What the cycle cost is not
    the same question as which answer it kept.
    """
    heuristic = jax.vmap(_cold_start)(batch)
    pinned = jax.vmap(lambda slice_data: _pinned_mask(slice_data, settings))(batch)
    # A restart offset is a different *search*, and a pinned coordinate is not searched. Offsetting
    # one anyway would turn the multi-start into a three-point grid over the very shape parameters
    # a thin slice was refused the freedom to fit, which is ADR-008's pinning undone by the back
    # door -- and it would make "held at the value the start gave them" false for the one case the
    # sentence is about.
    offsets = jnp.where(
        pinned[None, :, :], 0.0, jnp.asarray(RESTART_OFFSETS, dtype=DTYPE)[:, None, :]
    )
    starts = heuristic[None, :, :] + offsets

    def from_one_start(start: jax.Array) -> _Fit:
        fitted: _Fit = jax.vmap(lambda x0, slice_data: _lbfgs_fit(x0, slice_data, settings))(
            start, batch
        )
        return fitted

    attempts = jax.vmap(from_one_start)(starts)

    best = jnp.argmin(attempts.cost, axis=0)
    slices = jnp.arange(attempts.cost.shape[1])
    return _Fit(
        x=attempts.x[best, slices],
        cost=attempts.cost[best, slices],
        evaluations=jnp.sum(attempts.evaluations, axis=0),
        converged=attempts.converged[best, slices],
    )


@cache
def _compiled(settings: JaxFitSettings, shape: PadShape) -> tuple[_HotCycle, _ColdCycle]:
    """The two compiled cycles for one (settings, shape) pair, built at most once per process.

    Cached because a compilation is the expensive thing ADR-009 is about, and because a test suite
    that builds a dozen calibrators with the same configuration should pay for one. Keying on the
    settings as well as the shape is what keeps that safe: the settings are closed over and baked
    into the compiled code, so two calibrators with different Huber scales must not share it.

    Frozen dataclasses on both sides of the key, which is why both are hashable.
    """
    hot = jax.jit(lambda warm, has_warm, batch: _hot_cycle(warm, has_warm, batch, settings))
    cold = jax.jit(lambda batch: _cold_cycle(batch, settings))
    return hot, cold


def _to_batch(padded: PaddedTask) -> _Batch:
    """The padded rectangle as device arrays, at this adapter's one precision.

    The dtype is pinned here rather than inherited from whatever the caller built: dtype is part of
    the compiled signature, so a task that arrived as ``float32`` and one that arrived as
    ``float64`` would otherwise be two compilations of the same function -- ADR-009's failure by a
    different door.
    """
    return _Batch(
        log_moneyness=jnp.asarray(padded.log_moneyness, dtype=DTYPE),
        implied_vol=jnp.asarray(padded.implied_vol, dtype=DTYPE),
        weights=jnp.asarray(padded.weights, dtype=DTYPE),
        quote_mask=jnp.asarray(padded.quote_mask, dtype=jnp.bool_),
        tenor_years=jnp.asarray(padded.tenor_years, dtype=DTYPE)[:, None],
        mesh=jnp.asarray(padded.mesh, dtype=DTYPE),
    )


def _metrics(params: SVIParams, padded: PaddedTask, index: int) -> tuple[float, float, int]:
    """``(rmse_vol_bp, max_err_vol_bp, n_quotes_used)`` for one fitted slice, on the host.

    In ``float64`` and through ``SVIParams.implied_vol`` -- the domain's own curve -- rather than
    off the device. Two reasons, and both are about the comparison this stage exists to make:
    the baseline's metrics are computed exactly this way, so any difference between the two
    producers' RMSEs is a difference between the fits; and the invariant ``max >= rmse`` that
    ``SliceResult`` enforces is arithmetic that single precision can round the wrong way when
    every error on a slice is the same size.

    Both errors are taken over the quotes that **entered the fit** -- masked in and weighing more
    than zero. A zero weight is how the ACL keeps a flagged quote visible without letting it steer
    the parameters, so the model's error against one is the error against something deliberately
    ignored. The RMSE is weighted, because it is the loss's own view of the fit and the acceptance
    threshold is read against it; the maximum is not, because its job is to expose the single quote
    the mean absorbed.
    """
    used = padded.quote_mask[index] & (padded.weights[index] > 0.0)
    k = padded.log_moneyness[index][used]
    market = padded.implied_vol[index][used]
    weights = padded.weights[index][used]

    errors_bp = np.abs(params.implied_vol(k, float(padded.tenor_years[index])) - market) * (
        BASIS_POINTS_PER_VOL
    )
    rmse = float(np.sqrt(np.average(np.square(errors_bp), weights=weights)))
    return rmse, float(errors_bp.max()), int(used.sum())


def _at_bound(params: SVIParams, pinned: bool) -> bool:
    """Whether any *searched* parameter finished at or past a practical bound.

    The bounds are the baseline's, imported rather than restated: ``at_bound`` is half of ADR-006's
    acceptance rule and the rule has to be read against one ruler, or two producers are being
    judged by two rules wearing one name.

    "At or past", where the baseline says "at". The search here is unconstrained -- the box is not
    handed to an optimiser that keeps its iterates feasible -- so a fit can finish outside the
    region rather than pressed against its edge. Both mean what ADR-006 refuses: the optimiser
    wanted to leave the region a publishable fit lives in. The closeness test carries the
    baseline's tolerances so that a fit which merely *approaches* a bound is caught identically.

    ``pinned`` excludes ``m`` and ``sigma`` on a thin slice. A pinned parameter never moved, so
    reporting it as pinned by the optimiser would mean something the rule does not.
    """

    def touching(value: float, bound: float) -> bool:
        return bool(np.isclose(value, bound, rtol=BOUND_RTOL, atol=BOUND_ATOL))

    if abs(params.a) >= A_LIMIT or touching(abs(params.a), A_LIMIT):
        return True
    if params.b >= B_MAX or touching(params.b, B_MAX):
        return True
    if abs(params.rho) >= RHO_MAX or touching(abs(params.rho), RHO_MAX):
        return True
    if pinned:
        return False
    if abs(params.m) >= M_LIMIT or touching(abs(params.m), M_LIMIT):
        return True
    if params.sigma <= SIGMA_MIN or touching(params.sigma, SIGMA_MIN):
        return True
    return bool(params.sigma >= SIGMA_MAX or touching(params.sigma, SIGMA_MAX))


def _healthy(result: SliceResult) -> bool:
    """The two structural halves of ADR-006's acceptance rule, without the configured threshold.

    The RMSE half is deliberately absent: the threshold is TOML (ADR-012) and belongs to the use
    case, and a calibrator that read it would take the publication decision away from the layer
    that owns it. What is left is enough to answer the only question asked here -- was this warm
    start worth keeping, or is this a cycle for the cold path?
    """
    return result.converged and not result.at_bound


class JaxCalibrator:
    """A ``Calibrator`` backed by JAX: padded to a fixed shape, compiled once, run on every cycle.

    Structural conformance like every adapter here -- a ``producer_id`` property and a
    ``calibrate`` method, no base class, nothing the domain could import back. Stateless between
    calls: the compiled functions and the settings are configuration, not memory, so two runs over
    the same recording produce the same surfaces.
    """

    def __init__(
        self,
        settings: JaxFitSettings | None = None,
        shape: PadShape | None = None,
        producer_id: str = PRODUCER_ID,
    ) -> None:
        """Build the calibrator **and compile it**, which is the whole of ADR-009's timing claim.

        The constructor runs both cycles once on a dummy rectangle of exactly the padded shape, so
        the compilation happens at start-up rather than inside the first snapshot's latency budget.
        A configuration this process has already compiled costs nothing: :func:`_compiled` caches
        on the pair below.

        Args:
            settings: The empirical knobs of the loss and of the search. Defaults to a working
                crypto configuration.
            shape: The reserved rectangle, which *is* the compiled signature. Defaults to ADR-009's
                sixty-four strikes by sixteen expiries.
            producer_id: What this instance calls itself. A parameter rather than a constant so
                that two of them can run side by side under different names, which is the
                arrangement Design 5.7 exists to compare.
        """
        if not producer_id.strip():
            raise ValueError(f"The producer id must not be empty, got {producer_id!r}")
        self._producer_id = producer_id
        self._settings = JaxFitSettings() if settings is None else settings
        self._shape = PadShape() if shape is None else shape
        self._hot, self._cold = _compiled(self._settings, self._shape)
        self._warm_up()

    @property
    def producer_id(self) -> str:
        return self._producer_id

    @property
    def settings(self) -> JaxFitSettings:
        """The knobs this instance was built with. Read-only, and read by nothing in the pipeline:
        it is the seam a wiring test asserts a configuration arrived through."""
        return self._settings

    @property
    def shape(self) -> PadShape:
        """The rectangle every task of this instance is fitted on."""
        return self._shape

    def calibrate(
        self,
        previous: Mapping[datetime, SVIParams] | None,
        task: CalibrationTask,
    ) -> CalibrationResult:
        """Fit every slice of the task, warm-starting each expiry that has a history.

        Args:
            previous: The parameters each expiry was last fitted to, keyed by expiry. ``None`` is a
                cold start, and so is a key this task does not carry -- a tenor born since the last
                cycle (ADR-013), which is an ordinary event rather than an error.
            task: The slices to fit, already inverted, weighted and ordered by tenor.

        Returns:
            One ``SliceResult`` per slice, in the task's own order, **including the slices that
            fitted badly**: the acceptance rule of ADR-006 is the use case's, and it needs to see
            the rejects to report them.

            ``duration_ms`` is zero and deliberately not measured, exactly as in the baseline and
            in ``flat_vol.py``: reading a clock is the one thing this port forbids outright, and it
            would make two runs over the same recording differ. What a consumer sees is the use
            case's own measurement of the call, taken from the injected ``Clock``.

            ``n_iterations`` is the number of **objective evaluations, summed over the slices that
            were fitted** -- one per Adam step, and one per linesearch evaluation in the cold cycle,
            where several go into a single L-BFGS iteration. ADR-027 left this stage the choice and
            it is made deliberately: the baseline reports ``least_squares``'s residual evaluations
            summed the same way, so the two producers report the same physical quantity -- how many
            times the objective of a slice had to be looked at -- and Design 5.7's table compares
            like with like. Reporting Adam steps instead would have been the more flattering number
            and would have compared a step against an evaluation.

        Raises:
            CalibrationError: If the task does not fit the reserved rectangle (``padding.pad``). A
                poor fit is a return value; a chain wider than the compiled shape is not a fit at
                all.
        """
        padded = pad(task, self._shape, self._settings.durrleman_mesh_margin)
        batch = _to_batch(padded)

        warm, has_warm = self._warm_start(padded, task, previous)
        if bool(has_warm.any()):
            fit = self._hot(warm, has_warm, batch)
            results = self._read(fit, padded, task)
            if all(_healthy(one) for one in results):
                return self._result(results, fit, padded)

            # Design 5.6's cold cycle is "periodic or after a failure". A pure function cannot know
            # that a period has elapsed -- it reads no clock and keeps no state -- so what this
            # adapter can express is the second half, and it expresses it for the whole surface
            # rather than for the slice that failed: the compiled shape is the surface, and running
            # a subset would be a second signature. The periodic half belongs to the use case that
            # owns the state, and it is a seam (`docs/SEAMS.md`).
            #
            # A slice that fitted healthily keeps its warm answer even if the retry found a lower
            # residual: a calm expiry should not be re-seated on a different optimum because its
            # neighbour failed, which is the same stability argument the ridge makes one layer
            # down. Only the slices that failed are re-decided, and `_prefer` decides them.
            retry = self._cold(batch)
            recovered = self._read(retry, padded, task)
            return self._result(
                [
                    warm_result if _healthy(warm_result) else _prefer(warm_result, cold_result)
                    for warm_result, cold_result in zip(results, recovered, strict=True)
                ],
                fit,
                padded,
                extra=retry,
            )

        fit = self._cold(batch)
        return self._result(self._read(fit, padded, task), fit, padded)

    def _warm_up(self) -> None:
        """Run both cycles once on an empty rectangle so the compilation lands in ``__init__``.

        Empty rather than plausible: every lane is masked out, which exercises exactly the shapes
        the real cycles use and none of the arithmetic anyone could mistake for a result. The
        padding's fill values are what make an all-masked rectangle safe to differentiate through,
        which is the same property the inert-mask test asserts.
        """
        rows, columns = self._shape.max_slices, self._shape.max_quotes
        batch = _Batch(
            log_moneyness=jnp.zeros((rows, columns), dtype=DTYPE),
            implied_vol=jnp.ones((rows, columns), dtype=DTYPE),
            weights=jnp.zeros((rows, columns), dtype=DTYPE),
            quote_mask=jnp.zeros((rows, columns), dtype=jnp.bool_),
            tenor_years=jnp.ones((rows, 1), dtype=DTYPE),
            mesh=jnp.zeros((rows, self._shape.mesh_nodes), dtype=DTYPE),
        )
        warm = jnp.zeros((rows, 5), dtype=DTYPE)
        has_warm = jnp.zeros(rows, dtype=jnp.bool_)
        # Tracing and compilation happen inside the call and are synchronous; only the
        # execution that follows is dispatched asynchronously. So the compilation ADR-009 cares
        # about is paid for by the time this returns, and there is nothing to wait on.
        self._hot(warm, has_warm, batch)
        self._cold(batch)

    def _warm_start(
        self,
        padded: PaddedTask,
        task: CalibrationTask,
        previous: Mapping[datetime, SVIParams] | None,
    ) -> tuple[jax.Array, jax.Array]:
        """The previous cycle's parameters as free coordinates, plus which rows actually have one.

        A per-row flag rather than a shorter array: an expiry born since the last cycle has no
        history and the row it occupies has to be filled with something, so the flag is what says
        the contents are meaningless and the cold heuristic should be used instead. That keeps the
        shape fixed while a chain's composition changes underneath it, which is the whole exercise.
        """
        rows = padded.log_moneyness.shape[0]
        warm = np.zeros((rows, 5), dtype=np.float64)
        has_warm = np.zeros(rows, dtype=np.bool_)
        if previous is None:
            return jnp.asarray(warm, dtype=DTYPE), jnp.asarray(has_warm)

        for index, one in enumerate(task.slices):
            params = previous.get(one.expiry)
            if params is None:
                continue
            warm[index] = _encode(params)
            has_warm[index] = True
        return jnp.asarray(warm, dtype=DTYPE), jnp.asarray(has_warm)

    def _read(self, fit: _Fit, padded: PaddedTask, task: CalibrationTask) -> list[SliceResult]:
        """One device round trip, then every reported number rebuilt on the host in ``float64``."""
        coordinates = np.asarray(fit.x, dtype=np.float64)
        converged = np.asarray(fit.converged)
        thin = [
            int((padded.quote_mask[index] & (padded.weights[index] > 0.0)).sum())
            < self._settings.min_quotes_for_free_shape
            for index in range(len(task.slices))
        ]

        results: list[SliceResult] = []
        for index, one in enumerate(task.slices):
            params = _to_params(coordinates[index])
            rmse_vol_bp, max_err_vol_bp, n_quotes_used = _metrics(params, padded, index)
            results.append(
                SliceResult(
                    expiry=one.expiry,
                    tenor_years=one.tenor_years,
                    params=params,
                    rmse_vol_bp=rmse_vol_bp,
                    max_err_vol_bp=max_err_vol_bp,
                    n_quotes_used=n_quotes_used,
                    converged=bool(converged[index]),
                    at_bound=_at_bound(params, thin[index]),
                )
            )
        return results

    def _result(
        self,
        slices: list[SliceResult],
        fit: _Fit,
        padded: PaddedTask,
        extra: _Fit | None = None,
    ) -> CalibrationResult:
        """Assemble the surface, counting what every cycle that ran cost.

        The evaluations of a cold retry are added to those of the warm attempt whichever answer
        won, and only the rows that carry a real expiry are counted: a padded lane's search is an
        artefact of the fixed shape and charging the caller for it would make ``n_iterations``
        depend on how much margin the grid was given.
        """
        active = np.asarray(padded.slice_mask)
        spent = int(np.asarray(fit.evaluations)[active].sum())
        if extra is not None:
            spent += int(np.asarray(extra.evaluations)[active].sum())
        return CalibrationResult(
            slices=tuple(slices),
            n_iterations=spent,
            duration_ms=0.0,
        )


def _prefer(first: SliceResult, second: SliceResult) -> SliceResult:
    """Which of two attempts on the same slice to keep.

    A converged, unpinned fit beats one that is not, whatever the residuals say: a fit that stopped
    against a bound reports the residual of a projection onto that bound, and comparing it against
    an honest one on RMSE alone would prefer the projection whenever the bound happened to sit near
    the data. Among two of the same kind the lower RMSE wins, and ties go to the first -- which is
    the warm start, so a calm market keeps returning the same parameters.
    """
    if _healthy(first) != _healthy(second):
        return first if _healthy(first) else second
    return first if first.rmse_vol_bp <= second.rmse_vol_bp else second
