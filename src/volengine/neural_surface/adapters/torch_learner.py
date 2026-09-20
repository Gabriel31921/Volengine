"""The neural producer: a small MLP in PyTorch behind the ``SurfaceLearner`` port (Design 6.2-6.4).

The third implementer this engine publishes ``CalibratedSurface`` through, and the first that is
not a parametrisation. Where SVI fits one smile per expiry from five numbers, this module fits the
whole surface at once as a function of two inputs -- log-forward-moneyness and tenor -- with the
parameters shared across every expiry. That is the entire modelling difference between the two
engines, and everything else here exists so that the difference is the *only* thing the comparison
of Design 5.7 and 6.5 measures.

**A pure function, as the port demands.** ``update(previous, batch)`` reads no clock, performs no
I/O and keeps no state on the learner between calls: the network weights, the optimiser's moment
estimates and the version counter all travel inside the returned :class:`TorchSurface`, and a warm
start begins by *copying* them out of ``previous`` rather than continuing on them in place. The
copy is not fastidiousness. ``torch.optim.Optimizer.load_state_dict`` hands back the very tensor it
was given when the dtype and device already match, and Adam updates its moments in place, so
without the deep copy a fine-tuning step would silently rewrite the history held by the surface it
was handed -- and the same ``(previous, batch)`` pair would give a different answer the second
time. ADR-004's replay rests on that never happening, and
``test_the_same_previous_and_batch_give_the_same_surface_twice`` is what watches it.

**Float64 throughout.** The network is tens of neurons, so double precision costs nothing, and it
means the number the gate judges and the number the contract publishes are the number the model
computed, with no reconstruction on the host. The JAX calibrator had to argue its way around a
single-precision seam (ADR-029); this adapter does not open one.

**No global state is touched.** No ``torch.manual_seed``, no ``set_default_dtype``, no
``set_num_threads``, no ``use_deterministic_algorithms``: every one of those is process-wide and an
optional adapter has no business mutating the process on behalf of the whole engine. Cold-start
weights come from a ``torch.Generator`` local to the call and seeded from configuration, which is
what makes ``update(None, batch)`` reproducible without the module claiming the global generator.

**The three tiers of ADR-010, as this module sees them.** The *soft* tier is in the loss below:
Durrleman's function on a uniform moneyness mesh and monotonicity of total variance in the tenor,
both as squared hinges weighted by configuration. The *architectural* tier is deliberately not
built -- the network is a plain MLP, and a construction with guaranteed convexity is Design §10.2.
The *hard* tier is nowhere in this file, on purpose: a surface this module suspects of arbitrage is
still returned, because deciding whether it may be published belongs to the use case and the
domain gate, which measure on a mesh of their own. Soft constraints train; hard constraints govern.

**Total variance out, volatility in the loss.** The network parametrises the volatility itself
through a softplus, so ``w = sigma^2 * T`` is strictly positive by construction and no iterate can
hand the gate a non-positive variance. The data term is a weighted squared error *in volatility*,
which is the unit ``FitMetrics.rmse_vol_bp`` reports in and the unit the scipy baseline minimises
in (ADR-018's weights, consumed exactly as given); a loss in total variance would weight a
one-year quote nine times as heavily as a one-month one for the same volatility error, and the two
producers would no longer be fitting one problem.

**Two step budgets, and two rate regimes.** ``previous is None`` is the cold start -- the first
snapshot of a market, the recovery after a failure and the scheduled restart of Design 6.4
(ADR-019) -- and it retrains from scratch over the whole batch with the larger budget, at a rate
that is **annealed** (cosine) from the cold rate down to the warm one over the budget. Annealed
rather than constant because Adam at a constant rate does not settle: measured on the synthetic
chain, a constant ``1e-2`` reached seven basis points at step 1000 and was back at thirty-eight by
step 2000, so which iterate a cold start returned was a matter of where the budget happened to
end. Under the schedule the last iterate was the best one in every run measured, and the moments
it leaves behind describe that iterate. Every other call is the continuous fine-tuning of Design
6.4: a handful of Adam steps at the small *constant* warm rate -- the rate the cold run ended on --
continuing the optimiser's own trajectory because its moments travelled with the surface. Fresh
moments on a converged network are not a small perturbation: ten steps of a fresh Adam took a
42 bp fit to 150 bp, and the same ten steps on carried moments moved it by a hundredth of a basis
point. That measurement is why the moments are in the surface and not discarded between calls.

**Where the constants come from.** Every number a deployment might tune -- widths, activation,
input scales, step budgets, learning rates, penalty weights, the seed -- is a field of
:class:`NetworkSpec` or :class:`TorchFitSettings` and arrives through the constructor (ADR-012).
The defaults were measured on the synthetic SVI chain the tests use and are argued for on each
field; none of them has a TOML home yet, and that gap is recorded in ``docs/SEAMS.md`` with the
rest of what this stage left open.
"""

from __future__ import annotations

import copy
import math
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from itertools import pairwise
from typing import Any

import numpy as np
import torch
import torch.nn.functional as functional
from numpy.typing import NDArray
from torch import Tensor
from torch.optim.lr_scheduler import CosineAnnealingLR, LRScheduler

from volengine.neural_surface.domain.errors import NeuralSurfaceError
from volengine.neural_surface.domain.invariants import ArbitrageMesh
from volengine.neural_surface.domain.learned_surface import LearnedSurface
from volengine.neural_surface.domain.training_batch import TrainingBatch

PRODUCER_ID = "mlp-torch"
"""What this producer is called everywhere downstream: on its topic, in the contract, in metrics."""

DTYPE = torch.float64
"""Every tensor this module creates. Passed explicitly rather than set as the process default."""

INPUTS = 2
"""Log-forward-moneyness and tenor: the two coordinates of Design 6.2, and nothing else."""


class Activation(StrEnum):
    """The smooth nonlinearities this adapter offers. **ReLU is absent by design.**

    A volatility smile is a smooth profile and its no-arbitrage condition involves the second
    derivative in moneyness. ReLU is piecewise linear, so a network built on it has a piecewise
    linear output whose second derivative is zero almost everywhere and undefined at the kinks --
    Durrleman's function would then be measured on artefacts of the architecture rather than on
    the surface. Both members below are infinitely differentiable. Explicit string values, because
    this is what a configuration file will name.
    """

    TANH = "tanh"
    SOFTPLUS = "softplus"


def _require_positive_finite(value: float, what: str) -> None:
    """Reject anything that is not a usable positive number, as a plain ``ValueError``.

    Finiteness first, and the bad conditions joined with ``or``, because ``float("nan") <= 0`` is
    ``False`` and a NaN learning rate would otherwise pass and produce a network of NaN on the
    first step.
    """
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"The {what} must be positive and finite, got {value}")


def _require_non_negative_finite(value: float, what: str) -> None:
    """The same, admitting zero: a penalty weight of zero is how the soft tier is switched off."""
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"The {what} must be non-negative and finite, got {value}")


@dataclass(frozen=True, slots=True)
class NetworkSpec:
    """What a surface needs in order to evaluate itself: the architecture and the input scaling.

    Kept apart from :class:`TorchFitSettings` because the two have different owners. A
    :class:`TorchSurface` must carry *this* -- the weights mean nothing without the shape they
    fill and the scaling the inputs were trained under -- while the step budgets and learning
    rates belong to the learner and may change between two calls that fine-tune the same surface.
    A learner handed a surface built on a different spec refuses it (``NeuralSurfaceError``)
    rather than silently cold-starting, which is the failure the port's docstring warns is the
    worst one because nothing raises.
    """

    hidden: tuple[int, ...] = (32, 32)
    """Width of each hidden layer, in order. Two or three layers of tens of neurons (Design 6.2).

    Non-empty and every width positive. Two layers of thirty-two fit the synthetic chain to a
    handful of basis points in a thousand cold steps; the architecture is not where this
    producer's accuracy is decided, the update regime is.
    """

    activation: Activation = Activation.TANH
    """The hidden nonlinearity. Bounded and smooth; softplus is the alternative offered."""

    k_scale: float = 0.5
    """Log-moneyness is divided by this before it enters the network. Positive, finite.

    Half a unit of log-moneyness is roughly the width of a quoted crypto chain, so the first
    input lands in about ``[-1, 1]`` -- the range a tanh layer is sensitive over. A fixed scale
    rather than a statistic of the batch, so that a surface's inputs mean the same thing on every
    cycle and a warm start never has to re-normalise the weights it inherited.
    """

    tenor_scale: float = 1.0
    """The tenor enters as ``log(T / tenor_scale)``. Positive, finite, in years.

    Logarithmic because tenors span three decades -- a day to a year -- and a network fed the raw
    year fraction would see every expiry under a month as the same point. One year is the natural
    unit: the input is zero at a year, about ``-2.5`` at a month and ``-4`` at a week.
    """

    def __post_init__(self) -> None:
        if not self.hidden:
            raise ValueError("The network must have at least one hidden layer")
        if any(width <= 0 for width in self.hidden):
            raise ValueError(f"Every hidden width must be positive, got {self.hidden}")
        _require_positive_finite(self.k_scale, "moneyness scale")
        _require_positive_finite(self.tenor_scale, "tenor scale")

    @property
    def sizes(self) -> tuple[int, ...]:
        """Layer widths from the two inputs to the single output: ``(2, *hidden, 1)``."""
        return (INPUTS, *self.hidden, 1)


@dataclass(frozen=True, slots=True)
class TorchFitSettings:
    """How the network is trained: budgets, step sizes, the soft tier's weights, and the seed.

    All configuration (ADR-012). The defaults are the ones measured on the synthetic SVI chain in
    ``tests/neural_surface/test_torch_learner.py`` and each is argued below; a deployment tunes them
    against its own market, which is what a TOML section is for once one exists.
    """

    cold_steps: int = 2000
    """Adam steps on a cold start -- ``previous is None``. Positive.

    A cold start is the first snapshot, the recovery after a failure and the scheduled restart, and
    on every one of them the answer has to be a surface and not a warm-up: two thousand annealed
    steps take a random network to about five basis points on the synthetic chain in a little over
    a second on one CPU, where a thousand stop at twenty-five. The rare path, and the expensive
    one, deliberately.
    """

    warm_steps: int = 10
    """Adam steps per snapshot when fine-tuning -- Design 6.4's ``K = 5-20``. Positive.

    Few, because the network is expected to be near the answer already and the point of continuous
    training is to track a moving market cheaply rather than to refit it. Ten steps cost single
    milliseconds and, on carried moments, move a converged fit by hundredths of a basis point.
    """

    cold_learning_rate: float = 1e-2
    """Adam's step size at the *start* of a cold run. Positive, finite.

    The run anneals from here down to ``warm_learning_rate`` on a cosine over ``cold_steps``, so
    this is the ceiling of the schedule rather than a constant. Large enough to cross from the
    initial flat surface to the market inside the budget; a run capped at a tenth of this leaves
    the fit at forty basis points rather than five.
    """

    warm_learning_rate: float = 1e-3
    """Adam's step size when fine-tuning: Design 6.4's "small constant LR". Positive, finite.

    Also the floor the cold schedule anneals to, so a warm start continues at exactly the rate the
    cold run ended on and the hand-over between the two regimes introduces no step of its own. The
    moments carried in the surface already encode the direction and scale of the trajectory, so
    the rate only has to be small enough not to overshoot a market that moved a little since the
    last snapshot.
    """

    butterfly_penalty: float = 1000.0
    """Weight of the squared hinge on ``-g`` over the penalty mesh. Non-negative, finite.

    Zero switches the term off, which is the experiment ADR-010 invites rather than a
    misconfiguration. A thousand, because the data term is a squared volatility error -- about
    ``1e-5`` at thirty basis points -- while ``g`` is dimensionless and of order one, so a breach
    has to outweigh the data by orders of magnitude before the optimiser prefers the admissible
    surface to the closer one. Measured: on a smile whose data carry a butterfly violation of
    depth sixteen, a weight of a hundred already drives the mesh violation to zero and the weight
    of zero leaves it there; the same number as the calendar weight below so that the two
    conditions are traded against the data on equal terms.
    """

    calendar_penalty: float = 1000.0
    """Weight of the squared hinge on ``w_i - w_{i+1}`` between consecutive mesh tenors. Same terms.

    In total-variance units, like the gate's own ``calendar_violation``, so the two tiers measure
    one quantity. Large because a calendar crossing is small in those units -- a hundredth of a
    unit of variance is a large crossing -- and on the synthetic chain, whose data are exactly
    calendar-monotone, the crossing left in the extrapolated wings of the mesh went from ``6e-5``
    at a weight of a hundred to under ``1e-5`` at a thousand without costing the fit a basis
    point. Ten thousand does cost it: the term then dominates the data and the fit stalls at forty.
    """

    init_seed: int = 0
    """Seed of the local generator the cold-start weights are drawn from. Any integer.

    Configuration rather than a constant so that two deployments can differ deliberately, and so
    that the multi-seed comparison of a training run is a change in a file rather than in code.
    Never fed to the global generator.
    """

    def __post_init__(self) -> None:
        if self.cold_steps <= 0:
            raise ValueError(f"The cold step budget must be positive, got {self.cold_steps}")
        if self.warm_steps <= 0:
            raise ValueError(f"The warm step budget must be positive, got {self.warm_steps}")
        _require_positive_finite(self.cold_learning_rate, "cold learning rate")
        _require_positive_finite(self.warm_learning_rate, "warm learning rate")
        _require_non_negative_finite(self.butterfly_penalty, "butterfly penalty")
        _require_non_negative_finite(self.calendar_penalty, "calendar penalty")


@dataclass(frozen=True, slots=True)
class TorchSurface:
    """A trained network, as the domain sees it: something with a ``version`` you can evaluate.

    The concrete ``LearnedSurface`` of this adapter. Beyond the two members the Protocol asks for
    it carries what a warm start needs -- the spec, the weights and the optimiser's moments -- and
    the domain never looks at any of them, which is what the port's docstring means by "an
    implementation is free to hide network weights, optimiser moments and input normalisation
    statistics inside its own concrete type".

    **Immutable by discipline as well as by declaration.** The dataclass is frozen, but a tensor is
    not, so the invariant this class actually rests on is that nothing that can write to these
    tensors ever holds a reference to them: ``TorchLearner`` clones the weights out before
    training, deep-copies the moments before handing them to an optimiser, and clones both again
    on the way into a new surface. A surface handed to ``update`` comes back byte-identical.
    """

    spec: NetworkSpec
    """The architecture and input scaling these weights were trained under."""

    weights: tuple[Tensor, ...]
    """``(W_1, b_1, ..., W_L, b_L)``, float64, detached, never written to."""

    moments: Mapping[str, Any]
    """Adam's state as ``Optimizer.state_dict()`` returns it, deep-copied: first and second moment
    per weight and the step count. What lets a warm start continue the optimiser's own trajectory
    rather than start a new one on a converged network."""

    version: int
    """How many updates produced this surface since the last cold start. ``1`` on a cold start.

    Monotone within one training lineage and reset by a restart, necessarily: ``update(None, ...)``
    receives no history to continue the count from, and that is the design (ADR-019). The reset is
    informative rather than a flaw -- ``producer_meta["weights_version"]`` dropping to ``1.0`` is
    what a scheduled restart looks like from outside.
    """

    def total_variance(
        self, k: NDArray[np.float64], tenors: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """``w = sigma(k, T)^2 * T`` on the full ``(len(tenors), len(k))`` mesh, tenor-major.

        Raw and unpoliced, as the Protocol asks: shape, finiteness and positivity are imposed by
        ``learned_surface.evaluate_total_variance``, once, for every implementation. A network that
        has diverged into NaN answers with NaN here and is caught there.
        """
        with torch.no_grad():
            k_mesh, tenor_mesh = _mesh(
                torch.as_tensor(np.asarray(k, dtype=np.float64), dtype=DTYPE),
                torch.as_tensor(np.asarray(tenors, dtype=np.float64), dtype=DTYPE),
            )
            w = _total_variance(self.spec, self.weights, k_mesh, tenor_mesh)
        return np.asarray(w.numpy(), dtype=np.float64)


class TorchLearner:
    """The ``SurfaceLearner`` this context runs in production. See the module docstring.

    One instance per market is the natural arrangement but not a requirement: the learner holds
    only its configuration and its penalty mesh, both immutable, so one instance may serve any
    number of markets and any number of threads at once.
    """

    def __init__(
        self,
        penalty_mesh: ArbitrageMesh,
        settings: TorchFitSettings | None = None,
        spec: NetworkSpec | None = None,
    ) -> None:
        """Configure the learner.

        Args:
            penalty_mesh: Where the soft tier is evaluated: a uniform moneyness mesh and a tenor
                axis, exactly the shape the hard gate judges on, and reused from the domain for
                that reason -- its constructor already refuses the non-uniform mesh a central
                difference would silently corrupt. Deliberately *not* required to be the gate's
                mesh: training on a wider or denser one than the gate looks at is a legitimate
                choice, and training on the gate's own is another. The composition root decides.
            settings: Budgets, rates, penalty weights and seed. The shipped defaults when omitted.
            spec: Architecture and input scaling. The shipped defaults when omitted.
        """
        self._settings = TorchFitSettings() if settings is None else settings
        self._spec = NetworkSpec() if spec is None else spec
        self._penalty_k = torch.as_tensor(penalty_mesh.k_array, dtype=DTYPE)
        self._penalty_tenors = torch.as_tensor(penalty_mesh.tenor_array, dtype=DTYPE)
        # `ArbitrageMesh` guarantees uniform spacing and at least three points, which is what a
        # central difference needs; the step is taken endpoint to endpoint for the same reason the
        # domain's `_uniform_step` takes it that way -- one subtraction rather than `n`.
        self._penalty_step = float(
            (penalty_mesh.log_moneyness[-1] - penalty_mesh.log_moneyness[0])
            / (len(penalty_mesh.log_moneyness) - 1)
        )

    @property
    def producer_id(self) -> str:
        """``"mlp-torch"``."""
        return PRODUCER_ID

    def update(self, previous: LearnedSurface | None, batch: TrainingBatch) -> TorchSurface:
        """Train from one batch: a cold start when ``previous`` is ``None``, a fine-tune otherwise.

        Args:
            previous: A :class:`TorchSurface` this learner produced, or ``None`` for a cold start.
                Anything else -- a surface of another kind, or one built on a different
                :class:`NetworkSpec` -- is refused with ``NeuralSurfaceError`` rather than
                silently cold-started, because a learner that accepted a history and ignored it
                would turn continuous fine-tuning into a restart on every snapshot with nothing
                raising, and the use case's response to the error (refuse, republish the last good
                surface) is the visible one.
            batch: The points to fit. Consumed exactly as given -- weights included, fresh and
                replayed alike -- because the batch is where the use case has already decided how
                much history this step sees.

        Returns:
            A new :class:`TorchSurface` -- the concrete type, which is narrower than the port's
            ``LearnedSurface`` and satisfies it. ``previous`` is not modified: its weights are
            cloned and its moments deep-copied before either is touched. The surface may be
            arbitrageable; judging it is the gate's job.

        Raises:
            NeuralSurfaceError: If ``previous`` is not a surface this learner can continue from.
        """
        if previous is None:
            weights = _initial_weights(self._spec, self._settings.init_seed)
            moments: Mapping[str, Any] | None = None
            steps = self._settings.cold_steps
            learning_rate = self._settings.cold_learning_rate
            version = 1
        else:
            if not isinstance(previous, TorchSurface):
                raise NeuralSurfaceError(
                    f"{PRODUCER_ID} can only fine-tune a surface it trained, got "
                    f"{type(previous).__name__}"
                )
            if previous.spec != self._spec:
                raise NeuralSurfaceError(
                    f"{PRODUCER_ID} was configured with {self._spec} and cannot fine-tune a "
                    f"surface trained under {previous.spec}"
                )
            weights = tuple(w.detach().clone().requires_grad_(True) for w in previous.weights)
            moments = previous.moments
            steps = self._settings.warm_steps
            learning_rate = self._settings.warm_learning_rate
            version = previous.version + 1

        k, tenors, observed, normalised_weights = _as_tensors(batch)
        optimizer = torch.optim.Adam(list(weights), lr=learning_rate)
        schedule: LRScheduler | None = None
        if moments is None:
            # Cold: anneal from the cold rate down to the warm one over the budget, so the last
            # iterate is a settled one and the warm steps that follow continue at the rate this
            # run ended on. See the module docstring for the measurement behind it.
            schedule = CosineAnnealingLR(
                optimizer, T_max=steps, eta_min=self._settings.warm_learning_rate
            )
        else:
            # The deep copy is load-bearing. `load_state_dict` returns the tensor it was given
            # whenever dtype and device already match, and Adam updates its moments in place --
            # so without it this call would rewrite the history inside `previous`.
            optimizer.load_state_dict(copy.deepcopy(dict(moments)))
            # `load_state_dict` restores the parameter groups too, learning rate included, so
            # without this line a warm start would train at whatever rate the previous run ended
            # on rather than at the one this learner is configured with.
            for group in optimizer.param_groups:
                group["lr"] = learning_rate

        for _ in range(steps):
            optimizer.zero_grad()
            loss = self._loss(weights, k, tenors, observed, normalised_weights)
            torch.autograd.backward(loss)
            optimizer.step()
            if schedule is not None:
                schedule.step()

        return TorchSurface(
            spec=self._spec,
            weights=tuple(w.detach().clone() for w in weights),
            moments=copy.deepcopy(optimizer.state_dict()),
            version=version,
        )

    def _loss(
        self,
        weights: tuple[Tensor, ...],
        k: Tensor,
        tenors: Tensor,
        observed: Tensor,
        normalised_weights: Tensor,
    ) -> Tensor:
        """Weighted squared volatility error plus the soft tier, as one scalar.

        The data term is ``sum(weight_i * (sigma_i - iv_i)^2)`` with the weights normalised to one,
        so its square root is the weighted RMSE in volatility that ``fit_metrics`` will report. The
        penalties are squared hinges over the penalty mesh, each multiplied by its configured
        weight; a weight of zero contributes exactly nothing and costs one multiplication.
        """
        predicted = _sigma(self._spec, weights, k, tenors)
        data = torch.sum(normalised_weights * (predicted - observed) ** 2)
        if self._settings.butterfly_penalty == 0.0 and self._settings.calendar_penalty == 0.0:
            return data

        k_mesh, tenor_mesh = _mesh(self._penalty_k, self._penalty_tenors)
        w = _total_variance(self._spec, weights, k_mesh, tenor_mesh)
        butterfly = _butterfly_penalty(w, k_mesh, self._penalty_step)
        calendar = _calendar_penalty(w)
        return (
            data
            + self._settings.butterfly_penalty * butterfly
            + self._settings.calendar_penalty * calendar
        )


# --- the network


def _activation(spec: NetworkSpec, h: Tensor) -> Tensor:
    """Apply the configured hidden nonlinearity."""
    if spec.activation is Activation.TANH:
        return torch.tanh(h)
    return functional.softplus(h)


def _sigma(spec: NetworkSpec, weights: tuple[Tensor, ...], k: Tensor, tenors: Tensor) -> Tensor:
    """The volatility the network assigns to each ``(k, T)`` pair, elementwise over any shape.

    Inputs are scaled as :class:`NetworkSpec` describes, pass through the hidden layers under the
    configured activation, and the single output is put through a softplus -- so the volatility
    is strictly positive by construction and ``0.69`` where the output is zero, which is where
    the zero-initialised output bias puts the whole surface on a cold start: flat, and therefore
    free of arbitrage, before the first gradient step.
    """
    h = torch.stack([k / spec.k_scale, torch.log(tenors / spec.tenor_scale)], dim=-1)
    n_layers = len(weights) // 2
    for layer in range(n_layers - 1):
        h = _activation(spec, functional.linear(h, weights[2 * layer], weights[2 * layer + 1]))
    z = functional.linear(h, weights[-2], weights[-1]).squeeze(-1)
    return functional.softplus(z)


def _total_variance(
    spec: NetworkSpec, weights: tuple[Tensor, ...], k: Tensor, tenors: Tensor
) -> Tensor:
    """``sigma^2 * T``, elementwise: what the port publishes and what the penalties are taken on."""
    sigma = _sigma(spec, weights, k, tenors)
    return sigma * sigma * tenors


def _mesh(k: Tensor, tenors: Tensor) -> tuple[Tensor, Tensor]:
    """Broadcast two axes to the ``(len(tenors), len(k))`` mesh the port's layout promises.

    ``indexing="xy"`` is what makes the first axis the *second* argument: row ``i`` is
    ``tenors[i]``, column ``j`` is ``k[j]``. The transposed answer is the mistake an adapter makes
    on its first day, and a square mesh would hide it; the tests evaluate on rectangular ones.
    """
    k_mesh, tenor_mesh = torch.meshgrid(k, tenors, indexing="xy")
    return k_mesh, tenor_mesh


def _initial_weights(spec: NetworkSpec, seed: int) -> tuple[Tensor, ...]:
    """Xavier-uniform weights and zero biases from a generator local to this call.

    Built as bare tensors rather than through ``torch.nn.Linear`` because the module constructor
    draws its initial weights from the *global* generator -- consuming it as a side effect even if
    the values were overwritten afterwards -- and this adapter touches no global state. The bound
    is ``sqrt(6 / (fan_in + fan_out))``, the usual choice for a tanh network; zero biases put the
    output at exactly zero and the initial surface at a flat ``softplus(0)``.
    """
    generator = torch.Generator().manual_seed(seed)
    weights: list[Tensor] = []
    for fan_in, fan_out in pairwise(spec.sizes):
        bound = math.sqrt(6.0 / (fan_in + fan_out))
        uniform = torch.rand(fan_out, fan_in, generator=generator, dtype=DTYPE)
        weights.append((uniform * 2.0 - 1.0).mul_(bound).requires_grad_(True))
        weights.append(torch.zeros(fan_out, dtype=DTYPE, requires_grad=True))
    return tuple(weights)


def _as_tensors(batch: TrainingBatch) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """The batch as four float64 vectors: ``k``, ``T``, observed vol, weights normalised to one.

    The weights are normalised here so that the data term is a weighted *mean* and its scale does
    not depend on how many points the buffer happened to contribute -- which is also what keeps
    one penalty weight meaning the same thing on a thin snapshot and on a full one.
    ``TrainingBatch`` guarantees the sum is positive.
    """
    k = torch.tensor([sample.log_moneyness for sample in batch.samples], dtype=DTYPE)
    tenors = torch.tensor([sample.tenor_years for sample in batch.samples], dtype=DTYPE)
    observed = torch.tensor([sample.implied_vol for sample in batch.samples], dtype=DTYPE)
    weights = torch.tensor([sample.weight for sample in batch.samples], dtype=DTYPE)
    return k, tenors, observed, weights / weights.sum()


# --- the soft tier


def _durrleman_g(w: Tensor, k_mesh: Tensor, step: float) -> Tensor:
    """Durrleman's function on the interior of every mesh row, by central differences.

    Term for term the domain's ``invariants.durrleman_g`` -- the same second-order stencil on the
    same kind of uniform mesh, so that what the soft tier pushes on is what the hard tier will
    measure. Reimplemented in torch rather than imported because the domain's is numpy and a
    penalty has to be differentiable; ``test_the_torch_durrleman_agrees_with_the_domain`` holds
    the two copies together. Interior only, for the same reason as there: a one-sided difference
    is first-order accurate and would put a less trustworthy number in exactly the wings where the
    violations are.

    Args:
        w: Total variance on the mesh, ``(n_tenors, n_k)``.
        k_mesh: The moneyness coordinate broadcast to the same shape.
        step: The uniform spacing of the moneyness axis.

    Returns:
        ``g`` of shape ``(n_tenors, n_k - 2)``.
    """
    w_prime = (w[:, 2:] - w[:, :-2]) / (2.0 * step)
    w_second = (w[:, 2:] - 2.0 * w[:, 1:-1] + w[:, :-2]) / (step * step)
    k_interior = k_mesh[:, 1:-1]
    w_interior = w[:, 1:-1]
    return (
        (1.0 - k_interior * w_prime / (2.0 * w_interior)) ** 2
        - (w_prime * w_prime / 4.0) * (1.0 / w_interior + 0.25)
        + w_second / 2.0
    )


def _butterfly_penalty(w: Tensor, k_mesh: Tensor, step: float) -> Tensor:
    """Mean squared hinge on ``-g`` over the interior of every mesh row.

    Squared rather than linear so that the term is smooth where it switches on: a linear hinge has
    a constant gradient right up to the boundary and an optimiser holding a surface just inside it
    would be pushed back and forth across ``g = 0`` on every step.
    """
    return torch.mean(functional.relu(-_durrleman_g(w, k_mesh, step)) ** 2)


def _calendar_penalty(w: Tensor) -> Tensor:
    """Mean squared hinge on ``w_i - w_{i+1}`` between consecutive tenor rows, every column.

    Positive where the nearer expiry carries more total variance than the further one -- the
    crossing itself, in the units the gate reports it in. A single-tenor mesh has no consecutive
    pair and contributes zero, which is "not applicable" rather than "clean", exactly as the
    domain's report treats it.
    """
    if w.shape[0] < 2:
        return torch.zeros((), dtype=DTYPE)
    return torch.mean(functional.relu(w[:-1] - w[1:]) ** 2)
