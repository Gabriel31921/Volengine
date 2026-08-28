"""The hard gate of ADR-010: no-arbitrage judged on a mesh, before anything is published.

Three tiers of constraints govern the neural surface, and this module is the third one. The soft
tier lives in the training loop, where Durrleman's condition and calendar monotonicity are
penalties the optimiser can trade against the fit. The architectural tier is not built in v1. The
hard tier is here, in ordinary numpy, in a layer that imports no torch: the surface is evaluated
on a mesh, the two arbitrage conditions are measured on it, and the use case declines to publish
when the measurement is worse than the configured tolerance. *Soft constraints train, hard
constraints govern* -- and a constraint the network can push against during training is not a
constraint at all, which is why the governing one is stated where no gradient can reach it.

**Nothing here raises when it finds arbitrage.** A violation is a number in an
:class:`ArbitrageReport`, and refusing a surface is the gate working rather than the system
breaking; ``domain/errors.py`` argues the point at length. The one thing that *does* raise is a
surface that stopped being a surface -- and that check is not this module's, it belongs to
``learned_surface.evaluate_total_variance``, which every consumer goes through so that "usable"
means one thing engine-wide.

**The same condition as the parametric context, computed the other way round.** The formula::

    g(k) = (1 - k * w'(k) / (2 * w(k)))^2  -  (w'(k)^2 / 4) * (1 / w(k) + 1/4)  +  w''(k) / 2

is Durrleman's function, and ``g >= 0`` everywhere on a slice is the statement that the
risk-neutral density that slice implies is non-negative -- that a butterfly spread, whose payoff
is never negative in any state of the world, is not being priced at a negative cost.
``parametric_pricing/domain/durrleman.py`` evaluates exactly this expression and gets ``w'`` and
``w''`` in closed form, because raw SVI is five parameters and an algebraic curve: differentiating
it is a line of calculus and the answer is exact. **A network has no closed form.** An MLP with
softplus activations is differentiable, but its derivative lives inside a framework this layer is
forbidden to import (rule 3), and asking the adapter for it would move the invariant into the
thing being judged. So this module takes the only derivative available to it -- second-order
central differences of the evaluated surface -- and owns the truncation error the other module
refuses. That is a real cost, paid deliberately: the differences are exact only to ``O(h^2)``, and
a reported violation of ``1e-6`` on a mesh of step ``0.05`` is indistinguishable from the
discretisation. It is the price of judging an artefact that has no algebra.

The two modules are never unified and never import each other. Rule 6 forbids one context
importing another, and ADR-010's last consequence says this duplication is on purpose: a single
shared "Durrleman module" would be the canonical-model anti-pattern, and it would also hide the
fact that the two contexts genuinely compute the thing differently and fail differently.

**What the mesh can and cannot see.** Two limits are structural and both are documented on the
functions that carry them. The mesh is a *sample* of a statement about the whole line, so a
violation narrower than the spacing is invisible -- and the module cannot tell that case from a
clean surface, which is why Design 6.3 asks for a dense mesh and why the mesh is configuration
(ADR-012) rather than a constant. And central differences need a neighbour on each side, so the
two outermost moneyness points are never judged: the mesh has to extend past the region anyone
cares about, or the wings -- where butterfly violations actually live -- fall in the blind band.

numpy is used for the vectorised evaluation and nothing else. torch, jax and scipy are barred
from this layer, which is what lets the gate be tested against a surface written in four lines of
numpy, long before there is a trained model to point it at.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from itertools import pairwise

import numpy as np
from numpy.typing import NDArray

from volengine.neural_surface.domain.learned_surface import (
    LearnedSurface,
    evaluate_total_variance,
)

_SPACING_RTOL = 1e-9
"""How far consecutive mesh spacings may drift from their mean, relatively, and still count as
uniform.

Two failure directions bound this number, and there are six orders of magnitude between them.

Too tight and a legitimate mesh is refused. A mesh is built in floating point, so each point
carries a rounding error of about ``eps * |k|``, and the *difference* of two neighbours carries
that error against a divisor of ``h``: the relative drift of the spacings is therefore around
``eps * max|k| / h``. The mesh the tests are written on -- 25 points across +-60% -- drifts by
about ``1.5e-15``; a mesh a hundred times finer, ``h = 5e-4`` across the same band, drifts by
about ``3e-13``. Even an absurdly fine ``h = 1e-5`` stays under ``1.5e-11``.

Too loose and the check stops meaning anything. What it is really there to refuse is the mesh a
person would naturally reach for -- coarse in the wings, fine around the money -- whose spacings
differ by *tens of percent*, not by parts per billion. Any such mesh is rejected by a margin of
eight orders of magnitude, so the threshold never has to be a judgement call at the boundary.

The check is not cosmetic. A central difference derives its second-order accuracy from the two
neighbours being equidistant: on an unequal stencil the first-order error terms stop cancelling
and ``w'`` degrades to ``O(h)``, silently. This gate's whole job is to be trusted near zero, and
an error term that shows up as a small positive or negative ``g`` is exactly the answer it must
not invent.
"""


def _require_non_negative_finite(value: float, what: str) -> None:
    """Reject anything that is not a usable non-negative number, as a plain ``ValueError``.

    Finiteness first, and the bad cases joined with ``or``, because ``float("nan") < 0`` is
    ``False``: written the other way round a NaN walks straight through, which in this module
    would mean a tolerance of NaN against which no violation ever compares greater and a gate
    that has quietly stopped gating.
    """
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"The {what} must be non-negative and finite, got {value}")


def _uniform_step(k: NDArray[np.float64], what: str) -> float:
    """The spacing of a mesh central differences may be taken on, or a ``ValueError``.

    Four conditions, each of which would otherwise corrupt a derivative rather than announce
    itself. A one-dimensional axis, because a stray second dimension would broadcast into ``g``
    and produce a plausible array of the wrong meaning. At least three points, since a central
    difference needs a neighbour on each side and a two-point mesh has no interior at all.
    Finite everywhere -- checked *before* any ordering, because a NaN compares ``False`` against
    every bound and would sail past the increasing test to poison ``w'`` and ``w''`` and then, via
    ``max(0, -min(g))``, be reported as a perfectly clean surface. And strictly increasing, which
    both fixes the sign of ``h`` and rules out a repeated point that would make the spacing zero
    and the second difference infinite.

    The step is taken as ``(k[-1] - k[0]) / (n - 1)`` rather than as the mean of the differences.
    The two agree exactly in real arithmetic -- the sum telescopes -- but the endpoint form
    involves one subtraction instead of ``n``, so it does not accumulate the rounding of every
    interval into the divisor that the whole measurement is scaled by.
    """
    if k.ndim != 1:
        raise ValueError(f"The {what} must be one-dimensional, got shape {k.shape}")
    if k.size < 3:
        raise ValueError(
            f"The {what} must hold at least three points for a central difference, got {k.size}"
        )
    if not np.all(np.isfinite(k)):
        raise ValueError(f"The {what} must be finite at every point")

    spacings = np.diff(k)
    if np.any(spacings <= 0.0):
        raise ValueError(
            f"The {what} must be strictly increasing, got a spacing of {spacings.min()}"
        )

    step = float(k[-1] - k[0]) / (k.size - 1)
    if np.any(np.abs(spacings - step) > _SPACING_RTOL * step):
        raise ValueError(
            f"The {what} must be uniformly spaced for a central difference, got spacings from "
            f"{spacings.min()} to {spacings.max()}"
        )
    return step


@dataclass(frozen=True, slots=True)
class ArbitrageMesh:
    """Where the hard gate looks: a uniform moneyness axis crossed with a tenor axis.

    Not the training data and not the published grid -- a third thing, chosen for the judgement
    rather than for the fit or for the consumer. It is configuration (ADR-012), because how dense
    and how wide it should be is a policy about how much arbitrage the engine is willing to be
    blind to, and that is a decision someone makes in a TOML file with the wings of a real market
    in front of them, not a constant compiled into a domain module.

    The two axes are asymmetric on purpose, and the asymmetry is the whole shape of this class.
    Moneyness is differentiated, so it must be uniform and must have an interior; tenors are only
    compared, so they need neither. Stating that as two different sets of invariants on one object
    is what stops a caller from handing the calendar check a mesh that was only ever fit for the
    butterfly one.
    """

    log_moneyness: tuple[float, ...]
    """``k = ln(K / F)``, finite, strictly increasing and **uniformly spaced**. At least three
    points.

    Three is the smallest mesh with an interior point, and an interior point is the only kind that
    gets judged. Uniformity is the requirement that makes the central differences second-order
    accurate; see :data:`_SPACING_RTOL` for the tolerance and why it sits where it does. The
    natural instinct -- pack points around the money where the quotes are and spread them out in
    the wings -- is refused here, and refusing it costs nothing: the wings are where butterfly
    violations live, so a mesh that is coarse there is a mesh that cannot see the failure it was
    built to catch.

    **Extend it past what you care about.** The outermost two points are consumed as neighbours
    and never judged, so a mesh clipped to the quoted strike band leaves its own edge unexamined.
    """

    tenors: tuple[float, ...]
    """Year fractions, finite, strictly positive and strictly increasing. At least one.

    One tenor is legal, and it is the honest representation of a market quoting a single expiry:
    the butterfly condition is a statement about one slice and says everything it has to say, and
    the calendar condition simply has nothing to compare. No uniformity is required because
    nothing is differentiated along this axis -- ``w`` at neighbouring tenors is subtracted, not
    differenced -- so an axis mirroring the real expiry ladder, dense at the front and sparse at
    the back, is exactly right here and would be wrong on the other one.

    Strictly increasing because the calendar condition is directional: ``w`` must not fall as the
    expiry lengthens, and a shuffled axis would report the same surface as violating or clean
    depending on the order it was handed.
    """

    def __post_init__(self) -> None:
        _uniform_step(np.asarray(self.log_moneyness, dtype=np.float64), "moneyness mesh")

        if not self.tenors:
            raise ValueError("The mesh must hold at least one tenor")
        for index, tenor in enumerate(self.tenors):
            if not math.isfinite(tenor) or tenor <= 0:
                raise ValueError(
                    f"The tenor at index {index} must be positive and finite, got {tenor}"
                )
        # `pairwise`, never `zip(xs, xs[1:], strict=True)`: consecutive pairs differ in length by
        # one by design, so the strict flag this repo requires elsewhere would be wrong here.
        for near, far in pairwise(self.tenors):
            if near >= far:
                raise ValueError(f"The tenors must be strictly increasing, got {near} before {far}")

    @property
    def k_array(self) -> NDArray[np.float64]:
        """The moneyness axis as an array, for the surface and the differences.

        Rebuilt on every access rather than cached: a frozen dataclass with ``slots=True`` has
        nowhere to put a cache that would not also be a mutable attribute on an immutable object,
        and a numpy array is mutable, so a shared one could be written through by any caller and
        would silently change what the gate measured. Copying a few hundred floats once per
        snapshot is not a cost worth trading an aliasing bug for.
        """
        return np.asarray(self.log_moneyness, dtype=np.float64)

    @property
    def tenor_array(self) -> NDArray[np.float64]:
        """The tenor axis as an array, on the same terms as :attr:`k_array`."""
        return np.asarray(self.tenors, dtype=np.float64)


@dataclass(frozen=True, slots=True)
class ArbitrageReport:
    """What the gate found, as numbers: two depths, and how widely each breach ran.

    This object is published whether or not the surface is, and that is the point of it. When the
    gate refuses, the previous surface is republished and *this* is the explanation of why -- so a
    report that only said "rejected" would leave the operator with a stalled producer and no way
    to tell a wing that grazes zero from a fit that has come apart. When the gate passes, the same
    numbers are the metric series that says how close to the boundary the network has been
    drifting, which is the early warning that a refusal is coming.

    **Depths and counts sit next to each other because they answer different questions.** One
    isolated dip of 0.4 between two mesh points is a surface with a defect; four hundred points of
    0.4 across an entire wing is a surface that is wrong. A single number cannot separate those,
    and picking either one alone would make the tolerance in the TOML mean something different in
    the two cases.
    """

    butterfly_violation: float
    """Depth of the worst butterfly breach on the judged points: ``max(0, -min g)``. Non-negative
    and finite.

    Zero means clean and larger is worse, which is the orientation that lets one number serve as
    both the published metric and, in the training loop, the soft penalty of ADR-010's first tier:
    a quantity that is zero on the admissible set and grows with the breach is exactly what a
    penalty has to be. Without the clamp at zero a comfortably arbitrage-free surface would report
    a large *negative* number and the penalty would go on rewarding the optimiser for lifting
    ``g`` long after the constraint stopped binding.

    The units are those of ``g`` itself, which are those of a density: it says how negative the
    implied density gets, not how far the weights are from admissible.
    """

    calendar_violation: float
    """Largest amount by which total variance falls as the expiry lengthens. Non-negative, finite.

    ``max(0, max(w_near - w_far))`` over every consecutive pair of tenors at every moneyness
    point. Variance accumulates: whatever uncertainty the market has priced by the near expiry is
    still priced by the far one, plus whatever happens in between. A surface claiming otherwise
    prices a calendar spread -- long the far option, short the near -- at a negative cost.

    In total-variance units, so it is directly comparable across tenors and directly comparable to
    the ``w`` the network emits, which is why Design 6.2 has the network output total variance in
    the first place.
    """

    n_points_judged: int
    """How many ``(tenor, interior k)`` points ``g`` was evaluated at. At least one.

    The denominator that makes ``n_butterfly_violations`` readable, and the answer to a question
    the depths cannot be trusted without: *how much was actually looked at*. Two edge columns are
    dropped by the differences, so this is ``len(tenors) * (len(log_moneyness) - 2)`` and never
    the size of the mesh -- reporting the mesh size here would overstate the coverage by exactly
    the band where the wings are.
    """

    n_butterfly_violations: int
    """How many judged points came out with ``g < 0``. Between zero and
    :attr:`n_points_judged`.

    Counted strictly below zero, matching the condition: ``g == 0`` is the boundary of the
    admissible set and a density that touches zero is degenerate, not arbitrageable.
    """

    n_calendar_violations: int
    """How many ``(tenor pair, k)`` points came out with total variance falling. Non-negative.

    Its ceiling is not :attr:`n_points_judged` and deliberately has no invariant tying it there:
    the calendar check runs over a different population -- every moneyness point including the two
    edges, against one fewer tenor -- so a bound copied from the butterfly count would be wrong in
    both directions.

    This is also what disambiguates the one genuinely ambiguous number in the report. A
    single-tenor mesh reports a calendar violation of ``0.0``, and so does a two-tenor mesh that
    was checked and found clean. The depths are identical because there is no third value that
    honestly means "not applicable"; the count is ``0`` in both cases too, but read together with
    ``len(tenors)`` the caller can always tell the two apart -- and neither reading changes what
    the gate does, since nothing was found either way.
    """

    def __post_init__(self) -> None:
        _require_non_negative_finite(self.butterfly_violation, "butterfly violation")
        _require_non_negative_finite(self.calendar_violation, "calendar violation")

        if self.n_points_judged < 1:
            raise ValueError(
                "A report must judge at least one point -- nothing was found and nothing was "
                f"looked at are different verdicts, got {self.n_points_judged}"
            )
        if not 0 <= self.n_butterfly_violations <= self.n_points_judged:
            raise ValueError(
                "The butterfly violation count must lie between zero and the number of judged "
                f"points, got {self.n_butterfly_violations} of {self.n_points_judged}"
            )
        if self.n_calendar_violations < 0:
            raise ValueError(
                f"The calendar violation count cannot be negative, got {self.n_calendar_violations}"
            )

    def exceeds(self, butterfly_tol: float, calendar_tol: float) -> bool:
        """Whether this surface must not be published. The gate itself, in one line.

        **The thresholds arrive as arguments and the rule lives here**, and that split is the
        whole of ADR-010's third tier. How much arbitrage is tolerable is a number someone tunes
        in a TOML file per market (ADR-012) and will keep tuning; *that a surface over the
        threshold is refused* is a domain rule, and putting it in the use case -- or worse, in the
        training loop -- would put it somewhere a gradient or a config edit could get around. A
        model cannot train its way past a method it never sees.

        The comparison is ``>``, so a violation landing exactly on the tolerance passes. That is
        the right way round for a threshold that will be set to zero: with ``>=`` a tolerance of
        ``0.0`` would refuse every surface, including the arbitrage-free ones, since a clean report
        holds exactly ``0.0``. Tolerances are compared independently and combined with ``or``: the
        two conditions are different statements about the surface and a butterfly breach is not
        excused by a clean term structure.

        Args:
            butterfly_tol: Largest butterfly depth still publishable. Non-negative and finite;
                ``0.0`` is meaningful and means no breach at all is tolerated.
            calendar_tol: Largest calendar depth still publishable, in total-variance units, on
                the same terms.

        Returns:
            ``True`` when either measured depth is strictly above its tolerance.

        Raises:
            ValueError: If either tolerance is negative or not finite. A NaN tolerance is the one
                that has to be caught here: every ``>`` against it is ``False``, so a
                misconfigured gate would not fail loudly, it would silently pass every surface
                for the rest of the session -- which is precisely the outcome this class exists
                to prevent.
        """
        _require_non_negative_finite(butterfly_tol, "butterfly tolerance")
        _require_non_negative_finite(calendar_tol, "calendar tolerance")
        return self.butterfly_violation > butterfly_tol or self.calendar_violation > calendar_tol


def durrleman_g(w: NDArray[np.float64], k: NDArray[np.float64]) -> NDArray[np.float64]:
    """Durrleman's function of one slice, differenced from sampled total variance.

    ``g >= 0`` everywhere on a slice is equivalent to that slice being free of butterfly
    arbitrage: it is the risk-neutral density the smile implies, up to a strictly positive factor,
    so a negative value is a fit pricing a butterfly spread -- a payoff that is never negative in
    any state of the world -- at a negative cost. The formula and the contrast with the analytic
    treatment in the parametric context are in the module docstring.

    The derivatives are second-order central differences on the uniform mesh ``k``::

        w'(k_i)  = (w[i+1] - w[i-1]) / (2 h)
        w''(k_i) = (w[i+1] - 2 w[i] + w[i-1]) / h^2

    Both stencils reach one point either side, so ``g`` is defined on the **interior only** and
    the returned array is two shorter than the input. The outermost two points of the mesh are
    consumed as neighbours and never judged -- there is no one-sided fallback here on purpose,
    because a one-sided difference is first-order accurate and would put a systematically less
    trustworthy number in exactly the wings where the violations are, wearing the same units as
    the rest and indistinguishable in the report.

    Args:
        w: Total variance along one tenor, one value per point of ``k``. Finite and strictly
            positive.
        k: The moneyness mesh: one-dimensional, finite, strictly increasing, uniformly spaced,
            at least three points.

    Returns:
        ``g`` at ``k[1:-1]``, elementwise, of length ``len(k) - 2``. Positive is healthy, negative
        is arbitrage, and the magnitude of a negative value is how deep the breach runs there.

    Raises:
        ValueError: If ``k`` is not a mesh differences can be taken on, if ``w`` does not match it
            in shape, if ``w`` is not finite and strictly positive everywhere, or if ``g`` itself
            comes out non-finite.

        The condition on ``w`` is a plain ``ValueError`` and not a ``SurfaceEvaluationError``,
        even though a diverged model is the likeliest way to produce such a ``w``. The reason is
        that this function does not know where its argument came from: reached through
        :func:`check_surface` the array has already been through
        ``learned_surface.evaluate_total_variance``, which raises the domain error for exactly
        this case, so anything arriving here broken came from a caller that built the row itself
        -- a bug in this process, not a market condition. Guarding anyway is not redundant: ``g``
        divides by ``w`` twice, and an ``inf`` or a NaN produced there survives
        ``max(0, -min(g))`` as a clean ``0.0`` in both cases, because ``nan < 0`` is ``False``.
        A broken slice reported as arbitrage-free is the worst failure mode a gate has.
    """
    step = _uniform_step(k, "moneyness mesh")

    if w.shape != k.shape:
        raise ValueError(
            f"The total variance must hold one value per mesh point, got shape {w.shape} "
            f"against a mesh of shape {k.shape}"
        )
    # Finiteness first, then the sign, joined with `or`: a NaN sails through `w <= 0` and would
    # leave this function as a NaN that every downstream `max` accepts as a clean verdict.
    if not np.all(np.isfinite(w)) or np.any(w <= 0.0):
        raise ValueError(
            "The total variance must be finite and strictly positive at every mesh point to "
            f"evaluate Durrleman's function, got a minimum of {np.min(w)}"
        )

    w_prime = (w[2:] - w[:-2]) / (2.0 * step)
    w_second = (w[2:] - 2.0 * w[1:-1] + w[:-2]) / (step * step)
    k_interior = k[1:-1]
    w_interior = w[1:-1]

    g: NDArray[np.float64] = (
        (1.0 - k_interior * w_prime / (2.0 * w_interior)) ** 2
        - (w_prime * w_prime / 4.0) * (1.0 / w_interior + 0.25)
        + w_second / 2.0
    )
    if not np.all(np.isfinite(g)):
        raise ValueError("Durrleman's function is not finite over the given moneyness mesh")
    return g


def check_surface(surface: LearnedSurface, mesh: ArbitrageMesh) -> ArbitrageReport:
    """Judge a learned surface on the mesh, and report. The only entry point the gate needs.

    One evaluation of the surface feeds both conditions, which is not merely an optimisation: the
    butterfly and calendar numbers in a report have to describe the *same* surface, and evaluating
    twice would leave a stateful or non-deterministic model free to answer differently between the
    two and produce a verdict that never existed.

    The two conditions are then measured over different populations, and the difference is
    structural rather than an oversight. The butterfly condition differentiates along moneyness,
    so it lives on the interior columns only. The calendar condition subtracts one tenor row from
    the next, and subtraction needs no neighbours -- so it runs across **every** moneyness point
    including the two the differences dropped, and losing them there would be throwing away the
    wings for no reason at all.

    A mesh with a single tenor has no consecutive pair, so there is no calendar condition to check
    and the report carries ``0.0`` with a count of ``0``. See
    :attr:`ArbitrageReport.n_calendar_violations` on why that is honest rather than a silent
    conflation of "clean" with "not applicable".

    Args:
        surface: Any evaluable surface, of any origin. The gate never asks what produced it, which
            is what lets it be tested against a hand-written one and lets the same invariant judge
            a parametric surface if anyone ever wants the comparison.
        mesh: Where to look. Its density and width are how much arbitrage the engine agrees to be
            blind to; see :class:`ArbitrageMesh`.

    Returns:
        The verdict as numbers. Publication is decided by
        :meth:`ArbitrageReport.exceeds` against the configured tolerances, by the use case, never
        here -- this function measures and does not judge, which is what keeps a threshold change
        out of the domain.

    Raises:
        SurfaceEvaluationError: If the surface returns the wrong shape, a non-finite value, or a
            total variance that is not strictly positive. Raised by
            ``learned_surface.evaluate_total_variance``, deliberately not caught, and deliberately
            not folded into the report: a NaN grid is a **diverged model, not an arbitrageable
            one**, and it would score a butterfly violation of exactly zero -- the cleanest report
            of the session -- because ``nan < 0`` is ``False``. Both outcomes end in nothing being
            published, but only one of them belongs in a metric series that is supposed to mean
            "how close to arbitrage-free is this fit".
    """
    k = mesh.k_array
    # Every guarantee `durrleman_g` and the calendar check need -- shape, finiteness, strict
    # positivity -- is established here, once, by the module that owns what "usable" means. It is
    # not repeated below.
    w = evaluate_total_variance(surface, k, mesh.tenor_array)

    g: NDArray[np.float64] = np.vstack([durrleman_g(row, k) for row in w])
    butterfly = max(0.0, -float(np.min(g)))
    n_butterfly = int(np.count_nonzero(g < 0.0))

    if w.shape[0] < 2:
        calendar = 0.0
        n_calendar = 0
    else:
        # Row i minus row i+1: positive where the nearer expiry carries more total variance than
        # the further one, which is the crossing itself.
        excess = w[:-1] - w[1:]
        calendar = max(0.0, float(np.max(excess)))
        n_calendar = int(np.count_nonzero(excess > 0.0))

    return ArbitrageReport(
        butterfly_violation=butterfly,
        calendar_violation=calendar,
        n_points_judged=int(g.size),
        n_butterfly_violations=n_butterfly,
        n_calendar_violations=n_calendar,
    )
