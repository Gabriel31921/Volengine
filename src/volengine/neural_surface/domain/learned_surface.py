"""The surface a network produces, as the domain is willing to look at it.

The parametric context can hand its rules a *description* of a smile: five numbers, and
``SVIParams`` knows how to turn them into total variance and into its own derivatives. A network
has no such description. Its weights are not a vocabulary anyone can reason about, and the
domain is forbidden to import the framework that holds them (rule 3). What is left is the only
thing a trained model can honestly offer, and it is enough: **something you can evaluate**.
:class:`LearnedSurface` is that abstraction -- Design 6.1's "abstraccion evaluable sigma(k,T)"
-- and it is the whole surface of contact between the hard gate of ADR-010 and the artefact the
gate judges.

**Why the validation lives here, in module-level functions, rather than in the adapter.** Every
consumer of a surface in this context goes through :func:`evaluate_total_variance`: the gate in
``invariants.py``, the metrics that measure the neural-parametric distance of Design 6.5, and
the ACL that turns a surface into a published grid. If each of them decided for itself what
"usable" means, they would disagree the first time one of them was written in a hurry, and the
disagreement would show up as a surface that the gate accepted and the ACL then serialised as a
row of NaN. Putting the check in one function is what makes the guarantee worth stating at all:
if you got an array back from this module, it is rectangular, finite and strictly positive, and
nothing else in the context has to ask. An adapter evaluating its own model directly and
handing the result around would defeat exactly that, which is why the Protocol's own
``total_variance`` is deliberately raw and unpoliced -- it is the thing being checked, not the
check.

**A diverged model is not an arbitrageable one, and the two must never be confused.** This is
the distinction the module owns, and it is the reason there are exceptions in a context whose
central mechanism is a report. Arbitrage is a judgement about numbers: here is a total variance,
here is its curvature, and the density it implies is negative by this much. That judgement is
published as an :class:`ArbitrageReport`, compared against a configured tolerance (ADR-012), and
answered by declining to publish. A wrong shape, an infinity, a NaN or a total variance of zero
is not a bad number -- it is the absence of a number, and there is nothing for a tolerance to
compare. Folding it into the report would be worse than useless, because of the trap this repo
keeps walking into: ``nan < 0`` is ``False``, so a mesh full of NaN reports a butterfly
violation of exactly ``0.0`` and a calendar violation of exactly ``0.0``, and a model whose
weights all went to NaN on one bad gradient step would be published as the cleanest surface of
the session. One misplaced ``or`` and the loudest possible failure becomes the quietest. So the
rule is stated once, plainly: **exceptions are for "there is no honest number", the report is
for "here is the number, and it is bad"**.

The other half of that rule is that a bad *argument* is neither. An empty moneyness axis, a NaN
in the tenor axis, a tenor of zero -- none of those say anything about the model, because the
model was never asked a well-formed question. They are the caller's bug, they raise plain
``ValueError``, and they are visibly not a ``SurfaceEvaluationError``: ``NeuralSurfaceError``
does not inherit from ``ValueError``, so a caller that catches one never catches the other by
accident. The failure the operator has to act on -- retrain, republish the last good surface --
must not be reachable by passing an empty array.

**Tenor-major, once and everywhere.** ``total_variance`` answers with shape
``(len(tenors), len(k))``: one row per expiry, one column per moneyness node. That is the layout
the published grid uses -- a smile per tenor, indexed by tenor first -- so the ACL reshapes
nothing and there is no axis convention to get wrong at the boundary. The contract is not
imported here and must not be (rule 3); the layout is restated as this module's own rule, and
the shape check below is what makes the restatement true rather than hoped for. It is also the
single most likely mistake a real adapter makes, and it hides on a square mesh, which is why
nothing in this context evaluates on one.

numpy and nothing else. torch is barred from this layer by rule 3, and that is precisely what
lets these rules be checked against a surface written in four lines of arithmetic, long before
the model they will judge has been trained once.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from volengine.neural_surface.domain.errors import SurfaceEvaluationError


class LearnedSurface(Protocol):
    """A trained surface, reduced to the one thing the domain can use: evaluate it.

    Structural conformance, so nothing inherits from this. The production implementation is a
    PyTorch module living in ``adapters/``, which this layer neither imports nor names; a frozen
    dataclass wrapping a closure satisfies it just as well, and the test doubles that exercise
    the gate are exactly that. Two implementations of one shape, judged identically -- the same
    arrangement ``Calibrator`` gives the parametric context, for the same reason.

    **Not** ``@runtime_checkable``, and the omission is deliberate rather than forgotten.
    ``isinstance`` against a runtime-checkable Protocol compares *member names* and nothing more,
    so a ``total_variance`` that took its arguments in the wrong order, returned a list, or --
    the mistake that actually happens -- answered with the axes transposed would sail through
    it. A check that passes on the one bug it was written to catch is worse than no check,
    because it reads like reassurance. The real guards are two and neither is an ``isinstance``:
    ``mypy --strict`` where the concrete object is assigned to this port at the composition
    root, and :func:`evaluate_total_variance`, which looks at the array that actually came back.

    The Protocol deliberately stops here. No ``fit``, no ``parameters``, no ``to_dict``: training
    belongs to ``SurfaceLearner`` in ``ports.py``, and a surface that could describe itself would
    be a parameterisation, which is the other context's answer to the same problem.
    """

    @property
    def version(self) -> int:
        """Monotone counter identifying the state of the weights this surface evaluates.

        A published surface has to be traceable back to what produced it, and
        ``CalibratedSurface.producer_meta`` is a ``Mapping[str, float]`` -- numbers only, by
        design, so that the contract stays serialisable without a codec. An integer counter is
        what survives that boundary intact, where a hash or a checkpoint path would not.

        It is also the measurement of Design 6.4. The scheduled restart retrains from the whole
        buffer and hands back a surface with a new version; comparing the surfaces on either side
        of that boundary is the honest reading of how far continuous fine-tuning had drifted, and
        without an identity on each of them the comparison has nothing to name.

        A read-only property rather than a plain attribute so the Protocol constrains neither how
        it is stored nor whether it is computed: a dataclass field, a class constant and a
        derived expression all satisfy it.
        """
        ...

    def total_variance(
        self, k: NDArray[np.float64], tenors: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """Total variance on the full ``(tenors x k)`` mesh, tenor-major.

        Total variance rather than volatility because that is what the network outputs (Design
        6.2): the calendar condition is monotonicity of ``w`` in the tenor, which is trivial to
        state and to penalise on this quantity and needs an extra factor of ``T`` on any other.
        Everything downstream that wants a volatility divides and takes a root, which is what
        :func:`implied_vol_grid` is.

        Args:
            k: Log-forward-moneyness nodes, ``ln(K / F)``, one-dimensional. Zero is at-the-money
                forward and is an ordinary value, so nothing here may test it for truthiness.
            tenors: Tenor axis in years, one-dimensional, already resolved under the producing
                market's daycount before it ever reached this context. The two axes are passed
                separately rather than as a flat list of points because the answer is a mesh: the
                gate differentiates along ``k`` at fixed tenor and compares along the tenor at
                fixed ``k``, and both are row and column operations on the result.

        Returns:
            Shape ``(len(tenors), len(k))``. Row ``i`` is the smile at ``tenors[i]``, column ``j``
            is the term structure at ``k[j]``.

        Implementations are not asked to validate anything and should not try: the guarantees
        callers rely on are imposed by :func:`evaluate_total_variance`, in one place, so that
        they mean the same thing for every implementation. An implementation that raised on its
        own terms would put a second vocabulary of failure in front of the one this context
        defines.
        """
        ...


def _require_axis(values: NDArray[np.float64], what: str) -> None:
    """Reject an axis that cannot carry a question, as a plain ``ValueError``.

    Three failures, all of them the caller's. An empty axis makes the mesh empty, and an empty
    mesh is not a clean surface -- "nothing was looked at" and "nothing was found" are different
    statements and only one of them belongs anywhere near a tolerance. A non-finite node poisons
    the row or column it sits in and every reduction over it afterwards, and it would arrive at
    the gate looking like a model failure when it is nothing of the kind. A multi-dimensional
    axis makes ``len`` mean the first dimension only, which would silently redefine the shape
    every consumer of this module is promised.

    Emptiness is tested with ``.size`` rather than truthiness: numpy raises on the truth value of
    a multi-element array, so ``if not values`` crashes on every axis but the degenerate ones.
    """
    if values.ndim != 1:
        raise ValueError(f"The {what} must be one-dimensional, got {values.ndim} dimensions")
    if values.size == 0:
        raise ValueError(f"The {what} must hold at least one point, got an empty array")
    if not np.all(np.isfinite(values)):
        raise ValueError(f"The {what} must be finite at every point")


def _require_tenors(tenors: NDArray[np.float64]) -> None:
    """Everything :func:`_require_axis` asks, plus strict positivity.

    A tenor is a length of time to an expiry, so zero is the expiry itself and negative is the
    past. Neither has a total variance: at ``T = 0`` all uncertainty has collapsed onto a point,
    and :func:`implied_vol_grid` would divide by it. Refusing here rather than at the division is
    what keeps that failure attributable -- a zero tenor is a caller passing a stale expiry, not
    a model that has diverged, and the two must not arrive at the operator wearing the same face.

    Finiteness first and the bad conditions joined with ``or``, because ``float("nan") <= 0`` is
    ``False`` and a NaN tenor would otherwise pass a positivity test and be reported as a
    perfectly ordinary expiry.
    """
    _require_axis(tenors, "tenor axis")
    if np.any(tenors <= 0.0):
        raise ValueError(
            f"The tenor axis must be strictly positive at every point, got a minimum of "
            f"{np.min(tenors)}"
        )


def evaluate_total_variance(
    surface: LearnedSurface, k: NDArray[np.float64], tenors: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Evaluate a learned surface on a mesh and refuse anything that is not a surface.

    The single door every consumer of a :class:`LearnedSurface` in this context goes through. It
    is a thin function on purpose: it asks the model one question and then decides whether the
    answer is a number at all. What it never does is judge the answer -- a surface can be
    perfectly usable and deeply arbitrageable at the same time, and saying so is the job of the
    report in ``invariants.py``, not of an exception here.

    Args:
        surface: The model to evaluate. Taken as an argument rather than as ``self`` because this
            is a rule about surfaces, not a capability of one: it applies identically to a
            PyTorch model, to a hand-written double in a test, and to anything else that satisfies
            the Protocol, and none of them should be able to opt out of it by overriding a method.
        k: Log-forward-moneyness nodes, one-dimensional, non-empty, finite. Not required to be
            sorted or unique -- this function evaluates, it does not build a grid, and the mesh
            invariants that the gate needs are ``ArbitrageMesh``'s to enforce.
        tenors: Tenor axis in years, one-dimensional, non-empty, finite and strictly positive.
            Passed separately from ``k`` for the same reason the Protocol takes them separately:
            the result is a mesh with two meaningful axes, and the caller has to be able to say
            which is which.

    Returns:
        Total variance of shape ``(len(tenors), len(k))``, finite and strictly positive
        everywhere. Row ``i`` is the smile at ``tenors[i]``. Whatever comes out of here can be
        divided by, taken the log of, and differenced without a further check, which is the only
        reason the modules downstream are as short as they are.

    Raises:
        SurfaceEvaluationError: If the model answered with the wrong shape, with a non-finite
            value, or with a total variance that is not strictly positive.

            All three mean the same thing -- the model has stopped describing a market -- and
            none of them is an arbitrage violation. The shape case is the transposition an
            adapter gets wrong on its first day, invisible on a square mesh and catastrophic
            everywhere else: it would not raise downstream, it would quietly report the term
            structure as a smile. The finiteness case is one bad gradient step, and it is the one
            that has to raise rather than be reported, because ``nan < 0`` is ``False`` and a
            grid of NaN scores zero on every violation measure in the engine. The positivity case
            is the degenerate limit: total variance is a volatility squared times a tenor, so
            zero says the distribution has collapsed onto a point and negative has no reading at
            all, and both make the square root in :func:`implied_vol_grid` meaningless.

        ValueError: If ``k`` or ``tenors`` is empty, not one-dimensional or non-finite, or if
            ``tenors`` is not strictly positive. That is the caller's bug: the model was never
            asked a well-formed question, so nothing it did or did not do is being reported.
            Deliberately a different kind of failure from the one above, and observably so --
            ``SurfaceEvaluationError`` is not a ``ValueError``, so the handler that republishes
            the last good surface cannot be triggered by an empty array.
    """
    _require_axis(k, "moneyness axis")
    _require_tenors(tenors)

    w = surface.total_variance(k, tenors)
    expected = (tenors.size, k.size)
    if w.shape != expected:
        raise SurfaceEvaluationError(
            f"The surface must answer one total variance per mesh node, tenor-major: expected "
            f"shape {expected}, got {w.shape}"
        )
    # Finiteness first, then the sign, joined with `or`. A NaN passes `w <= 0` and an infinity
    # passes it too, and either one would leave this function as a value every downstream
    # tolerance accepts as clean.
    if not np.all(np.isfinite(w)) or np.any(w <= 0.0):
        raise SurfaceEvaluationError(
            "The surface must be finite and strictly positive at every mesh node, got a minimum "
            f"of {np.min(w)}"
        )
    return w


def implied_vol_grid(
    surface: LearnedSurface, k: NDArray[np.float64], tenors: NDArray[np.float64]
) -> NDArray[np.float64]:
    """The same mesh as implied volatilities: ``sqrt(w / T)``, row by row.

    The network speaks total variance and the outside world speaks volatility, and this is the
    one place the conversion happens. It is here rather than in the ACL because it is the inverse
    of ``w = sigma^2 * T``, which is this context's own arithmetic and not a translation into
    anyone's published language; the ACL's job is to arrange numbers into a DTO, and it should
    not also be the only module that knows what they mean.

    The division is by ``tenors`` broadcast down the tenor axis, which is exactly the layout
    :func:`evaluate_total_variance` guarantees -- row ``i`` belongs to ``tenors[i]``. That is the
    second reason the shape check earns its place: a transposed answer would divide each smile by
    the wrong expiry and produce a grid of entirely plausible volatilities that describe nothing.

    Args:
        surface: The model to evaluate, on the same terms as :func:`evaluate_total_variance` --
            the conversion is applied to a validated grid, never to a raw one, so a diverged
            model cannot reach the square root.
        k: Log-forward-moneyness nodes. Passed straight through; this function adds no
            requirement of its own on them, because the conversion touches only the tenor axis.
        tenors: Tenor axis in years, strictly positive, which is what makes the division defined.
            The positivity is checked upstream rather than here so that a zero tenor is reported
            as the caller's bug it is, instead of surfacing as an infinity in the result.

    Returns:
        Implied volatilities in absolute terms (``0.65`` is 65%), shape ``(len(tenors), len(k))``,
        finite and strictly positive. The square root is defined because the total variance was
        validated strictly positive first, and it is why that check is strict rather than merely
        non-negative.

    Raises:
        SurfaceEvaluationError: Everything :func:`evaluate_total_variance` raises, on the same
            terms, plus the one failure this step can add on its own: a validated total variance
            divided by a legal but vanishingly small tenor can still overflow to infinity, and an
            infinite volatility is no more publishable than a NaN one. Rare, and checked anyway,
            because the alternative is an ``inf`` walking into a contract whose grid promises
            every vol is finite.

        ValueError: Everything :func:`evaluate_total_variance` raises for a malformed axis.
    """
    w = evaluate_total_variance(surface, k, tenors)
    vols: NDArray[np.float64] = np.sqrt(w / tenors[:, None])
    if not np.all(np.isfinite(vols)):
        raise SurfaceEvaluationError(
            "The implied volatilities must be finite at every mesh node; the total variance was "
            "usable, so the tenor axis has divided it into an overflow"
        )
    return vols
