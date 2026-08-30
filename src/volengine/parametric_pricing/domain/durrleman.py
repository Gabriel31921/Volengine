"""No-arbitrage diagnostics for SVI: butterfly on one slice, calendar between two.

This module is the other half of the split ``svi_slice.py`` describes: **the value object
admits, the metric judges**. ``SVIParams`` guarantees only that its five numbers describe a
curve that is a surface at all -- finite, non-negative variance, ``|rho| < 1`` -- and stops
deliberately short of arbitrage-freeness, because a constructor that refused an arbitrageable
slice would refuse the optimiser its own search path and would make the violation unmeasurable
at the same time. Everything that is *not* enforced there is measured here, as a plain
non-negative number the use case compares against a configured acceptance threshold (ADR-012).

**Butterfly arbitrage, and what ``g`` really is.** Fix one expiry. The call price as a function
of strike determines the risk-neutral density of the underlying at that expiry: the density is
the second derivative of the (undiscounted) call price in the strike, which is the limit of the
butterfly spread -- buy one call at ``K - dK``, sell two at ``K``, buy one at ``K + dK`` --
divided by ``dK^2``. That spread has a non-negative payoff in every state of the world, so it
must cost something non-negative; a smile that prices it negative is offering a portfolio that
can only pay out and yet is paid for holding. Durrleman's function ``g(k)`` is exactly that
density rewritten in the SVI coordinates ``(k, w)``, up to a strictly positive factor::

    g(k) = (1 - k * w'(k) / (2 * w(k)))^2  -  (w'(k)^2 / 4) * (1 / w(k) + 1/4)  +  w''(k) / 2

so ``g(k) >= 0`` for every ``k`` **is** the statement that the implied density is non-negative,
and ``g(k) < 0`` somewhere is a butterfly spread with negative cost, priced by a fit nobody
would want to publish. It is a condition on one slice alone: the tenor has already been absorbed
into the total variance ``w``, and no other expiry appears anywhere in the formula.

**Calendar arbitrage** is the other half, and it is the one condition that spans expiries.
Total variance at a fixed ``k`` cannot decrease as the expiry lengthens -- uncertainty
accumulates, it does not un-accumulate -- and a fit that says otherwise sells a calendar spread
for a negative price. Raw SVI fitted one slice at a time guarantees nothing of the sort, since
nothing couples the expiries during the fit. ADR-008 accepts that risk knowingly:
:func:`calendar_violation` **measures and reports** it, and never corrects it. Correcting it by
construction is SSVI, deferred to its own milestone.

**Analytic derivatives, not finite differences.** With ``y = k - m`` and
``r = sqrt(y^2 + sigma^2)``::

    w   = a + b * (rho * y + r)          (the shared kernel's, not restated here)
    w'  = b * (rho + y / r)
    w'' = b * sigma^2 / r^3

which is one differentiation of the raw SVI form each time: ``d r / d k = y / r``, and
``d (y / r) / d k = (r - y^2 / r) / r^2 = sigma^2 / r^3`` after substituting ``r^2 - y^2 =
sigma^2``. Three reasons this is written out rather than differenced. It is exact, so a reported
violation of 1e-9 is a real 1e-9 and not the truncation error of a step size nobody tuned. It is
cheaper, which matters because the calibrator evaluates ``g`` on a dense grid inside its soft
penalty at *every* iteration (Design 5.3), not once per snapshot. And a second-order central
difference of ``w`` loses about half its digits to cancellation, which is precisely the regime
-- ``g`` near zero -- where the answer has to be trusted.

numpy is used for the vectorised evaluation and nothing else; jax, torch and scipy are barred
from this layer by rule 3, which is what keeps these rules checkable without a GPU or a JIT.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from volengine.parametric_pricing.domain.svi_slice import SVIParams, SVISlice


def _require_grid(k_grid: NDArray[np.float64], what: str) -> None:
    """Reject a grid that cannot carry a measurement, as a plain ``ValueError``.

    Two failures, both of which would otherwise be reported as a suspiciously clean surface.
    An empty grid makes ``min`` and ``max`` undefined and the honest answer is not ``0.0`` --
    "no violation was found" and "nothing was looked at" are different statements, and only one
    of them should reach an acceptance threshold. A non-finite grid point poisons ``w`` and
    every derivative of it, and a NaN then walks straight through ``max(0, -min(g))`` because
    ``np.min`` of an array containing a NaN is a NaN and ``nan < 0`` is ``False``.

    Emptiness is tested with ``.size``, never with truthiness: numpy raises on the truth value
    of a multi-element array, so ``if not k_grid`` is a crash on every grid but the degenerate
    ones.
    """
    if k_grid.size == 0:
        raise ValueError(f"The {what} must hold at least one point, got an empty array")
    if not np.all(np.isfinite(k_grid)):
        raise ValueError(f"The {what} must be finite at every point")


def _curve_and_derivatives(
    params: SVIParams, k: NDArray[np.float64]
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """``(w, w', w'')`` of the raw SVI curve at every point of ``k``, in closed form.

    The three expressions are derived in the module docstring. **``w`` is not rewritten here**:
    it comes from :meth:`SVIParams.total_variance`, and so from the one copy of the raw SVI form
    the shared kernel holds (ADR-026). Only the two derivatives are written in this module,
    because they are different formulas rather than another spelling of the same one. The cost
    is that ``r`` is computed twice per grid point, which is one square root against the
    optimiser's own arithmetic and is not where this loop's time goes.

    ``r`` is bounded below by ``sigma``, which ``SVIParams`` guarantees strictly positive, so
    neither the division by ``r`` nor the one by ``r^3`` can fail here -- the only division that
    needs a guard in this module is the one by ``w``, and it lives in :func:`durrleman_g` where
    ``w`` is checked.

    Private because the published vocabulary of this module is the three measures below; the
    derivatives are the machinery under them. Its own tests reach in anyway, and say why: a sign
    slip in ``w'`` still produces a perfectly plausible-looking ``g``, so the derivative has to
    be pinned against an independent computation rather than only through the function that
    consumes it.
    """
    y = k - params.m
    r = np.sqrt(y * y + params.sigma * params.sigma)
    w = params.total_variance(k)
    w_prime = params.b * (params.rho + y / r)
    w_second = params.b * params.sigma * params.sigma / (r * r * r)
    return w, w_prime, w_second


def durrleman_g(params: SVIParams, k: NDArray[np.float64]) -> NDArray[np.float64]:
    """Durrleman's function of one slice, evaluated at every point of ``k``.

    ``g(k) >= 0`` everywhere is equivalent to the slice being free of butterfly arbitrage, which
    is to say that the risk-neutral density it implies is non-negative. Where ``g(k) < 0`` the
    fit is pricing a butterfly spread -- a payoff that is never negative in any state of the
    world -- at a negative cost. See the module docstring for the formula and its derivation.

    Args:
        params: The five raw SVI parameters of one expiry. The tenor is already inside ``w``,
            so no maturity is needed and none is accepted.
        k: Log-forward-moneyness values to evaluate at. Any shape; the result has the same one.

    Returns:
        ``g`` at each point, elementwise. Positive is healthy, negative is arbitrage, and the
        magnitude of a negative value is how deep the violation runs at that strike.

    Raises:
        ValueError: If ``k`` is empty or holds a non-finite value; if the total variance is not
            finite and strictly positive at some grid point; or if ``g`` itself comes out
            non-finite.

        The middle case is a real seam, not defensive noise. ``SVIParams`` requires
        ``min_total_variance >= 0``, **not** ``> 0``, so a legal slice can attain ``w = 0``
        exactly -- at the single interior minimum of a curve tuned to touch zero, or everywhere
        at once on the degenerate flat slice ``a = b = 0``. ``g`` divides by ``w`` twice, so at
        such a point it is ``inf`` or ``0/0``. Returning that is the one thing this function
        must not do: an ``inf`` survives ``max(0, -min(g))`` as a clean zero and a NaN survives
        it as a clean zero too, because ``nan < 0`` is ``False``. Either way a broken slice
        would be reported as arbitrage-free, which is the worst possible failure mode for a
        metric whose whole job is to say when something is wrong. Raising is also the honest
        answer mathematically: a slice with a vanishing total variance has no density to be
        non-negative about -- the distribution has collapsed onto a point -- so there is no
        number to return, in the same spirit as ``NoImpliedVolError``.

        The last case catches what an ordering test cannot. A very large ``|k|`` overflows
        ``y * y`` and leaves ``w`` infinite, and an infinite ``w`` passes ``w <= 0`` happily and
        then divides its way to a ``g`` of about 1 -- a clean bill of health computed at a point
        where the total variance is not a number at all. Push further and ``y`` itself overflows
        to infinity, at which point a negative ``rho`` makes the curve evaluate ``-inf + inf``,
        a NaN, which passes the same ordering test for the opposite reason: a NaN compares
        ``False`` against everything. That is the trap this repo keeps walking into, so
        finiteness is asserted with ``np.isfinite`` *before* any sign is tested, and the result
        is checked once more on the way out.
    """
    _require_grid(k, "moneyness grid")

    w, w_prime, w_second = _curve_and_derivatives(params, k)
    # Finiteness first, then the sign, joined with `or`: a NaN sails straight through
    # `w <= 0` and would leave this function as an `inf` that every downstream `max` accepts.
    if not np.all(np.isfinite(w)) or np.any(w <= 0.0):
        raise ValueError(
            "The total variance must be finite and strictly positive at every grid point to "
            f"evaluate Durrleman's function, got a minimum of {np.min(w)}"
        )

    g: NDArray[np.float64] = (
        (1.0 - k * w_prime / (2.0 * w)) ** 2
        - (w_prime * w_prime / 4.0) * (1.0 / w + 0.25)
        + w_second / 2.0
    )
    if not np.all(np.isfinite(g)):
        raise ValueError("Durrleman's function is not finite over the given moneyness grid")
    return g


def butterfly_violation(params: SVIParams, k_grid: NDArray[np.float64]) -> float:
    """Depth of the worst butterfly violation on the grid: ``max(0, -min(g))``.

    **Zero means clean, and larger is worse.** That orientation is what lets one number do two
    jobs without a second convention: it is the reported metric an acceptance rule compares
    against a threshold, and it is the soft penalty term the calibration loss adds to its
    residuals (Design 5.3), where a quantity that is zero on the admissible set and grows with
    the breach is exactly what a penalty has to be. The clamp at zero is not cosmetic -- without
    it a comfortably arbitrage-free slice would report a large *negative* number and the penalty
    would reward the optimiser for pushing ``g`` up long after the constraint stopped binding.

    The value is in the units of ``g`` itself, which are those of a density: it says how
    negative the implied density gets, not how far the parameters are from admissible.

    **This is a grid measure, and it can only see what the grid samples.** ``g`` is smooth, but
    a violation confined to a narrow dip between two grid points is invisible here, and the
    function has no way to tell that case from a genuinely clean slice. Choosing the density is
    therefore the caller's decision and a real one: Design 5.3 asks for a dense moneyness grid
    precisely because this is a sampled proxy for a statement about the whole line. The grid is
    also not clipped to the slice's quoted band; see :func:`calendar_violation` for why the
    caller owns that choice.

    Args:
        params: The five raw SVI parameters of one expiry.
        k_grid: Log-forward-moneyness values to test. Non-empty and finite.

    Returns:
        ``0.0`` if ``g >= 0`` at every sampled point, otherwise the magnitude of the most
        negative value it reaches.

    Raises:
        ValueError: Everything :func:`durrleman_g` raises, on the same terms.
    """
    return max(0.0, -float(np.min(durrleman_g(params, k_grid))))


def calendar_violation(near: SVISlice, far: SVISlice, k_grid: NDArray[np.float64]) -> float:
    """Worst calendar-spread crossing between two expiries: ``max(0, max(w_near - w_far))``.

    Total variance at a fixed ``k`` must be non-decreasing in the tenor, because variance
    accumulates: whatever uncertainty the market has priced by the near expiry is still priced
    by the far one, plus whatever else happens in between. A fit where ``w_near(k) > w_far(k)``
    is claiming the underlying becomes *more* certain by waiting, and it prices a calendar
    spread -- long the far option, short the near one -- at a negative cost.

    **This is ADR-008's accepted debt, made measurable.** Raw SVI fits each expiry
    independently, so nothing during the fit couples the slices and nothing prevents the
    crossing. That was chosen knowingly: the alternative, SSVI, ties every tenor into one global
    parameterisation and reshapes the whole calibration cycle, and it is deferred to its own
    milestone. Accepted does not mean ignored -- this number is computed, reported and kept as
    analysis material. It is **never** corrected: there is no repair step downstream that nudges
    the slices apart, because a silently repaired surface is no longer the surface that was
    fitted and the honest report of what v1 does not guarantee would disappear with it.

    **The two slices' validity bands are deliberately not applied.** ``k_min`` and ``k_max``
    describe where each fit is backed by real quotes, and intersecting them here would look
    tidy and would hide violations in exactly the wing where they live: crossings appear where
    an aggressive extrapolation of one slice runs under the other, which is outside at least one
    band by definition. The bands are published on ``SVISlice`` so the caller can decide what to
    measure over -- the same reason ``total_variance`` does not clamp to them either.

    Args:
        near: The shorter-dated slice.
        far: The longer-dated slice, strictly beyond ``near``.
        k_grid: Log-forward-moneyness values to compare over. Non-empty and finite. A grid
            measure, with the same sampling caveat as :func:`butterfly_violation`.

    Returns:
        ``0.0`` when the far slice sits weakly above the near one everywhere on the grid,
        otherwise the largest amount of total variance by which the near slice exceeds it.

    Raises:
        ValueError: If ``near`` is not strictly shorter-dated than ``far``, if ``k_grid`` is
            empty or non-finite, or if the comparison is not finite. The ordering is checked
            rather than sorted for: swapping the arguments is a caller bug, and quietly
            returning the mirror measure would answer a question nobody asked with a number
            that looks exactly like the right one.
    """
    if not near.tenor_years < far.tenor_years:
        raise ValueError(
            "The near slice must be strictly shorter-dated than the far one, got "
            f"{near.tenor_years} and {far.tenor_years}"
        )
    _require_grid(k_grid, "moneyness grid")

    excess = near.params.total_variance(k_grid) - far.params.total_variance(k_grid)
    if not np.all(np.isfinite(excess)):
        raise ValueError(
            "The total variance of both slices must be finite over the given moneyness grid"
        )
    return max(0.0, float(np.max(excess)))
