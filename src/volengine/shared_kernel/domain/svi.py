"""The raw SVI curve, the closed form itself, owned jointly by every context that writes one.

Raw SVI parameterises the *total variance* of one expiry against log-forward-moneyness::

    w(k) = a + b * (rho * (k - m) + sqrt((k - m)^2 + sigma^2))

with ``a`` the level, ``b`` the wing steepness, ``rho`` the skew, ``m`` the horizontal shift of
the minimum and ``sigma`` the curvature at the bottom -- a *shape* parameter of the hyperbola,
not a volatility, despite the name the literature settled on. Far from ``m`` the square root
becomes ``|k - m|`` and the curve is asymptotically linear, with slope ``b * (1 + rho)`` on the
right and ``b * (rho - 1)`` on the left.

**Why this is in the shared kernel.** ADR-014 admits a module here only if it passes all three of
its tests at once, and this one is the second module to (see ``black76.py`` for the first
statement of them):

1. **It is a fact, not a policy.** Given five numbers and a moneyness there is one right answer,
   independent of who is asking. Nothing here is tunable, so nothing belongs in the TOML of
   ADR-012. *Which* parametrisation this engine fits is very much a policy -- and it is already
   expressed as one, by the ``Calibrator`` port: SSVI or a stochastic-volatility model enters as
   another implementation behind that abstraction, one model in and another out. That is a
   different object arriving, not this expression coming to mean something else.
2. **It carries no vocabulary.** Five floats in, one float out. The value objects stay in their
   contexts and stay different, which is the whole point: ``parametric_pricing``'s ``SVIParams``
   is the *output of a fit* and must admit a collapsed slice, because an optimiser walks through
   them and the violation has to stay measurable, while ``market_data``'s ``SVIParamsSpec`` is a
   handwritten *specification of what to generate* and is strictly stricter. Sharing the
   arithmetic must not become sharing the model.
3. **It is frozen by nature.** It is Gatheral's published form. It changes only if it was written
   down wrong, and then every caller wants the fix.

**What the duplication cost.** The expression had been written three times before this module --
``svi_slice.py``, ``durrleman.py``, and the synthetic generator one context over -- and the
danger was never the typing. Market Data's generator produces the surface that
``parametric_pricing``'s calibrator is measured against, so a sign slip in one copy **cancels**
against a fit performed with the other and passes the known-truth test with the engine wrong.
That is the trap F2-03 was warned about, and two copies of one formula on the two sides of it is
exactly how it springs. ADR-026 records the extraction.

**What is deliberately left outside.** Three things, each for a reason that is not squeamishness:

* **The two value objects**, for admission test 2 above. Their invariants differ and must.
* **The annualisation** ``sqrt(w / T)``. It is one square root over a tenor the caller supplies,
  and what each caller actually owns there is the guard: a non-positive tenor is a construction
  bug whose ``ValueError`` names that context's parameter. Moving it here would relocate a
  message, not a formula.
* **The derivatives and the vectorised evaluation.** Rule 1 confines the shared kernel to the
  standard library, and both want numpy: ``durrleman.py`` differentiates ``w`` twice on a dense
  grid inside the calibrator's penalty, and ``SVIParams.total_variance`` evaluates whole grids of
  ``k`` at once. The array path is the elementwise image of :func:`total_variance` and is pinned
  to it, one point at a time, by a test that reaches across the boundary the way ``tests/`` is
  allowed to.

**Nothing here can fail on a finite input.** There is no division, no logarithm, and the radicand
is a sum of two squares, so :func:`total_variance` needs no guard and takes none -- a caller's
value object has already established that its five parameters are finite and admissible.
:func:`min_total_variance` is the one exception: its square root has a real domain, so ``rho`` is
checked there.
"""

from __future__ import annotations

import math


def total_variance(k: float, a: float, b: float, rho: float, m: float, sigma: float) -> float:
    """``w(k) = a + b * (rho * (k - m) + sqrt((k - m)^2 + sigma^2))``.

    Total variance, not an annualised rate: the tenor of the expiry is already inside ``w``,
    which is why no maturity is an argument here and none is accepted. It is also the coordinate
    the no-arbitrage conditions are stated in -- Durrleman's butterfly condition on one slice,
    monotonicity in ``T`` for the calendar one -- so this is the quantity every consumer of a
    slice actually wants.

    Args:
        k: Log-forward-moneyness, ``ln(K / F)``. Zero is at the money forward and is a perfectly
            ordinary value, so no caller may test it for truthiness.
        a: Vertical level of the curve, in total-variance units.
        b: Wing steepness, half the difference of the two asymptotic slopes. Zero is the
            degenerate flat slice ``w(k) = a``.
        rho: Skew between the wings, strictly inside ``(-1, 1)``.
        m: Where the smile's minimum sits, in log-forward-moneyness.
        sigma: Curvature at the bottom of the smile, strictly positive.

    Returns:
        The total variance at that moneyness. Non-negative exactly when
        :func:`min_total_variance` is, which is the caller's invariant to hold rather than this
        function's to check -- an optimiser is entitled to evaluate a curve it is about to reject.
    """
    y = k - m
    return a + b * (rho * y + math.sqrt(y * y + sigma * sigma))


def min_total_variance(a: float, b: float, rho: float, sigma: float) -> float:
    """Lowest value the curve attains, ``a + b * sigma * sqrt(1 - rho^2)``.

    Closed form rather than a search over ``k``. Differentiating gives
    ``w'(k) = b * (rho + y / r)`` with ``y = k - m`` and ``r = sqrt(y^2 + sigma^2)``, which
    vanishes at ``y = -rho * sigma / sqrt(1 - rho^2)``; substituting that back leaves this
    expression, and ``m`` drops out of it because shifting a curve sideways does not move its
    minimum value.

    It is the quantity a caller's constructor checks, because a curve dipping below zero
    somewhere is claiming a negative variance at that strike -- not an arbitrage to be measured
    and reported, but an object that is not a volatility surface at all.

    Args:
        a: Vertical level of the curve.
        b: Wing steepness, non-negative.
        rho: Skew between the wings.
        sigma: Curvature at the bottom of the smile.

    Returns:
        The minimum of ``w`` over all ``k``.

    Raises:
        ValueError: If ``rho`` is not strictly inside ``(-1, 1)``, which is the one input with a
            domain restriction here: ``1 - rho^2`` would be negative and the square root has no
            real value. Written as a bad-case test rather than as ``isfinite(rho) and ...``
            inverted, so that a NaN -- which compares ``False`` against every bound -- is rejected
            rather than walked through into a ``math domain error`` from inside the arithmetic.
    """
    if not -1.0 < rho < 1.0:
        raise ValueError(f"The SVI parameter rho must be inside (-1, 1), got {rho}")
    return a + b * sigma * math.sqrt(1.0 - rho * rho)
