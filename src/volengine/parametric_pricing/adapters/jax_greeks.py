"""Greeks by automatic differentiation, on demand, **outside the contract** (Design 5.8).

A fitted slice is a pricing function: give it a forward, a strike and a tenor and it answers with a
premium, through the smile it just learned. Every sensitivity anyone wants is a derivative of that
function, and this module takes them literally -- ``grad`` for delta, ``hessian`` for gamma,
``jacfwd`` for the sensitivity of a whole wing to the five parameters. Nothing here is a formula
anybody had to derive; the derivative of the code *is* the derivative of the model, which is the
argument for having a differentiable pricer at all.

**They do not travel in the contract, and that is ADR-001.** ``CalibratedSurface`` carries a grid
and no behaviour: no ``implied_vol(K, T)`` method, no greeks. A consumer that wants a delta computes
it with its own conventions -- Risk already does, by bumping (Design 7.4) -- and this module is for
the analyst asking the *model* what it thinks, not for the pipeline. Nothing in the engine imports
it, and that is the correct amount.

**The delta AD gives you is not the Black-76 delta**, and the difference is the whole point. A
strike is fixed in currency, so moving the forward moves the log-moneyness the slice is evaluated
at, which moves the volatility, which moves the price again:

    dPrice/dF = BS delta + vega * dSigma/dk * dk/dF,   with dk/dF = -1/F

That second term is the smile delta -- the reason a book hedged on the sticky-strike Black delta
drifts against a skewed market. Writing it by hand means differentiating raw SVI through a square
root and a chain rule; asking ``jax.grad`` for it means writing nothing. On a flat slice (``b = 0``)
the term vanishes identically, which is the oracle the test checks the whole path against.

**Forward mode for the parameter sensitivities**, and the choice is not incidental. ``jacfwd``
builds a Jacobian column by column, one tangent per *input*; ``jacrev`` builds it row by row, one
cotangent per *output*. A sensitivity surface has five inputs and as many outputs as there are
strikes, so forward mode costs five passes where reverse mode costs one per strike. It is the one
place in this engine where forward mode is the right tool, which is exactly why Design 5.8 names it.
"""

from __future__ import annotations

from collections.abc import Sequence

import jax
import jax.numpy as jnp

from volengine.parametric_pricing.adapters import jax_black76
from volengine.parametric_pricing.adapters.jax_black76 import DTYPE
from volengine.parametric_pricing.domain.svi_slice import SVIParams


def _theta(params: SVIParams) -> jax.Array:
    """The five raw parameters as one differentiable vector, in ``SVIParams``'s own order.

    A vector rather than five arguments because it is the object ``jacfwd`` differentiates against:
    one tangent per component, and the Jacobian's columns come back in the order written here.
    """
    return jnp.asarray([params.a, params.b, params.rho, params.m, params.sigma], dtype=DTYPE)


def _price(
    forward: jax.Array,
    strike: jax.Array,
    tenor_years: jax.Array,
    theta: jax.Array,
    is_call: jax.Array,
    discount: jax.Array,
) -> jax.Array:
    """The premium of one option under one SVI slice. **The function every greek differentiates.**

    Three steps, and the middle one is what makes the derivatives interesting: the log-moneyness is
    computed *from* the forward, so the volatility this option is priced at depends on the forward
    as well as on the strike. Differentiating through that dependency is what produces a smile
    delta rather than a Black-76 one.

    The total variance is floored before the square root for the reason every other module here
    floors it: a legal slice may touch zero, and ``d sqrt(w) / dw`` is infinite there, so an
    unguarded root turns a degenerate curve into a NaN that spreads to every output.
    """
    a, b, rho, m, sigma = theta[0], theta[1], theta[2], theta[3], theta[4]
    y = jnp.log(strike / forward) - m
    w = a + b * (rho * y + jnp.sqrt(y * y + sigma * sigma))
    vol = jnp.sqrt(jnp.maximum(w, jax_black76.TINY) / tenor_years)
    return jax_black76.price(forward, strike, tenor_years, vol, is_call, discount)


def _grid(
    params: SVIParams,
    forward: float,
    strikes: Sequence[float],
    tenor_years: float,
    is_call: bool,
    discount: float,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    """Everything a greek needs, as device scalars and one strike vector. Validated by the caller.

    Deliberately no guards. The scalar Black-76 in the shared kernel validates because it is called
    with market data straight off a feed; this module is called by an analyst who already holds a
    fitted ``SVIParams`` and a position, and a Python ``if`` on a traced value is no check at all.
    """
    return (
        jnp.asarray(forward, dtype=DTYPE),
        jnp.asarray(strikes, dtype=DTYPE),
        jnp.asarray(tenor_years, dtype=DTYPE),
        _theta(params),
        jnp.asarray(is_call),
        jnp.asarray(discount, dtype=DTYPE),
    )


def price(
    params: SVIParams,
    forward: float,
    strikes: Sequence[float],
    tenor_years: float,
    is_call: bool,
    discount: float = 1.0,
) -> tuple[float, ...]:
    """Premium per unit at each strike, under this slice. The function the greeks below derive.

    Published beside them so that a caller can check a sensitivity by bumping the very same pricer
    the derivative was taken of -- which is what the finite-difference tests do, and what makes an
    AD greek falsifiable rather than merely plausible.
    """
    f, k, t, theta, call, d = _grid(params, forward, strikes, tenor_years, is_call, discount)
    return tuple(
        float(one)
        for one in jax.vmap(_price, in_axes=(None, 0, None, None, None, None))(
            f, k, t, theta, call, d
        )
    )


def delta(
    params: SVIParams,
    forward: float,
    strikes: Sequence[float],
    tenor_years: float,
    is_call: bool,
    discount: float = 1.0,
) -> tuple[float, ...]:
    """``dPrice / dForward`` at each strike, **including the smile's own movement**.

    ``jax.grad`` of :func:`_price` with respect to the forward, vectorised over strikes. See the
    module docstring for why this is not ``N(d1)``: the strike is fixed in currency, so a moving
    forward slides the option along the smile, and the total derivative picks up the extra
    ``vega * dSigma/dk * dk/dF`` term that a sticky-strike Black delta leaves out.
    """
    f, k, t, theta, call, d = _grid(params, forward, strikes, tenor_years, is_call, discount)
    gradient = jax.grad(_price, argnums=0)
    return tuple(
        float(one)
        for one in jax.vmap(gradient, in_axes=(None, 0, None, None, None, None))(
            f, k, t, theta, call, d
        )
    )


def gamma(
    params: SVIParams,
    forward: float,
    strikes: Sequence[float],
    tenor_years: float,
    is_call: bool,
    discount: float = 1.0,
) -> tuple[float, ...]:
    """``d2Price / dForward2`` at each strike, by ``jax.hessian`` of the same pricer.

    The second derivative of a function AD already differentiates once is free to ask for, and it
    is exact -- no bump size to choose, and none of the cancellation that makes a finite-difference
    gamma noisy in the wings, where the premium being differenced is a fraction of a tick.
    """
    f, k, t, theta, call, d = _grid(params, forward, strikes, tenor_years, is_call, discount)
    second = jax.hessian(_price, argnums=0)
    return tuple(
        float(one)
        for one in jax.vmap(second, in_axes=(None, 0, None, None, None, None))(
            f, k, t, theta, call, d
        )
    )


def parameter_sensitivity(
    params: SVIParams,
    forward: float,
    strikes: Sequence[float],
    tenor_years: float,
    is_call: bool,
    discount: float = 1.0,
) -> tuple[tuple[float, ...], ...]:
    """How each strike's premium moves with each of the five SVI parameters. **The full surface.**

    ``jacfwd`` with respect to ``theta``, so the result is one row per strike and one column per
    parameter, in ``SVIParams``'s own order: ``(a, b, rho, m, sigma)``.

    What it is for: a calibration reports parameters, and the question a risk desk actually asks is
    what a parameter being wrong is worth. This is that number, in currency per unit of parameter,
    strike by strike -- and it is also the sensitivity of the book to the *fit* rather than to the
    market, which is the quantity nobody computes because nobody wants to derive it by hand.

    Forward mode because there are five inputs and many outputs; reverse mode would cost one pass
    per strike (module docstring).
    """
    f, k, t, theta, call, d = _grid(params, forward, strikes, tenor_years, is_call, discount)
    jacobian = jax.jacfwd(_price, argnums=3)
    rows = jax.vmap(jacobian, in_axes=(None, 0, None, None, None, None))(f, k, t, theta, call, d)
    return tuple(tuple(float(cell) for cell in row) for row in rows)
