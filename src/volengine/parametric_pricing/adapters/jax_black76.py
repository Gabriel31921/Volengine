"""Black-76 again, in JAX: batched over strikes and differentiable end to end.

**This is the one remaining copy of the formula, and it is deliberate.** ADR-014 admitted the
closed form to ``shared_kernel/domain/black76.py`` precisely so that it would live once -- and rule
1 of the import table says the shared kernel is stdlib only, so the kernel's version cannot be the
one an optimiser differentiates through. What follows is therefore not a convenience copy but a
genuine reimplementation with two properties the scalar one does not have: it evaluates a whole
chain at once, and ``jax.grad`` can walk through it. The duplication is paid for the same way
every duplication in this repo is: the kernel is this module's **oracle**, and
``tests/parametric_pricing/test_jax_black76.py`` asserts the two agree, deep wing included.

Nothing here is a calibration input. The fit works in ``(k, w)`` space with volatilities that
Market Data already inverted, so the calibrator never prices anything. This module exists for the
greeks of Design 5.8, which are AD over a *price*, and for the wings the SVI surface is asked about
after it is fitted.

**Single precision, on purpose.** JAX's default is ``float32`` and this module does not change it:
enabling ``jax_enable_x64`` is a process-wide mutation performed by importing a module, which is
exactly the kind of action at a distance an optional adapter must not take. What it costs is about
seven significant digits -- three orders of magnitude below the uncertainty of any fit -- and every
tolerance in the tests is written at that scale rather than at ``float64``'s. It does not cost the
deep wing either -- the recurring trap below is about *cancellation*, not about precision, and it
would delete the tail in any precision at all -- though it does end it sooner: a premium below
about 1e-38 of a currency unit underflows, which on a two-week option is a strike some twelve
standard deviations out and worth 1e-32. That is economically nothing and numerically stated
here rather than discovered later.

**The trap this module refuses to walk into.** The normal CDF as ``0.5 * (1 + erf(x / sqrt(2)))``
saturates in the left tail -- ``erf`` reaches exactly ``-1.0`` past about -6, the sum cancels, and
``N(-9)`` comes back as a clean ``0.0`` instead of 1.128e-19. That silently deletes the whole deep
wing of a crypto book, which is the half of the chain the smile says the most about. It is
``0.5 * erfc(-x / sqrt(2))`` here for the same reason it is in the kernel.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy.special import erfc

DTYPE = jnp.float32
"""The precision every array in this adapter is built in. Stated once so that a caller handing in
``float64`` numpy arrays gets a documented downcast rather than a silent one -- and so that the
compiled signature ADR-009 protects cannot change because someone passed a different dtype."""

TINY: float = float(np.finfo(np.float32).tiny)
"""Smallest positive normal in this module's precision, and the floor the total standard deviation
is clamped at. A zero volatility divides by zero and a zero tenor does the same; the limit of the
formula there is the intrinsic value, which is a far more useful answer to hand a differentiator
than a NaN. Taken from numpy rather than from ``jnp.finfo`` so that it is a plain float computed
once at import, not a device call inside a traced function."""

_INV_SQRT_2 = 0.7071067811865476
"""``1 / sqrt(2)``, the argument scaling of the CDF below."""

_INV_SQRT_2PI = 0.3989422804014327
"""``1 / sqrt(2 * pi)``, the leading constant of the normal density."""


def norm_cdf(x: jax.Array) -> jax.Array:
    """Standard normal CDF, elementwise, as ``0.5 * erfc(-x / sqrt(2))``.

    See the module docstring for why it is not the textbook spelling. ``erfc`` is differentiable in
    JAX, so this is the form both the price and its greeks are read from.
    """
    return 0.5 * erfc(-x * _INV_SQRT_2)


def norm_pdf(x: jax.Array) -> jax.Array:
    """Standard normal density, elementwise."""
    return _INV_SQRT_2PI * jnp.exp(-0.5 * x * x)


def price(
    forward: jax.Array,
    strike: jax.Array,
    tenor_years: jax.Array,
    vol: jax.Array,
    is_call: jax.Array,
    discount: jax.Array | float = 1.0,
) -> jax.Array:
    """Black-76 premium per unit, elementwise over whatever shape broadcasts.

    Args:
        forward: Forward price for delivery at expiry. Positive.
        strike: Strike, in the forward's units. Positive.
        tenor_years: Year fraction to expiry under the market's day count. Positive.
        vol: Black-76 implied volatility, annualised as a decimal. Positive.
        is_call: Boolean array, ``True`` for a call. An array rather than a Python ``bool`` so that
            a mixed chain prices in one call and so that the argument does not become a static
            operand that forces its own compilation.
        discount: ``exp(-r * T)``. One is exact on an inverse crypto book, where premium and
            settlement share a numeraire.

    Returns:
        The premium, elementwise, same broadcast shape as the inputs.

    **No validation, and that is the difference from the kernel.** A guard is a Python ``if`` on a
    value that does not exist while the function is being traced, so a check here would either be
    dropped under ``jit`` or force the whole batch through a host round trip. The caller is inside
    this adapter and holds the invariants instead: quotes come out of ``PaddedTask``, whose fill
    values are positive by construction, and greeks are asked about positions Risk has already
    validated. A zero volatility is the one input that would divide by zero, and it is floored
    below rather than rejected -- the ``vol -> 0`` limit of the formula is the intrinsic value, and
    returning it is more useful to a differentiator than a NaN.
    """
    total_stdev = jnp.maximum(vol * jnp.sqrt(tenor_years), TINY)
    d1 = (jnp.log(forward / strike) + 0.5 * total_stdev * total_stdev) / total_stdev
    d2 = d1 - total_stdev

    call = discount * (forward * norm_cdf(d1) - strike * norm_cdf(d2))
    put = discount * (strike * norm_cdf(-d2) - forward * norm_cdf(-d1))
    return jnp.where(is_call, call, put)


def vega(
    forward: jax.Array,
    strike: jax.Array,
    tenor_years: jax.Array,
    vol: jax.Array,
    discount: jax.Array | float = 1.0,
) -> jax.Array:
    """``dPrice / dVol``, elementwise. ``D * F * phi(d1) * sqrt(T)``.

    No side argument: put-call parity says ``C - P = D * (F - K)``, whose right-hand side has no
    volatility in it, so a call and a put on one strike have identical vega. Accepting a side would
    leave a parameter no code path reads.

    Written in closed form rather than as ``jax.grad(price)`` although the two agree to the last
    bit, and the test asserts that they do. The closed form is what a batched weight computation
    wants -- one pass, no tape -- and having both is how the AD path gets an oracle that is not
    itself AD.
    """
    root_t = jnp.sqrt(tenor_years)
    total_stdev = jnp.maximum(vol * root_t, TINY)
    d1 = (jnp.log(forward / strike) + 0.5 * total_stdev * total_stdev) / total_stdev
    return discount * forward * norm_pdf(d1) * root_t
