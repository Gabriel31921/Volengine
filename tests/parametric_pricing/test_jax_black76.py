"""The batched Black-76 against the shared kernel, which is its oracle.

This adapter holds the one deliberate second copy of the formula (ADR-014 admitted the closed form
to the kernel; rule 1 keeps JAX out of it). A duplication that is allowed to drift is not a
duplication, it is a fork, so every claim below is the same claim: **the two agree**, at the money
and nine standard deviations out.

Tolerances are written at ``float32``'s scale and never at ``float64``'s, because that is the
precision this module runs in. Every deep-wing assertion carries an explicit ``abs=`` at the scale
of the value being checked: ``pytest.approx`` passes on ``rel`` *or* ``abs`` and its default ``abs``
is 1e-12, so a bare relative bound on a 1e-19 probability accepts a clean zero -- which is exactly
the failure the ``erfc`` form exists to prevent.
"""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import pytest

from volengine.parametric_pricing.adapters import jax_black76
from volengine.shared_kernel.domain import black76 as kernel

FORWARD = 60_000.0
"""A BTC forward, and exactly representable in single precision."""

TENOR = 0.25
"""A three-month expiry."""

VOL = 0.65
"""Sixty-five vol: an ordinary crypto at-the-money level."""

STRIKES: tuple[float, ...] = (20_000.0, 45_000.0, 60_000.0, 80_000.0, 150_000.0)
"""A chain from the deep downside wing to the deep upside one."""


def test_the_price_agrees_with_the_shared_kernel_across_the_chain() -> None:
    """The oracle test the module's whole existence rests on."""
    ours = jax_black76.price(
        jnp.asarray(FORWARD),
        jnp.asarray(STRIKES),
        jnp.asarray(TENOR),
        jnp.asarray(VOL),
        jnp.asarray(True),
    )

    for strike, price in zip(STRIKES, ours, strict=True):
        expected = kernel.price(FORWARD, strike, TENOR, VOL, is_call=True)
        assert float(price) == pytest.approx(expected, rel=1e-5, abs=1e-5 * expected)


def test_the_put_agrees_with_the_shared_kernel_too() -> None:
    """Both branches of the ``where``, because a side chosen by an array is a place a
    transposition hides: the value would still be a plausible premium."""
    ours = jax_black76.price(
        jnp.asarray(FORWARD),
        jnp.asarray(STRIKES),
        jnp.asarray(TENOR),
        jnp.asarray(VOL),
        jnp.asarray(False),
    )

    for strike, price in zip(STRIKES, ours, strict=True):
        expected = kernel.price(FORWARD, strike, TENOR, VOL, is_call=False)
        assert float(price) == pytest.approx(expected, rel=1e-5, abs=1e-5 * expected)


def test_a_mixed_chain_prices_both_sides_in_one_call() -> None:
    """``is_call`` is an array rather than a Python bool so that a real book -- calls above the
    forward, puts below -- costs one call and one compilation."""
    sides = jnp.asarray([False, False, True, True, True])

    ours = jax_black76.price(
        jnp.asarray(FORWARD),
        jnp.asarray(STRIKES),
        jnp.asarray(TENOR),
        jnp.asarray(VOL),
        sides,
    )

    for strike, side, price in zip(STRIKES, sides, ours, strict=True):
        expected = kernel.price(FORWARD, strike, TENOR, VOL, is_call=bool(side))
        assert float(price) == pytest.approx(expected, rel=1e-5, abs=1e-5 * expected)


def test_the_normal_cdf_survives_the_deep_left_tail() -> None:
    """**The trap.** ``0.5 * (1 + erf(x / sqrt(2)))`` saturates past about -6 and returns exactly
    zero, which deletes the whole deep wing of a crypto book. The ``erfc`` form does not.

    The ``abs`` is explicit and at the scale of the value: without it ``approx`` would compare
    against its 1e-12 default and accept the zero this test exists to reject.
    """
    ours = float(jax_black76.norm_cdf(jnp.asarray(-9.0)))

    assert ours > 0.0
    assert ours == pytest.approx(1.1285884059538422e-19, rel=1e-4, abs=1e-23)


def test_the_naive_cdf_really_would_have_returned_zero_there() -> None:
    """The guard on the test above: without it, an assertion that a number is positive proves
    nothing about the formula that produced it."""
    naive = float(0.5 * (1.0 + jax.scipy.special.erf(jnp.asarray(-9.0) / math.sqrt(2.0))))

    assert naive == 0.0


def test_a_far_out_of_the_money_wing_is_priced_rather_than_zeroed() -> None:
    """What the tail actually costs downstream: a zero premium is uninvertible in Pricing and
    stays zero under a bump in Risk, so an unhedged wing reports as no risk at all.

    Nine standard deviations out on a one-week option, where the premium is 4e-17 of a currency
    unit -- comfortably inside single precision, and comfortably past where the naive normal CDF
    has already collapsed to zero.
    """
    strike = 100_000.0

    ours = float(
        jax_black76.price(
            jnp.asarray(FORWARD),
            jnp.asarray(strike),
            jnp.asarray(0.02),
            jnp.asarray(0.40),
            jnp.asarray(True),
        )
    )
    expected = kernel.price(FORWARD, strike, 0.02, 0.40, is_call=True)

    assert ours > 0.0
    assert ours == pytest.approx(expected, rel=1e-3, abs=1e-3 * expected)


def test_the_naive_cdf_would_already_have_deleted_that_wing() -> None:
    """The guard on the test above. Nine standard deviations is not an exotic corner: it is a
    perfectly ordinary short-dated crypto strike, and the textbook spelling of the CDF prices it
    at exactly nothing."""
    total_stdev = 0.40 * math.sqrt(0.02)
    d1 = (math.log(FORWARD / 100_000.0) + 0.5 * total_stdev**2) / total_stdev
    naive = float(0.5 * (1.0 + jax.scipy.special.erf(jnp.asarray(d1) / math.sqrt(2.0))))

    assert naive == 0.0


def test_the_vega_agrees_with_the_shared_kernel() -> None:
    ours = jax_black76.vega(
        jnp.asarray(FORWARD), jnp.asarray(STRIKES), jnp.asarray(TENOR), jnp.asarray(VOL)
    )

    for strike, value in zip(STRIKES, ours, strict=True):
        expected = kernel.vega(FORWARD, strike, TENOR, VOL)
        assert float(value) == pytest.approx(expected, rel=1e-4, abs=1e-4 * expected)


def test_the_closed_form_vega_is_the_derivative_of_the_price() -> None:
    """The one internal consistency check that makes the AD path trustworthy: the hand-written
    vega and ``jax.grad`` of the price are the same number, so a mistake in either shows up here
    rather than in a greek nobody has an oracle for."""
    differentiated = jax.grad(jax_black76.price, argnums=3)

    for strike in STRIKES:
        by_ad = float(
            differentiated(
                jnp.asarray(FORWARD),
                jnp.asarray(strike),
                jnp.asarray(TENOR),
                jnp.asarray(VOL),
                jnp.asarray(True),
                1.0,
            )
        )
        closed = float(
            jax_black76.vega(
                jnp.asarray(FORWARD), jnp.asarray(strike), jnp.asarray(TENOR), jnp.asarray(VOL)
            )
        )
        assert by_ad == pytest.approx(closed, rel=1e-4, abs=1e-4 * abs(closed))


def test_a_discount_factor_scales_the_premium() -> None:
    """The default of one is exact on an inverse crypto book and is not an assumption anyone
    should have to discover: a rate that was silently ignored would misprice every long-dated
    position by the carry."""
    plain = jax_black76.price(
        jnp.asarray(FORWARD),
        jnp.asarray(STRIKES),
        jnp.asarray(TENOR),
        jnp.asarray(VOL),
        jnp.asarray(True),
    )

    discounted = jax_black76.price(
        jnp.asarray(FORWARD),
        jnp.asarray(STRIKES),
        jnp.asarray(TENOR),
        jnp.asarray(VOL),
        jnp.asarray(True),
        jnp.asarray(0.9),
    )

    for one, other in zip(plain, discounted, strict=True):
        assert float(other) == pytest.approx(0.9 * float(one), rel=1e-5)


def test_a_vanishing_volatility_gives_the_intrinsic_value_rather_than_a_nan() -> None:
    """The limit, not an error. A traced function has nothing to raise to, and a NaN here would
    not stop a fit -- it would spread through the batch and let it finish on garbage."""
    ours = float(
        jax_black76.price(
            jnp.asarray(FORWARD),
            jnp.asarray(50_000.0),
            jnp.asarray(TENOR),
            jnp.asarray(0.0),
            jnp.asarray(True),
        )
    )

    assert ours == pytest.approx(10_000.0, rel=1e-5)
