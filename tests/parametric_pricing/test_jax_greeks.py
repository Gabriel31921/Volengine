"""AD greeks against two oracles: a closed form where one exists, and a bump where none does.

Every number here comes out of ``jax.grad``, ``jax.hessian`` or ``jax.jacfwd``, which is precisely
why it needs checking against something that is not AD. Two oracles do that:

* **A flat slice**, ``b = 0``. The volatility no longer depends on the strike, so the smile term of
  the delta vanishes identically and the answer must be the textbook Black-76 delta and gamma. That
  is an exact comparison against a formula nobody in this repository wrote twice.
* **A central difference of the very same pricer** on a real smile, where no closed form exists.
  Loose by construction -- a second-order difference in single precision is good to a few parts in
  a thousand -- and it is what makes the smile term falsifiable rather than merely plausible.

The pair matters. The flat test alone would pass on an implementation that ignored the smile, and
the bump test alone would pass on one that got the level wrong in the same way twice.
"""

from __future__ import annotations

import math

import pytest

from volengine.parametric_pricing.adapters import jax_greeks
from volengine.parametric_pricing.domain.svi_slice import SVIParams
from volengine.shared_kernel.domain import black76 as kernel

FORWARD = 60_000.0
TENOR = 0.25
STRIKES: tuple[float, ...] = (45_000.0, 60_000.0, 80_000.0)

FLAT = SVIParams(a=0.04, b=0.0, rho=0.0, m=0.0, sigma=0.2)
"""A slice with no smile: ``w(k) = a`` everywhere, so the implied volatility is
``sqrt(0.04 / 0.25) = 0.4`` at every strike and Black-76 is the exact answer."""

FLAT_VOL = math.sqrt(0.04 / TENOR)
"""The volatility ``FLAT`` implies, once, for the oracle to be priced at."""

SMILE = SVIParams(a=0.04, b=0.1, rho=-0.35, m=-0.02, sigma=0.2)
"""A skewed crypto smile: the downside wing lifted, the minimum a little below the forward."""


def bumped_price(params: SVIParams, forward: float, bump: float) -> tuple[float, ...]:
    """The pricer's own answer at a shifted forward, which is what a finite difference needs."""
    return jax_greeks.price(params, forward + bump, STRIKES, TENOR, is_call=True)


def test_the_price_under_a_flat_slice_is_the_black_76_price() -> None:
    """The path from parameters to premium, before any derivative is taken of it. A delta that
    matched an oracle while the price did not would mean the two errors cancelled."""
    ours = jax_greeks.price(FLAT, FORWARD, STRIKES, TENOR, is_call=True)

    for strike, premium in zip(STRIKES, ours, strict=True):
        expected = kernel.price(FORWARD, strike, TENOR, FLAT_VOL, is_call=True)
        assert premium == pytest.approx(expected, rel=1e-5, abs=1e-5 * expected)


def test_the_delta_of_a_flat_slice_is_the_black_76_delta() -> None:
    """With no smile there is nothing for the extra term to pick up, so ``dPrice/dF`` collapses to
    ``N(d1)``. This is the exact oracle for the whole AD path."""
    ours = jax_greeks.delta(FLAT, FORWARD, STRIKES, TENOR, is_call=True)

    for strike, value in zip(STRIKES, ours, strict=True):
        total_stdev = FLAT_VOL * math.sqrt(TENOR)
        d1 = (math.log(FORWARD / strike) + 0.5 * total_stdev**2) / total_stdev
        assert value == pytest.approx(kernel.norm_cdf(d1), rel=1e-4)


def test_the_gamma_of_a_flat_slice_is_the_black_76_gamma() -> None:
    """``phi(d1) / (F * sigma * sqrt(T))``, from ``jax.hessian`` of the same pricer."""
    ours = jax_greeks.gamma(FLAT, FORWARD, STRIKES, TENOR, is_call=True)

    for strike, value in zip(STRIKES, ours, strict=True):
        total_stdev = FLAT_VOL * math.sqrt(TENOR)
        d1 = (math.log(FORWARD / strike) + 0.5 * total_stdev**2) / total_stdev
        expected = kernel.norm_pdf(d1) / (FORWARD * total_stdev)
        assert value == pytest.approx(expected, rel=1e-3, abs=1e-3 * expected)


def test_the_delta_of_a_skewed_smile_is_not_the_black_76_delta() -> None:
    """**What AD is here for.** A strike is fixed in currency, so a moving forward slides the
    option along the smile and the volatility moves with it. On a downward-skewed slice that term
    is worth several delta points at the money -- the difference between a hedge that holds and one
    that bleeds, and a term nobody has to derive by hand."""
    smile_delta = jax_greeks.delta(SMILE, FORWARD, STRIKES, TENOR, is_call=True)

    at_the_money = smile_delta[1]
    vol = SMILE.implied_vol(0.0, TENOR)
    total_stdev = vol * math.sqrt(TENOR)
    black = kernel.norm_cdf(0.5 * total_stdev)

    assert abs(at_the_money - black) > 0.01


def test_the_delta_of_a_skewed_smile_matches_a_bump_of_the_same_pricer() -> None:
    """The oracle for the case with no closed form: a central difference of the published pricer.

    Loose on purpose. A first-order central difference in single precision, with a bump of a
    hundredth of the forward, is good to a few parts in a thousand -- and a tolerance any tighter
    would be a test of the bump size rather than of the derivative.
    """
    bump = FORWARD * 0.01
    up = bumped_price(SMILE, FORWARD, bump)
    down = bumped_price(SMILE, FORWARD, -bump)

    ours = jax_greeks.delta(SMILE, FORWARD, STRIKES, TENOR, is_call=True)

    for value, high, low in zip(ours, up, down, strict=True):
        assert value == pytest.approx((high - low) / (2 * bump), rel=5e-3)


def test_the_gamma_of_a_skewed_smile_matches_a_second_difference() -> None:
    """The second derivative, checked the same way. AD needs no bump size and suffers none of the
    cancellation that makes a differenced gamma noisy in the wings."""
    bump = FORWARD * 0.02
    up = bumped_price(SMILE, FORWARD, bump)
    middle = bumped_price(SMILE, FORWARD, 0.0)
    down = bumped_price(SMILE, FORWARD, -bump)

    ours = jax_greeks.gamma(SMILE, FORWARD, STRIKES, TENOR, is_call=True)

    for value, high, mid, low in zip(ours, up, middle, down, strict=True):
        assert value == pytest.approx((high - 2 * mid + low) / (bump * bump), rel=2e-2)


def test_a_put_and_a_call_on_one_strike_differ_in_delta_by_exactly_one() -> None:
    """Put-call parity, differentiated: ``C - P = D * (F - K)`` gives ``dC/dF - dP/dF = D``. It
    holds for the smile delta too, because the volatility term is identical on both sides -- which
    is the cheapest possible check that the extra term was added to both."""
    calls = jax_greeks.delta(SMILE, FORWARD, STRIKES, TENOR, is_call=True)
    puts = jax_greeks.delta(SMILE, FORWARD, STRIKES, TENOR, is_call=False)

    for call, put in zip(calls, puts, strict=True):
        assert call - put == pytest.approx(1.0, abs=1e-3)


def test_the_parameter_sensitivity_has_one_row_per_strike_and_one_column_per_parameter() -> None:
    """The shape *is* the interface: rows are strikes, columns are ``(a, b, rho, m, sigma)`` in
    ``SVIParams``'s own order, and a transposed Jacobian would still be full of plausible
    numbers."""
    surface = jax_greeks.parameter_sensitivity(SMILE, FORWARD, STRIKES, TENOR, is_call=True)

    assert len(surface) == len(STRIKES)
    assert all(len(row) == 5 for row in surface)
    assert all(math.isfinite(cell) for row in surface for cell in row)


def test_the_sensitivity_to_the_level_matches_a_bump_of_that_parameter() -> None:
    """One column against a finite difference in the parameter itself: the answer to "what is this
    fitted parameter being wrong worth", in currency per unit of ``a``."""
    bump = 1e-4
    up = jax_greeks.price(
        SVIParams(a=SMILE.a + bump, b=SMILE.b, rho=SMILE.rho, m=SMILE.m, sigma=SMILE.sigma),
        FORWARD,
        STRIKES,
        TENOR,
        is_call=True,
    )
    down = jax_greeks.price(
        SVIParams(a=SMILE.a - bump, b=SMILE.b, rho=SMILE.rho, m=SMILE.m, sigma=SMILE.sigma),
        FORWARD,
        STRIKES,
        TENOR,
        is_call=True,
    )

    surface = jax_greeks.parameter_sensitivity(SMILE, FORWARD, STRIKES, TENOR, is_call=True)

    for row, high, low in zip(surface, up, down, strict=True):
        assert row[0] == pytest.approx((high - low) / (2 * bump), rel=5e-3)


def test_a_flat_slice_has_no_sensitivity_to_the_skew_or_to_the_smile_position() -> None:
    """``b = 0`` erases the wings, and with them every parameter that only shapes them: ``rho``,
    ``m`` and ``sigma`` multiply ``b`` in the curve, so their columns must be exactly zero. An
    implementation that had differentiated a *different* curve would have to be wrong here."""
    surface = jax_greeks.parameter_sensitivity(FLAT, FORWARD, STRIKES, TENOR, is_call=True)

    for row in surface:
        assert row[2] == pytest.approx(0.0, abs=1e-6)
        assert row[3] == pytest.approx(0.0, abs=1e-6)
        assert row[4] == pytest.approx(0.0, abs=1e-6)
