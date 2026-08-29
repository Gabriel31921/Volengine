"""Tests of the shared kernel's own surface: what only it exposes, and what only it promises.

The four contexts that wrap this module carry the bulk of the behavioural tests already --
``tests/parametric_pricing/test_black76.py`` alone is 400-odd lines of smile, wing and underflow
cases -- and duplicating them here would be re-testing one implementation twice. What is tested
here is what those cannot see: that the kernel speaks in bools rather than any context's enum, that
it raises an exception belonging to no context, and that it stays admissible under rule 1.
"""

from __future__ import annotations

import ast
import math
from pathlib import Path

import pytest

from volengine.shared_kernel.domain import black76
from volengine.shared_kernel.domain.black76 import (
    PriceNotInvertibleError,
    ceiling,
    implied_vol,
    intrinsic,
    norm_cdf,
    price,
    vega,
)

FORWARD = 60_000.0
STRIKE = 60_000.0
TENOR = 0.25
VOL = 0.65

ACROSS_THE_SMILE = [30_000.0, 45_000.0, 60_000.0, 80_000.0, 120_000.0]


# --- rule 1: the shared kernel imports the standard library and nothing else


def test_the_kernel_imports_no_project_module() -> None:
    """The admission price of the Shared Kernel is joint ownership, and this is what pays it.

    A module every context depends on must not depend on any of them, or the exception to "no
    context imports another" becomes a route between two contexts with an extra hop. Rule 1 states
    it for the whole of ``shared_kernel/``; asserted here because this is the first module in it
    with a reason to reach for numpy, and because the temptation to vectorise will be real in F3.

    Read from the source rather than from ``sys.modules``: an import that is never executed still
    couples the file, and a test that only watches what ran would miss it.
    """
    source = Path(black76.__file__).read_text(encoding="utf-8")
    imported: list[str] = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            imported.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            imported.append(node.module)

    assert imported, "the AST walk found no imports at all, so it is not reading the module"
    forbidden = [name for name in imported if name.split(".")[0] not in {"__future__", "math"}]
    assert forbidden == []


# --- the side arrives as a bool, never as a context's vocabulary


def test_the_call_and_put_branches_satisfy_put_call_parity() -> None:
    """``is_call`` really selects the two sides, and it selects the *right* two.

    Parity is the check rather than two hard-coded premiums because it is falsifiable by exactly
    the mistakes a boolean flag invites -- a swapped branch, or a put priced with the call's ``d``
    signs -- and it holds at every strike, so one identity covers the whole smile.
    """
    for strike in ACROSS_THE_SMILE:
        call = price(FORWARD, strike, TENOR, VOL, True)
        put = price(FORWARD, strike, TENOR, VOL, False)
        assert call - put == pytest.approx(FORWARD - strike, rel=1e-12)


def test_the_two_sides_are_distinct_away_from_the_money() -> None:
    """Guards the test above from passing on an implementation that ignores the flag.

    At the money a call and a put are worth the same, so parity alone would still hold if
    ``is_call`` were dropped on the floor and both branches returned the call.
    """
    assert price(FORWARD, 45_000.0, TENOR, VOL, True) != price(FORWARD, 45_000.0, TENOR, VOL, False)


def test_the_price_sits_strictly_between_the_intrinsic_value_and_the_ceiling() -> None:
    """The two bounds the inversion brackets with, checked against the function they bound."""
    for strike in ACROSS_THE_SMILE:
        for is_call in (True, False):
            premium = price(FORWARD, strike, TENOR, VOL, is_call)
            assert intrinsic(FORWARD, strike, is_call, 1.0) < premium
            assert premium < ceiling(FORWARD, strike, is_call, 1.0)


# --- the exception belongs to no context


def test_a_price_below_intrinsic_raises_the_kernel_error() -> None:
    """It must not be any context's ``NoImpliedVolError``.

    The kernel has no context to borrow an error hierarchy from, and reaching into one would be
    the import rule 1 forbids. Each wrapper translates this into its own, which is what keeps a
    caller's ``except CalibrationError`` around a slice still catching the one genuine market
    outcome.
    """
    floor = intrinsic(FORWARD, 45_000.0, False, 1.0)
    with pytest.raises(PriceNotInvertibleError, match="intrinsic value"):
        implied_vol(floor, FORWARD, 45_000.0, TENOR, False)


def test_a_price_above_the_ceiling_raises_the_kernel_error() -> None:
    with pytest.raises(PriceNotInvertibleError, match="ceiling"):
        implied_vol(FORWARD * 1.01, FORWARD, STRIKE, TENOR, True)


def test_the_kernel_error_is_not_a_value_error() -> None:
    """A market outcome and a construction bug must stay separable by ``except``.

    ``PriceNotInvertibleError`` inheriting from ``ValueError`` would let a caller dropping
    uninvertible quotes in a loop swallow a NaN forward one quote at a time.
    """
    assert not issubclass(PriceNotInvertibleError, ValueError)


def test_a_non_finite_target_is_a_value_error_not_a_market_outcome() -> None:
    with pytest.raises(ValueError, match="target price"):
        implied_vol(float("nan"), FORWARD, STRIKE, TENOR, True)


# --- the numerics the whole engine rests on


def test_the_normal_cdf_survives_the_left_tail() -> None:
    """``0.5 * erfc(-x / sqrt(2))``, not ``0.5 * (1 + erf(x / sqrt(2)))``.

    The naive form returns exactly ``0.0`` here. This is the trap that reappeared once per copy of
    this formula, which is the concrete reason the module exists at all.
    """
    assert norm_cdf(-9.0) == pytest.approx(1.128588e-19, rel=1e-5)
    assert norm_cdf(-20.0) > 0.0


def test_the_naive_normal_cdf_really_would_have_returned_zero() -> None:
    """Guards the test above from being vacuous."""
    assert 0.5 * (1.0 + math.erf(-9.0 / math.sqrt(2.0))) == 0.0


def test_an_inverted_price_reproduces_the_volatility_that_made_it() -> None:
    """The round trip, on the out-of-the-money leg the ACLs actually invert."""
    for strike in ACROSS_THE_SMILE:
        is_call = strike >= FORWARD
        premium = price(FORWARD, strike, TENOR, VOL, is_call)
        assert implied_vol(premium, FORWARD, strike, TENOR, is_call) == pytest.approx(VOL, abs=1e-7)


def test_vega_needs_no_side_because_a_call_and_a_put_share_it() -> None:
    """The signature omits the side; this is the identity that entitles it to.

    Differentiating ``C - P = D * (F - K)`` in vol gives zero, so the two vegas are equal. Checked
    numerically against both branches rather than asserted, so a wrong ``d1`` cannot hide.
    """
    bump = 1e-6
    for strike in ACROSS_THE_SMILE:
        analytic = vega(FORWARD, strike, TENOR, VOL)
        for is_call in (True, False):
            up = price(FORWARD, strike, TENOR, VOL + bump, is_call)
            down = price(FORWARD, strike, TENOR, VOL - bump, is_call)
            assert (up - down) / (2 * bump) == pytest.approx(analytic, rel=1e-6)
