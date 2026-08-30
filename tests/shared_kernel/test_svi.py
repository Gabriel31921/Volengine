"""The raw SVI curve as the kernel owns it: the shape it promises, and the price of living here.

The two contexts that hold five SVI numbers already test the curve through their own value
objects -- ``tests/parametric_pricing/test_svi_slice.py`` for the fitted one,
``tests/market_data/test_synthetic_provider.py`` for the generated one -- so what is tested here
is what neither of them can see: that the closed form is right on its own terms, that it speaks
in floats rather than in either context's type, and that it stays admissible under rule 1.

The wings are checked numerically rather than by re-deriving them, because the asymptotic slopes
``b * (1 + rho)`` and ``b * (rho - 1)`` are exactly what a sign slip in ``rho`` corrupts, and a
sign slip here would now corrupt the generator and the calibrator fitted to it *identically* --
which is the one failure this module's existence makes structurally impossible to notice from
either side (ADR-026).
"""

from __future__ import annotations

import ast
import math
from pathlib import Path
from typing import get_type_hints

import pytest

from volengine.shared_kernel.domain import svi
from volengine.shared_kernel.domain.svi import min_total_variance, total_variance

A = 0.02
B = 0.05
RHO = -0.30
M = -0.01
SIGMA = 0.20

ACROSS_THE_SMILE = [-0.6, -0.2, 0.0, 0.15, 0.5]

FAR_LEFT = -1000.0
FAR_RIGHT = 1000.0
"""Log-moneyness values where the square root is indistinguishable from ``|k - m|``.

Far enough that the asymptotic slope is reached to eight digits, close enough that the total
variance itself stays in a range where a difference of two of them keeps its precision.
"""


def w(k: float) -> float:
    """The default slice, evaluated at one moneyness."""
    return total_variance(k, a=A, b=B, rho=RHO, m=M, sigma=SIGMA)


# --- rule 1: the shared kernel imports the standard library and nothing else


def test_the_curve_imports_no_project_module_and_no_array_library() -> None:
    """The admission price of the Shared Kernel, paid again by its second member.

    Read from the source rather than from ``sys.modules``: an import that is never executed
    still couples the file, and a test that only watches what ran would miss it. The temptation
    this module specifically invites is ``import numpy`` to serve ``SVIParams``'s array branch
    in one place -- which rule 1 forbids, and which is why that branch stays where it is.
    """
    source = Path(svi.__file__).read_text(encoding="utf-8")
    imported: list[str] = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            imported.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            imported.append(node.module)

    assert imported, "the AST walk found no imports at all, so it is not reading the module"
    forbidden = [name for name in imported if name.split(".")[0] not in {"__future__", "math"}]
    assert forbidden == []


def test_the_curve_takes_five_floats_and_never_a_value_object() -> None:
    """ADR-014's second admission test, made structural: primitives in, primitives out.

    The obvious "simplification" of these signatures is to take the five parameters as one
    object -- and there is no object it could take. ``SVIParams`` and ``SVIParamsSpec`` live in
    two contexts that must not import each other, and promoting either into the kernel would
    make one context's invariants binding on the other's, which is precisely what a shared
    kernel must not do. Sharing the arithmetic must not become sharing the model.
    """
    for function in (total_variance, min_total_variance):
        hints = get_type_hints(function)
        assert set(hints.values()) == {float}, function.__name__


# --- the shape of the curve


def test_a_slice_with_no_wings_is_flat_across_every_strike() -> None:
    """``b = 0`` is the degenerate slice ``w(k) = a``, and it is legal."""
    for k in ACROSS_THE_SMILE:
        assert total_variance(k, a=0.04, b=0.0, rho=0.0, m=0.0, sigma=0.2) == pytest.approx(0.04)


def test_the_right_wing_approaches_the_slope_b_times_one_plus_rho() -> None:
    """Far from ``m`` the square root becomes ``k - m`` and the curve is a straight line."""
    slope = w(FAR_RIGHT + 1.0) - w(FAR_RIGHT)
    assert slope == pytest.approx(B * (1.0 + RHO), rel=1e-6)


def test_the_left_wing_approaches_the_slope_b_times_rho_minus_one() -> None:
    slope = w(FAR_LEFT + 1.0) - w(FAR_LEFT)
    assert slope == pytest.approx(B * (RHO - 1.0), rel=1e-6)


def test_the_two_wings_have_different_slopes_when_rho_is_not_zero() -> None:
    """Guards the two tests above from passing on a curve that ignored ``rho``.

    A skew of zero makes the wings symmetric, so both assertions would still hold with the
    parameter dropped on the floor -- and ``rho`` is the one parameter whose sign error the
    known-truth test cannot catch, because the same slip would sit on both sides of it.

    Asserted on the curve rather than on the two target values, which would only restate
    arithmetic: it is the evaluated slopes that have to come out asymmetric.
    """
    right = w(FAR_RIGHT + 1.0) - w(FAR_RIGHT)
    left = w(FAR_LEFT + 1.0) - w(FAR_LEFT)

    assert abs(right) != pytest.approx(abs(left), rel=1e-3)


def test_the_minimum_is_the_lowest_value_the_curve_attains() -> None:
    """The closed form against a brute-force scan, which is the only independent check there is."""
    scanned = min(
        total_variance(k / 10_000.0, a=A, b=B, rho=RHO, m=M, sigma=SIGMA)
        for k in range(-50_000, 50_001)
    )
    assert min_total_variance(a=A, b=B, rho=RHO, sigma=SIGMA) == pytest.approx(scanned, rel=1e-9)


def test_the_minimum_does_not_move_when_the_smile_is_shifted_sideways() -> None:
    """``m`` is absent from the closed form, and this is the statement that entitles it to be.

    Shifting a curve horizontally moves *where* its minimum sits, not what it is worth there.
    """
    shifted = min(
        total_variance(k / 1_000.0, a=A, b=B, rho=RHO, m=0.75, sigma=SIGMA)
        for k in range(-5_000, 5_001)
    )
    assert min_total_variance(a=A, b=B, rho=RHO, sigma=SIGMA) == pytest.approx(shifted, rel=1e-6)


def test_a_moneyness_of_zero_is_an_ordinary_point_on_the_curve() -> None:
    """At the money forward, and nothing in the closed form may test ``k`` for truthiness."""
    assert w(0.0) == pytest.approx(A + B * (RHO * -M + math.sqrt(M * M + SIGMA * SIGMA)))


# --- the one input with a domain restriction


@pytest.mark.parametrize("rho", [1.0, -1.0, 1.5, float("inf"), float("nan")])
def test_the_minimum_refuses_a_rho_outside_the_open_interval(rho: float) -> None:
    """NaN included, and it is why the guard is written as a bad-case test: a NaN compares
    ``False`` against every bound, so ``-1 < nan < 1`` is ``False`` and the negation catches it
    instead of letting it reach ``sqrt`` of a negative number."""
    with pytest.raises(ValueError, match="rho must be inside"):
        min_total_variance(a=A, b=B, rho=rho, sigma=SIGMA)


def test_the_naive_guard_really_would_have_admitted_the_nan() -> None:
    """Guards the test above from being vacuous: this is the ordering test that reads correct."""
    assert not (float("nan") >= 1.0)


def test_the_curve_admits_the_parameters_its_own_minimum_would_reject() -> None:
    """:func:`total_variance` takes no guard, and that is deliberate rather than an oversight.

    An optimiser is entitled to evaluate a curve it is about to reject -- that is the whole
    reason ``SVIParams`` admits a collapsed slice -- and a negative ``a`` is a finite, perfectly
    computable point on a curve that is simply not a surface. Judging it belongs to the caller's
    constructor, one layer up, where the error can name the field.
    """
    assert total_variance(0.1, a=-1.0, b=B, rho=RHO, m=M, sigma=SIGMA) < 0.0
