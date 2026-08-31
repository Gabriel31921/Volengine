"""Raw SVI stated as laws over a family of slices, including one the other context has to agree to.

The example tests around this one pin the curve and its diagnostics at chosen parameters. These
quantify the same claims over every market-shaped slice hypothesis can build, which is the regime
the optimiser actually walks through: a calibration proposes hundreds of iterates per cycle and
none of them is a value anybody wrote in a test file.

**The last property is a cross-context one, and ``tests/`` is the only place it can live.**
``parametric_pricing`` differentiates ``w(k)`` in closed form because raw SVI is an algebraic
curve; ``neural_surface`` takes central differences of a sampled surface because a network has no
algebra to differentiate (rule 6 keeps the two apart, and ADR-010's last consequence says the
duplication is deliberate). They are two implementations of one equation, and nothing in ``src/``
may import both -- so if they ever drift apart, this is the only file that would notice. The
tolerance is the finite-difference truncation error and nothing else, which is why the mesh is
stated with the property and why a guard below shows the band is narrow enough to discriminate.
"""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from numpy.typing import NDArray

from volengine.neural_surface.domain.invariants import durrleman_g as differenced_g
from volengine.parametric_pricing.domain.durrleman import butterfly_violation
from volengine.parametric_pricing.domain.durrleman import durrleman_g as closed_form_g
from volengine.parametric_pricing.domain.svi_slice import SVIParams

SLICES = st.builds(
    SVIParams,
    a=st.floats(min_value=5e-3, max_value=0.5),
    b=st.floats(min_value=5e-3, max_value=0.3),
    rho=st.floats(min_value=-0.95, max_value=0.95),
    m=st.floats(min_value=-0.3, max_value=0.3),
    sigma=st.floats(min_value=0.1, max_value=1.0),
)
"""A market-shaped slice: a live level, a real wing, a skew short of the limit, a rounded bottom.

Every corner of the box is admissible, so the constructor never refuses a draw and hypothesis
never has to filter -- a strategy that threw most of its examples away would shrink towards the
same handful of survivors and quietly stop searching. The box is where a fitted crypto slice
lives, not where ``SVIParams`` stops: the value object deliberately admits collapsed and
arbitrageable slices because an optimiser walks through them, and those are the subject of the
example tests rather than of a law.
"""

MESH: NDArray[np.float64] = np.linspace(-1.0, 1.0, 801)
"""Log-forward-moneyness from -100% to +100%, spaced by 0.0025.

Uniform because the differenced form requires it, dense because the truncation error of a central
difference is second order in the spacing and it is the whole tolerance of the last property.
"""

MONEYNESS = st.floats(min_value=-2.0, max_value=2.0)
"""One point of the line, drawn wider than the mesh: the curve is defined everywhere."""


@given(params=SLICES)
def test_the_free_coordinates_come_back_as_the_slice_they_left(params: SVIParams) -> None:
    """``from_free(to_free(p)) == p``: the optimiser's change of variables loses nothing.

    A warm start stores fitted parameters and hands the optimiser their free image, so a map that
    was not a bijection would restart every cycle from a slightly different slice than the one
    that was published -- a drift that no residual would report because both ends are admissible.
    """
    recovered = SVIParams.from_free(params.to_free())

    assert recovered.a == pytest.approx(params.a)
    assert recovered.b == pytest.approx(params.b)
    assert recovered.rho == pytest.approx(params.rho)
    assert recovered.m == pytest.approx(params.m)
    assert recovered.sigma == pytest.approx(params.sigma)


@given(params=SLICES, k=MONEYNESS)
def test_the_curve_never_dips_below_the_minimum_its_parameters_promise(
    params: SVIParams, k: float
) -> None:
    """``w(k) >= a + b * sigma * sqrt(1 - rho^2)`` everywhere, which is what admits the slice.

    ``SVIParams`` accepts a set of parameters by evaluating that closed form once and checking its
    sign. The whole value of the check rests on it really being the minimum of the curve, and this
    is the only statement in the repository that quantifies over the curve to say so.
    """
    total_variance = params.total_variance(k)

    assert np.isfinite(total_variance)
    assert total_variance >= params.min_total_variance - 1e-12


@given(params=SLICES)
def test_the_butterfly_measure_is_a_finite_depth_and_never_a_sign(params: SVIParams) -> None:
    """Zero when clean, positive when breached, finite always -- the orientation a penalty needs.

    The calibration loss adds this number to its residuals, so a NaN surviving the ``max(0, ...)``
    clamp as a clean zero would report a broken iterate as arbitrage-free and let the optimiser
    settle on it. That trap is why the assertion is on finiteness *before* it is on the sign.
    """
    depth = butterfly_violation(params, MESH)

    assert np.isfinite(depth)
    assert depth >= 0.0


@settings(max_examples=100)
@given(params=SLICES)
def test_the_two_contexts_measure_the_same_durrleman_function(params: SVIParams) -> None:
    """The closed form and the differenced form agree to the truncation error of the mesh.

    ``parametric_pricing`` publishes a surface only if its own ``g`` says the density is
    non-negative; ``neural_surface`` refuses to publish one if its ``g`` says otherwise. Two
    producers judged by two implementations of one condition are only comparable while those
    implementations mean the same thing, and no test inside either context can say that.
    """
    exact = closed_form_g(params, MESH)[1:-1]
    approximate = differenced_g(params.total_variance(MESH), MESH)

    assert approximate == pytest.approx(exact, abs=5e-3)


def test_a_slice_one_wing_apart_would_have_missed_that_band() -> None:
    """The guard on the tolerance above: five thousandths is not a band anything would clear.

    The two implementations agree to within the mesh's truncation error, which is small enough
    that a slice differing only in the steepness of its wings lands far outside it. Without this,
    a tolerance chosen loosely enough to hide a real disagreement would look exactly the same.
    """
    params = SVIParams(a=0.04, b=0.10, rho=-0.35, m=-0.02, sigma=0.20)
    steeper = SVIParams(a=0.04, b=0.16, rho=-0.35, m=-0.02, sigma=0.20)

    exact = closed_form_g(params, MESH)[1:-1]
    elsewhere = differenced_g(steeper.total_variance(MESH), MESH)

    assert np.max(np.abs(elsewhere - exact)) > 5e-3
