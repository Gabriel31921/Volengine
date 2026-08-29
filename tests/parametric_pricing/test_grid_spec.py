"""Where the published moneyness nodes sit.

Small enough that the tests are mostly about the two ends of the axis, which is where a mesh
built by accumulating a step goes wrong: it lands *near* the configured maximum rather than on it,
and the published surface then claims a band it was not configured for.
"""

from __future__ import annotations

from itertools import pairwise

import pytest

from volengine.parametric_pricing.application.grid_spec import GridSpec


def test_the_axis_spans_the_configured_range_inclusively() -> None:
    nodes = GridSpec(k_min=-0.4, k_max=0.4, n_nodes=5).nodes()

    assert nodes[0] == -0.4
    assert nodes[-1] == 0.4


def test_the_endpoint_is_exact_rather_than_accumulated() -> None:
    """A step repeated 200 times lands near the maximum; the last node is assigned instead."""
    nodes = GridSpec(k_min=-1.0, k_max=1.0, n_nodes=201).nodes()

    assert nodes[-1] == 1.0


def test_the_axis_has_the_configured_number_of_nodes() -> None:
    assert len(GridSpec(k_min=-0.4, k_max=0.4, n_nodes=17).nodes()) == 17


def test_the_axis_is_strictly_increasing() -> None:
    """``VolGrid`` refuses anything else, and a duplicated node makes interpolation ambiguous."""
    nodes = GridSpec(k_min=-0.4, k_max=0.4, n_nodes=41).nodes()

    assert all(near < far for near, far in pairwise(nodes))


def test_the_nodes_are_evenly_spaced() -> None:
    nodes = GridSpec(k_min=-0.4, k_max=0.4, n_nodes=5).nodes()

    assert list(nodes) == pytest.approx([-0.4, -0.2, 0.0, 0.2, 0.4])


def test_at_the_money_forward_is_an_ordinary_node() -> None:
    """``0.0`` is the most informative point on the surface and the one truthiness would drop."""
    nodes = GridSpec(k_min=-0.4, k_max=0.4, n_nodes=5).nodes()

    assert 0.0 in nodes


def test_a_two_node_grid_is_legal() -> None:
    assert GridSpec(k_min=-0.1, k_max=0.1, n_nodes=2).nodes() == (-0.1, 0.1)


@pytest.mark.parametrize("n_nodes", [1, 0, -3])
def test_a_grid_with_no_width_is_refused(n_nodes: int) -> None:
    """One node returns the same volatility for every strike: a surface describing nothing."""
    with pytest.raises(ValueError, match="two nodes"):
        GridSpec(k_min=-0.4, k_max=0.4, n_nodes=n_nodes)


def test_an_empty_range_is_refused() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        GridSpec(k_min=0.4, k_max=0.4, n_nodes=5)


def test_an_inverted_range_is_refused() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        GridSpec(k_min=0.4, k_max=-0.4, n_nodes=5)


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_a_non_finite_bound_is_refused_before_it_becomes_a_step(bad: float) -> None:
    """``nan < x`` is ``False``, so the ordering guard alone would let a NaN through."""
    with pytest.raises(ValueError, match="finite"):
        GridSpec(k_min=bad, k_max=0.4, n_nodes=5)
