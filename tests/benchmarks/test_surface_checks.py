"""No-arbitrage on a published grid, measured the same way for every producer."""

from __future__ import annotations

import math

import pytest
from benchmarks.surface_checks import GridArbitrage, grid_arbitrage

from tests.benchmarks.builders import K_AXIS, TENORS, make_grid


def test_a_convex_smile_rising_with_tenor_shows_no_breach() -> None:
    assert grid_arbitrage(make_grid()) == GridArbitrage(butterfly=0.0, calendar=0.0)


def test_a_smile_with_a_hump_shows_a_butterfly_breach() -> None:
    """A bump in total variance near the money is a negative density there."""

    def hump(tenor: float, k: float) -> float:
        return tenor * (0.25 + 0.2 * k * k) + 0.02 * math.exp(-((k / 0.08) ** 2))

    assert grid_arbitrage(make_grid(hump)).butterfly > 0.0


def test_a_far_expiry_carrying_less_variance_shows_the_calendar_excess() -> None:
    """The year row sits below the three-month row by a known margin at every node."""

    def inverted(tenor: float, k: float) -> float:
        level = 0.05 if tenor == TENORS[-1] else tenor * 0.25
        return level + 0.01 * k * k

    near = TENORS[1] * 0.25
    expected = near - 0.05

    assert grid_arbitrage(make_grid(inverted)).calendar == pytest.approx(expected, abs=1e-12)


def test_the_calendar_check_is_not_vacuous_on_equal_rows() -> None:
    """Equal total variance on two rows is the boundary: no excess, not a breach."""

    def flat(tenor: float, k: float) -> float:
        return 0.04 + 0.01 * k * k

    assert grid_arbitrage(make_grid(flat)).calendar == pytest.approx(0.0, abs=1e-15)


def test_a_non_uniform_moneyness_axis_is_refused() -> None:
    axis = (*K_AXIS[:-1], K_AXIS[-1] + 0.01)

    with pytest.raises(ValueError, match="uniform"):
        grid_arbitrage(make_grid(log_moneyness=axis))


@pytest.mark.parametrize("bad", [-1e-9, math.nan, math.inf])
def test_a_violation_is_finite_and_non_negative(bad: float) -> None:
    with pytest.raises(ValueError, match="finite and non-negative"):
        GridArbitrage(butterfly=bad, calendar=0.0)
