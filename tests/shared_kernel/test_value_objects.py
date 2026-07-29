from dataclasses import FrozenInstanceError

import pytest

from volengine.shared_kernel.domain.value_objects import Moneyness, Strike, Tenor


@pytest.mark.parametrize("bad", [0.0, -1.0, float("nan"), float("inf"), float("-inf")])
def test_strike_rejects_bad_inputs(bad: float) -> None:
    with pytest.raises(ValueError):
        Strike(value=bad)


def test_strike_is_immutable() -> None:
    strike = Strike(value=60000)
    with pytest.raises(FrozenInstanceError):
        strike.value = 5  # type: ignore[misc]


def test_strike_equality_is_structural() -> None:
    assert Strike(60000.0) == Strike(60000.0)
    assert Strike(60000.0) != Strike(61000.0)


# --- Tenor


@pytest.mark.parametrize("bad", [0.0, -1.0, float("nan"), float("inf"), float("-inf")])
def test_tenor_rejects_bad_inputs(bad: float) -> None:
    with pytest.raises(ValueError):
        Tenor(years=bad)


def test_tenor_is_immutable() -> None:
    tenor = Tenor(years=2)
    with pytest.raises(FrozenInstanceError):
        tenor.years = 5  # type: ignore[misc]


def test_tenor_equality_is_structural() -> None:
    assert Tenor(1) == Tenor(1)
    assert Tenor(1) != Tenor(1.5)


# --- Moneyness


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_moneyness_rejects_bad_inputs(bad: float) -> None:
    with pytest.raises(ValueError):
        Moneyness(log_forward=bad)


def test_moneyness_is_immutable() -> None:
    moneyness = Moneyness(log_forward=0.15)
    with pytest.raises(FrozenInstanceError):
        moneyness.log_forward = 5  # type: ignore[misc]


def test_moneyness_equality_is_structural() -> None:
    assert Moneyness(0.5) == Moneyness(0.5)
    assert Moneyness(0.5) != Moneyness(0.6)


def test_moneynes_accepts_at_the_money() -> None:
    moneyness = Moneyness(log_forward=0.0)
    assert moneyness.log_forward == 0.0
