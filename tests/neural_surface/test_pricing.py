"""Tests of this context's Black-76 wrapper: the enum it owns and the error it translates.

The formula is the shared kernel's and is tested there and in
``tests/parametric_pricing/test_black76.py``. What this module adds is a mapping and an exception,
and both are exactly the kind of one-line seam that no behavioural test elsewhere would notice
breaking.
"""

from __future__ import annotations

import pytest

from volengine.neural_surface.domain.errors import NeuralSurfaceError, NoImpliedVolError
from volengine.neural_surface.domain.pricing import OptionKindN, implied_vol, price, vega
from volengine.shared_kernel.domain import black76 as kernel

FORWARD = 60_000.0
TENOR = 0.25
VOL = 0.65


def test_the_enum_maps_to_the_kernels_boolean_on_both_sides() -> None:
    """The one line of translation between this context's vocabulary and the shared arithmetic.

    A swapped mapping would price every put as a call and still return a plausible number at every
    strike, so the two sides are pinned against the kernel directly rather than against each other.
    """
    for strike in (45_000.0, 60_000.0, 80_000.0):
        assert price(FORWARD, strike, TENOR, VOL, OptionKindN.CALL) == kernel.price(
            FORWARD, strike, TENOR, VOL, True
        )
        assert price(FORWARD, strike, TENOR, VOL, OptionKindN.PUT) == kernel.price(
            FORWARD, strike, TENOR, VOL, False
        )


def test_vega_passes_straight_through_because_it_takes_no_side() -> None:
    assert vega(FORWARD, 45_000.0, TENOR, VOL) == kernel.vega(FORWARD, 45_000.0, TENOR, VOL)


def test_an_uninvertible_price_raises_this_contexts_error_not_the_kernels() -> None:
    """The seam that keeps this context's error hierarchy whole.

    ``PriceNotInvertibleError`` belongs to no context, so it would sail straight through a caller's
    ``except NeuralSurfaceError`` around a batch and take a routine market outcome -- a mid below
    intrinsic, which a crossed book produces regularly -- out as an unhandled failure of the whole
    training step.
    """
    floor = kernel.intrinsic(FORWARD, 45_000.0, False, 1.0)
    with pytest.raises(NoImpliedVolError, match="intrinsic value"):
        implied_vol(floor, FORWARD, 45_000.0, TENOR, OptionKindN.PUT)


def test_that_error_is_reachable_by_the_contexts_own_base_class() -> None:
    """Guards the test above: catching the leaf would still pass if the hierarchy were wrong."""
    floor = kernel.intrinsic(FORWARD, 45_000.0, False, 1.0)
    with pytest.raises(NeuralSurfaceError):
        implied_vol(floor, FORWARD, 45_000.0, TENOR, OptionKindN.PUT)


def test_the_kernels_error_is_chained_rather_than_discarded() -> None:
    """``raise ... from exc``: the original is what says which bound was crossed and by how much."""
    floor = kernel.intrinsic(FORWARD, 45_000.0, False, 1.0)
    with pytest.raises(NoImpliedVolError) as caught:
        implied_vol(floor, FORWARD, 45_000.0, TENOR, OptionKindN.PUT)
    assert isinstance(caught.value.__cause__, kernel.PriceNotInvertibleError)


def test_a_construction_bug_still_arrives_as_a_value_error() -> None:
    """A NaN forward is not a market condition, and must not be catchable as one."""
    with pytest.raises(ValueError, match="forward"):
        price(float("nan"), 60_000.0, TENOR, VOL, OptionKindN.CALL)
