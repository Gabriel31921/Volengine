"""What the fixed grid guarantees, and what it refuses.

Padding is the whole of ADR-009 and it has exactly two jobs: make every task the same shape, and
leave the reserved cells unable to influence anything. The first is asserted here directly; the
second is asserted here as a property of the *fill values* -- finite, positive where a divisor --
and again in ``test_jax_calibrator.py`` as a property of the loss. Both halves are needed. A fill
of ``nan`` would satisfy every mask in the loss and still destroy the fit, because the gradient of
``where`` touches the branch it did not take.

No JAX in this module: the layout is numpy and is meant to be testable without the optional extra.
"""

from __future__ import annotations

import numpy as np
import pytest

from tests.parametric_pricing.builders import (
    FAR,
    FAR_TENOR,
    NEAR,
    NEAR_TENOR,
    make_calibration_task,
    make_slice_task,
)
from tests.support import replace_field
from volengine.parametric_pricing.adapters.padding import (
    FILL_TENOR_YEARS,
    PaddedTask,
    PadShape,
    pad,
)
from volengine.parametric_pricing.domain.errors import CalibrationError

SHAPE = PadShape(max_slices=4, max_quotes=8, mesh_nodes=11)
"""A rectangle with room to spare for the two-slice, four-quote task the builders produce, so
every test here has real padding to look at."""

MARGIN = 0.5
"""The mesh margin the defaults use, in log-moneyness."""


def padded(shape: PadShape = SHAPE, margin: float = MARGIN) -> PaddedTask:
    """The standard two-slice task, laid out on the grid."""
    return pad(make_calibration_task(), shape, margin)


def test_every_array_has_the_reserved_shape_whatever_the_task_holds() -> None:
    """The compiled signature is the reservation, not the chain. This is the claim ADR-009 makes."""
    one_slice = pad(make_calibration_task(slices=(make_slice_task(),)), SHAPE, MARGIN)
    two_slices = padded()

    for layout in (one_slice, two_slices):
        assert layout.shape == (SHAPE.max_slices, SHAPE.max_quotes)
        assert layout.log_moneyness.shape == (SHAPE.max_slices, SHAPE.max_quotes)
        assert layout.implied_vol.shape == (SHAPE.max_slices, SHAPE.max_quotes)
        assert layout.weights.shape == (SHAPE.max_slices, SHAPE.max_quotes)
        assert layout.quote_mask.shape == (SHAPE.max_slices, SHAPE.max_quotes)
        assert layout.tenor_years.shape == (SHAPE.max_slices,)
        assert layout.slice_mask.shape == (SHAPE.max_slices,)
        assert layout.mesh.shape == (SHAPE.max_slices, SHAPE.mesh_nodes)


def test_the_real_quotes_land_in_the_leading_cells_of_their_row() -> None:
    """Index ``[s, q]`` of every quote-shaped array describes one quote, and the fit reads them
    positionally. A layout that shifted one array and not another would pair a volatility with
    someone else's weight and nothing downstream could notice."""
    task = make_calibration_task()
    layout = padded()

    for index, one in enumerate(task.slices):
        width = len(one.log_moneyness)
        assert layout.log_moneyness[index, :width] == pytest.approx(one.log_moneyness)
        assert layout.implied_vol[index, :width] == pytest.approx(one.implied_vol)
        assert layout.tenor_years[index] == pytest.approx(one.tenor_years)


def test_the_mask_marks_exactly_the_real_cells() -> None:
    task = make_calibration_task()
    layout = padded()

    assert int(layout.quote_mask.sum()) == sum(len(one.log_moneyness) for one in task.slices)
    assert int(layout.slice_mask.sum()) == len(task.slices)
    assert layout.n_slices == len(task.slices)
    assert bool(layout.quote_mask[len(task.slices) :].any()) is False


def test_the_weights_of_a_slice_are_renormalised_to_one() -> None:
    """The loss reads the weights as a distribution, so a task built with unnormalised ones must
    produce the same fit as the same task normalised. The division happens here, once."""
    task = make_calibration_task(
        slices=(make_slice_task(weights=(3.0, 1.0, 4.0, 2.0)),),
    )

    layout = pad(task, SHAPE, MARGIN)

    assert float(layout.weights[0].sum()) == pytest.approx(1.0)
    assert layout.weights[0, :4] == pytest.approx((0.3, 0.1, 0.4, 0.2))


def test_a_zero_weight_quote_keeps_its_place_in_the_mask() -> None:
    """A flagged quote weighs nothing and stays visible: it is still a quote, and the metrics
    computed after the fit distinguish "in the chain" from "steered the fit"."""
    task = make_calibration_task(slices=(make_slice_task(weights=(0.0, 0.5, 0.3, 0.2)),))

    layout = pad(task, SHAPE, MARGIN)

    assert bool(layout.quote_mask[0, 0]) is True
    assert float(layout.weights[0, 0]) == 0.0


def test_the_padded_cells_are_finite_and_safe_to_differentiate_through() -> None:
    """**The trap this fill exists for.** ``where(mask, err, 0)`` protects the *value* of the loss
    and not its gradient: reverse-mode AD multiplies the untaken branch by zero, and ``0 * nan`` is
    ``nan``, so one NaN parked under the mask would poison every parameter of every slice. Nothing
    in the loss can repair that, which is why it is asserted on the layout instead."""
    layout = padded()

    assert bool(np.isfinite(layout.log_moneyness).all())
    assert bool(np.isfinite(layout.implied_vol).all())
    assert bool(np.isfinite(layout.weights).all())
    assert bool(np.isfinite(layout.mesh).all())
    assert bool(np.isfinite(layout.tenor_years).all())


def test_a_padded_cell_carries_a_positive_volatility_and_tenor() -> None:
    """Both are divided by and rooted -- ``vol**2 * T`` and ``sqrt(w / T)`` -- so a zero would be
    an infinity in the gradient rather than a harmless value under a mask."""
    layout = padded()
    empty = ~layout.slice_mask

    assert float(layout.implied_vol[~layout.quote_mask].min()) > 0.0
    assert layout.tenor_years[empty] == pytest.approx(FILL_TENOR_YEARS)
    assert float(layout.tenor_years.min()) > 0.0


def test_the_mesh_spans_the_quoted_band_plus_the_configured_margin() -> None:
    """The published grid is wider than the quoted band (ADR-001), so the wings a consumer prices
    against are extrapolation -- and that is exactly where raw SVI produces a negative density.
    A mesh that stopped at the outermost quote would leave the fit free to be arbitrage-free only
    where somebody already had a price."""
    task = make_calibration_task()
    layout = padded()

    for index, one in enumerate(task.slices):
        assert float(layout.mesh[index].min()) == pytest.approx(min(one.log_moneyness) - MARGIN)
        assert float(layout.mesh[index].max()) == pytest.approx(max(one.log_moneyness) + MARGIN)


def test_a_zero_margin_is_legal_and_pins_the_mesh_to_the_quoted_band() -> None:
    """The margin is a threshold, not a shape: it moves where the nodes sit and never how many
    there are, so turning it off must not change a single array's dimensions."""
    layout = pad(make_calibration_task(), SHAPE, 0.0)

    assert layout.mesh.shape == (SHAPE.max_slices, SHAPE.mesh_nodes)
    assert float(layout.mesh[0].min()) == pytest.approx(min(make_slice_task().log_moneyness))


def test_more_expiries_than_reserved_is_refused_rather_than_truncated() -> None:
    """Truncation would drop expiries the venue really published, silently, and the surface would
    still be released as healthy. Refusing makes the use case republish the last good one."""
    tenors = ((NEAR, NEAR_TENOR), (FAR, FAR_TENOR))
    slices = tuple(make_slice_task(expiry=expiry, tenor_years=tenor) for expiry, tenor in tenors)

    with pytest.raises(CalibrationError, match="reserves 1 expiries"):
        pad(make_calibration_task(slices=slices), PadShape(max_slices=1), MARGIN)


def test_more_quotes_than_reserved_is_refused_and_the_message_names_the_expiry() -> None:
    """An operator has to know which expiry outgrew the reservation to widen it by the right
    amount."""
    with pytest.raises(CalibrationError, match="2026-08-27"):
        pad(make_calibration_task(), PadShape(max_slices=4, max_quotes=3), MARGIN)


@pytest.mark.parametrize("margin", [-0.1, float("nan"), float("inf")])
def test_a_margin_that_is_not_a_non_negative_number_is_refused(margin: float) -> None:
    """NaN included, and deliberately first in the guard: ``nan < 0`` is ``False``, so an ordering
    test written the other way round would let it through into every mesh."""
    with pytest.raises(ValueError, match="mesh margin"):
        pad(make_calibration_task(), SHAPE, margin)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("max_slices", 0, "at least one slice"),
        ("max_quotes", 0, "at least one quote"),
        ("mesh_nodes", 1, "at least two nodes"),
    ],
)
def test_a_reservation_that_cannot_hold_a_task_is_refused_at_construction(
    field: str, value: int, message: str
) -> None:
    """The shape is the compiled signature: a grid with no room, or a mesh with one node, is a
    configuration error and belongs at start-up rather than at the first snapshot."""
    with pytest.raises(ValueError, match=message):
        replace_field(SHAPE, field, value)
