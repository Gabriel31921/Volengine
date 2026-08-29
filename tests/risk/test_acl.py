"""The translation a published surface makes into the one Risk holds.

Short, because the translation is short: axes copied, volatilities squared into total variance,
status dropped. What the tests below pin is the squaring -- which is the one place a wrong answer
would still look like a surface -- and the two deliberate absences.
"""

from __future__ import annotations

from dataclasses import replace
from datetime import timedelta

import pytest

from tests.risk.builders import (
    EXPIRIES,
    FORWARDS,
    K_AXIS,
    NOW,
    TENORS,
    make_calibrated_surface,
    make_view,
    smile_vol,
)
from volengine.contracts.calibrated_surface import SurfaceStatus
from volengine.risk.application.acl import to_surface_view
from volengine.risk.domain.surface_view import SurfaceView


def test_the_volatilities_become_total_variance() -> None:
    """``w = vol**2 * T``, and the tenor it is multiplied by has to be that row's own."""
    view = to_surface_view(make_calibrated_surface())

    expected = [smile_vol(TENORS[1], k) ** 2 * TENORS[1] for k in K_AXIS]
    assert list(view.total_variance[1]) == pytest.approx(expected)


def test_each_row_is_scaled_by_its_own_tenor() -> None:
    """The vacuous-pass guard: scaling every row by the first tenor would still look plausible."""
    view = to_surface_view(make_calibrated_surface())

    first = view.total_variance[0][2]
    last = view.total_variance[-1][2]
    assert last > first


def test_the_translation_reproduces_the_view_a_test_builds_by_hand() -> None:
    """The two builders describe one surface, so the ACL is what makes them agree."""
    assert to_surface_view(make_calibrated_surface()) == make_view()


def test_the_axes_are_carried_across_unchanged() -> None:
    view = to_surface_view(make_calibrated_surface())

    assert view.log_moneyness == K_AXIS
    assert view.tenors == TENORS
    assert view.expiries == EXPIRIES
    assert view.forwards == FORWARDS


def test_the_producer_travels_inside_the_view() -> None:
    """A composite provider serves several producers, so identity cannot be attached by a caller."""
    surface = make_calibrated_surface(producer_id="mlp-torch")

    assert to_surface_view(surface).producer_id == "mlp-torch"


def test_the_snapshot_instant_is_carried_and_not_the_calibration_one() -> None:
    """The whole of ADR-006: staleness is measured against the market, not the calculation."""
    surface = make_calibrated_surface(ts_snapshot=NOW, ts_calibrated=NOW + timedelta(seconds=45))

    assert to_surface_view(surface).ts_snapshot == NOW


def test_a_republished_stale_surface_translates_like_any_other() -> None:
    """It carries its original instant, which is the only channel the freshness policy needs."""
    surface = make_calibrated_surface(status=SurfaceStatus.STALE_REPUBLISH)

    view = to_surface_view(surface)

    assert view.ts_snapshot == NOW
    assert not hasattr(view, "status")


def test_the_view_has_no_status_field_at_all() -> None:
    """An architectural assertion: a second channel for one fact only ever disagrees with it."""
    assert "status" not in SurfaceView.__dataclass_fields__


def test_a_surface_whose_front_expiry_has_run_out_is_refused() -> None:
    """``VolGrid`` allows it, ``SurfaceView`` does not: a node with no time left has no variance."""
    surface = make_calibrated_surface(ts_snapshot=EXPIRIES[0] + timedelta(days=1))

    with pytest.raises(ValueError, match="first expiry"):
        to_surface_view(surface)


def test_a_ragged_published_grid_cannot_be_translated() -> None:
    """Both types refuse it, and the translation must not be where a ragged grid slips through."""
    surface = make_calibrated_surface()
    with pytest.raises(ValueError):
        to_surface_view(
            replace(
                surface,
                grid=replace(surface.grid, vols=surface.grid.vols[:2]),
            )
        )
