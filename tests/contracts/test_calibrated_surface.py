from __future__ import annotations

import json
from dataclasses import FrozenInstanceError, replace
from datetime import UTC, datetime, timedelta

import pytest

from tests.support import replace_field
from volengine.contracts.calibrated_surface import (
    SCHEMA_VERSION,
    CalibratedSurface,
    FitMetrics,
    SurfaceStatus,
    VolGrid,
)

TS = datetime(2026, 7, 27, 8, 0, tzinfo=UTC)
CALIBRATED_AT = TS + timedelta(milliseconds=120)

# Log-forward-moneyness axis: k = 0 is at the money forward, the wings reach roughly
# +-22% away from it. Tenors in years: one month, three months, one year.
K_AXIS = (-0.2, -0.1, 0.0, 0.1, 0.2)
T_AXIS = (30 / 365.0, 90 / 365.0, 1.0)


# --- builders
# One valid object per class, with a single knob per test. A test that needs an
# invalid variant starts from a valid one and changes the minimum: the test then
# states what it is probing instead of drowning the intent in twenty fields.


def _smile(tenor: float, log_moneyness: tuple[float, ...]) -> tuple[float, ...]:
    """A plausible smile: minimum at the money, wings curving up, level decaying with tenor.

    The shape is not what the tests are about, but plausible numbers make a failure
    readable — 0.65 is a believable BTC vol, 4.2 would send you hunting the wrong bug.
    """
    at_the_money = 0.65 - 0.05 * tenor
    return tuple(at_the_money + 0.35 * k * k for k in log_moneyness)


def make_grid(
    log_moneyness: tuple[float, ...] = K_AXIS,
    tenors: tuple[float, ...] = T_AXIS,
    vols: tuple[tuple[float, ...], ...] | None = None,
) -> VolGrid:
    """Coherent by default; pass `vols` explicitly to build a ragged or negative grid."""
    return VolGrid(
        log_moneyness=log_moneyness,
        tenors=tenors,
        vols=vols if vols is not None else tuple(_smile(t, log_moneyness) for t in tenors),
    )


def make_fit() -> FitMetrics:
    return FitMetrics(
        rmse_vol_bp=18.0,
        max_err_vol_bp=64.0,
        n_quotes_used=180,
        n_iterations=7,
        duration_ms=3.4,
    )


def make_surface() -> CalibratedSurface:
    return CalibratedSurface(
        surface_id="01JZQ0T4M2",
        source_snapshot_id="01JZQ0S9K7",
        market_id="BTC-DERIBIT",
        ts_snapshot=TS,
        ts_calibrated=CALIBRATED_AT,
        producer_id="svi-scipy",
        grid=make_grid(),
        fit=make_fit(),
        status=SurfaceStatus.OK,
        producer_meta={"a": 0.041, "b": 0.13, "rho": -0.42, "m": 0.02, "sigma": 0.19},
    )


# --- SurfaceStatus


@pytest.mark.parametrize(
    ("status", "wire_value"),
    [
        (SurfaceStatus.OK, "OK"),
        (SurfaceStatus.DEGRADED, "DEGRADED"),
        (SurfaceStatus.STALE_REPUBLISH, "STALE_REPUBLISH"),
    ],
)
def test_surface_status_has_stable_wire_values(
    status: SurfaceStatus,
    wire_value: str,
) -> None:
    assert status.value == wire_value
    assert SurfaceStatus(wire_value) is status


# --- VolGrid


def test_grid_accepts_a_single_tenor_and_moneyness_node() -> None:
    grid = make_grid(log_moneyness=(0.0,), tenors=(1.0,))

    assert grid.vols == ((0.6,),)


def test_grid_rejects_an_empty_tenor_axis() -> None:
    with pytest.raises(ValueError, match="at least one tenor"):
        make_grid(tenors=())


def test_grid_rejects_an_empty_moneyness_axis() -> None:
    with pytest.raises(ValueError, match="at least one moneyness node"):
        make_grid(log_moneyness=())


@pytest.mark.parametrize(
    "bad_value",
    [float("nan"), float("inf"), float("-inf")],
    ids=["nan", "positive-infinity", "negative-infinity"],
)
def test_grid_rejects_non_finite_moneyness_nodes(bad_value: float) -> None:
    with pytest.raises(ValueError, match="moneyness axis must be finite"):
        make_grid(log_moneyness=(-0.2, bad_value, 0.2))


@pytest.mark.parametrize(
    "bad_value",
    [float("nan"), float("inf"), float("-inf")],
    ids=["nan", "positive-infinity", "negative-infinity"],
)
def test_grid_rejects_non_finite_tenors(bad_value: float) -> None:
    with pytest.raises(ValueError, match="tenor axis must be finite"):
        make_grid(tenors=(30 / 365.0, bad_value, 1.0))


@pytest.mark.parametrize(
    "axis",
    [(-0.2, -0.1, -0.1, 0.2), (-0.2, 0.1, 0.0, 0.2)],
    ids=["duplicate", "decreasing"],
)
def test_grid_requires_strictly_increasing_moneyness(axis: tuple[float, ...]) -> None:
    with pytest.raises(ValueError, match="moneyness axis must be a strictly increasing"):
        make_grid(log_moneyness=axis)


@pytest.mark.parametrize(
    "axis",
    [(30 / 365.0, 30 / 365.0, 1.0), (30 / 365.0, 1.0, 90 / 365.0)],
    ids=["duplicate", "decreasing"],
)
def test_grid_requires_strictly_increasing_tenors(axis: tuple[float, ...]) -> None:
    with pytest.raises(ValueError, match="tenor axis must be a strictly increasing"):
        make_grid(tenors=axis)


@pytest.mark.parametrize("smile_count", [2, 4], ids=["too-few", "too-many"])
def test_grid_requires_one_smile_per_tenor(smile_count: int) -> None:
    smiles = tuple(_smile(T_AXIS[0], K_AXIS) for _ in range(smile_count))

    with pytest.raises(ValueError, match="one smile per tenor"):
        make_grid(vols=smiles)


def test_grid_requires_one_vol_per_moneyness_node() -> None:
    smiles = list(make_grid().vols)
    smiles[1] = smiles[1][:-1]

    with pytest.raises(ValueError, match="one vol per moneyness node"):
        make_grid(vols=tuple(smiles))


@pytest.mark.parametrize(
    "bad_vol",
    [0.0, -0.01, float("nan"), float("inf"), float("-inf")],
    ids=["zero", "negative", "nan", "positive-infinity", "negative-infinity"],
)
def test_grid_rejects_non_positive_or_non_finite_vols(bad_vol: float) -> None:
    smiles = [list(smile) for smile in make_grid().vols]
    smiles[1][2] = bad_vol

    with pytest.raises(ValueError, match="volatility must be positive and finite"):
        make_grid(vols=tuple(tuple(smile) for smile in smiles))


def test_grid_does_not_impose_smile_monotonicity_or_convexity() -> None:
    irregular_smile = (0.7, 0.9, 0.6, 0.85, 0.72)
    grid = make_grid(vols=tuple(irregular_smile for _ in T_AXIS))

    assert grid.vols[0] == irregular_smile


def test_grid_dict_round_trip_uses_json_friendly_lists() -> None:
    grid = VolGrid(
        log_moneyness=(-0.1, 0.1),
        tenors=(0.5,),
        vols=((0.4, 0.45),),
    )

    raw = grid.to_dict()

    assert raw == {
        "log_moneyness": [-0.1, 0.1],
        "tenors": [0.5],
        "vols": [[0.4, 0.45]],
    }
    assert VolGrid.from_dict(raw) == grid


# --- FitMetrics


def test_fit_accepts_zero_errors_iterations_and_duration() -> None:
    fit = FitMetrics(
        rmse_vol_bp=0.0,
        max_err_vol_bp=0.0,
        n_quotes_used=1,
        n_iterations=0,
        duration_ms=0.0,
    )

    assert fit.rmse_vol_bp == 0.0
    assert fit.n_iterations == 0
    assert fit.duration_ms == 0.0


@pytest.mark.parametrize(
    ("field", "bad_value", "message"),
    [
        ("rmse_vol_bp", -0.01, "RMSE must be non-negative and finite"),
        ("rmse_vol_bp", float("nan"), "RMSE must be non-negative and finite"),
        ("rmse_vol_bp", float("inf"), "RMSE must be non-negative and finite"),
        ("max_err_vol_bp", -0.01, "maximum vol bp error must be non-negative and finite"),
        ("max_err_vol_bp", float("nan"), "maximum vol bp error must be non-negative and finite"),
        ("max_err_vol_bp", float("inf"), "maximum vol bp error must be non-negative and finite"),
        ("n_quotes_used", 0, "number of quotes used must be positive"),
        ("n_quotes_used", -1, "number of quotes used must be positive"),
        ("n_iterations", -1, "number of iterations must be non-negative"),
        ("duration_ms", -0.01, "fitting duration .* must be non-negative and finite"),
        ("duration_ms", float("nan"), "fitting duration .* must be non-negative and finite"),
        ("duration_ms", float("inf"), "fitting duration .* must be non-negative and finite"),
    ],
)
def test_fit_rejects_invalid_individual_metrics(
    field: str,
    bad_value: float | int,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        replace_field(make_fit(), field, bad_value)


def test_fit_rejects_rmse_greater_than_maximum_error() -> None:
    with pytest.raises(ValueError, match="RMSE error can't be greater"):
        replace(make_fit(), rmse_vol_bp=65.0)


def test_fit_dict_round_trip() -> None:
    fit = make_fit()

    assert FitMetrics.from_dict(fit.to_dict()) == fit


# --- CalibratedSurface


@pytest.mark.parametrize(
    "field",
    ["surface_id", "source_snapshot_id", "market_id", "producer_id"],
)
def test_surface_rejects_empty_identifiers(field: str) -> None:
    with pytest.raises(ValueError, match=rf"The {field} must not be empty"):
        replace_field(make_surface(), field, "")


@pytest.mark.parametrize("field", ["ts_snapshot", "ts_calibrated"])
def test_surface_rejects_naive_timestamps(field: str) -> None:
    naive = TS.replace(tzinfo=None)

    with pytest.raises(ValueError, match=rf"{field} must be timezone-aware"):
        replace_field(make_surface(), field, naive)


def test_surface_allows_calibration_at_the_snapshot_time() -> None:
    surface = replace(make_surface(), ts_calibrated=TS)

    assert surface.ts_calibrated == surface.ts_snapshot


def test_surface_rejects_calibration_before_the_snapshot() -> None:
    with pytest.raises(ValueError, match="can't be calibrated before"):
        replace(make_surface(), ts_calibrated=TS - timedelta(microseconds=1))


def test_surface_dict_round_trip_through_json() -> None:
    surface = make_surface()

    payload = json.loads(json.dumps(surface.to_dict()))
    restored = CalibratedSurface.from_dict(payload)

    assert restored == surface
    assert isinstance(restored.status, SurfaceStatus)
    assert restored.ts_snapshot.tzinfo is not None


def test_surface_round_trip_with_no_producer_metadata() -> None:
    surface = replace(make_surface(), producer_meta=None)

    assert CalibratedSurface.from_dict(surface.to_dict()) == surface


def test_surface_from_dict_defaults_a_missing_schema_version() -> None:
    raw = make_surface().to_dict()
    raw.pop("schema_version")

    surface = CalibratedSurface.from_dict(raw)

    assert surface.schema_version == 1


def test_surface_from_dict_rejects_a_future_schema_version() -> None:
    raw = make_surface().to_dict()
    raw["schema_version"] = SCHEMA_VERSION + 1

    with pytest.raises(ValueError, match="cannot read schema version"):
        CalibratedSurface.from_dict(raw)


def test_surface_from_dict_rejects_an_unknown_status() -> None:
    raw = make_surface().to_dict()
    raw["status"] = "BROKEN"

    with pytest.raises(ValueError):
        CalibratedSurface.from_dict(raw)


def test_surface_to_dict_copies_producer_metadata() -> None:
    surface = make_surface()
    raw = surface.to_dict()
    assert raw["producer_meta"] is not None

    raw["producer_meta"]["a"] = 999.0

    assert surface.producer_meta is not None
    assert surface.producer_meta["a"] == 0.041


def test_surface_from_dict_copies_producer_metadata() -> None:
    raw = make_surface().to_dict()
    surface = CalibratedSurface.from_dict(raw)
    assert raw["producer_meta"] is not None

    raw["producer_meta"]["a"] = 999.0

    assert surface.producer_meta is not None
    assert surface.producer_meta["a"] == 0.041


@pytest.mark.parametrize(
    ("value", "field"),
    [
        (make_grid(), "tenors"),
        (make_fit(), "duration_ms"),
        (make_surface(), "status"),
    ],
)
def test_contract_values_are_frozen_and_slotted(value: object, field: str) -> None:
    assert not hasattr(value, "__dict__")

    with pytest.raises(FrozenInstanceError):
        setattr(value, field, None)
