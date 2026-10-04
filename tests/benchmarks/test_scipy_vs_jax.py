"""The scipy-vs-JAX benchmark's table helpers and its report, on synthetic rows and surfaces."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest
from benchmarks.metrics_file import read_metrics
from benchmarks.report import DASH
from benchmarks.scipy_vs_jax import (
    SLUG,
    Run,
    _cycles,
    _evaluations,
    _refusals,
    _slices,
    build_report,
    construction_seconds,
)
from benchmarks.session import SessionResult

from tests.benchmarks.builders import write_metrics
from tests.risk.builders import make_calibrated_surface, make_report
from volengine.contracts.calibrated_surface import CalibratedSurface, SurfaceStatus
from volengine.entrypoints.config import load_config
from volengine.parametric_pricing.adapters.flat_vol import PRODUCER_ID as FLAT_ID
from volengine.parametric_pricing.adapters.scipy_calibrator import PRODUCER_ID as SCIPY_ID
from volengine.risk.domain.comparison import ComparativeReport, pair_lines

EXAMPLES = Path(__file__).resolve().parents[2] / "examples"
P = {"producer": "svi-scipy"}
OTHER = {"producer": "svi-jax"}


def with_iterations(n: int, status: SurfaceStatus = SurfaceStatus.OK) -> CalibratedSurface:
    surface = make_calibrated_surface(status=status)
    return replace(surface, fit=replace(surface.fit, n_iterations=n))


def result(
    surfaces: dict[str, tuple[CalibratedSurface, ...]],
    comparison: ComparativeReport | None = None,
) -> SessionResult:
    return SessionResult(
        market_id="BTC-DERIBIT",
        producers=tuple(surfaces),
        surfaces=surfaces,
        comparison=comparison,
        wall_seconds=1.0,
    )


def session_rows(path: Path) -> Path:
    return write_metrics(
        path,
        [
            (0.0, "timing", "pricing.cycle_ms", 2.0, P),
            (1.0, "timing", "pricing.cycle_ms", 80.0, P),
            (2.0, "timing", "pricing.cycle_ms", 100.0, P),
            (2.5, "timing", "pricing.cycle_ms", 999.0, OTHER),
            (0.0, "gauge", "pricing.slices.accepted", 2.0, P),
            (0.0, "gauge", "pricing.slices.attempted", 3.0, P),
            (1.0, "gauge", "pricing.slices.accepted", 3.0, P),
            (1.0, "gauge", "pricing.slices.attempted", 3.0, P),
            (1.0, "gauge", "pricing.slices.accepted", 9.0, OTHER),
            (1.0, "counter", "pricing.slice.not_converged", 2, P),
            (2.0, "counter", "pricing.slice.rmse_exceeded", 1, P),
            (2.0, "counter", "pricing.slice.at_bound", 7, OTHER),
            (2.0, "gauge", "pricing.rmse_vol_bp", 37.0, P),
            (2.0, "gauge", "risk.comparison.distance_rms_vol_bp", 80.0, {"challenger": "svi-jax"}),
        ],
    )


def test_the_first_cycle_is_reported_alone_and_the_rest_are_summarised(tmp_path: Path) -> None:
    first, later = _cycles(read_metrics(session_rows(tmp_path / "m.csv")), "svi-scipy")

    assert first == "2.0"
    # 80 and 100: the median, then the 95th percentile and maximum; the other producer's 999
    # never enters.
    assert later == "90.0 / 99.0 / 100.0 (n=2)"


def test_a_producer_without_cycles_prints_dashes_for_both(tmp_path: Path) -> None:
    rows = read_metrics(session_rows(tmp_path / "m.csv"))

    assert _cycles(rows, "mlp-torch") == (DASH, DASH)


def test_a_single_cycle_has_a_first_value_and_no_warm_statistics(tmp_path: Path) -> None:
    path = write_metrics(tmp_path / "m.csv", [(0.0, "timing", "pricing.cycle_ms", 5.0, P)])

    assert _cycles(read_metrics(path), "svi-scipy") == ("5.0", DASH)


def test_slices_sum_accepted_and_attempted_over_the_producers_cycles(tmp_path: Path) -> None:
    rows = read_metrics(session_rows(tmp_path / "m.csv"))

    assert _slices(rows, "svi-scipy") == "5 / 6"


def test_refusals_are_listed_not_converged_then_at_bound_then_over_rmse(tmp_path: Path) -> None:
    rows = read_metrics(session_rows(tmp_path / "m.csv"))

    assert _refusals(rows, "svi-scipy") == "2 / 0 / 1"


def test_evaluations_report_the_first_fit_then_the_warm_median() -> None:
    surfaces = (with_iterations(3), with_iterations(200), with_iterations(240))

    assert _evaluations(result({"svi-scipy": surfaces}), "svi-scipy") == "3 / 220"


def test_evaluations_leave_republished_surfaces_out() -> None:
    """A republished surface carries the old fit's count; counting it would double that fit."""
    surfaces = (
        with_iterations(999, SurfaceStatus.STALE_REPUBLISH),
        with_iterations(3),
        with_iterations(200),
    )

    assert _evaluations(result({"svi-scipy": surfaces}), "svi-scipy") == "3 / 200"


def test_evaluations_of_a_producer_that_never_fitted_are_a_dash() -> None:
    surfaces = (with_iterations(5, SurfaceStatus.STALE_REPUBLISH),)

    assert _evaluations(result({"svi-scipy": surfaces}), "svi-scipy") == DASH
    assert _evaluations(result({"svi-scipy": surfaces}), "svi-jax") == DASH


def test_construction_is_timed_for_registered_calibrators_and_skips_unknown_names() -> None:
    config = load_config(EXAMPLES / "synthetic-svi.toml")
    config = replace(
        config,
        calibration=replace(config.calibration, calibrators=(SCIPY_ID, FLAT_ID, "unregistered")),
    )

    timings = construction_seconds(config)

    assert set(timings) == {SCIPY_ID, FLAT_ID}
    assert all(seconds >= 0.0 for seconds in timings.values())


def runs(tmp_path: Path, comparison: ComparativeReport | None) -> tuple[dict[str, Run], Run]:
    rows = read_metrics(session_rows(tmp_path / "m.csv"))
    surfaces: dict[str, tuple[CalibratedSurface, ...]] = {
        "svi-scipy": (with_iterations(3), with_iterations(200)),
        "svi-jax": (with_iterations(250), with_iterations(2800)),
    }
    solo = {producer: Run(result({producer: found}), rows) for producer, found in surfaces.items()}
    return solo, Run(result(surfaces, comparison), rows)


def test_the_report_carries_the_table_the_distance_and_its_charts(tmp_path: Path) -> None:
    solo, joint = runs(tmp_path, comparison=None)
    out = tmp_path / "out"
    out.mkdir()

    text = build_report(tmp_path / "x.toml", out, solo, joint, {"svi-scipy": 0.0}, cycles=7)

    assert "| Slices accepted / attempted | 5 / 6 |" in text
    assert "| Objective evaluations, first / warm p50 | 3 / 200 | 250 / 2800 |" in text
    assert "80.0 / 80.0 / 80.0 (n=1)" in text
    assert "overridden to 7" in text
    assert f"![{SLUG}-cycle-ms]({SLUG}-cycle-ms.svg)" in text
    assert (out / f"{SLUG}-cycle-ms.svg").is_file()


def test_the_report_says_so_when_no_comparison_was_possible(tmp_path: Path) -> None:
    solo, joint = runs(tmp_path, comparison=None)

    text = build_report(tmp_path / "x.toml", tmp_path, solo, joint, {})

    assert "No comparison: a producer never published." in text


def test_the_report_prints_the_comparison_and_what_each_side_was_valued_off(
    tmp_path: Path,
) -> None:
    baseline = make_report(producer_id="svi-scipy")
    challenger = make_report(producer_id="svi-jax")
    comparison = ComparativeReport(
        market_id="BTC-DERIBIT",
        baseline=baseline,
        challenger=challenger,
        lines=pair_lines(baseline, challenger),
        distance=None,
    )
    solo, joint = runs(tmp_path, comparison=comparison)

    text = build_report(tmp_path / "x.toml", tmp_path, solo, joint, {})

    assert "- `svi-scipy` valued off a `OK` surface" in text
    assert "Total value difference over paired lines" in text


@pytest.mark.parametrize("name", ["svi-scipy", "svi-jax"])
def test_every_engine_module_counted_for_lines_exists(name: str) -> None:
    from benchmarks.scipy_vs_jax import ENGINE_MODULES

    assert all(path.is_file() for path in ENGINE_MODULES[name])
