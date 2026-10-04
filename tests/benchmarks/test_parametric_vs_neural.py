"""The parametric-vs-neural benchmark's measures and its report, on synthetic surfaces and rows."""

from __future__ import annotations

import math
from dataclasses import replace
from pathlib import Path

from benchmarks.parametric_vs_neural import SLUG, build_report, fitted, steps, violations
from benchmarks.session import SessionResult

from tests.benchmarks.builders import TENORS, make_grid, write_metrics
from tests.risk.builders import make_calibrated_surface, make_report
from volengine.contracts.calibrated_surface import CalibratedSurface, SurfaceStatus, VolGrid
from volengine.risk.domain.comparison import ComparativeReport, pair_lines

NEURAL = {"market": "BTC-DERIBIT"}


def on_grid(grid: VolGrid, status: SurfaceStatus = SurfaceStatus.OK) -> CalibratedSurface:
    return replace(make_calibrated_surface(status=status), grid=grid)


def healthy(status: SurfaceStatus = SurfaceStatus.OK) -> CalibratedSurface:
    return on_grid(make_grid(), status)


def shifted(by: float) -> CalibratedSurface:
    """The healthy surface with every volatility moved by ``by``: a known step of ``by`` vol."""
    grid = make_grid()
    vols = tuple(tuple(vol + by for vol in row) for row in grid.vols)
    return on_grid(replace(grid, vols=vols))


def butterfly_only() -> CalibratedSurface:
    def hump(tenor: float, k: float) -> float:
        return tenor * (0.25 + 0.2 * k * k) + 0.02 * math.exp(-((k / 0.08) ** 2))

    return on_grid(make_grid(hump))


def calendar_only() -> CalibratedSurface:
    def inverted(tenor: float, k: float) -> float:
        level = 0.05 if tenor == TENORS[-1] else tenor * 0.25
        return level + 0.01 * k * k

    return on_grid(make_grid(inverted))


def test_fitted_leaves_republished_surfaces_out() -> None:
    kept = healthy()

    assert fitted((healthy(SurfaceStatus.STALE_REPUBLISH), kept)) == (kept,)


def test_a_step_is_the_rms_distance_between_consecutive_fits() -> None:
    summary = steps((healthy(), shifted(0.01)))

    assert summary.count == 1
    assert math.isclose(summary.p50 or 0.0, 100.0, rel_tol=1e-9)


def test_a_republished_surface_is_not_a_step() -> None:
    """A republish is the previous fit again; counting it would report a zero step."""
    summary = steps((healthy(), healthy(SurfaceStatus.STALE_REPUBLISH), shifted(0.01)))

    assert summary.count == 1
    assert math.isclose(summary.maximum or 0.0, 100.0, rel_tol=1e-9)


def test_fewer_than_two_fits_have_no_step_rather_than_a_zero_one() -> None:
    assert steps(()).count == 0
    assert steps((healthy(),)).maximum is None


def test_two_fits_with_no_common_region_are_skipped_not_counted_as_zero() -> None:
    left = make_calibrated_surface(log_moneyness=(-0.4, -0.3, -0.2))
    right = make_calibrated_surface(log_moneyness=(0.2, 0.3, 0.4))

    summary = steps((left, right))

    assert (summary.count, summary.mean) == (0, None)


def test_violations_count_a_butterfly_only_breach() -> None:
    butterfly, calendar, breached = violations((healthy(), butterfly_only()))

    assert breached == 1
    assert (butterfly.maximum or 0.0) > 0.0
    assert calendar.maximum == 0.0


def test_violations_count_a_calendar_only_breach() -> None:
    butterfly, calendar, breached = violations((calendar_only(),))

    assert breached == 1
    assert butterfly.maximum == 0.0
    assert (calendar.maximum or 0.0) > 0.0


def test_violations_leave_republished_surfaces_out() -> None:
    _, _, breached = violations(
        (
            on_grid(make_grid(), SurfaceStatus.OK),
            replace(butterfly_only(), status=SurfaceStatus.STALE_REPUBLISH),
        )
    )

    assert breached == 0


def session(
    comparison: ComparativeReport | None,
) -> SessionResult:
    surfaces = {"svi-scipy": (healthy(), shifted(0.01)), "mlp-torch": (healthy(), butterfly_only())}
    return SessionResult(
        market_id="BTC-DERIBIT",
        producers=("svi-scipy", "mlp-torch"),
        surfaces=surfaces,
        comparison=comparison,
        wall_seconds=1.0,
    )


def test_the_report_without_a_restart_or_a_comparison_says_both(tmp_path: Path) -> None:
    write_metrics(
        tmp_path / f"{SLUG}.metrics.csv",
        [
            (0.0, "gauge", "pricing.rmse_vol_bp", 36.0, {"producer": "svi-scipy"}),
            (0.5, "gauge", "neural.rmse_vol_bp", 680.0, NEURAL),
        ],
    )

    text = build_report(tmp_path / "x.toml", tmp_path, session(None))

    assert "never restarted" in text
    assert "No comparison: a producer never published." in text
    assert "| Grids with any breach on the published nodes | 0 | 1 |" in text
    assert (tmp_path / f"{SLUG}-rmse.svg").is_file()


def test_the_report_splits_the_regimes_and_prints_the_comparison(tmp_path: Path) -> None:
    write_metrics(
        tmp_path / f"{SLUG}.metrics.csv",
        [
            (0.0, "gauge", "pricing.rmse_vol_bp", 36.0, {"producer": "svi-scipy"}),
            (1.0, "gauge", "neural.rmse_vol_bp", 680.0, NEURAL),
            (3.0, "counter", "neural.restart", 1, NEURAL),
            (3.0, "gauge", "neural.restart.drift_vol_bp", 300.0, NEURAL),
            (4.0, "gauge", "neural.rmse_vol_bp", 47.0, NEURAL),
        ],
    )
    baseline = make_report(producer_id="svi-scipy")
    challenger = make_report(producer_id="mlp-torch")
    comparison = ComparativeReport(
        market_id="BTC-DERIBIT",
        baseline=baseline,
        challenger=challenger,
        lines=pair_lines(baseline, challenger),
        distance=None,
    )

    text = build_report(tmp_path / "x.toml", tmp_path, session(comparison), cycles=60)

    assert "first restarted 3.0 s into the session; restarts in all: 1." in text
    assert "| Before the network's first restart |  | 680.0 / 680.0 / 680.0 (n=1) |" in text
    assert "| From the first restart on |  | 47.0 / 47.0 / 47.0 (n=1) |" in text
    assert "300.0 / 300.0 / 300.0 (n=1)" in text
    assert "- `mlp-torch` valued off a `OK` surface" in text
    assert "overridden to 60" in text
