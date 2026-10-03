"""The comparative report of Design 7.3, end to end inside Risk: a cache, two producers, one book.

Everything runs on a ``ManualClock`` and a ``RecordingMetrics``, so freshness and the emitted
series are assertions rather than observations.
"""

from __future__ import annotations

from dataclasses import replace
from datetime import timedelta

import pytest

from tests.risk.builders import (
    NOW,
    make_bumps,
    make_cache,
    make_calibrated_surface,
    make_policy,
    make_portfolio,
)
from tests.support import RecordingMetrics
from volengine.contracts.calibrated_surface import CalibratedSurface
from volengine.platform.clock import ManualClock
from volengine.risk.application.compare_producers import CompareProducersUseCase
from volengine.risk.application.compute_report import ComputeReportUseCase, ReportSettings
from volengine.risk.application.surface_cache import LastValueSurfaceProvider, ProducerSurfaces
from volengine.risk.domain.freshness_policy import FreshnessDecision

MARKET = "BTC-DERIBIT"
BASELINE = "svi-scipy"
CHALLENGER = "mlp-torch"


def shifted(surface: CalibratedSurface, vol_shift: float) -> CalibratedSurface:
    """The same surface with every volatility moved by ``vol_shift``: a known disagreement."""
    vols = tuple(tuple(vol + vol_shift for vol in row) for row in surface.grid.vols)
    return replace(surface, grid=replace(surface.grid, vols=vols))


def make_use_case(
    now: float = 2.0,
) -> tuple[CompareProducersUseCase, LastValueSurfaceProvider, RecordingMetrics]:
    clock = ManualClock(NOW + timedelta(seconds=now))
    metrics = RecordingMetrics()
    cache = make_cache(clock=clock)
    use_case = CompareProducersUseCase(
        cache=cache,
        portfolio=make_portfolio(),
        policy=make_policy(),
        clock=clock,
        metrics=metrics,
        settings=ReportSettings(bumps=make_bumps()),
        baseline_producer_id=BASELINE,
        challenger_producer_id=CHALLENGER,
    )
    return use_case, cache, metrics


def test_both_producers_value_the_whole_book() -> None:
    use_case, cache, _ = make_use_case()
    cache.accept(make_calibrated_surface(producer_id=BASELINE))
    cache.accept(shifted(make_calibrated_surface(producer_id=CHALLENGER), 0.01))

    comparison = use_case.compute(MARKET)

    assert comparison.comparable
    assert comparison.baseline.producer_id == BASELINE
    assert comparison.challenger.producer_id == CHALLENGER
    assert len(comparison.lines) == len(make_portfolio().positions)


def test_each_side_is_the_report_that_producer_would_have_had_alone() -> None:
    """Nothing about valuation exists twice: the comparison is two ordinary reports, paired."""
    use_case, cache, _ = make_use_case()
    cache.accept(make_calibrated_surface(producer_id=BASELINE))
    cache.accept(shifted(make_calibrated_surface(producer_id=CHALLENGER), 0.01))
    alone = ComputeReportUseCase(
        provider=ProducerSurfaces(cache, CHALLENGER),
        portfolio=make_portfolio(),
        policy=make_policy(),
        clock=ManualClock(NOW + timedelta(seconds=2)),
        metrics=RecordingMetrics(),
        settings=ReportSettings(bumps=make_bumps()),
        expected_producer_id=CHALLENGER,
    ).compute(MARKET)

    assert use_case.compute(MARKET).challenger == alone


def test_a_higher_vol_surface_values_a_long_option_book_line_higher() -> None:
    """The sign convention, against a disagreement whose direction is known in advance."""
    use_case, cache, _ = make_use_case()
    cache.accept(make_calibrated_surface(producer_id=BASELINE))
    cache.accept(shifted(make_calibrated_surface(producer_id=CHALLENGER), 0.01))

    long_call = use_case.compute(MARKET).lines[0]

    assert long_call.baseline.position.quantity > 0
    assert long_call.vol_diff == pytest.approx(0.01, rel=0.05)
    assert long_call.value_diff > 0


def test_two_identical_surfaces_differ_by_nothing() -> None:
    """The control: with no disagreement every difference and the distance are exactly zero."""
    use_case, cache, _ = make_use_case()
    cache.accept(make_calibrated_surface(producer_id=BASELINE))
    cache.accept(make_calibrated_surface(producer_id=CHALLENGER))

    comparison = use_case.compute(MARKET)

    assert comparison.total_value_diff == 0.0
    assert comparison.distance is not None
    assert comparison.distance.rms_vol_bp == 0.0


def test_the_distance_between_producers_is_emitted_in_basis_points() -> None:
    use_case, cache, metrics = make_use_case()
    cache.accept(make_calibrated_surface(producer_id=BASELINE))
    cache.accept(shifted(make_calibrated_surface(producer_id=CHALLENGER), 0.01))

    use_case.compute(MARKET)

    assert metrics.gauge_value("risk.comparison.distance_rms_vol_bp") == pytest.approx(100.0)
    assert metrics.gauge_value("risk.comparison.distance_max_vol_bp") == pytest.approx(100.0)
    [(_, _, tags)] = [g for g in metrics.gauges if g[0] == "risk.comparison.distance_rms_vol_bp"]
    assert tags == {"market": MARKET, "baseline": BASELINE, "challenger": CHALLENGER}


def test_the_total_value_difference_is_emitted() -> None:
    use_case, cache, metrics = make_use_case()
    cache.accept(make_calibrated_surface(producer_id=BASELINE))
    cache.accept(shifted(make_calibrated_surface(producer_id=CHALLENGER), 0.01))

    comparison = use_case.compute(MARKET)

    assert metrics.gauge_value("risk.comparison.value_diff") == comparison.total_value_diff
    assert comparison.total_value_diff != 0.0


def test_a_producer_that_has_published_nothing_makes_the_comparison_incomparable() -> None:
    use_case, cache, metrics = make_use_case()
    cache.accept(make_calibrated_surface(producer_id=BASELINE))

    comparison = use_case.compute(MARKET)

    assert not comparison.comparable
    assert comparison.challenger.freshness is FreshnessDecision.REJECT
    assert comparison.challenger.producer_id == CHALLENGER
    assert comparison.distance is None
    assert "risk.comparison.incomparable" in metrics.counter_names()


def test_a_stale_side_is_not_compared_against_a_fresh_one() -> None:
    """A model disagreement measured against an obsolete market would mostly measure the market."""
    use_case, cache, metrics = make_use_case(now=60.0)
    cache.accept(make_calibrated_surface(producer_id=BASELINE))
    cache.accept(
        make_calibrated_surface(producer_id=CHALLENGER, ts_snapshot=NOW + timedelta(seconds=59))
    )

    comparison = use_case.compute(MARKET)

    assert comparison.baseline.freshness is FreshnessDecision.REJECT
    assert comparison.challenger.freshness is FreshnessDecision.NORMAL
    assert comparison.lines == ()
    assert all(not name.startswith("risk.comparison.distance") for name, _, _ in metrics.gauges)


def test_a_producer_cannot_be_compared_with_itself() -> None:
    with pytest.raises(ValueError, match="two different producers"):
        CompareProducersUseCase(
            cache=make_cache(),
            portfolio=make_portfolio(),
            policy=make_policy(),
            clock=ManualClock(NOW),
            metrics=RecordingMetrics(),
            settings=ReportSettings(bumps=make_bumps()),
            baseline_producer_id=BASELINE,
            challenger_producer_id=BASELINE,
        )
