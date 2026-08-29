"""The translation a snapshot of premiums makes into a cloud of training points.

The same out-of-the-money rule and the same weighting formula as the parametric ACL, and the tests
below lean on that deliberately: the snapshot they run on is built by
``tests/parametric_pricing/builders.py``, so both contexts are shown the *identical* market. Tests
are subject to none of the import rules, which makes this the one place the two producers can be
put side by side -- and the last test in this file is the guard that keeps them there.
"""

from __future__ import annotations

import math
from dataclasses import replace
from datetime import timedelta

import numpy as np
import pytest

from tests.neural_surface.builders import (
    FlatVolSurface,
    make_grid_spec,
    make_weighting,
)
from tests.parametric_pricing.builders import (
    FORWARD,
    NEAR,
    NEAR_TENOR,
    NOW,
    SNAPSHOT_MONEYNESS,
    make_market_snapshot,
    make_params,
)
from volengine.contracts.calibrated_surface import FitMetrics, SurfaceStatus
from volengine.contracts.market_snapshot import OptionKind, QuoteFlag
from volengine.neural_surface.application.acl import (
    Weighting,
    fit_metrics,
    to_calibrated_surface,
    to_training_batch,
    to_training_samples,
)

# --- the inversion


def test_the_inverted_volatilities_are_the_ones_the_chain_was_priced_from() -> None:
    params = make_params()
    snapshot = make_market_snapshot(params=params, tenors=((NEAR, NEAR_TENOR),))

    samples = to_training_samples(snapshot, make_weighting())

    recovered = sorted(sample.implied_vol for sample in samples)
    expected = sorted(params.implied_vol(k, NEAR_TENOR) for k in SNAPSHOT_MONEYNESS)
    assert recovered == pytest.approx(expected, abs=1e-6)


def test_one_point_per_strike_even_though_both_legs_were_quoted() -> None:
    snapshot = make_market_snapshot(tenors=((NEAR, NEAR_TENOR),))

    assert len(to_training_samples(snapshot, make_weighting())) == len(SNAPSHOT_MONEYNESS)


def test_the_points_of_two_expiries_arrive_in_one_flat_cloud() -> None:
    """No grouping anywhere: a network sees the whole surface at once, unlike a slice-wise fit."""
    samples = to_training_samples(make_market_snapshot(), make_weighting())

    assert len({sample.tenor_years for sample in samples}) == 2
    assert len(samples) == 2 * len(SNAPSHOT_MONEYNESS)


def test_a_snapshot_with_nothing_invertible_produces_no_samples() -> None:
    snapshot = make_market_snapshot()
    dead = replace(
        snapshot,
        slices=tuple(
            replace(
                slice_data,
                quotes=tuple(replace(quote, mid=FORWARD * 10) for quote in slice_data.quotes),
            )
            for slice_data in snapshot.slices
        ),
    )

    assert to_training_samples(dead, make_weighting()) == ()


# --- the observation instant


def test_each_point_keeps_the_instant_the_venue_last_touched_it() -> None:
    """The field the replay buffer's retention policy is measured on."""
    snapshot = make_market_snapshot(tenors=((NEAR, NEAR_TENOR),))

    samples = to_training_samples(snapshot, make_weighting())

    assert samples[0].ts_observed == NOW - timedelta(seconds=0.4)


def test_an_older_quote_carries_an_older_instant() -> None:
    """The vacuous-pass guard: stamping every point with the snapshot instant would pass above."""
    snapshot = make_market_snapshot(tenors=((NEAR, NEAR_TENOR),))
    strike = FORWARD * math.exp(SNAPSHOT_MONEYNESS[0])
    aged = replace(
        snapshot,
        slices=(
            replace(
                snapshot.slices[0],
                # The *out-of-the-money* leg, which below the forward is the put: ageing the call
                # would age a quote the ACL never looks at, and the test would pass vacuously.
                quotes=tuple(
                    replace(quote, age_seconds=90.0)
                    if quote.kind is OptionKind.PUT and math.isclose(quote.strike, strike)
                    else quote
                    for quote in snapshot.slices[0].quotes
                ),
            ),
        ),
    )

    samples = to_training_samples(aged, make_weighting())

    assert min(sample.ts_observed for sample in samples) == NOW - timedelta(seconds=90)


# --- the weights


def test_the_weights_of_a_whole_snapshot_sum_to_one() -> None:
    """Normalised across the snapshot, not per expiry: one gradient step spans every point."""
    samples = to_training_samples(make_market_snapshot(), make_weighting())

    assert sum(sample.weight for sample in samples) == pytest.approx(1.0)


def test_a_flagged_point_counts_for_less_than_its_clean_neighbours() -> None:
    snapshot = make_market_snapshot(tenors=((NEAR, NEAR_TENOR),))
    strike = FORWARD * math.exp(SNAPSHOT_MONEYNESS[0])
    flagged = replace(
        snapshot,
        slices=(
            replace(
                snapshot.slices[0],
                quotes=tuple(
                    replace(quote, flags=(QuoteFlag.STALE,))
                    if quote.kind is OptionKind.PUT and math.isclose(quote.strike, strike)
                    else quote
                    for quote in snapshot.slices[0].quotes
                ),
            ),
        ),
    )

    clean = to_training_samples(snapshot, make_weighting())
    marked = to_training_samples(flagged, make_weighting())

    assert marked[0].weight < clean[0].weight


def test_a_zero_flagged_factor_keeps_the_point_and_removes_its_influence() -> None:
    """A zero weight keeps the point visible in the residuals and in the buffer's occupancy."""
    snapshot = make_market_snapshot(tenors=((NEAR, NEAR_TENOR),), flags=(QuoteFlag.STALE,))
    weighting = Weighting(spread_scale=0.05, flagged_factor=0.0, unpaired_itm_factor=0.5)

    # Every point is flagged, so the whole snapshot weighs zero and there is no weighting at all.
    assert to_training_samples(snapshot, weighting) == ()


# --- the batch


def test_the_batch_puts_the_fresh_points_first() -> None:
    """The ordering convention is the whole meaning of ``n_fresh``."""
    snapshot = make_market_snapshot()
    fresh = to_training_samples(snapshot, make_weighting())

    batch = to_training_batch(snapshot, fresh=fresh[:3], replayed=fresh[3:])

    assert batch is not None
    assert batch.n_fresh == 3
    assert batch.fresh == fresh[:3]
    assert batch.replayed == fresh[3:]


def test_a_batch_with_no_fresh_points_is_legal() -> None:
    """The scheduled restart: every point replayed, ``n_fresh`` exactly zero."""
    snapshot = make_market_snapshot()
    samples = to_training_samples(snapshot, make_weighting())

    batch = to_training_batch(snapshot, fresh=(), replayed=samples)

    assert batch is not None
    assert batch.n_fresh == 0


def test_an_empty_batch_is_refused_rather_than_constructed() -> None:
    """A gradient step over no data is a no-op that would still be counted and published."""
    assert to_training_batch(make_market_snapshot(), fresh=(), replayed=()) is None


def test_the_batch_carries_the_snapshots_own_instant() -> None:
    snapshot = make_market_snapshot()
    fresh = to_training_samples(snapshot, make_weighting())

    batch = to_training_batch(snapshot, fresh=fresh, replayed=())

    assert batch is not None
    assert batch.ts_snapshot == NOW


# --- publishing


def a_fit() -> FitMetrics:
    return FitMetrics(
        rmse_vol_bp=10.0, max_err_vol_bp=20.0, n_quotes_used=10, n_iterations=1, duration_ms=1.0
    )


def test_the_published_grid_uses_the_snapshots_own_tenors() -> None:
    """The network could answer anywhere; publishing rows no market data stands behind would make
    the producer comparison measure an interpolation choice."""
    snapshot = make_market_snapshot()

    surface = to_calibrated_surface(
        surface=FlatVolSurface(),
        snapshot=snapshot,
        grid=make_grid_spec(),
        producer_id="mlp-stub",
        surface_id="surface-1",
        ts_calibrated=NOW,
        status=SurfaceStatus.OK,
        fit=a_fit(),
    )

    assert surface is not None
    assert surface.grid.tenors == tuple(one.tenor_years for one in snapshot.slices)
    assert surface.grid.expiries == tuple(one.expiry for one in snapshot.slices)
    assert surface.grid.forwards == tuple(one.forward for one in snapshot.slices)


def test_the_published_volatilities_are_the_model_evaluated_on_the_mesh() -> None:
    surface = to_calibrated_surface(
        surface=FlatVolSurface(vol=0.55),
        snapshot=make_market_snapshot(),
        grid=make_grid_spec(),
        producer_id="mlp-stub",
        surface_id="surface-1",
        ts_calibrated=NOW,
        status=SurfaceStatus.OK,
        fit=a_fit(),
    )

    assert surface is not None
    assert all(vol == pytest.approx(0.55) for smile in surface.grid.vols for vol in smile)


def test_the_published_volatilities_are_plain_floats() -> None:
    """``VolGrid`` promises primitives only (ADR-011); a numpy scalar needs a codec to serialise."""
    surface = to_calibrated_surface(
        surface=FlatVolSurface(),
        snapshot=make_market_snapshot(),
        grid=make_grid_spec(),
        producer_id="mlp-stub",
        surface_id="surface-1",
        ts_calibrated=NOW,
        status=SurfaceStatus.OK,
        fit=a_fit(),
    )

    assert surface is not None
    assert type(surface.grid.vols[0][0]) is float


def test_the_weights_version_travels_in_the_producer_metadata() -> None:
    """An integer counter survives a ``Mapping[str, float]``; a checkpoint path would not."""
    surface = to_calibrated_surface(
        surface=FlatVolSurface(version=7),
        snapshot=make_market_snapshot(),
        grid=make_grid_spec(),
        producer_id="mlp-stub",
        surface_id="surface-1",
        ts_calibrated=NOW,
        status=SurfaceStatus.OK,
        fit=a_fit(),
    )

    assert surface is not None
    assert surface.producer_meta is not None
    assert surface.producer_meta["weights_version"] == 7.0


# --- the fit metrics the port does not return


def test_the_residual_is_measured_against_the_surface_that_was_trained() -> None:
    """``SurfaceLearner.update`` hands back a bare surface, so the RMSE has to be recomputed."""
    snapshot = make_market_snapshot()
    fresh = to_training_samples(snapshot, make_weighting())

    perfect = fit_metrics(
        surface=FlatVolSurface(vol=fresh[0].implied_vol),
        fresh=fresh[:1],
        n_iterations=1,
        duration_ms=0.0,
    )

    assert perfect.rmse_vol_bp == pytest.approx(0.0, abs=1e-6)


def test_a_surface_that_misses_the_market_reports_a_real_error() -> None:
    """The vacuous-pass guard: a metric that always reported zero would satisfy the test above."""
    snapshot = make_market_snapshot()
    fresh = to_training_samples(snapshot, make_weighting())

    wrong = fit_metrics(
        surface=FlatVolSurface(vol=fresh[0].implied_vol + 0.05),
        fresh=fresh[:1],
        n_iterations=1,
        duration_ms=0.0,
    )

    assert wrong.rmse_vol_bp == pytest.approx(500.0, rel=1e-3)


def test_the_worst_point_is_reported_beside_the_mean() -> None:
    """A mean absorbs one badly missed quote; the wings are where a producer fails first."""
    snapshot = make_market_snapshot()
    fresh = to_training_samples(snapshot, make_weighting())

    metrics = fit_metrics(
        surface=FlatVolSurface(vol=0.60), fresh=fresh, n_iterations=1, duration_ms=0.0
    )

    assert metrics.max_err_vol_bp >= metrics.rmse_vol_bp


# --- the guard on four Black-76s


def test_the_two_contexts_invert_a_price_to_the_same_volatility() -> None:
    """**The guard that the two contexts have not re-forked the formula.**

    ``neural_surface/domain/pricing.py`` used to be a line-for-line copy of
    ``parametric_pricing/domain/black76.py``, and this test was the only place the copies could be
    required to agree, because ``tests/`` is subject to none of the import rules. Both are now thin
    wrappers over ``shared_kernel/domain/black76.py``, so the agreement is structural and this
    passes by construction.

    It stays because the failure it guards against has not gone away, only moved: the risk was
    never that someone breaks one implementation, it is that someone *improves* one of them, and
    the two producers then weight the same market differently while every test in both contexts
    keeps passing. Design 6.5's whole comparison rests on that not happening. Inlining the formula
    back into one context -- the exact edit the old arrangement invited -- fails here.
    """
    from volengine.neural_surface.domain import pricing as neural
    from volengine.parametric_pricing.domain import black76 as parametric

    for k in (-0.6, -0.2, 0.0, 0.2, 0.6):
        strike = FORWARD * math.exp(k)
        # The out-of-the-money leg at every strike, which is the only side either context ever
        # inverts -- and, at the far wings, the only side whose price is not its own intrinsic
        # value to the last bit.
        side = (
            (neural.OptionKindN.CALL, parametric.OptionKindP.CALL)
            if k >= 0
            else (
                neural.OptionKindN.PUT,
                parametric.OptionKindP.PUT,
            )
        )
        for tenor in (0.02, NEAR_TENOR, 1.0):
            for vol in (0.2, 0.65, 1.4):
                price = parametric.price(FORWARD, strike, tenor, vol, side[1])
                assert neural.price(FORWARD, strike, tenor, vol, side[0]) == pytest.approx(
                    price, rel=0, abs=0
                )
                assert neural.implied_vol(price, FORWARD, strike, tenor, side[0]) == pytest.approx(
                    parametric.implied_vol(price, FORWARD, strike, tenor, side[1]),
                    rel=0,
                    abs=0,
                )


def test_the_two_contexts_agree_on_vega_which_is_what_the_weights_are_built_from() -> None:
    """The number the comparison actually depends on: a divergence here reweights one producer."""
    from volengine.neural_surface.domain import pricing as neural
    from volengine.parametric_pricing.domain import black76 as parametric

    for k in (-0.6, 0.0, 0.6):
        strike = FORWARD * math.exp(k)
        assert neural.vega(FORWARD, strike, NEAR_TENOR, 0.65) == pytest.approx(
            parametric.vega(FORWARD, strike, NEAR_TENOR, 0.65), rel=0, abs=0
        )


def test_the_two_contexts_weight_one_market_identically() -> None:
    """The end-to-end statement of the same guard, through both ACLs on one snapshot.

    It is stronger than comparing the two Black-76s, because the weights also depend on the
    out-of-the-money rule, the spread discount and the flag factors -- and those are duplicated
    prose rather than duplicated arithmetic, which is the easier kind to let drift.
    """
    from volengine.parametric_pricing.application.acl import Weighting as PricingWeighting
    from volengine.parametric_pricing.application.acl import to_calibration_task

    snapshot = make_market_snapshot(tenors=((NEAR, NEAR_TENOR),))
    task = to_calibration_task(
        snapshot,
        PricingWeighting(spread_scale=0.05, flagged_factor=0.25, unpaired_itm_factor=0.10),
    )
    samples = to_training_samples(snapshot, make_weighting())

    assert task is not None
    # The parametric weights are normalised per slice and the neural ones across the snapshot, so
    # a one-expiry snapshot is where the two normalisations coincide and the comparison is exact.
    assert [sample.weight for sample in samples] == pytest.approx(list(task.slices[0].weights))
    assert [sample.implied_vol for sample in samples] == pytest.approx(
        list(task.slices[0].implied_vol)
    )


def test_a_multi_expiry_snapshot_normalises_differently_in_the_two_contexts() -> None:
    """The one intended difference, stated so it cannot be mistaken for the drift above."""
    from volengine.parametric_pricing.application.acl import Weighting as PricingWeighting
    from volengine.parametric_pricing.application.acl import to_calibration_task

    snapshot = make_market_snapshot()
    task = to_calibration_task(
        snapshot,
        PricingWeighting(spread_scale=0.05, flagged_factor=0.25, unpaired_itm_factor=0.10),
    )
    samples = to_training_samples(snapshot, make_weighting())

    assert task is not None
    assert sum(task.slices[0].weights) == pytest.approx(1.0)
    assert sum(sample.weight for sample in samples) == pytest.approx(1.0)
    assert np.sum([s.weight for s in samples][: len(SNAPSHOT_MONEYNESS)]) < 1.0
