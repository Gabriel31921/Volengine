"""The translation a snapshot of premiums makes into a problem in volatility space.

The inbound half is where this context's most consequential rule lives -- one volatility per
strike, inverted from the out-of-the-money leg -- so most of what follows is about which quote
was chosen and what it ended up weighing. The tests can be that direct because the builder prices
its chain *forwards* from a known SVI slice: every inversion below has a right answer that was
put there on purpose, rather than a plausible number nobody can check.
"""

from __future__ import annotations

import math
from dataclasses import replace

import pytest

from tests.parametric_pricing.builders import (
    FAR,
    FAR_TENOR,
    FORWARD,
    NEAR,
    NEAR_TENOR,
    NOW,
    SNAPSHOT_MONEYNESS,
    make_calibration_task,
    make_grid_spec,
    make_market_snapshot,
    make_params,
    make_slice_result,
    make_weighting,
)
from volengine.contracts.calibrated_surface import SurfaceStatus
from volengine.contracts.market_snapshot import MarketSnapshot, OptionKind, QuoteFlag
from volengine.parametric_pricing.application.acl import (
    Weighting,
    as_stale_republish,
    to_calibrated_surface,
    to_calibration_task,
)
from volengine.parametric_pricing.domain.black76 import OptionKindP, implied_vol
from volengine.parametric_pricing.domain.calibration import SliceTask


def only_slice_task(
    both_sides: bool = True,
    spread_rel: float = 0.02,
    log_moneyness: tuple[float, ...] = SNAPSHOT_MONEYNESS,
) -> SliceTask:
    """The near slice of a one-expiry snapshot, which is what most tests below look at."""
    snapshot = make_market_snapshot(
        tenors=((NEAR, NEAR_TENOR),),
        both_sides=both_sides,
        spread_rel=spread_rel,
        log_moneyness=log_moneyness,
    )
    task = to_calibration_task(snapshot, make_weighting())
    assert task is not None
    return task.slices[0]


def replace_otm(snapshot: MarketSnapshot, k: float, **changes: object) -> MarketSnapshot:
    """Change the out-of-the-money leg at one moneyness, leaving its twin untouched.

    Poisoning ``quotes[0]`` would not do: the builder emits a call and a put at every strike and
    the ACL deliberately ignores whichever of them is in the money, so a test that bent the wrong
    leg would assert that nothing happened and pass for the wrong reason.
    """
    strike = FORWARD * math.exp(k)
    wanted = OptionKind.CALL if strike >= FORWARD else OptionKind.PUT
    quotes = tuple(
        replace(quote, **changes)  # type: ignore[arg-type]
        if quote.kind is wanted and math.isclose(quote.strike, strike)
        else quote
        for quote in snapshot.slices[0].quotes
    )
    return replace(snapshot, slices=(replace(snapshot.slices[0], quotes=quotes),))


# --- the inversion


def test_the_inverted_volatility_is_the_one_the_chain_was_priced_from() -> None:
    """The whole rule rests on this: our own Black-76 vol, recovered exactly."""
    params = make_params()
    snapshot = make_market_snapshot(params=params, tenors=((NEAR, NEAR_TENOR),))

    task = to_calibration_task(snapshot, make_weighting())

    assert task is not None
    expected = [params.implied_vol(k, NEAR_TENOR) for k in SNAPSHOT_MONEYNESS]
    assert task.slices[0].implied_vol == pytest.approx(expected, abs=1e-6)


def test_the_venues_own_implied_vol_is_never_used() -> None:
    """An exchange's IV is the output of the exchange's model, and fitting to it fits theirs."""
    snapshot = make_market_snapshot(tenors=((NEAR, NEAR_TENOR),))
    poisoned = replace(
        snapshot,
        slices=(
            replace(
                snapshot.slices[0],
                quotes=tuple(
                    replace(quote, exchange_iv=9.99) for quote in snapshot.slices[0].quotes
                ),
            ),
        ),
    )

    task = to_calibration_task(poisoned, make_weighting())

    assert task is not None
    assert all(vol < 2.0 for vol in task.slices[0].implied_vol)


def test_one_volatility_per_strike_even_though_both_legs_were_quoted() -> None:
    """Two entries at one ``k`` would be one strike counted twice in the loss."""
    task_slice = only_slice_task()

    assert len(task_slice.log_moneyness) == len(SNAPSHOT_MONEYNESS)


def test_the_moneyness_axis_is_strictly_increasing() -> None:
    task_slice = only_slice_task()
    axis = task_slice.log_moneyness

    assert list(axis) == sorted(axis)


def test_a_quote_whose_price_admits_no_volatility_is_dropped() -> None:
    """A mid above the no-arbitrage ceiling admits no volatility at all; the quote goes.

    Above rather than below, because it is the bound a quote can be pushed past from either side
    of the forward: a put's ceiling is its strike and a call's is the forward, while the floor of
    an out-of-the-money option is zero and ``QuoteData`` already refuses a non-positive mid.
    """
    snapshot = make_market_snapshot(tenors=((NEAR, NEAR_TENOR),))
    broken = replace_otm(snapshot, SNAPSHOT_MONEYNESS[0], mid=FORWARD * 10)

    task = to_calibration_task(broken, make_weighting())

    assert task is not None
    assert len(task.slices[0].log_moneyness) == len(SNAPSHOT_MONEYNESS) - 1


def test_a_snapshot_with_nothing_invertible_produces_no_task() -> None:
    """``CalibrationTask`` refuses to hold no slices, and rightly: it is not a problem."""
    snapshot = make_market_snapshot(tenors=((NEAR, NEAR_TENOR),))
    dead = replace(
        snapshot,
        slices=(
            replace(
                snapshot.slices[0],
                quotes=tuple(
                    replace(quote, mid=FORWARD * 10) for quote in snapshot.slices[0].quotes
                ),
            ),
        ),
    )

    assert to_calibration_task(dead, make_weighting()) is None


def test_an_expiry_with_nothing_invertible_is_dropped_without_the_others() -> None:
    snapshot = make_market_snapshot()
    dead_near = replace(
        snapshot,
        slices=(
            replace(
                snapshot.slices[0],
                quotes=tuple(
                    replace(quote, mid=FORWARD * 10) for quote in snapshot.slices[0].quotes
                ),
            ),
            snapshot.slices[1],
        ),
    )

    task = to_calibration_task(dead_near, make_weighting())

    assert task is not None
    assert [one.expiry for one in task.slices] == [FAR]


# --- choosing the out-of-the-money leg


def test_a_strike_above_the_forward_is_inverted_from_its_call() -> None:
    """Above the forward the call is the leg with real time value in it."""
    snapshot = make_market_snapshot(
        tenors=((NEAR, NEAR_TENOR),), log_moneyness=(0.30,), both_sides=True
    )
    call = next(q for q in snapshot.slices[0].quotes if q.kind is OptionKind.CALL)

    task = to_calibration_task(snapshot, make_weighting())

    assert task is not None
    expected = implied_vol(
        target_price=call.mid,
        forward=FORWARD,
        strike=call.strike,
        tenor_years=NEAR_TENOR,
        kind=OptionKindP.CALL,
    )
    assert task.slices[0].implied_vol[0] == pytest.approx(expected)


def test_a_strike_below_the_forward_is_inverted_from_its_put() -> None:
    snapshot = make_market_snapshot(
        tenors=((NEAR, NEAR_TENOR),), log_moneyness=(-0.30,), both_sides=True
    )
    put = next(q for q in snapshot.slices[0].quotes if q.kind is OptionKind.PUT)

    task = to_calibration_task(snapshot, make_weighting())

    assert task is not None
    expected = implied_vol(
        target_price=put.mid,
        forward=FORWARD,
        strike=put.strike,
        tenor_years=NEAR_TENOR,
        kind=OptionKindP.PUT,
    )
    assert task.slices[0].implied_vol[0] == pytest.approx(expected)


def test_the_two_legs_of_one_strike_imply_the_same_volatility() -> None:
    """Put-call parity, and the reason nothing is lost by preferring one leg over the other.

    If this failed, the rule would be discarding information rather than choosing the readable
    copy of it -- so it is worth pinning independently of which leg the ACL happens to pick.
    """
    snapshot = make_market_snapshot(
        tenors=((NEAR, NEAR_TENOR),), log_moneyness=(0.15,), both_sides=True
    )
    call = next(q for q in snapshot.slices[0].quotes if q.kind is OptionKind.CALL)
    put = next(q for q in snapshot.slices[0].quotes if q.kind is OptionKind.PUT)

    from_call = implied_vol(call.mid, FORWARD, call.strike, NEAR_TENOR, OptionKindP.CALL)
    from_put = implied_vol(put.mid, FORWARD, put.strike, NEAR_TENOR, OptionKindP.PUT)

    assert from_call == pytest.approx(from_put, abs=1e-6)


def test_the_at_the_money_strike_has_a_rule_rather_than_a_coin_flip() -> None:
    """At ``k = 0`` both legs are equivalent, so what matters is only that the rule is fixed."""
    snapshot = make_market_snapshot(
        tenors=((NEAR, NEAR_TENOR),), log_moneyness=(0.0,), both_sides=True
    )

    first = to_calibration_task(snapshot, make_weighting())
    second = to_calibration_task(snapshot, make_weighting())

    assert first is not None and second is not None
    assert first.slices[0].implied_vol == second.slices[0].implied_vol


# --- the unpaired in-the-money leg


def test_an_in_the_money_leg_with_no_twin_is_still_inverted() -> None:
    """Dropping it would leave the fit blind exactly where the market is thinnest."""
    snapshot = make_market_snapshot(tenors=((NEAR, NEAR_TENOR),), both_sides=False)

    task = to_calibration_task(snapshot, make_weighting())

    assert task is not None
    assert len(task.slices[0].log_moneyness) == len(SNAPSHOT_MONEYNESS)


def test_a_uniform_discount_vanishes_under_the_per_slice_normalisation() -> None:
    """Worth stating, because it is why the next test has to build a *mixed* slice.

    The weights are normalised to sum to one within each slice, so multiplying every quote in a
    slice by the same factor changes nothing. A discount is only ever observable as a ratio
    against a quote that kept its full weight.
    """
    paired = only_slice_task(both_sides=True)
    unpaired = only_slice_task(both_sides=False)

    assert paired.weights == pytest.approx(unpaired.weights)


def test_the_discount_shows_when_only_some_strikes_lost_their_twin() -> None:
    """A mixed slice is where the down-weighting is observable, and it is the realistic case."""
    snapshot = make_market_snapshot(
        tenors=((NEAR, NEAR_TENOR),), log_moneyness=(-0.15, 0.15), both_sides=True
    )
    # Strip the out-of-the-money put at k = -0.15, leaving only its in-the-money call.
    quotes = tuple(
        quote
        for quote in snapshot.slices[0].quotes
        if not (quote.strike < FORWARD and quote.kind is OptionKind.PUT)
    )
    stripped = replace(snapshot, slices=(replace(snapshot.slices[0], quotes=quotes),))

    full = to_calibration_task(snapshot, make_weighting())
    mixed = to_calibration_task(stripped, make_weighting())

    assert full is not None and mixed is not None
    assert mixed.slices[0].weights[0] < full.slices[0].weights[0]


# --- the weights


def test_the_weights_of_a_slice_sum_to_one() -> None:
    task_slice = only_slice_task()

    assert sum(task_slice.weights) == pytest.approx(1.0)


def test_widening_every_quote_equally_changes_no_weight() -> None:
    """The same normalisation fact, on the spread half of the formula."""
    tight = only_slice_task(spread_rel=0.001)
    wide = only_slice_task(spread_rel=0.400)

    assert tight.weights == pytest.approx(wide.weights)


def test_a_wide_quote_beside_tight_ones_loses_influence() -> None:
    snapshot = make_market_snapshot(
        tenors=((NEAR, NEAR_TENOR),), log_moneyness=(-0.15, 0.15), spread_rel=0.001
    )
    widened = replace_otm(snapshot, -0.15, spread_rel=0.5)

    even = to_calibration_task(snapshot, make_weighting())
    lopsided = to_calibration_task(widened, make_weighting())

    assert even is not None and lopsided is not None
    assert lopsided.slices[0].weights[0] < even.slices[0].weights[0]


def test_a_flagged_quote_counts_for_less_than_its_clean_neighbours() -> None:
    """Ingestion flags, the calibrator weights -- and this is the weighting half."""
    snapshot = make_market_snapshot(tenors=((NEAR, NEAR_TENOR),), log_moneyness=(-0.15, 0.15))
    flagged = replace_otm(snapshot, -0.15, flags=(QuoteFlag.STALE,))

    clean = to_calibration_task(snapshot, make_weighting())
    marked = to_calibration_task(flagged, make_weighting())

    assert clean is not None and marked is not None
    assert marked.slices[0].weights[0] < clean.slices[0].weights[0]


def test_a_zero_flagged_factor_keeps_the_quote_and_removes_its_influence() -> None:
    """A zero weight is how a point stays visible in the residuals without steering a parameter."""
    snapshot = make_market_snapshot(tenors=((NEAR, NEAR_TENOR),), log_moneyness=(-0.15, 0.15))
    flagged = replace_otm(snapshot, -0.15, flags=(QuoteFlag.STALE,))

    task = to_calibration_task(
        flagged, Weighting(spread_scale=0.05, flagged_factor=0.0, unpaired_itm_factor=0.1)
    )

    assert task is not None
    assert task.slices[0].weights[0] == 0.0
    assert len(task.slices[0].log_moneyness) == 2


def test_a_slice_whose_every_quote_was_zeroed_is_dropped() -> None:
    """Legal configuration meeting a legal market: not a reason to fail the whole snapshot."""
    snapshot = make_market_snapshot(flags=(QuoteFlag.STALE,))

    task = to_calibration_task(
        snapshot, Weighting(spread_scale=0.05, flagged_factor=0.0, unpaired_itm_factor=0.1)
    )

    assert task is None


# --- the weighting configuration


@pytest.mark.parametrize("bad", [0.0, -1.0, float("nan"), float("inf")])
def test_an_unusable_spread_scale_is_refused(bad: float) -> None:
    with pytest.raises(ValueError, match="spread scale"):
        Weighting(spread_scale=bad, flagged_factor=0.5, unpaired_itm_factor=0.1)


@pytest.mark.parametrize("bad", [-0.1, 1.5, float("nan")])
def test_a_factor_outside_the_unit_interval_is_refused(bad: float) -> None:
    with pytest.raises(ValueError, match="factor"):
        Weighting(spread_scale=0.05, flagged_factor=bad, unpaired_itm_factor=0.1)


# --- the outbound half


def test_the_published_grid_uses_the_configured_moneyness_axis() -> None:
    task = make_calibration_task()
    grid = make_grid_spec()

    surface = to_calibrated_surface(
        task=task,
        accepted=[make_slice_result(expiry=NEAR, tenor_years=NEAR_TENOR)],
        n_iterations=3,
        duration_ms=1.0,
        grid=grid,
        producer_id="svi-stub",
        surface_id="surface-1",
        ts_calibrated=NOW,
        status=SurfaceStatus.OK,
    )

    assert surface is not None
    assert surface.grid.log_moneyness == grid.nodes()


def test_the_published_tenor_axis_is_the_accepted_slices() -> None:
    """A rejected expiry leaves a hole in the term structure rather than a fabricated row."""
    surface = to_calibrated_surface(
        task=make_calibration_task(),
        accepted=[make_slice_result(expiry=FAR, tenor_years=FAR_TENOR)],
        n_iterations=3,
        duration_ms=1.0,
        grid=make_grid_spec(),
        producer_id="svi-stub",
        surface_id="surface-1",
        ts_calibrated=NOW,
        status=SurfaceStatus.OK,
    )

    assert surface is not None
    assert surface.grid.expiries == (FAR,)


def test_the_forward_travels_with_the_published_grid() -> None:
    """A moneyness axis with no forward is a coordinate system with no origin."""
    surface = to_calibrated_surface(
        task=make_calibration_task(),
        accepted=[make_slice_result(expiry=NEAR, tenor_years=NEAR_TENOR)],
        n_iterations=3,
        duration_ms=1.0,
        grid=make_grid_spec(),
        producer_id="svi-stub",
        surface_id="surface-1",
        ts_calibrated=NOW,
        status=SurfaceStatus.OK,
    )

    assert surface is not None
    assert surface.grid.forwards == (FORWARD,)


def test_the_published_vols_are_the_parameters_evaluated_on_the_mesh() -> None:
    params = make_params()
    grid = make_grid_spec()

    surface = to_calibrated_surface(
        task=make_calibration_task(),
        accepted=[make_slice_result(expiry=NEAR, tenor_years=NEAR_TENOR, params=params)],
        n_iterations=3,
        duration_ms=1.0,
        grid=grid,
        producer_id="svi-stub",
        surface_id="surface-1",
        ts_calibrated=NOW,
        status=SurfaceStatus.OK,
    )

    assert surface is not None
    expected = [params.implied_vol(k, NEAR_TENOR) for k in grid.nodes()]
    assert list(surface.grid.vols[0]) == pytest.approx(expected)


def test_a_slice_that_collapses_to_zero_variance_is_not_published() -> None:
    """``SVIParams`` admits it so the metrics can measure it; ``VolGrid`` cannot carry it."""
    collapsed = make_params(a=0.0, b=0.0, rho=0.0, m=0.0, sigma=0.20)

    surface = to_calibrated_surface(
        task=make_calibration_task(),
        accepted=[make_slice_result(expiry=NEAR, tenor_years=NEAR_TENOR, params=collapsed)],
        n_iterations=3,
        duration_ms=1.0,
        grid=make_grid_spec(),
        producer_id="svi-stub",
        surface_id="surface-1",
        ts_calibrated=NOW,
        status=SurfaceStatus.OK,
    )

    assert surface is None


def test_the_pooled_rmse_weights_the_slices_by_their_quote_counts() -> None:
    """Not the mean of the slices' own RMSEs: a thin slice must not outvote a hundred-quote one."""
    thin = make_slice_result(
        expiry=NEAR,
        tenor_years=NEAR_TENOR,
        rmse_vol_bp=100.0,
        max_err_vol_bp=200.0,
        n_quotes_used=1,
    )
    fat = make_slice_result(
        expiry=FAR,
        tenor_years=FAR_TENOR,
        rmse_vol_bp=10.0,
        max_err_vol_bp=20.0,
        n_quotes_used=99,
    )

    surface = to_calibrated_surface(
        task=make_calibration_task(),
        accepted=[thin, fat],
        n_iterations=3,
        duration_ms=1.0,
        grid=make_grid_spec(),
        producer_id="svi-stub",
        surface_id="surface-1",
        ts_calibrated=NOW,
        status=SurfaceStatus.OK,
    )

    assert surface is not None
    pooled = math.sqrt((1 * 100.0**2 + 99 * 10.0**2) / 100)
    assert surface.fit.rmse_vol_bp == pytest.approx(pooled)
    assert surface.fit.max_err_vol_bp == 200.0
    assert surface.fit.n_quotes_used == 100


# --- the stale republish


def test_a_republished_surface_keeps_its_original_snapshot_instant() -> None:
    """Restamping it would make a surface from ten minutes ago look current (ADR-006)."""
    surface = to_calibrated_surface(
        task=make_calibration_task(),
        accepted=[make_slice_result(expiry=NEAR, tenor_years=NEAR_TENOR)],
        n_iterations=3,
        duration_ms=1.0,
        grid=make_grid_spec(),
        producer_id="svi-stub",
        surface_id="surface-1",
        ts_calibrated=NOW,
        status=SurfaceStatus.OK,
    )

    assert surface is not None
    stale = as_stale_republish(surface)

    assert stale.ts_snapshot == surface.ts_snapshot
    assert stale.ts_calibrated == surface.ts_calibrated
    assert stale.surface_id == surface.surface_id
    assert stale.status is SurfaceStatus.STALE_REPUBLISH
