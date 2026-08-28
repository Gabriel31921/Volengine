from __future__ import annotations

from dataclasses import replace
from datetime import timedelta

import numpy as np
import pytest

from tests.neural_surface.builders import (
    FAR_TENOR,
    NAIVE,
    NEAR_TENOR,
    NOW,
    make_buffer,
    make_sample,
    make_spec,
)
from tests.support import replace_field
from volengine.neural_surface.domain.errors import EmptyBufferError
from volengine.neural_surface.domain.replay_buffer import ReplayBuffer

# --- StratificationSpec: the two axes

# make_spec cuts moneyness at (-0.40, -0.10, 0.10, 0.40) and the tenor at (0.02, 0.10, 0.50),
# so the default sample -- k = -0.05 at one month -- lands in cell (1, 0), the middle moneyness
# band of the near tenor. Every expected cell index below is read off those two tuples.
ATM_CELL = (1, 0)


@pytest.mark.parametrize("field", ["moneyness_edges", "tenor_edges"])
def test_spec_rejects_an_axis_with_fewer_than_two_edges(field: str) -> None:
    """One edge bounds no cell, and a spec with nothing to stratify is not a stratification."""
    with pytest.raises(ValueError, match="at least two edges"):
        replace_field(make_spec(), field, (0.25,))


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_spec_rejects_a_non_finite_moneyness_edge(bad: float) -> None:
    with pytest.raises(ValueError, match="moneyness edges must be finite"):
        replace(make_spec(), moneyness_edges=(-0.40, bad, 0.10, 0.40))


@pytest.mark.parametrize("bad", [float("nan"), float("inf")])
def test_spec_rejects_a_non_finite_tenor_edge(bad: float) -> None:
    with pytest.raises(ValueError, match="tenor edges must be finite"):
        replace(make_spec(), tenor_edges=(0.02, bad, 0.50))


@pytest.mark.parametrize("bad", [0.0, -0.10])
def test_spec_rejects_a_non_positive_tenor_edge(bad: float) -> None:
    """A tenor a TrainingSample cannot carry bounds a cell nothing can ever reach."""
    with pytest.raises(ValueError, match="tenor edges must be positive"):
        replace(make_spec(), tenor_edges=(bad, 0.10, 0.50))


def test_spec_accepts_negative_moneyness_edges() -> None:
    """The moneyness axis is a partition of the line, not a range around the forward."""
    spec = replace(make_spec(), moneyness_edges=(-0.90, -0.60, -0.30))

    assert spec.n_cells == 2 * (len(spec.tenor_edges) - 1)


def test_spec_rejects_moneyness_edges_out_of_order() -> None:
    with pytest.raises(ValueError, match="strictly increasing"):
        replace(make_spec(), moneyness_edges=(-0.40, 0.10, -0.10, 0.40))


def test_spec_rejects_tenor_edges_out_of_order() -> None:
    with pytest.raises(ValueError, match="strictly increasing"):
        replace(make_spec(), tenor_edges=(0.50, 0.10, 0.02))


def test_spec_rejects_a_repeated_edge() -> None:
    """Strictly increasing, not merely non-decreasing: equal edges bound an unreachable cell."""
    with pytest.raises(ValueError, match="strictly increasing"):
        replace(make_spec(), moneyness_edges=(-0.40, -0.10, -0.10, 0.40))


@pytest.mark.parametrize("bad", [0, -1])
def test_spec_rejects_a_capacity_below_one(bad: int) -> None:
    with pytest.raises(ValueError, match="at least one"):
        replace(make_spec(), capacity_per_cell=bad)


@pytest.mark.parametrize("bad", [0.0, -60.0, float("nan"), float("inf")])
def test_spec_rejects_a_non_positive_max_age(bad: float) -> None:
    with pytest.raises(ValueError, match="maximum age must be positive and finite"):
        replace(make_spec(), max_age_seconds=bad)


def test_n_cells_is_the_product_of_the_two_axes() -> None:
    """Three moneyness bands crossed with two tenor bands."""
    assert make_spec().n_cells == 6


def test_n_cells_follows_a_refinement_of_one_axis() -> None:
    refined = replace(make_spec(), tenor_edges=(0.02, 0.10, 0.50, 2.00))

    assert refined.n_cells == 9


# --- StratificationSpec.cell_of: clamping and which side of an edge


def test_cell_of_places_a_sample_in_the_band_that_contains_it() -> None:
    assert make_spec().cell_of(make_sample()) == ATM_CELL


def test_cell_of_clamps_a_moneyness_below_the_first_edge_into_the_first_band() -> None:
    """A 90% down wing is beyond every configured edge, and is exactly what must be kept."""
    assert make_spec().cell_of(make_sample(log_moneyness=-2.30)) == (0, 0)


def test_cell_of_clamps_a_moneyness_above_the_last_edge_into_the_last_band() -> None:
    assert make_spec().cell_of(make_sample(log_moneyness=1.50)) == (2, 0)


def test_cell_of_clamps_a_tenor_below_the_first_edge_into_the_first_band() -> None:
    """An expiry hours away sits below the 0.02-year edge and still belongs somewhere."""
    assert make_spec().cell_of(make_sample(tenor_years=0.001)) == (1, 0)


def test_cell_of_clamps_a_tenor_above_the_last_edge_into_the_last_band() -> None:
    assert make_spec().cell_of(make_sample(tenor_years=5.0)) == (1, 1)


def test_cell_of_puts_an_interior_moneyness_edge_in_the_band_above_it() -> None:
    """Half-open upwards: -0.10 is the lower bound of band 1, not the upper bound of band 0."""
    assert make_spec().cell_of(make_sample(log_moneyness=-0.10)) == (1, 0)


def test_cell_of_puts_an_interior_tenor_edge_in_the_band_above_it() -> None:
    assert make_spec().cell_of(make_sample(tenor_years=0.10)) == (1, 1)


def test_cell_of_puts_the_outermost_lower_edge_in_the_first_band() -> None:
    assert make_spec().cell_of(make_sample(log_moneyness=-0.40)) == (0, 0)


def test_cell_of_puts_the_outermost_upper_edge_in_the_last_band() -> None:
    """The top band is closed at its upper edge, by the same clamp that catches the outliers."""
    assert make_spec().cell_of(make_sample(log_moneyness=0.40, tenor_years=0.50)) == (2, 1)


# --- Writing: placement, capacity and eviction


def test_add_places_a_sample_in_its_own_cell() -> None:
    buffer = make_buffer(samples=(make_sample(),))

    assert buffer.occupancy() == {ATM_CELL: 1}


def test_add_keeps_a_sample_beyond_the_outermost_edge_in_the_outermost_cell() -> None:
    """Clamping seen through the aggregate: the wing observation is remembered, not dropped."""
    buffer = make_buffer(samples=(make_sample(log_moneyness=-2.30),))

    assert len(buffer) == 1
    assert buffer.occupancy() == {(0, 0): 1}


def test_extend_adds_every_sample_it_is_given() -> None:
    buffer = make_buffer(
        samples=(
            make_sample(log_moneyness=-0.30),
            make_sample(log_moneyness=-0.05),
            make_sample(log_moneyness=0.25),
        )
    )

    assert len(buffer) == 3
    assert buffer.occupancy() == {(0, 0): 1, (1, 0): 1, (2, 0): 1}


def test_the_buffer_holds_two_observations_of_the_same_point() -> None:
    """The same strike seen twice is two observations, not one fact recorded twice."""
    buffer = make_buffer(
        samples=(
            make_sample(ts_observed=NOW - timedelta(minutes=2)),
            make_sample(ts_observed=NOW),
        )
    )

    assert len(buffer) == 2


def test_a_full_cell_never_grows_beyond_its_capacity() -> None:
    buffer = make_buffer(
        samples=tuple(make_sample(log_moneyness=-0.09 + 0.001 * step) for step in range(20))
    )

    assert buffer.occupancy() == {ATM_CELL: 3}


def test_a_full_cell_evicts_by_observation_time_and_not_by_arrival_order() -> None:
    """The oldest quote goes, whichever position it happens to occupy in the cell."""
    recent = make_sample(log_moneyness=-0.05, ts_observed=NOW - timedelta(minutes=1))
    oldest = make_sample(log_moneyness=0.00, ts_observed=NOW - timedelta(minutes=9))
    middle = make_sample(log_moneyness=0.05, ts_observed=NOW - timedelta(minutes=5))
    arriving = make_sample(log_moneyness=0.09, ts_observed=NOW)

    buffer = make_buffer(samples=(recent, oldest, middle, arriving))

    # The first-inserted sample survives and the second-inserted one is gone: a FIFO eviction
    # would have produced exactly the opposite set.
    assert set(buffer.snapshot()) == {recent, middle, arriving}


def test_an_arriving_sample_older_than_the_whole_cell_still_enters() -> None:
    """Arrival order is the venue's, not the market's; staleness is prune's judgement alone."""
    held = tuple(
        make_sample(log_moneyness=-0.05 + 0.01 * step, ts_observed=NOW - timedelta(minutes=step))
        for step in range(3)
    )
    backdated = make_sample(log_moneyness=0.08, ts_observed=NOW - timedelta(minutes=30))

    buffer = make_buffer(samples=(*held, backdated))

    assert backdated in buffer.snapshot()


def test_a_busy_cell_cannot_evict_a_sample_from_another_cell() -> None:
    """The whole reason for stratifying: at-the-money volume must not cost the wing its memory."""
    wing = make_sample(log_moneyness=-0.30, tenor_years=FAR_TENOR)
    ticks = tuple(
        make_sample(
            log_moneyness=-0.09 + 0.001 * step,
            ts_observed=NOW - timedelta(seconds=50 - step),
        )
        for step in range(50)
    )

    buffer = make_buffer(samples=(wing, *ticks))

    # Guard against a vacuous assertion: eviction really did run, so "the wing survived" is not
    # merely "no cell ever filled up". Forty-seven of the fifty ticks were displaced.
    assert buffer.occupancy() == {(0, 1): 1, ATM_CELL: 3}
    assert ticks[0] not in buffer.snapshot()
    assert wing in buffer.snapshot()


# --- prune: the age horizon, with the instant handed in


def test_prune_drops_samples_older_than_the_horizon_and_returns_the_count() -> None:
    """make_spec remembers for ten minutes; the two older quotes go and the recent one stays."""
    fresh = make_sample(log_moneyness=-0.05, ts_observed=NOW - timedelta(seconds=100))
    stale = make_sample(log_moneyness=0.00, ts_observed=NOW - timedelta(seconds=700))
    ancient = make_sample(log_moneyness=0.05, ts_observed=NOW - timedelta(seconds=1200))

    buffer = make_buffer(samples=(fresh, stale, ancient))

    assert buffer.prune(NOW) == 2
    assert buffer.snapshot() == (fresh,)


def test_prune_keeps_a_sample_exactly_at_the_horizon() -> None:
    buffer = make_buffer(samples=(make_sample(ts_observed=NOW - timedelta(seconds=600)),))

    assert buffer.prune(NOW) == 0
    assert len(buffer) == 1


def test_prune_drops_a_sample_one_second_past_the_horizon() -> None:
    """The boundary from the other side, so "strictly older" is pinned rather than assumed."""
    buffer = make_buffer(samples=(make_sample(ts_observed=NOW - timedelta(seconds=601)),))

    assert buffer.prune(NOW) == 1
    assert len(buffer) == 0


def test_prune_removes_a_cell_it_empties() -> None:
    """An emptied cell disappears, so coverage never has to tell "empty" from "absent"."""
    buffer = make_buffer(samples=(make_sample(ts_observed=NOW - timedelta(hours=1)),))

    buffer.prune(NOW)

    assert buffer.occupancy() == {}
    assert buffer.cell_coverage == 0.0


def test_prune_on_an_empty_buffer_drops_nothing() -> None:
    assert make_buffer().prune(NOW) == 0


def test_prune_rejects_a_naive_now() -> None:
    buffer = make_buffer(samples=(make_sample(),))

    with pytest.raises(ValueError, match="timezone-aware"):
        buffer.prune(NAIVE)


def test_the_first_prune_accepts_any_instant() -> None:
    """There is no reference to run backwards from until a prune has established one."""
    buffer = make_buffer(samples=(make_sample(ts_observed=NOW - timedelta(hours=2)),))

    assert buffer.prune(NOW - timedelta(hours=1)) == 1


def test_prune_rejects_a_now_that_precedes_the_previous_prune() -> None:
    """A clock running backwards makes every age negative and silently retires the horizon."""
    buffer = make_buffer(samples=(make_sample(ts_observed=NOW),))
    buffer.prune(NOW)

    with pytest.raises(ValueError, match="clock ran backwards"):
        buffer.prune(NOW - timedelta(seconds=1))


def test_prune_keeps_a_sample_stamped_after_now() -> None:
    """A venue clock running ahead is routine, and must not take the training loop down.

    ``ChainSnapshot.ts_exchange`` is deliberately not clamped to ``ts_local``, so quotes from the
    future arrive as a matter of course. They age negatively for a moment and then age normally.
    """
    from_the_future = make_sample(log_moneyness=0.05, ts_observed=NOW + timedelta(seconds=60))
    buffer = make_buffer(samples=(from_the_future,))

    assert buffer.prune(NOW) == 0
    assert buffer.snapshot() == (from_the_future,)
    # Guard against the vacuous reading: the sample is prunable once it is genuinely stale, so
    # its survival above is the negative age being tolerated rather than the cell being untouched.
    assert buffer.prune(NOW + timedelta(seconds=1200)) == 1


def test_a_refused_prune_drops_nothing_at_all() -> None:
    """All-or-nothing: the stale sample survives the rejection, and is provably prunable."""
    stale = make_sample(log_moneyness=-0.05, ts_observed=NOW - timedelta(seconds=1200))
    recent = make_sample(log_moneyness=0.05, ts_observed=NOW)
    buffer = make_buffer(samples=(stale, recent))
    buffer.prune(NOW - timedelta(seconds=1000))

    with pytest.raises(ValueError, match="clock ran backwards"):
        buffer.prune(NOW - timedelta(seconds=1001))

    assert len(buffer) == 2
    # Guard against the vacuous reading: that sample really was old enough to be dropped, so
    # its survival above is the refusal working rather than the horizon being generous.
    assert buffer.prune(NOW) == 1


def test_a_refused_prune_does_not_move_the_reference_instant() -> None:
    """The rejected instant must not become the one the next prune is compared against."""
    buffer = make_buffer(samples=(make_sample(ts_observed=NOW),))
    buffer.prune(NOW)

    with pytest.raises(ValueError, match="clock ran backwards"):
        buffer.prune(NOW - timedelta(seconds=60))

    with pytest.raises(ValueError, match="clock ran backwards"):
        buffer.prune(NOW - timedelta(seconds=30))


# --- sample: the stratified draw


@pytest.mark.parametrize("bad", [0, -1])
def test_sample_rejects_a_non_positive_size(bad: int) -> None:
    buffer = make_buffer(samples=(make_sample(),))

    with pytest.raises(ValueError, match="at least one sample"):
        buffer.sample(bad, np.random.default_rng(0))


def test_sample_raises_on_an_empty_buffer() -> None:
    """A market condition at the very first snapshot, not a bug: the caller waits for data."""
    with pytest.raises(EmptyBufferError):
        make_buffer().sample(4, np.random.default_rng(0))


def test_sample_returns_everything_when_the_buffer_is_thinner_than_asked() -> None:
    held = (
        make_sample(log_moneyness=-0.30),
        make_sample(log_moneyness=-0.05),
        make_sample(log_moneyness=0.25),
    )
    buffer = make_buffer(samples=held)

    drawn = buffer.sample(10, np.random.default_rng(0))

    assert sorted(one.log_moneyness for one in drawn) == [-0.30, -0.05, 0.25]


def test_sample_never_returns_the_same_held_sample_twice() -> None:
    """A draw is a subset of what is held, not a resampling of it."""
    held = tuple(make_sample(log_moneyness=k) for k in (-0.30, -0.20, -0.05, 0.00, 0.05))
    buffer = make_buffer(samples=held)

    drawn = buffer.sample(5, np.random.default_rng(3))

    assert sorted(one.log_moneyness for one in drawn) == [-0.30, -0.20, -0.05, 0.00, 0.05]


def test_the_same_seed_gives_the_identical_draw() -> None:
    """ADR-004's deterministic replay reaches all the way into the batch the network saw."""
    buffer = _mixed_buffer()

    first = buffer.sample(8, np.random.default_rng(20260824))
    second = buffer.sample(8, np.random.default_rng(20260824))

    assert first == second


def test_a_different_seed_gives_a_different_draw() -> None:
    """The counterpart: the injected generator really is what orders the draw."""
    buffer = _mixed_buffer()

    orders = {buffer.sample(8, np.random.default_rng(seed)) for seed in range(8)}

    assert len(orders) > 1


def test_the_draw_visits_every_non_empty_cell_before_taking_a_second_from_any() -> None:
    """Round-robin, so a cell holding one wing quote is served before a cell holding three."""
    wing = make_sample(log_moneyness=-0.30, tenor_years=FAR_TENOR)
    ticks = tuple(make_sample(log_moneyness=-0.09 + 0.01 * step) for step in range(3))
    buffer = make_buffer(samples=(wing, *ticks))

    for seed in range(20):
        assert wing in buffer.sample(2, np.random.default_rng(seed))


def test_a_stratified_draw_reaches_a_wing_cell_that_a_uniform_draw_would_starve() -> None:
    """The reason the class exists, measured against the draw it replaces.

    A hundred at-the-money ticks against three wing quotes is an ordinary ten minutes of a live
    chain. Drawing six points uniformly from that population finds a wing quote about one time in
    six; drawing six the stratified way finds one every time, because the wing cell is served
    before the busy cell is served twice. Without the uniform baseline the stratified assertion
    would be vacuous -- it would pass just as happily on a buffer that was never unbalanced.
    """
    spec = make_spec(capacity_per_cell=100)
    ticks = tuple(make_sample(log_moneyness=-0.099 + 0.0019 * step) for step in range(100))
    wings = tuple(
        make_sample(log_moneyness=k, tenor_years=FAR_TENOR) for k in (-0.30, -0.28, -0.26)
    )
    buffer = make_buffer(spec=spec, samples=ticks + wings)
    population = buffer.snapshot()
    assert len(population) == 103

    stratified_hits = sum(
        any(one.tenor_years == FAR_TENOR for one in buffer.sample(6, np.random.default_rng(seed)))
        for seed in range(100)
    )
    uniform_hits = 0
    for seed in range(100):
        picks = np.random.default_rng(seed).choice(len(population), size=6, replace=False)
        if any(population[int(index)].tenor_years == FAR_TENOR for index in picks):
            uniform_hits += 1

    assert stratified_hits == 100
    assert uniform_hits < 40


# --- Reading: coverage, occupancy and the restart's view


def test_cell_coverage_is_zero_on_an_empty_buffer() -> None:
    assert make_buffer().cell_coverage == 0.0


def test_cell_coverage_counts_cells_rather_than_samples() -> None:
    """Twenty ticks in one cell of six is the same coverage as one tick in that cell."""
    buffer = make_buffer(
        samples=tuple(make_sample(log_moneyness=-0.09 + 0.001 * step) for step in range(20))
    )

    assert buffer.cell_coverage == pytest.approx(1.0 / 6.0)


def test_cell_coverage_rises_as_distinct_cells_fill() -> None:
    buffer = make_buffer(
        samples=(
            make_sample(log_moneyness=-0.30),
            make_sample(log_moneyness=-0.05),
            make_sample(log_moneyness=0.25),
        )
    )

    assert buffer.cell_coverage == pytest.approx(0.5)


def test_occupancy_lists_only_the_non_empty_cells() -> None:
    buffer = make_buffer(
        samples=(
            make_sample(log_moneyness=-0.30, tenor_years=FAR_TENOR),
            make_sample(log_moneyness=-0.05),
            make_sample(log_moneyness=0.00),
        )
    )

    assert buffer.occupancy() == {(0, 1): 1, ATM_CELL: 2}


def test_occupancy_is_a_copy_rather_than_a_live_view() -> None:
    buffer = make_buffer(samples=(make_sample(),))
    taken = buffer.occupancy()

    buffer.add(make_sample(log_moneyness=0.25))

    assert taken == {ATM_CELL: 1}


def test_snapshot_returns_everything_held() -> None:
    held = (
        make_sample(log_moneyness=-0.30),
        make_sample(log_moneyness=-0.05),
        make_sample(log_moneyness=0.25, tenor_years=FAR_TENOR),
    )
    buffer = make_buffer(samples=held)

    assert set(buffer.snapshot()) == set(held)
    assert len(buffer.snapshot()) == len(buffer) == 3


def test_snapshot_is_ordered_by_cell_and_not_by_arrival() -> None:
    """The scheduled restart trains on this, so the same contents must give the same tuple."""
    held = (
        make_sample(log_moneyness=-0.30),
        make_sample(log_moneyness=-0.05),
        make_sample(log_moneyness=0.25),
    )

    assert make_buffer(samples=held).snapshot() == make_buffer(samples=held[::-1]).snapshot()


def test_snapshot_does_not_change_when_the_buffer_does() -> None:
    buffer = make_buffer(samples=(make_sample(),))
    taken = buffer.snapshot()

    buffer.add(make_sample(log_moneyness=0.25))

    assert len(taken) == 1


def test_len_counts_every_cell() -> None:
    buffer = make_buffer(
        samples=(
            make_sample(log_moneyness=-0.30),
            make_sample(log_moneyness=-0.05),
            make_sample(log_moneyness=0.00),
        )
    )

    assert len(buffer) == 3


def _mixed_buffer() -> ReplayBuffer:
    """A buffer spread over four cells, deep enough for an order to be worth asserting on."""
    return make_buffer(
        samples=(
            make_sample(log_moneyness=-0.30, tenor_years=NEAR_TENOR),
            make_sample(log_moneyness=-0.32, tenor_years=FAR_TENOR),
            make_sample(log_moneyness=-0.05, tenor_years=NEAR_TENOR),
            make_sample(log_moneyness=0.00, tenor_years=NEAR_TENOR),
            make_sample(log_moneyness=0.05, tenor_years=NEAR_TENOR),
            make_sample(log_moneyness=0.25, tenor_years=FAR_TENOR),
            make_sample(log_moneyness=0.28, tenor_years=FAR_TENOR),
            make_sample(log_moneyness=0.31, tenor_years=FAR_TENOR),
        )
    )
