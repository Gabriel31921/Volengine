from __future__ import annotations

from datetime import timedelta

import pytest

from tests.market_data.builders import (
    FAR,
    FORWARD,
    NAIVE,
    NEAR,
    NOW,
    make_chain,
    make_instrument,
    make_update,
)
from volengine.market_data.domain.admissibility import QuoteFlagD
from volengine.market_data.domain.option_quote import (
    QuoteUpdate,
)
from volengine.market_data.domain.quote_chain import ChainStats, QuoteChain

# --- builders


def populated(*updates: QuoteUpdate) -> QuoteChain:
    chain = make_chain()
    for one in updates or (make_update(),):
        chain.apply(one)
    return chain


# --- apply


def test_a_quote_reaches_the_snapshot() -> None:
    snapshot = populated().snapshot(NOW)
    assert len(snapshot.slices) == 1
    assert snapshot.slices[0].quotes[0].strike == FORWARD


def test_an_update_replaces_the_previous_state_of_its_instrument() -> None:
    """The ticker republishes the whole top of book, so a merge would be wrong as well as
    pointless: the second message is the state, not a correction to the first.
    """
    chain = populated(make_update(bid=0.050, ask=0.054), make_update(bid=0.070, ask=0.074))
    quotes = chain.snapshot(NOW).slices[0].quotes
    assert len(quotes) == 1
    assert quotes[0].mid == pytest.approx(0.072)


def test_an_undiscovered_instrument_is_registered_on_the_spot() -> None:
    """Discovery polls on its own schedule; a strike born between two polls still trades."""
    assert populated().stats(NOW).n_instruments_known == 1


def test_a_chain_refuses_an_update_for_another_underlying() -> None:
    """One chain covers one underlying. Mixing two corrupts every slice rule silently."""
    with pytest.raises(ValueError, match="ETH"):
        populated(make_update(underlying="ETH"))


def test_the_forward_is_kept_per_expiry() -> None:
    """The venue publishes the synchronous future for *that* expiry: in contango the near and
    far forwards differ by percent, and one global value would misplace a whole slice.
    """
    chain = populated(
        make_update(expiry=NEAR, underlying_price=60_000.0),
        make_update(expiry=FAR, underlying_price=63_000.0),
    )
    forwards = [chain_slice.forward for chain_slice in chain.snapshot(NOW).slices]
    assert forwards == [60_000.0, 63_000.0]


def test_an_update_without_an_underlying_price_keeps_the_last_one() -> None:
    chain = populated(
        make_update(strike=55_000.0, underlying_price=60_000.0),
        make_update(strike=65_000.0, underlying_price=None),
    )
    assert chain.snapshot(NOW).slices[0].forward == 60_000.0


def test_an_expiry_with_no_forward_yet_is_not_published() -> None:
    """Without it there is no moneyness and no comparable price; raw strikes would hand the
    calibrator numbers it cannot place.
    """
    assert populated(make_update(underlying_price=None)).snapshot(NOW).slices == ()


# --- set_live_instruments


def test_a_delisted_instrument_loses_its_state() -> None:
    chain = populated(make_update(strike=55_000.0), make_update(strike=65_000.0))
    chain.set_live_instruments([make_instrument(strike=55_000.0)])
    strikes = [quote.strike for quote in chain.snapshot(NOW).slices[0].quotes]
    assert strikes == [55_000.0]


def test_setting_the_live_set_twice_changes_nothing() -> None:
    """Idempotence is the property that makes conflating the composition event sound
    (ADR-013): receiving only the last of five must leave the same state as receiving all.
    """
    chain = populated(make_update(strike=55_000.0), make_update(strike=65_000.0))
    live = [make_instrument(strike=55_000.0)]
    chain.set_live_instruments(live)
    once = chain.snapshot(NOW)
    chain.set_live_instruments(live)
    assert chain.snapshot(NOW) == once


def test_the_live_set_is_the_denominator_of_coverage() -> None:
    """Two strikes exist, one has ticked: coverage is one half, not one."""
    chain = populated(make_update(strike=55_000.0))
    chain.set_live_instruments([make_instrument(strike=55_000.0), make_instrument(strike=65_000.0)])
    assert chain.stats(NOW).coverage_ratio == pytest.approx(0.5)


def test_delisting_a_whole_expiry_drops_its_forward() -> None:
    chain = populated(make_update(expiry=NEAR), make_update(expiry=FAR))
    chain.set_live_instruments([make_instrument(expiry=FAR)])
    assert [s.expiry for s in chain.snapshot(NOW).slices] == [FAR]


def test_the_live_set_refuses_another_underlying() -> None:
    with pytest.raises(ValueError, match="ETH"):
        make_chain().set_live_instruments([make_instrument(underlying="ETH")])


# --- snapshot: what is excluded


def test_an_expired_expiry_is_dropped() -> None:
    """Venues keep them listed until settlement, and there is no honest tenor for them."""
    chain = populated(make_update(expiry=NEAR), make_update(expiry=FAR))
    after_near_expired = NEAR + timedelta(seconds=1)
    assert [s.expiry for s in chain.snapshot(after_near_expired).slices] == [FAR]


@pytest.mark.parametrize(("bid", "ask"), [(None, 0.054), (0.050, None)], ids=["no-bid", "no-ask"])
def test_a_one_sided_book_is_not_published(bid: float | None, ask: float | None) -> None:
    """No mid means no datum to publish -- an absent observation, not a bad one."""
    assert populated(make_update(bid=bid, ask=ask)).snapshot(NOW).slices == ()


def test_a_market_bid_and_offered_at_zero_is_not_published() -> None:
    assert populated(make_update(bid=0.0, ask=0.0)).snapshot(NOW).slices == ()


def test_an_expiry_whose_quotes_are_all_unusable_produces_no_slice() -> None:
    assert populated(make_update(bid=None, ask=None)).snapshot(NOW).slices == ()


# --- snapshot: judgement happens here, not on apply


def test_flags_are_computed_against_the_snapshot_instant() -> None:
    """Age advances on its own, so a flag computed on arrival is stale before it is read.

    The same stored quote is clean now and stale ten seconds later, with no update in between.
    """
    chain = populated()
    assert chain.snapshot(NOW).slices[0].quotes[0].flags == ()
    later = chain.snapshot(NOW + timedelta(seconds=10)).slices[0].quotes[0]
    assert QuoteFlagD.STALE in later.flags


def test_slice_flags_are_recomputed_from_the_current_neighbours() -> None:
    """Monotonicity is a property of a quote's neighbours, so it cannot be decided when a
    single quote arrives. Here the third call breaks a curve the first two agreed on.
    """
    chain = populated(
        make_update(strike=55_000.0, bid=0.090, ask=0.094),
        make_update(strike=60_000.0, bid=0.050, ask=0.054),
    )
    assert chain.snapshot(NOW).slices[0].flags == ()
    chain.apply(make_update(strike=65_000.0, bid=0.190, ask=0.194))
    assert QuoteFlagD.SLICE_MONOTONICITY in chain.snapshot(NOW).slices[0].flags


def test_the_age_of_a_venue_clock_running_ahead_is_floored_at_zero() -> None:
    """QualityBlock rejects a negative age, so the skew is absorbed here rather than crashing
    the ACL two layers away.
    """
    ahead = populated(make_update(ts_exchange=NOW + timedelta(seconds=30)))
    assert ahead.snapshot(NOW).slices[0].quotes[0].age_seconds == 0.0


# --- snapshot: shape and ordering


def test_slices_come_out_sorted_by_expiry() -> None:
    """MarketSnapshot requires strictly increasing expiries, and a chain is keyed by
    instrument, so its iteration order guarantees nothing.
    """
    chain = populated(make_update(expiry=FAR), make_update(expiry=NEAR))
    assert [s.expiry for s in chain.snapshot(NOW).slices] == [NEAR, FAR]


def test_quotes_come_out_sorted_by_strike() -> None:
    chain = populated(make_update(strike=65_000.0), make_update(strike=55_000.0))
    assert [q.strike for q in chain.snapshot(NOW).slices[0].quotes] == [55_000.0, 65_000.0]


def test_the_snapshot_is_disconnected_from_the_chain() -> None:
    """A live view would change under a calibration that is halfway through reading it."""
    chain = populated()
    snapshot = chain.snapshot(NOW)
    chain.apply(make_update(strike=65_000.0))
    assert len(snapshot.slices[0].quotes) == 1


def test_the_tenor_comes_from_the_conventions() -> None:
    """Resolved before the boundary, so no calibrator ever redoes a daycount (ADR-002)."""
    expected = (NEAR - NOW).total_seconds() / 86_400 / 365
    assert populated().snapshot(NOW).slices[0].tenor_years == pytest.approx(expected)


def test_the_exchange_stamp_is_the_freshest_evidence_held() -> None:
    chain = populated(
        make_update(strike=55_000.0, ts_exchange=NOW - timedelta(seconds=3)),
        make_update(strike=65_000.0, ts_exchange=NOW - timedelta(seconds=1)),
    )
    assert chain.snapshot(NOW).ts_exchange == NOW - timedelta(seconds=1)


def test_the_snapshot_carries_the_market_identity() -> None:
    snapshot = populated().snapshot(NOW)
    assert (snapshot.market_id, snapshot.underlying) == ("BTC-DERIBIT", "BTC")


@pytest.mark.parametrize("method", ["snapshot", "stats"])
def test_a_naive_instant_is_refused(method: str) -> None:
    with pytest.raises(ValueError, match="timezone-aware"):
        getattr(populated(), method)(NAIVE)


# --- stats


def test_stats_cannot_hold_more_quotes_than_instruments() -> None:
    """The bound the snapshotting policy divides by: without it, n_quotes_admissible over
    n_instruments_known is not a ratio, and a chain could report itself better than complete.
    """
    with pytest.raises(ValueError, match="instruments known"):
        ChainStats(
            n_instruments_known=1,
            n_quotes_total=2,
            n_quotes_admissible=2,
            max_age_seconds=0.0,
            coverage_ratio=1.0,
        )


def test_an_empty_chain_has_zero_coverage_rather_than_dividing_by_zero() -> None:
    stats = make_chain().stats(NOW)
    assert (stats.coverage_ratio, stats.n_quotes_total) == (0.0, 0)


def test_a_flagged_quote_is_not_admissible() -> None:
    """Strict by design: deciding which flags are forgivable is the calibrator's job."""
    chain = populated(make_update(bid_size=0.1, ask_size=0.1))
    stats = chain.stats(NOW)
    assert (stats.n_quotes_total, stats.n_quotes_admissible) == (1, 0)


def test_the_oldest_quote_sets_the_max_age() -> None:
    chain = populated(
        make_update(strike=55_000.0, ts_exchange=NOW - timedelta(seconds=1)),
        make_update(strike=65_000.0, ts_exchange=NOW - timedelta(seconds=30)),
    )
    assert chain.stats(NOW).max_age_seconds == pytest.approx(30.0)


def test_an_unusable_quote_counts_towards_the_total_but_not_coverage() -> None:
    """Coverage answers "how much of the chain can I actually fit", not "how much arrived"."""
    chain = populated(make_update(strike=55_000.0), make_update(strike=65_000.0, bid=None))
    stats = chain.stats(NOW)
    assert (stats.n_quotes_total, stats.coverage_ratio) == (2, 0.5)


# --- the material-move baseline


def test_nothing_has_moved_before_a_baseline_exists() -> None:
    assert populated().max_relative_move_since_baseline() == 0.0


def test_a_move_is_measured_against_the_baseline() -> None:
    chain = populated(make_update(bid=0.050, ask=0.050))
    chain.reset_move_baseline()
    chain.apply(make_update(bid=0.055, ask=0.055))
    assert chain.max_relative_move_since_baseline() == pytest.approx(0.10)


def test_the_largest_move_wins() -> None:
    chain = populated(
        make_update(strike=55_000.0, bid=0.100, ask=0.100),
        make_update(strike=65_000.0, bid=0.050, ask=0.050),
    )
    chain.reset_move_baseline()
    chain.apply(make_update(strike=55_000.0, bid=0.101, ask=0.101))
    chain.apply(make_update(strike=65_000.0, bid=0.060, ask=0.060))
    assert chain.max_relative_move_since_baseline() == pytest.approx(0.20)


def test_resetting_the_baseline_forgets_the_move() -> None:
    chain = populated(make_update(bid=0.050, ask=0.050))
    chain.reset_move_baseline()
    chain.apply(make_update(bid=0.055, ask=0.055))
    chain.reset_move_baseline()
    assert chain.max_relative_move_since_baseline() == 0.0


def test_a_newborn_strike_is_not_a_move() -> None:
    """Its birth is a composition change with its own event. Counting it as an infinite move
    would make every discovery force a snapshot, whatever the market did.
    """
    chain = populated(make_update(strike=55_000.0, bid=0.050, ask=0.050))
    chain.reset_move_baseline()
    chain.apply(make_update(strike=65_000.0, bid=0.500, ask=0.500))
    assert chain.max_relative_move_since_baseline() == 0.0
