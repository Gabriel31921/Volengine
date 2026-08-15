from __future__ import annotations

import math
from dataclasses import replace
from datetime import datetime, timedelta

import pytest

from tests.market_data.builders import (
    FORWARD,
    NAIVE,
    NOW,
    make_observation,
    make_thresholds,
)
from tests.support import replace_field
from volengine.market_data.domain.admissibility import (
    QuoteFlagD,
    flag_quote,
    flag_slice,
    flag_surface,
)
from volengine.market_data.domain.option_quote import OptionKindD, QuoteObservation


def judge(
    observation: QuoteObservation | None = None,
    strike: float = FORWARD,
    own_iv: float | None = 0.62,
    now: datetime = NOW,
) -> tuple[QuoteFlagD, ...]:
    """flag_quote with everything at its clean default, so a test sets one knob."""
    return flag_quote(
        observation=observation if observation is not None else make_observation(),
        strike=strike,
        forward=FORWARD,
        own_iv=own_iv,
        thresholds=make_thresholds(),
        now=now,
    )


def at_moneyness(k: float) -> float:
    """The strike sitting at log-forward-moneyness ``k``, since k = log(K / F)."""
    return FORWARD * math.exp(k)


# --- AdmissibilityThresholds


@pytest.mark.parametrize("field", ["max_spread_rel", "max_age_seconds", "max_iv_divergence_bp"])
@pytest.mark.parametrize("bad", [0.0, -1.0, float("nan"), float("inf")])
def test_thresholds_reject_a_non_positive_limit(field: str, bad: float) -> None:
    with pytest.raises(ValueError, match=field):
        replace_field(make_thresholds(), field, bad)


@pytest.mark.parametrize("field", ["convexity_tolerance", "min_size"])
@pytest.mark.parametrize("bad", [-1.0, float("nan"), float("inf")])
def test_thresholds_reject_a_negative_slack(field: str, bad: float) -> None:
    with pytest.raises(ValueError, match=field):
        replace_field(make_thresholds(), field, bad)


@pytest.mark.parametrize("field", ["convexity_tolerance", "min_size"])
def test_thresholds_accept_a_zero_slack(field: str) -> None:
    """Zero is how a deployment switches the rule off, not an invalid setting."""
    assert getattr(replace_field(make_thresholds(), field, 0.0), field) == 0.0


@pytest.mark.parametrize("bad", [(1.5, -1.5), (0.0, 0.0)])
def test_thresholds_reject_an_empty_moneyness_range(bad: tuple[float, float]) -> None:
    with pytest.raises(ValueError, match="moneyness range"):
        replace(make_thresholds(), moneyness_range=bad)


# --- flag_quote: preconditions that are errors, not flags


@pytest.mark.parametrize("bad", [0.0, -1.0, float("nan"), float("inf")])
def test_flag_quote_rejects_an_unusable_forward(bad: float) -> None:
    """log(K / F) is meaningless without it, so this is a broken call rather than a bad quote."""
    with pytest.raises(ValueError, match="forward"):
        flag_quote(
            observation=make_observation(),
            strike=FORWARD,
            forward=bad,
            own_iv=0.62,
            thresholds=make_thresholds(),
            now=NOW,
        )


def test_flag_quote_rejects_a_naive_now() -> None:
    with pytest.raises(ValueError, match="timezone-aware"):
        judge(now=NAIVE)


# --- flag_quote: one rule at a time


def test_a_clean_quote_earns_no_flags() -> None:
    assert judge() == ()


def test_crossed_book_is_flagged() -> None:
    assert QuoteFlagD.CROSSED in judge(make_observation(bid=0.060, ask=0.054))


def test_a_locked_market_is_not_crossed() -> None:
    """bid == ask is a zero spread, not an inverted book."""
    assert judge(make_observation(bid=0.052, ask=0.052)) == ()


def test_a_wide_spread_is_flagged() -> None:
    assert QuoteFlagD.WIDE_SPREAD in judge(make_observation(bid=0.020, ask=0.100))


def test_a_crossed_book_does_not_also_count_as_wide() -> None:
    """Crossing makes spread_rel negative, and a negative spread is not a wide one.

    Reporting both would double-count one anomaly and make the flags useless as a tally.
    """
    assert QuoteFlagD.WIDE_SPREAD not in judge(make_observation(bid=0.060, ask=0.054))


def test_a_one_sided_book_has_no_spread_to_judge() -> None:
    assert QuoteFlagD.WIDE_SPREAD not in judge(make_observation(bid=None))


def test_an_old_quote_is_flagged_stale() -> None:
    assert QuoteFlagD.STALE in judge(now=NOW + timedelta(seconds=6))


def test_staleness_is_measured_from_the_exchange_stamp_not_ours() -> None:
    """Our arrival time is transport latency. Measuring from it reports our own lag as a dead
    market, and the two are different failures with different owners.
    """
    fresh_at_the_venue = replace(
        make_observation(), ts_local=NOW - timedelta(hours=1), ts_exchange=NOW
    )
    assert QuoteFlagD.STALE not in judge(fresh_at_the_venue)


def test_a_venue_clock_running_ahead_is_not_stale() -> None:
    """Negative age is skew, which QuoteObservation deliberately allows. It must not raise
    and must not be reported as staleness.
    """
    assert judge(make_observation(ts_exchange=NOW + timedelta(seconds=30))) == ()


def test_a_thin_side_is_flagged() -> None:
    assert QuoteFlagD.LOW_SIZE in judge(make_observation(bid_size=0.5))


def test_size_is_judged_on_the_thinner_side() -> None:
    """A quote is only as good as the leg you would struggle to fill."""
    assert QuoteFlagD.LOW_SIZE in judge(make_observation(bid_size=500.0, ask_size=0.5))


def test_size_is_not_confused_with_price() -> None:
    """Regression: comparing premiums against min_size flags every cheap wing forever.

    Both are floats, so nothing but this assertion tells the two rules apart.
    """
    cheap_but_deep = make_observation(bid=0.0004, ask=0.0006, bid_size=900.0, ask_size=900.0)
    assert QuoteFlagD.LOW_SIZE not in judge(cheap_but_deep)


@pytest.mark.parametrize("k", [-2.0, 2.0])
def test_a_quote_outside_the_moneyness_range_is_flagged(k: float) -> None:
    assert QuoteFlagD.EXTREME_MONEYNESS in judge(strike=at_moneyness(k))


@pytest.mark.parametrize("k", [-1.5, 0.0, 1.5])
def test_the_moneyness_bounds_are_inclusive(k: float) -> None:
    assert QuoteFlagD.EXTREME_MONEYNESS not in judge(strike=at_moneyness(k))


def test_disagreement_with_the_venue_vol_is_flagged() -> None:
    """0.70 against 0.62 is 800 bp, above the 500 bp threshold."""
    assert QuoteFlagD.IV_DIVERGENCE in judge(own_iv=0.70)


def test_small_disagreement_with_the_venue_vol_is_tolerated() -> None:
    """0.64 against 0.62 is 200 bp: the two use different conventions, some gap is expected."""
    assert QuoteFlagD.IV_DIVERGENCE not in judge(own_iv=0.64)


def test_divergence_needs_our_own_vol() -> None:
    assert QuoteFlagD.IV_DIVERGENCE not in judge(own_iv=None)


def test_divergence_needs_the_venue_vol() -> None:
    assert QuoteFlagD.IV_DIVERGENCE not in judge(make_observation(exchange_iv=None))


def test_a_failed_inversion_is_not_silent_agreement() -> None:
    """abs(nan - x) > threshold is False, so leaning on the comparison reports a NaN inversion
    as a quote that agrees with the venue. The explicit isfinite check is what prevents that.
    """
    assert judge(own_iv=float("nan")) == ()


# --- flag_quote: the rules are independent


def test_flags_accumulate_in_a_fixed_order() -> None:
    """The order is part of the contract: a test can assert the tuple, and two runs on the
    same quote produce byte-identical output.
    """
    bad = make_observation(bid_size=0.1, ask_size=0.1, ts_exchange=NOW - timedelta(seconds=60))
    assert judge(bad, strike=at_moneyness(2.0)) == (
        QuoteFlagD.STALE,
        QuoteFlagD.LOW_SIZE,
        QuoteFlagD.EXTREME_MONEYNESS,
    )


# --- flag_slice


def calls(*mids: float) -> list[tuple[float, OptionKindD, float]]:
    """Three calls at 55k / 60k / 65k with the given mids, in that strike order."""
    strikes = (55_000.0, 60_000.0, 65_000.0)
    return [(strike, OptionKindD.CALL, mid) for strike, mid in zip(strikes, mids, strict=True)]


def puts(*mids: float) -> list[tuple[float, OptionKindD, float]]:
    strikes = (55_000.0, 60_000.0, 65_000.0)
    return [(strike, OptionKindD.PUT, mid) for strike, mid in zip(strikes, mids, strict=True)]


def test_a_well_formed_slice_earns_no_flags() -> None:
    assert flag_slice(calls(6000.0, 3000.0, 1000.0), make_thresholds()) == ()


def test_calls_rising_with_strike_are_flagged() -> None:
    """A call struck higher pays off in strictly fewer states, so it cannot cost more."""
    assert QuoteFlagD.SLICE_MONOTONICITY in flag_slice(
        calls(1000.0, 3000.0, 6000.0), make_thresholds()
    )


def test_puts_falling_with_strike_are_flagged() -> None:
    """Mirror image: a put struck higher is worth at least as much."""
    assert QuoteFlagD.SLICE_MONOTONICITY in flag_slice(
        puts(6000.0, 3000.0, 1000.0), make_thresholds()
    )


def test_a_well_formed_put_curve_earns_no_flags() -> None:
    assert flag_slice(puts(1000.0, 3000.0, 6000.0), make_thresholds()) == ()


def test_equal_neighbouring_prices_are_not_disorder() -> None:
    """Two dead wing strikes worth the same is equality, not a broken chain."""
    assert QuoteFlagD.SLICE_MONOTONICITY not in flag_slice(
        calls(6000.0, 1000.0, 1000.0), make_thresholds()
    )


def test_input_order_does_not_manufacture_violations() -> None:
    """A chain is keyed by instrument, so its iteration order is arbitrary. Comparing
    neighbours without sorting first turns that arbitrariness into random flags.
    """
    ordered = calls(6000.0, 3000.0, 1000.0)
    shuffled = [ordered[2], ordered[0], ordered[1]]
    assert flag_slice(shuffled, make_thresholds()) == ()


def test_calls_and_puts_are_judged_as_separate_curves() -> None:
    """Merged into one sequence these prices look wildly disordered; they are two valid
    curves, and mixing them invents a violation that no market committed.
    """
    both = calls(6000.0, 3000.0, 1000.0) + puts(1000.0, 3000.0, 6000.0)
    assert flag_slice(both, make_thresholds()) == ()


def test_a_slice_is_marked_once_even_when_both_curves_break() -> None:
    both = calls(1000.0, 3000.0, 6000.0) + puts(6000.0, 3000.0, 1000.0)
    assert flag_slice(both, make_thresholds()).count(QuoteFlagD.SLICE_MONOTONICITY) == 1


def test_two_quotes_at_the_same_strike_are_rejected() -> None:
    """A malformed chain, not a suspicious market -- and it divides by zero in the chord."""
    duplicated = [
        (60_000.0, OptionKindD.CALL, 3000.0),
        (60_000.0, OptionKindD.CALL, 2900.0),
    ]
    with pytest.raises(ValueError, match="share strike"):
        flag_slice(duplicated, make_thresholds())


def test_a_pair_of_quotes_cannot_form_a_butterfly() -> None:
    """Convexity needs three strikes; with two the rule does not apply and must stay quiet."""
    two = calls(6000.0, 3000.0, 1000.0)[:2]
    assert QuoteFlagD.SLICE_CONVEXITY not in flag_slice(two, make_thresholds())


def test_an_empty_slice_earns_no_flags() -> None:
    assert flag_slice([], make_thresholds()) == ()


# --- flag_surface


def test_the_surface_check_is_deliberately_empty() -> None:
    """Not an oversight: comparing total variance at fixed k needs interpolation, which is a
    modelling choice. It lives in durrleman.calendar_violation over fitted slices (ADR-008).
    """
    assert flag_surface([(30 / 365, calls(6000.0, 3000.0, 1000.0))]) == ()
