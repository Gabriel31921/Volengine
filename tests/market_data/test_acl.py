"""The translation from the chain's own view to the published snapshot.

Every test here is about a place where the two vocabularies genuinely disagree. A field that is
copied straight across is not worth a test of its own -- the DTO's constructor already refuses a
bad one -- so what is pinned below is the reconciliation of the clocks, the crossed-market
spread, the flag and kind tables, the forward cross-check, and the identity.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from tests.market_data.builders import (
    FAR,
    FORWARD,
    NEAR,
    NOW,
    make_chain,
    make_instrument,
    make_update,
)
from volengine.contracts.market_snapshot import MarketSnapshot, OptionKind, QuoteFlag
from volengine.market_data.application.acl import (
    _FLAGS,
    _KINDS,
    build_snapshot_id,
    clock_skew_seconds,
    forward_crosscheck_error,
    instrument_key,
    reconcile_exchange_instant,
    to_composition_changed,
    to_market_snapshot,
)
from volengine.market_data.domain.admissibility import QuoteFlagD
from volengine.market_data.domain.option_quote import OptionKindD
from volengine.market_data.domain.quote_chain import QuoteChain
from volengine.market_data.domain.snapshot_policy import DegradationReason, QualityAssessment

CLEAN = QualityAssessment(reasons=())
DEGRADED = QualityAssessment(reasons=(DegradationReason.LOW_COVERAGE,))
SKEW_TOLERANCE = 30.0


def build_snapshot(
    chain: QuoteChain | None = None,
    now: datetime = NOW,
    quality: QualityAssessment = CLEAN,
    skew: float = SKEW_TOLERANCE,
) -> MarketSnapshot:
    """Freeze a chain and translate it, which is what every test below starts from."""
    chain = chain if chain is not None else make_chain()
    return to_market_snapshot(
        snapshot=chain.snapshot(now),
        snapshot_id="BTC-DERIBIT:00000000",
        quality=quality,
        max_skew_seconds=skew,
    )


def populated_chain() -> QuoteChain:
    """A chain with one two-sided strike on each of two expiries."""
    chain = make_chain()
    for expiry in (NEAR, FAR):
        for kind in (OptionKindD.CALL, OptionKindD.PUT):
            chain.apply(make_update(expiry=expiry, kind=kind))
    return chain


# --- the two clocks


def test_the_venue_stamp_is_published_when_it_agrees_with_our_clock() -> None:
    chain = populated_chain()
    published = build_snapshot(chain, now=NOW + timedelta(seconds=2))

    assert published.ts_exchange == NOW


def test_a_venue_clock_far_ahead_is_replaced_by_our_own() -> None:
    """A stamp in the future would make every surface fitted to it fail construction."""
    chain = make_chain()
    ahead = NOW + timedelta(seconds=120)
    for kind in (OptionKindD.CALL, OptionKindD.PUT):
        chain.apply(make_update(kind=kind, ts_exchange=ahead))

    published = build_snapshot(chain, now=NOW)

    assert published.ts_exchange == NOW


def test_a_venue_clock_far_behind_is_replaced_by_our_own() -> None:
    """Otherwise the snapshot is born stale and Risk rejects a market that is quoting fine."""
    chain = make_chain()
    behind = NOW - timedelta(seconds=600)
    for kind in (OptionKindD.CALL, OptionKindD.PUT):
        chain.apply(make_update(kind=kind, ts_exchange=behind))

    published = build_snapshot(chain, now=NOW)

    assert published.ts_exchange == NOW


def test_latency_within_the_tolerance_survives_into_the_published_stamp() -> None:
    """The vacuous-pass guard on the test above: a small lag must *not* be reconciled away.

    Without this, a reconciliation that simply always returned ``ts_local`` would pass every
    other test in this section, and the transport latency the freshness policy is supposed to see
    would be deleted at the boundary.
    """
    chain = make_chain()
    lagging = NOW - timedelta(seconds=3)
    for kind in (OptionKindD.CALL, OptionKindD.PUT):
        chain.apply(make_update(kind=kind, ts_exchange=lagging))

    published = build_snapshot(chain, now=NOW)

    assert published.ts_exchange == lagging


def test_reconciliation_keeps_the_local_stamp_untouched() -> None:
    chain = populated_chain()
    now = NOW + timedelta(seconds=5)

    published = build_snapshot(chain, now=now)

    assert published.ts_local == now


def test_the_skew_is_reported_signed() -> None:
    chain = make_chain()
    chain.apply(make_update(ts_exchange=NOW + timedelta(seconds=4)))

    assert clock_skew_seconds(chain.snapshot(NOW)) == pytest.approx(4.0)
    assert clock_skew_seconds(chain.snapshot(NOW + timedelta(seconds=10))) == pytest.approx(-6.0)


def test_a_non_positive_skew_tolerance_is_refused() -> None:
    with pytest.raises(ValueError, match="clock skew"):
        reconcile_exchange_instant(NOW, NOW, 0.0)


def test_a_nan_skew_tolerance_is_refused_before_it_disables_the_rule() -> None:
    """``nan <= 0`` is ``False``, so the guard has to test finiteness first or it lets NaN pass."""
    with pytest.raises(ValueError, match="clock skew"):
        reconcile_exchange_instant(NOW, NOW, float("nan"))


# --- the two vocabularies


def test_every_domain_flag_has_a_published_counterpart() -> None:
    """A member added on one side alone fails here, rather than stopping crossing in a market."""
    assert set(_FLAGS) == set(QuoteFlagD)


def test_no_two_domain_flags_collapse_onto_one_published_flag() -> None:
    """The vacuous-pass guard: a table mapping everything to ``STALE`` satisfies the test above."""
    assert set(_FLAGS.values()) == set(QuoteFlag)


def test_every_domain_option_side_has_a_published_counterpart() -> None:
    assert set(_KINDS) == set(OptionKindD)
    assert set(_KINDS.values()) == set(OptionKind)


def test_a_flagged_quote_carries_its_flag_across_the_boundary() -> None:
    chain = make_chain()
    chain.apply(make_update(bid_size=0.0))

    published = build_snapshot(chain)

    assert QuoteFlag.LOW_SIZE in published.slices[0].quotes[0].flags


def test_the_option_side_is_translated_to_the_wire_spelling() -> None:
    """The domain says ``"PUT"`` for a human; the wire says ``"P"``, and must keep saying it."""
    chain = make_chain()
    chain.apply(make_update(kind=OptionKindD.PUT))

    published = build_snapshot(chain)

    assert published.slices[0].quotes[0].kind is OptionKind.PUT
    assert published.slices[0].quotes[0].kind.value == "P"


def test_the_published_kind_is_an_enum_member_not_a_bare_string() -> None:
    """A ``StrEnum`` *is* a ``str``, so equality alone cannot catch a translation that forgot."""
    chain = make_chain()
    chain.apply(make_update())

    published = build_snapshot(chain)

    assert isinstance(published.slices[0].quotes[0].kind, OptionKind)


# --- the crossed book


def test_a_crossed_quote_publishes_a_zero_spread_and_keeps_the_flag() -> None:
    """``ChainQuote`` keeps the negative sign as evidence; ``QuoteData`` refuses it outright."""
    chain = make_chain()
    chain.apply(make_update(bid=0.060, ask=0.050))

    quote = build_snapshot(chain).slices[0].quotes[0]

    assert quote.spread_rel == 0.0
    assert QuoteFlag.CROSSED in quote.flags


def test_an_ordinary_spread_is_published_unchanged() -> None:
    """The vacuous-pass guard: clamping must not flatten every spread to zero."""
    chain = make_chain()
    chain.apply(make_update(bid=0.050, ask=0.054))

    assert build_snapshot(chain).slices[0].quotes[0].spread_rel > 0.0


# --- the forward cross-check


def test_the_cross_check_is_absent_when_no_strike_is_quoted_on_both_sides() -> None:
    """ "We could not look" is not "we looked and found nothing wrong"."""
    chain = make_chain()
    chain.apply(make_update(kind=OptionKindD.CALL))

    assert forward_crosscheck_error(chain.snapshot(NOW)) is None


def test_a_consistent_pair_implies_the_providers_own_forward() -> None:
    """Parity on a call and a put priced at the same mid implies exactly the strike.

    The default strike sits on the forward, so the two routes agree and the error is zero.
    """
    chain = make_chain()
    for kind in (OptionKindD.CALL, OptionKindD.PUT):
        chain.apply(make_update(kind=kind, strike=FORWARD))

    assert forward_crosscheck_error(chain.snapshot(NOW)) == pytest.approx(0.0)


def test_a_disagreeing_pair_is_reported_as_a_relative_error() -> None:
    chain = make_chain()
    chain.apply(make_update(kind=OptionKindD.CALL, strike=FORWARD, bid=0.10, ask=0.10))
    chain.apply(make_update(kind=OptionKindD.PUT, strike=FORWARD, bid=0.04, ask=0.04))

    # F_parity = K + (C - P) = 60000 + 0.06, against a provider forward of 60000.
    assert forward_crosscheck_error(chain.snapshot(NOW)) == pytest.approx(0.06 / FORWARD)


def test_the_worst_expiry_decides_rather_than_the_average() -> None:
    """A single broken expiry must not disappear into the healthy ones beside it."""
    chain = make_chain()
    for kind in (OptionKindD.CALL, OptionKindD.PUT):
        chain.apply(make_update(expiry=NEAR, kind=kind, strike=FORWARD))
    chain.apply(make_update(expiry=FAR, kind=OptionKindD.CALL, strike=FORWARD, bid=1.05, ask=1.05))
    chain.apply(make_update(expiry=FAR, kind=OptionKindD.PUT, strike=FORWARD, bid=0.05, ask=0.05))

    error = forward_crosscheck_error(chain.snapshot(NOW))

    assert error is not None
    assert error == pytest.approx(1.0 / FORWARD)


def test_the_cross_check_reaches_the_published_quality_block() -> None:
    published = build_snapshot(populated_chain())

    assert published.quality.forward_crosscheck_error == pytest.approx(0.0)


# --- the quality block


def test_the_policys_verdict_is_copied_rather_than_recomputed() -> None:
    published = build_snapshot(populated_chain(), quality=DEGRADED)

    assert published.quality.degraded is True


def test_a_clean_assessment_publishes_an_undegraded_block() -> None:
    published = build_snapshot(populated_chain(), quality=CLEAN)

    assert published.quality.degraded is False


# --- identity


def test_the_snapshot_id_is_derived_from_the_market_and_the_sequence() -> None:
    assert build_snapshot_id("BTC-DERIBIT", 42) == "BTC-DERIBIT:00000042"


def test_two_snapshots_of_one_market_get_different_ids() -> None:
    assert build_snapshot_id("BTC-DERIBIT", 0) != build_snapshot_id("BTC-DERIBIT", 1)


def test_the_same_sequence_always_mints_the_same_id() -> None:
    """Determinism is the whole point: a random id would make two replays incomparable."""
    assert build_snapshot_id("BTC-DERIBIT", 7) == build_snapshot_id("BTC-DERIBIT", 7)


def test_an_empty_market_id_is_refused() -> None:
    with pytest.raises(ValueError, match="market id"):
        build_snapshot_id("", 0)


def test_a_negative_sequence_is_refused() -> None:
    with pytest.raises(ValueError, match="sequence"):
        build_snapshot_id("BTC-DERIBIT", -1)


# --- the composition event


def test_the_composition_event_carries_the_whole_live_set_as_strings() -> None:
    instruments = frozenset({make_instrument(strike=60_000.0), make_instrument(strike=70_000.0)})

    event = to_composition_changed("BTC-DERIBIT", NOW, instruments)

    assert len(event.instruments) == 2
    assert all(isinstance(one, str) for one in event.instruments)


def test_the_instrument_list_is_ordered_so_a_replay_reproduces_it() -> None:
    instruments = frozenset({make_instrument(strike=70_000.0), make_instrument(strike=60_000.0)})

    event = to_composition_changed("BTC-DERIBIT", NOW, instruments)

    assert list(event.instruments) == sorted(event.instruments)


def test_the_instrument_key_is_the_engines_spelling_and_not_the_venues() -> None:
    """Deribit writes ``BTC-27AUG26-60000-C``; the adapter parsed that away on purpose."""
    key = instrument_key(make_instrument(strike=60_000.0, expiry=NEAR, kind=OptionKindD.CALL))

    assert key == "BTC|2026-08-27T08:00:00+00:00|60000|CALL"


def test_the_expiry_in_a_key_is_an_instant_rather_than_a_date() -> None:
    """The venue's expiry time of day is part of the identity (ADR-002)."""
    key = instrument_key(make_instrument(expiry=NEAR))

    assert "08:00:00" in key
