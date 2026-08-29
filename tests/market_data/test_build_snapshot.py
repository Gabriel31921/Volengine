"""The snapshotting cycle: when a snapshot goes out, and what state moves when it does.

Everything here runs on a ``ManualClock``, so there is no waiting anywhere and no test depends on
how fast the machine is. That is ADR-004 paying for itself at the first opportunity: the cadence
is a business rule about elapsed time, and it is asserted by moving time by hand.
"""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timedelta

import pytest

from tests.market_data.builders import (
    FORWARD,
    NEAR,
    NOW,
    make_chain,
    make_conventions,
    make_instrument,
    make_snapshot_policy,
    make_thresholds,
    make_update,
)
from tests.support import RecordingMetrics
from volengine.market_data.application.build_snapshot import BuildSnapshotUseCase
from volengine.market_data.domain.option_quote import OptionKindD
from volengine.market_data.domain.quote_chain import ChainSnapshot, QuoteChain
from volengine.market_data.domain.snapshot_policy import SnapshotPolicy
from volengine.platform.clock import ManualClock


def make_use_case(
    chain: QuoteChain | None = None,
    policy: SnapshotPolicy | None = None,
    clock: ManualClock | None = None,
    metrics: RecordingMetrics | None = None,
    max_skew_seconds: float = 30.0,
) -> tuple[BuildSnapshotUseCase, QuoteChain, ManualClock, RecordingMetrics]:
    """The use case and the three collaborators a test needs to reach into."""
    chain = chain if chain is not None else make_chain()
    clock = clock if clock is not None else ManualClock(NOW)
    metrics = metrics if metrics is not None else RecordingMetrics()
    use_case = BuildSnapshotUseCase(
        chain=chain,
        policy=policy if policy is not None else make_snapshot_policy(),
        clock=clock,
        metrics=metrics,
        max_skew_seconds=max_skew_seconds,
    )
    return use_case, chain, clock, metrics


def quote_the_chain(chain: QuoteChain, mid: float = 0.052) -> None:
    """Put one two-sided strike on the chain, tight enough to earn no flags."""
    half = 0.002
    for kind in (OptionKindD.CALL, OptionKindD.PUT):
        chain.apply(make_update(kind=kind, bid=mid - half, ask=mid + half))


# --- the emission decision


def test_the_first_quoted_cycle_publishes() -> None:
    use_case, chain, _, _ = make_use_case()
    quote_the_chain(chain)

    assert use_case.build() is not None


def test_an_empty_chain_publishes_nothing() -> None:
    use_case, _, _, _ = make_use_case()

    assert use_case.build() is None


def test_a_second_cycle_inside_the_cadence_publishes_nothing() -> None:
    use_case, chain, clock, _ = make_use_case(policy=make_snapshot_policy(cadence_seconds=10.0))
    quote_the_chain(chain)
    use_case.build()

    clock.advance(1.0)

    assert use_case.build() is None


def test_a_cycle_past_the_cadence_publishes_again() -> None:
    use_case, chain, clock, _ = make_use_case(policy=make_snapshot_policy(cadence_seconds=10.0))
    quote_the_chain(chain)
    use_case.build()

    clock.advance(11.0)

    assert use_case.build() is not None


def test_a_chain_whose_expiries_have_no_forward_publishes_nothing() -> None:
    """Without a forward there is no moneyness, so the slice never reaches the view."""
    use_case, chain, _, _ = make_use_case()
    chain.apply(make_update(underlying_price=None))

    assert use_case.build() is None


class SliceLosingChain(QuoteChain):
    """A chain whose stats claim coverage its view does not deliver.

    **Unreachable through the real aggregate today**, and deliberately reachable here.
    ``QuoteChain.stats`` and ``QuoteChain.snapshot`` both build their slices from the same private
    call at the same instant, so coverage above zero implies at least one slice and the guard in
    the use case cannot fire. That equivalence is an implementation detail of one class, stated
    nowhere and free to change; the guard is what stops a future version of it from publishing an
    empty snapshot that only the calibrator would notice, by failing there and being blamed for it.

    Subclassing the concrete aggregate rather than faking the port, because there is no port: the
    use case holds a ``QuoteChain``, which is this context's aggregate root and not a seam.
    """

    def snapshot(self, now: datetime) -> ChainSnapshot:
        real = super().snapshot(now)
        return replace(real, slices=())


def test_a_view_with_no_slices_is_never_published() -> None:
    """A calibrator handed an empty snapshot can only fail, and would be blamed for it."""
    chain = SliceLosingChain(make_conventions(), make_thresholds())
    use_case, _, _, metrics = make_use_case(chain=chain)
    quote_the_chain(chain)

    assert use_case.build() is None
    assert "marketdata.snapshot.empty" in metrics.counter_names()


def test_an_empty_view_does_not_consume_the_cadence() -> None:
    """The next usable cycle must publish at once rather than wait out another interval.

    This is the reason ``last_emit`` is advanced after the emptiness check and not before it.
    """
    chain = SliceLosingChain(make_conventions(), make_thresholds())
    policy = make_snapshot_policy(cadence_seconds=60.0)
    use_case, _, _, _ = make_use_case(chain=chain, policy=policy)
    quote_the_chain(chain)
    use_case.build()

    assert use_case.last_emit is None


# --- the movement filter, and the baseline the use case owns


def test_a_motionless_chain_is_not_republished() -> None:
    use_case, chain, clock, _ = make_use_case(
        policy=make_snapshot_policy(cadence_seconds=1.0, material_move_threshold=0.01)
    )
    quote_the_chain(chain)
    use_case.build()

    clock.advance(10.0)

    assert use_case.build() is None


def test_a_chain_that_moved_past_the_threshold_is_republished() -> None:
    use_case, chain, clock, _ = make_use_case(
        policy=make_snapshot_policy(cadence_seconds=1.0, material_move_threshold=0.01)
    )
    quote_the_chain(chain)
    use_case.build()

    clock.advance(10.0)
    quote_the_chain(chain, mid=0.060)

    assert use_case.build() is not None


def test_publishing_resets_the_movement_baseline() -> None:
    """Without the reset the same move would keep re-triggering long after it happened."""
    use_case, chain, clock, _ = make_use_case(
        policy=make_snapshot_policy(cadence_seconds=1.0, material_move_threshold=0.01)
    )
    quote_the_chain(chain)
    use_case.build()
    clock.advance(10.0)
    quote_the_chain(chain, mid=0.060)
    use_case.build()

    clock.advance(10.0)

    assert use_case.build() is None


# --- identity and state


def test_successive_snapshots_carry_different_ids() -> None:
    use_case, chain, clock, _ = make_use_case()
    quote_the_chain(chain)
    first = use_case.build()
    clock.advance(5.0)
    quote_the_chain(chain, mid=0.055)
    second = use_case.build()

    assert first is not None and second is not None
    assert first.snapshot_id != second.snapshot_id


def test_the_id_is_built_from_the_chains_own_market() -> None:
    use_case, chain, _, _ = make_use_case()
    quote_the_chain(chain)

    published = use_case.build()

    assert published is not None
    assert published.snapshot_id.startswith("BTC-DERIBIT:")


def test_a_cycle_that_published_nothing_does_not_consume_a_sequence_number() -> None:
    """A gap in the ids would look like a lost snapshot to anyone reading a recording."""
    use_case, chain, _, _ = make_use_case()
    use_case.build()
    quote_the_chain(chain)

    published = use_case.build()

    assert published is not None
    assert published.snapshot_id.endswith("00000000")


def test_the_last_emission_is_readable_and_starts_absent() -> None:
    use_case, chain, _, _ = make_use_case()

    assert use_case.last_emit is None

    quote_the_chain(chain)
    use_case.build()

    assert use_case.last_emit == NOW


# --- what the cycle reports


def test_a_published_snapshot_is_counted() -> None:
    use_case, chain, _, metrics = make_use_case()
    quote_the_chain(chain)
    use_case.build()

    assert "marketdata.snapshot.published" in metrics.counter_names()


def test_the_clock_skew_is_gauged_even_when_it_is_within_the_tolerance() -> None:
    """A venue drifting towards the threshold has to be visible before it arrives."""
    use_case, chain, _, metrics = make_use_case()
    for kind in (OptionKindD.CALL, OptionKindD.PUT):
        chain.apply(make_update(kind=kind, ts_exchange=NOW + timedelta(seconds=3)))

    use_case.build()

    assert metrics.gauge_value("marketdata.clock_skew_seconds") == pytest.approx(3.0)


def test_a_degraded_snapshot_is_counted_separately() -> None:
    use_case, chain, _, metrics = make_use_case(
        policy=make_snapshot_policy(min_coverage_ratio=0.99)
    )
    chain.set_live_instruments(
        [make_instrument(strike=FORWARD + step, expiry=NEAR) for step in (0.0, 1000.0, 2000.0)]
    )
    quote_the_chain(chain)
    use_case.build()

    assert "marketdata.snapshot.degraded" in metrics.counter_names()


def test_a_clean_snapshot_is_not_counted_as_degraded() -> None:
    """The vacuous-pass guard on the test above."""
    use_case, chain, _, metrics = make_use_case()
    quote_the_chain(chain)
    use_case.build()

    assert "marketdata.snapshot.degraded" not in metrics.counter_names()


# --- configuration


@pytest.mark.parametrize("bad", [0.0, -1.0, float("nan"), float("inf")])
def test_an_unusable_skew_tolerance_is_refused_at_construction(bad: float) -> None:
    """Fail when the deployment is built, not minutes into a session on its first snapshot."""
    with pytest.raises(ValueError, match="clock skew"):
        make_use_case(max_skew_seconds=bad)
