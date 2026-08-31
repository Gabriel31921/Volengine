"""The walking skeleton's feed: what it publishes, and that the engine can actually use it.

Two kinds of test here. The first kind is about the port -- a fixed set, a stream that ends, a
``close`` that is idempotent -- and would apply to any provider. The second kind is the one that
matters: a provider whose premiums no volatility reproduces would be dropped quote by quote by the
ACL one context over, and the pipeline would compose, subscribe, publish nothing and look exactly
like a market that is not moving. So the prices are inverted back here, and the chain is pushed
through a real ``QuoteChain``, because those are the two ways this adapter can be silently useless.

Inverting with ``parametric_pricing``'s own Black-76 rather than with the shared kernel's, even
though the kernel is what priced them: the question is not whether the arithmetic round-trips, it
is whether the *next context* recovers the volatility this one intended. Only ``tests/`` may ask
that, being subject to none of the import rules.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from datetime import UTC, datetime, timedelta

import pytest

from tests.market_data.builders import make_conventions, make_thresholds
from volengine.market_data.adapters.constant import (
    FORWARD,
    LOG_MONEYNESS,
    SPREAD_REL,
    TENOR_DAYS,
    VOL,
    ConstantProvider,
)
from volengine.market_data.domain.option_quote import OptionKindD, QuoteUpdate
from volengine.market_data.domain.quote_chain import ChainStats, QuoteChain
from volengine.parametric_pricing.domain.black76 import OptionKindP, implied_vol
from volengine.parametric_pricing.domain.errors import NoImpliedVolError

START = datetime(2026, 7, 27, 10, 30, tzinfo=UTC)
"""A fixed instant to pin the chain to, so every expiry below is predictable."""


def make_provider(
    cycles: int = 1,
    interval_seconds: float = 0.0,
    now: datetime = START,
    spread_rel: float = SPREAD_REL,
) -> ConstantProvider:
    """A provider on the default chain, pinned to one instant so nothing here reads a clock."""
    return ConstantProvider(
        make_conventions(),
        spread_rel=spread_rel,
        cycles=cycles,
        interval_seconds=interval_seconds,
        now=lambda: now,
    )


async def drain(provider: ConstantProvider) -> list[QuoteUpdate]:
    return [update async for update in provider.stream()]


async def stats_of(provider: ConstantProvider) -> ChainStats:
    """What the real aggregate makes of this provider's chain, one second after it was quoted.

    Through ``QuoteChain`` rather than through ``flag_quote`` directly, because the slice-level
    rules -- monotonicity and convexity across strikes -- only exist once a whole expiry is in
    place, and they are the ones a hand-written price ladder gets wrong.
    """
    chain = QuoteChain(make_conventions(), make_thresholds())
    chain.set_live_instruments(await provider.discover())
    for update in await drain(provider):
        chain.apply(update)
    return chain.snapshot(now=START + timedelta(seconds=1)).stats


# --- the port


async def test_every_instrument_is_quoted_on_both_sides_of_every_strike() -> None:
    provider = make_provider()

    assert len(provider.instruments) == len(TENOR_DAYS) * len(LOG_MONEYNESS) * 2


async def test_the_live_set_is_exactly_what_the_stream_quotes() -> None:
    """A ``discover`` that disagreed with the stream would report phantom missing instruments.

    Coverage is admissible quotes over *known* instruments, so a live set naming anything the
    stream never sends drags the ratio down for the whole session and marks every snapshot
    degraded for a reason nobody can find in the data.
    """
    provider = make_provider()

    discovered = await provider.discover()

    assert set(discovered) == {update.instrument for update in await drain(provider)}


async def test_the_stream_ends_after_the_configured_number_of_cycles() -> None:
    """Finite on purpose: the skeleton has to be something a person can watch end."""
    provider = make_provider(cycles=3)

    updates = await drain(provider)

    assert len(updates) == 3 * len(provider.instruments)


async def test_closing_ends_the_stream_and_can_be_done_twice() -> None:
    provider = make_provider(cycles=1000)
    await provider.close()
    await provider.close()

    assert await drain(provider) == []


async def test_the_timestamps_come_from_the_injected_clock_and_carry_a_zone() -> None:
    """Aware, always: staleness is a subtraction of instants and a naive one raises TypeError."""
    provider = make_provider(now=START)

    observation = (await drain(provider))[0].observation

    assert observation.ts_exchange == START
    assert observation.ts_local > observation.ts_exchange
    assert observation.ts_local.tzinfo is not None


async def test_the_same_instrument_is_republished_unchanged_across_cycles() -> None:
    """A full replacement of the top of book, never a delta -- and constant is what it says."""
    updates = await drain(make_provider(cycles=2))
    first, second = updates[0], updates[len(updates) // 2]

    assert first.instrument == second.instrument
    assert (first.observation.bid, first.observation.ask) == (
        second.observation.bid,
        second.observation.ask,
    )


# --- the quotes, which is what makes it usable


async def test_every_mid_inverts_back_to_the_volatility_it_was_priced_from() -> None:
    """The whole reason the premiums are computed rather than invented.

    One basis point of tolerance, which is a hundred times finer than any acceptance threshold
    anyone would configure and a thousand times coarser than the inversion's own tolerance.
    """
    provider = make_provider()

    for update in await drain(provider):
        observation = update.observation
        assert observation.mid is not None
        recovered = implied_vol(
            target_price=observation.mid,
            forward=FORWARD,
            strike=update.instrument.strike,
            tenor_years=make_conventions().tenor_years(update.instrument.expiry, START),
            kind=(
                OptionKindP.CALL if update.instrument.kind is OptionKindD.CALL else OptionKindP.PUT
            ),
        )
        assert recovered == pytest.approx(VOL, abs=1e-4)


async def test_a_premium_invented_instead_of_priced_would_not_survive_the_inversion() -> None:
    """The guard on the test above: it is not passing because inversion accepts anything.

    Halving the deepest in-the-money call's mid puts it below intrinsic, where no volatility
    reproduces it and the ACL drops the quote. That is precisely the failure a provider making up
    plausible numbers would produce, one quote at a time, in silence.
    """
    update = next(
        one
        for one in await drain(make_provider())
        if one.instrument.kind is OptionKindD.CALL
        and one.instrument.strike < FORWARD * math.exp(-0.15)
    )
    mid = update.observation.mid
    assert mid is not None

    with pytest.raises(NoImpliedVolError):
        implied_vol(
            target_price=mid / 2,
            forward=FORWARD,
            strike=update.instrument.strike,
            tenor_years=make_conventions().tenor_years(update.instrument.expiry, START),
            kind=OptionKindP.CALL,
        )


async def test_the_whole_chain_is_admissible_under_ordinary_thresholds() -> None:
    """A chain that arrived pre-flagged would publish snapshots nothing downstream trusts.

    Counted in ``n_quotes_admissible``, which is the number of published quotes carrying **no
    flag**, against every instrument the chain knows about. Deliberately *not* ``coverage_ratio``:
    that one counts every quote a usable mid arrived for, stale, crossed and absurdly wide ones
    included, so it reaches 1.0 on a chain that is entirely flagged -- see the guard below.
    """
    stats = await stats_of(make_provider())

    assert stats.n_quotes_admissible == stats.n_instruments_known


async def test_a_chain_quoted_too_wide_really_would_have_been_caught() -> None:
    """The guard on the test above: its assertion is one this provider can genuinely fail.

    A spread of 90 per cent against a 50 per cent threshold flags every quote in the chain, and
    ``n_quotes_admissible`` collapses to zero. The second assertion is why the first test does not
    use ``coverage_ratio``: the fully flagged chain still covers every instrument, so a test
    written on coverage would have passed here and guarded nothing.
    """
    stats = await stats_of(make_provider(spread_rel=0.9))

    assert stats.n_quotes_admissible == 0
    assert stats.coverage_ratio == 1.0


async def test_the_expiries_land_on_the_venue_expiry_hour() -> None:
    """Built through the conventions, so the chain expires when the venue says, not at start-up."""
    provider = make_provider()

    for instrument in provider.instruments:
        assert (instrument.expiry.hour, instrument.expiry.minute) == (8, 0)


async def test_the_near_expiry_is_published_first() -> None:
    """Ordering is load-bearing: the first update of a session is what the first snapshot rests on.

    An out-of-the-money put on the nearest expiry is the best-conditioned quote in the chain, and
    leaving the order to a dictionary would make the first surface of every run depend on nothing
    anybody chose.
    """
    updates = await drain(make_provider())

    first = updates[0].instrument

    assert first.expiry == min(one.instrument.expiry for one in updates)
    assert first.strike == pytest.approx(FORWARD * math.exp(min(LOG_MONEYNESS)))
    assert first.kind is OptionKindD.PUT


# --- construction


@pytest.mark.parametrize(
    ("build", "message"),
    [
        (lambda: ConstantProvider(make_conventions(), forward=0.0), "forward must be positive"),
        (
            lambda: ConstantProvider(make_conventions(), forward=float("nan")),
            "forward must be positive",
        ),
        (lambda: ConstantProvider(make_conventions(), vol=-0.1), "volatility must be positive"),
        (
            lambda: ConstantProvider(make_conventions(), size=float("-inf")),
            "size must be non-negative",
        ),
        (lambda: ConstantProvider(make_conventions(), spread_rel=0.0), "relative spread"),
        (lambda: ConstantProvider(make_conventions(), spread_rel=2.0), "relative spread"),
        (lambda: ConstantProvider(make_conventions(), cycles=0), "at least one cycle"),
        (
            lambda: ConstantProvider(make_conventions(), interval_seconds=-1.0),
            "interval between cycles",
        ),
    ],
)
def test_a_chain_that_could_not_be_quoted_is_refused(
    build: Callable[[], ConstantProvider], message: str
) -> None:
    """Plain ``ValueError``: a bad argument here is a wiring bug, not a market condition.

    Parametrised over constructor calls rather than over ``(field, value)`` pairs, because a
    ``**{field: value}`` splat is exactly the shape that defeats mypy at every call site -- the
    same reason ``tests/support.py`` owns the one deliberately dynamic ``replace``.
    """
    with pytest.raises(ValueError, match=message):
        build()


def test_a_chain_with_no_strike_or_no_expiry_is_refused() -> None:
    with pytest.raises(ValueError, match="at least one strike"):
        ConstantProvider(make_conventions(), log_moneyness=())
    with pytest.raises(ValueError, match="at least one expiry"):
        ConstantProvider(make_conventions(), tenor_days=())


def test_a_tenor_that_has_already_expired_is_refused_at_construction() -> None:
    """Not emptiness but an element, and the reason the elements are checked at all: a zero tenor
    is a legal float that escapes mid-stream as ``ExpiredInstrumentError``, blaming the instrument
    for an argument. The constructor documents a plain ``ValueError``, so it has to raise one."""
    with pytest.raises(ValueError, match="positive finite number of days"):
        ConstantProvider(make_conventions(), tenor_days=(0.0,))


def test_a_tenor_of_nan_is_refused_rather_than_handed_to_the_calendar() -> None:
    """``nan > 0`` is ``False``, so this is caught by the ordering half of the guard as much as by
    ``isfinite`` -- but unguarded it reaches ``int()`` and comes back as the stdlib's "cannot
    convert float NaN to integer", which names neither the argument nor the provider."""
    with pytest.raises(ValueError, match="positive finite number of days"):
        ConstantProvider(make_conventions(), tenor_days=(30.0, float("nan")))


def test_a_strike_ladder_carrying_a_nan_is_refused() -> None:
    """A NaN log-moneyness prices a NaN premium at every expiry, and the whole chain arrives
    downstream as quotes no constructor rejects one at a time."""
    with pytest.raises(ValueError, match="log-moneyness must be finite"):
        ConstantProvider(make_conventions(), log_moneyness=(0.0, float("inf")))
