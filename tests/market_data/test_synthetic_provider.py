"""The generator the engine is measured on: is its truth true, and is it the same truth twice?

Three kinds of test, and the order matters.

The first kind is about the port -- a fixed live set, a stream that ends, a ``close`` that is
idempotent -- and would apply to any provider.

The second is **known truth**, and it is the reason this adapter exists. Every mid is inverted back
with ``parametric_pricing``'s Black-76 and compared against the total variance the configured SVI
slice claims at that quote's own moneyness. Inverting with the *other* context's code rather than
with the shared kernel that priced it is deliberate: the question is not whether the arithmetic
round-trips, it is whether the next context recovers the surface this one intended. ``tests/`` is
subject to none of the import rules, which is what makes it the only place the two may meet -- and
the same freedom is what lets ``test_the_two_spellings_of_svi_describe_the_same_curve`` watch the
one duplication import rule 6 forces, now that the curve itself is shared (ADR-026) and only the
two value objects are still written twice.

The third is **the spoiling**, which is the half a naive generator skips. Noise that does not move
the quotes, junk that trips no rule and a seed nothing reads are all failures that make every
downstream test pass for the wrong reason, so each of those properties ships with the guard that
would have caught its absence.

The plan's trap for this stage is a shared error that cancels between this generator and the
calibrator that is fitted to it. Nothing in a single stage can rule that out completely -- F2-08
asserts the recovered *parameters* -- but two things here narrow it: the truth is checked through
a second, independently written implementation of Black-76, and the moneyness convention is
recomputed in this file from the published ``underlying_price`` rather than read back out of the
provider.
"""

from __future__ import annotations

import math
import random
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from itertools import pairwise
from typing import Any

import numpy as np
import pytest

from tests.market_data.builders import make_conventions, make_thresholds
from volengine.market_data.adapters.synthetic import (
    DEFAULT_EXPIRIES,
    DEFAULT_TRUE_PARAMS,
    JUNK_STALE_SECONDS,
    SVIParamsSpec,
    SyntheticConfig,
    SyntheticProvider,
)
from volengine.market_data.domain.admissibility import QuoteFlagD, flag_quote
from volengine.market_data.domain.option_quote import OptionKindD, QuoteUpdate
from volengine.market_data.domain.quote_chain import ChainStats, QuoteChain
from volengine.parametric_pricing.domain.black76 import OptionKindP, implied_vol
from volengine.parametric_pricing.domain.durrleman import butterfly_violation, calendar_violation
from volengine.parametric_pricing.domain.svi_slice import SVIParams, SVISlice

START = datetime(2026, 7, 27, 10, 30, tzinfo=UTC)
"""A fixed instant to pin the session to, so every expiry and every stamp below is predictable."""

SESSION: dict[str, Any] = {"cycles": 2, "interval_seconds": 0.0}
"""A short, silent session: no test sleeps, and two cycles is enough to see the forward move."""

TRUE_SLICE: dict[str, Any] = {"a": 0.02, "b": 0.05, "rho": -0.3, "m": 0.0, "sigma": 0.2}
"""One valid slice as keywords, so a rejection test can poison exactly one of the five."""


def make_config(**knobs: Any) -> SyntheticConfig:
    """The default surface over a short session, with the knobs a test is about bent.

    ``**knobs`` rather than sixteen restated parameters: a builder that respelled every field of
    ``SyntheticConfig`` would be a mirror of it, free to drift, and the drift would be silent
    because both sides would still construct. The cost is that a misspelled knob is a ``TypeError``
    at runtime instead of a type error, which a failing test reports just as loudly.
    """
    return SyntheticConfig(**(SESSION | knobs))


def make_provider(config: SyntheticConfig | None = None) -> SyntheticProvider:
    return SyntheticProvider(make_conventions(), config or make_config(), start=START)


async def drain(provider: SyntheticProvider) -> list[QuoteUpdate]:
    return [update async for update in provider.stream()]


def expiry_instants() -> dict[datetime, SVIParamsSpec]:
    """The generating slice for each expiry instant, recomputed here from the configuration.

    Deliberately rebuilt from ``DEFAULT_EXPIRIES`` and the venue's conventions rather than read
    out of the provider: a truth test that asked the provider what the truth was would agree with
    it by construction.
    """
    conventions = make_conventions()
    return {
        conventions.expiry_instant((START + offset).date()): DEFAULT_TRUE_PARAMS[offset]
        for offset in DEFAULT_EXPIRIES
    }


def recovered_total_variance(update: QuoteUpdate) -> float:
    """Invert one mid back to a volatility and re-annualise it into total variance.

    Through ``parametric_pricing``'s inversion, on the forward the update itself published, at the
    tenor the venue's conventions imply. Every convention-dependent step is redone here.
    """
    observation = update.observation
    assert observation.mid is not None
    forward = update.underlying_price
    assert forward is not None
    tenor_years = make_conventions().tenor_years(
        update.instrument.expiry, update.observation.ts_exchange
    )
    vol = implied_vol(
        target_price=observation.mid,
        forward=forward,
        strike=update.instrument.strike,
        tenor_years=tenor_years,
        kind=OptionKindP.CALL if update.instrument.kind is OptionKindD.CALL else OptionKindP.PUT,
    )
    return vol * vol * tenor_years


def true_total_variance(update: QuoteUpdate) -> float:
    """What the generating slice says the total variance is at this quote's own moneyness."""
    forward = update.underlying_price
    assert forward is not None
    return expiry_instants()[update.instrument.expiry].total_variance(
        math.log(update.instrument.strike / forward)
    )


async def stats_of(provider: SyntheticProvider) -> ChainStats:
    """What the real aggregate makes of this feed, one second after the session's first stamp.

    Through ``QuoteChain`` rather than through ``flag_quote`` directly, because the slice-level
    rules -- monotonicity and convexity across strikes -- only exist once a whole expiry is in
    place, and they are the ones a generated price ladder gets wrong.
    """
    chain = QuoteChain(make_conventions(), make_thresholds())
    chain.set_live_instruments(await provider.discover())
    for update in await drain(provider):
        chain.apply(update)
    return chain.snapshot(now=START + timedelta(seconds=1)).stats


# --- the port


async def test_every_strike_of_every_expiry_is_quoted_on_both_sides() -> None:
    """Both legs, though only the out-of-the-money one is fitted (ADR-017).

    The twin is what lets Market Data's put-call parity cross-check compute a second, independent
    forward, so a feed quoting one leg per strike would leave the quality block at ``None``.
    """
    config = make_config()
    provider = make_provider(config)

    assert len(provider.instruments) == len(config.expiries) * config.strikes_per_expiry * 2


async def test_the_live_set_is_exactly_what_the_stream_quotes() -> None:
    """A ``discover`` that disagreed with the stream would report phantom missing instruments.

    Coverage is quotes over *known* instruments, so a live set naming anything the stream never
    sends drags the ratio down for the whole session and marks every snapshot degraded for a
    reason nobody can find in the data.
    """
    provider = make_provider()

    discovered = await provider.discover()

    assert set(discovered) == {update.instrument for update in await drain(provider)}


async def test_the_stream_ends_after_the_configured_number_of_cycles() -> None:
    provider = make_provider(make_config(cycles=3))

    updates = await drain(provider)

    assert len(updates) == 3 * len(provider.instruments)


async def test_closing_ends_the_stream_and_can_be_done_twice() -> None:
    provider = make_provider(make_config(cycles=1000))
    await provider.close()
    await provider.close()

    assert await drain(provider) == []


async def test_the_expiries_land_on_the_venue_expiry_hour() -> None:
    """Built through the conventions, so the chain expires when the venue says (ADR-002)."""
    for instrument in make_provider().instruments:
        assert (instrument.expiry.hour, instrument.expiry.minute) == (8, 0)


async def test_the_nearest_expirys_lowest_strike_is_published_first() -> None:
    """Ordering is load-bearing: the first update of a session is what the first snapshot rests on.

    An out-of-the-money put on the nearest expiry is the best-conditioned quote in the chain, and
    leaving the order to a mapping's iteration order would make the first surface of every run
    depend on how the configuration happened to be written.
    """
    updates = await drain(make_provider())

    first = updates[0].instrument

    assert first.expiry == min(one.instrument.expiry for one in updates)
    assert first.strike == min(one.instrument.strike for one in updates)
    assert first.kind is OptionKindD.PUT


async def test_both_timestamps_are_aware_and_arrival_follows_the_exchange_stamp() -> None:
    """Aware always: staleness is a subtraction of instants and a naive one raises ``TypeError``.

    ``ts_local`` strictly after ``ts_exchange`` however the latency jitter lands, which is what
    keeps the two from ever being accidentally interchangeable.
    """
    for update in await drain(make_provider(make_config(junk_quote_rate=0.0))):
        observation = update.observation
        assert observation.ts_exchange.tzinfo is not None
        assert observation.ts_local > observation.ts_exchange


async def test_the_stamps_jitter_instead_of_freezing_a_whole_cycle_at_one_instant() -> None:
    """A venue does not stamp a chain with one instant, and a feed that did would make every
    measured age an artefact of the generator rather than a property of the data."""
    updates = await drain(make_provider(make_config(cycles=1, junk_quote_rate=0.0)))

    stamps = {update.observation.ts_exchange for update in updates}

    assert len(stamps) == len(updates)


async def test_a_zero_jitter_really_would_have_frozen_them() -> None:
    """The guard on the test above: its assertion is one this feed can genuinely fail."""
    updates = await drain(
        make_provider(make_config(cycles=1, jitter_seconds=0.0, junk_quote_rate=0.0))
    )

    assert len({update.observation.ts_exchange for update in updates}) == 1


# --- reproducibility, which is the requirement rather than the convenience


async def test_the_same_seed_and_start_produce_the_identical_stream() -> None:
    """Bit for bit, including the stamps: this is what makes an end-to-end run assertable.

    Compared as whole ``QuoteUpdate`` objects, which are frozen dataclasses and so compare by
    value across every field -- premiums, sizes, both instants and the walking forward.
    """
    first = await drain(make_provider())
    second = await drain(make_provider())

    assert first == second


async def test_opening_the_feed_again_replays_the_same_session() -> None:
    """The generator is re-seeded when the stream opens, not when the provider is built, so
    reproducibility is a property of the provider rather than of the moment it was constructed."""
    provider = make_provider()

    assert await drain(provider) == await drain(provider)


async def test_a_different_seed_produces_a_different_stream() -> None:
    """The guard on the test above: the streams are equal because the seed is read, not because
    the generator ignores randomness altogether."""
    first = await drain(make_provider())
    second = await drain(make_provider(make_config(seed=1)))

    assert first != second


async def test_a_different_start_moves_the_whole_timeline() -> None:
    provider = SyntheticProvider(
        make_conventions(), make_config(), start=START + timedelta(hours=1)
    )

    assert (await drain(provider))[0].observation.ts_exchange > START + timedelta(minutes=59)


async def test_the_stream_does_not_read_the_module_level_random_generator() -> None:
    """The trap the seed exists to avoid: ``random.gauss`` and friends share one global state, so
    any other module -- a test, a library -- seeding it would silently change this feed."""
    random.seed(1)
    first = await drain(make_provider())
    random.seed(2)
    second = await drain(make_provider())

    assert first == second


# --- known truth


async def test_every_mid_inverts_back_to_the_generating_total_variance() -> None:
    """The whole reason the premiums are computed rather than invented.

    With the noise switched off the feed is exactly the configured surface, so the recovered total
    variance has to match to within the inversion's own tolerance. An explicit ``abs`` at the scale
    of ``w`` -- around 0.03 -- because ``pytest.approx`` passes on ``rel`` *or* ``abs`` and its
    default ``abs`` of 1e-12 would never let the relative bound bind.
    """
    updates = await drain(make_provider(make_config(vol_noise_bp=0.0, junk_quote_rate=0.0)))

    for update in updates:
        assert recovered_total_variance(update) == pytest.approx(
            true_total_variance(update), abs=1e-6
        )


async def test_the_configured_noise_really_moves_the_quotes_off_the_true_surface() -> None:
    """The guard on the test above: it passes because the feed is exact, not because the
    comparison is blind. Half a volatility point of noise moves ``w`` by orders of magnitude more
    than the tolerance the exact case is asserted at."""
    updates = await drain(make_provider(make_config(vol_noise_bp=200.0, junk_quote_rate=0.0)))

    worst = max(
        abs(recovered_total_variance(update) - true_total_variance(update)) for update in updates
    )

    assert worst > 1e-4


async def test_both_legs_of_a_strike_are_priced_from_one_volatility() -> None:
    """Put-call parity, which is what makes Market Data's forward cross-check a check on *us*.

    Perturbing each leg separately would break parity by the size of the noise, and the
    cross-check error the quality block publishes would then be a measurement of this generator
    rather than of the forward the engine derived. Asserted with the noise *on*, since that is the
    case the shared draw exists for.
    """
    updates = await drain(make_provider(make_config(cycles=1, junk_quote_rate=0.0)))
    by_instrument = {
        (update.instrument.expiry, update.instrument.strike, update.instrument.kind): update
        for update in updates
    }

    for (expiry, strike, kind), update in by_instrument.items():
        if kind is not OptionKindD.CALL:
            continue
        put = by_instrument[(expiry, strike, OptionKindD.PUT)]
        call_mid, put_mid = update.observation.mid, put.observation.mid
        assert call_mid is not None and put_mid is not None
        forward = update.underlying_price
        assert forward is not None
        # Undiscounted, because the feed prices at `discount = 1.0`. The scale is the forward
        # itself, tens of thousands, so a micro-unit tolerance is already far tighter than any
        # cross-check threshold anyone would configure.
        assert call_mid - put_mid == pytest.approx(forward - strike, abs=1e-6)


def test_the_generating_surface_is_free_of_butterfly_and_calendar_arbitrage() -> None:
    """A generator quietly producing an arbitrageable truth would make every downstream acceptance
    test meaningless, and nothing in the pipeline would report it.

    Measured with ``parametric_pricing``'s own diagnostics rather than a restatement of them here,
    over a moneyness grid four times wider than the ladder the feed actually quotes: the wings are
    where both conditions break, and they break outside the quoted band first.
    """
    grid = np.linspace(-1.0, 1.0, 401)
    slices = [
        SVISlice(
            expiry=START + offset,
            tenor_years=offset.total_seconds() / (365.0 * 24 * 3600),
            params=_as_fitted(DEFAULT_TRUE_PARAMS[offset]),
            k_min=-1.0,
            k_max=1.0,
        )
        for offset in sorted(DEFAULT_EXPIRIES)
    ]

    for slice_ in slices:
        assert butterfly_violation(slice_.params, grid) == 0.0
    for near, far in pairwise(slices):
        assert calendar_violation(near, far, grid) == 0.0


def test_the_two_spellings_of_svi_describe_the_same_curve() -> None:
    """Import rule 6 forces this generator to carry its own SVI *type*. Not its own curve.

    The two types are deliberately different -- a specification of what to generate is not the
    output of a fit -- but the mathematics was never allowed to diverge, and since ADR-026 it
    cannot: both spellings call ``shared_kernel.domain.svi``, so this passes by construction.

    It stays for the reason the same guard stays in ``tests/neural_surface/test_acl.py``. The
    risk was never that someone breaks one copy, it is that someone *re-inlines* one -- and here
    that edit is invisible from either side, because this context generates the surface the
    other context's calibrator is measured against and a sign slip on both sides cancels. That
    is the trap this stage was warned about, and this is the assertion that fails if the formula
    comes back.
    """
    spec = SVIParamsSpec(a=0.02, b=0.05, rho=-0.3, m=-0.01, sigma=0.2)
    fitted = _as_fitted(spec)

    for k in (-0.4, -0.1, 0.0, 0.1, 0.4):
        assert spec.total_variance(k) == pytest.approx(fitted.total_variance(k), abs=1e-15)
    assert spec.min_total_variance == pytest.approx(fitted.min_total_variance, abs=1e-15)


def test_a_slice_annualises_its_total_variance_by_the_tenor() -> None:
    spec = SVIParamsSpec(a=0.04, b=0.0, rho=0.0, m=0.0, sigma=0.2)

    assert spec.volatility(k=0.0, tenor_years=0.25) == pytest.approx(0.4, abs=1e-12)


@pytest.mark.parametrize("tenor", [0.0, -1.0, float("nan"), float("inf")])
def test_a_slice_refuses_to_annualise_over_a_tenor_that_is_not_one(tenor: float) -> None:
    """NaN included, and it is why the guard is written ``not isfinite(x) or x <= 0``: a NaN
    compares ``False`` against everything, so it walks straight through an ordering test."""
    with pytest.raises(ValueError, match="tenor must be positive"):
        SVIParamsSpec(a=0.04, b=0.0, rho=0.0, m=0.0, sigma=0.2).volatility(k=0.0, tenor_years=tenor)


# --- the spoiling, which is what makes the filters testable at all


async def test_a_clean_chain_is_wholly_admissible_under_ordinary_thresholds() -> None:
    """With the junk switched off, ordinary microstructure noise must not flag anything.

    Counted in ``n_quotes_admissible``, the number of quotes carrying **no** flag, against every
    instrument the chain knows about. Deliberately not ``coverage_ratio``, which counts every
    quote a usable mid arrived for -- stale, crossed and absurdly wide ones included -- and so
    reaches 1.0 on a chain that is entirely flagged.
    """
    stats = await stats_of(make_provider(make_config(junk_quote_rate=0.0)))

    assert stats.n_quotes_admissible == stats.n_instruments_known


async def test_junk_quotes_reach_the_chain_as_flagged_and_uncovered_quotes() -> None:
    """The other half: with every quote spoiled, the same two numbers must both collapse.

    ``coverage_ratio`` falls because ``ONE_SIDED`` leaves no mid to cover with, and
    ``n_quotes_admissible`` falls much further because the other four kinds arrive flagged.
    """
    stats = await stats_of(make_provider(make_config(junk_quote_rate=1.0)))

    assert stats.n_quotes_admissible < stats.n_instruments_known
    assert stats.coverage_ratio < 1.0


async def test_every_junk_kind_trips_the_rule_it_is_named_for() -> None:
    """Junk exists to make the rules of Design 4.3 fire, and a rule that never fires in any test
    is a rule nobody knows works.

    Deterministic despite being random: the seed is fixed, so this is an assertion about one known
    stream rather than a probabilistic one. ``ONE_SIDED`` is checked separately because it is the
    kind that raises no flag at all -- an empty side has no mid, so there is no width to judge and
    no premium to invert.
    """
    updates = await drain(make_provider(make_config(cycles=1, junk_quote_rate=1.0)))

    raised = {
        flag
        for update in updates
        for flag in flag_quote(
            update.observation,
            strike=update.instrument.strike,
            forward=update.underlying_price or 1.0,
            own_iv=None,
            thresholds=make_thresholds(),
            now=START + timedelta(seconds=1),
        )
    }

    assert raised == {
        QuoteFlagD.CROSSED,
        QuoteFlagD.WIDE_SPREAD,
        QuoteFlagD.LOW_SIZE,
        QuoteFlagD.STALE,
    }
    assert any(update.observation.mid is None for update in updates)


async def test_a_stale_junk_quote_is_backdated_far_enough_for_any_threshold() -> None:
    """The provider cannot read the thresholds -- they are the engine's judgement, not the venue's
    -- so the backdate has to be past any ``max_age_seconds`` a deployment would configure."""
    updates = await drain(make_provider(make_config(cycles=1, junk_quote_rate=1.0)))

    oldest = min(update.observation.ts_exchange for update in updates)

    assert (START - oldest).total_seconds() >= JUNK_STALE_SECONDS - 1.0


async def test_no_junk_is_produced_at_a_zero_rate() -> None:
    """The guard on the tests above: the chain is dirty because junk was asked for."""
    updates = await drain(make_provider(make_config(junk_quote_rate=0.0)))

    for update in updates:
        observation = update.observation
        assert observation.bid is not None and observation.ask is not None
        assert observation.bid < observation.ask
        assert observation.ts_exchange > START - timedelta(seconds=1)


async def test_the_forward_walks_between_cycles() -> None:
    """What makes this a stream rather than a still life, and what gives the snapshot policy's
    material-move filter something to have an opinion about."""
    provider = make_provider(make_config(cycles=4))
    updates = await drain(provider)
    per_cycle = len(provider.instruments)

    forwards = {updates[cycle * per_cycle].underlying_price for cycle in range(4)}

    assert len(forwards) == 4


async def test_a_pinned_forward_does_not_move() -> None:
    """The guard: the forward moves because a move was configured, not because it is unstable."""
    provider = make_provider(make_config(cycles=4, forward_move_rel=0.0))

    assert {update.underlying_price for update in await drain(provider)} == {
        provider.config.forward0
    }


async def test_the_forward_walking_moves_the_moneyness_of_a_fixed_strike() -> None:
    """A venue lists absolute strikes and does not relist them when the underlying moves, so the
    ladder is fixed and the smile slides across it. That drift is what a calibrator has to track,
    and a feed that recentred its strikes every cycle would never present it."""
    provider = make_provider(make_config(cycles=2))
    updates = await drain(provider)
    per_cycle = len(provider.instruments)

    first, second = updates[0], updates[per_cycle]

    assert first.instrument == second.instrument
    assert first.underlying_price != second.underlying_price


async def test_the_sizes_wobble_around_the_configured_one() -> None:
    updates = await drain(make_provider(make_config(cycles=1, junk_quote_rate=0.0)))
    size = make_config().size

    sizes = [update.observation.bid_size for update in updates]

    assert len(set(sizes)) > 1
    assert all(0 < one < 2 * size for one in sizes)


async def test_no_exchange_volatility_is_published() -> None:
    """A synthetic feed publishing the number its own premiums were built from would make
    ``IV_DIVERGENCE`` agree with itself by construction, and a check that cannot fail is worse
    than an absent one because it reads like a check."""
    assert all(update.observation.exchange_iv is None for update in await drain(make_provider()))


async def test_a_ludicrous_noise_level_still_produces_quotable_premiums() -> None:
    """The volatility floor, which exists because additive noise on a low wing can cross zero and
    Black-76 refuses a non-positive volatility -- one quote in ten thousand, a long way from the
    line that chose the number."""
    updates = await drain(make_provider(make_config(vol_noise_bp=50_000.0, junk_quote_rate=0.0)))

    for update in updates:
        assert update.observation.mid is not None
        assert math.isfinite(update.observation.mid)


# --- construction


@pytest.mark.parametrize(
    ("build", "message"),
    [
        (lambda: make_config(forward0=0.0), "forward0 must be positive"),
        (lambda: make_config(forward0=float("nan")), "forward0 must be positive"),
        (lambda: make_config(expiries=()), "at least one expiry"),
        (lambda: make_config(strikes_per_expiry=1), "at least two strikes"),
        (lambda: make_config(log_moneyness_range=(0.3, 0.1)), "log_moneyness_range"),
        (lambda: make_config(log_moneyness_range=(float("nan"), 0.1)), "log_moneyness_range"),
        (lambda: make_config(spread_bp=0.0), "spread_bp must be positive"),
        (lambda: make_config(vol_noise_bp=-1.0), "vol_noise_bp must be non-negative"),
        (lambda: make_config(size=float("-inf")), "size must be non-negative"),
        (lambda: make_config(junk_quote_rate=1.5), "junk_quote_rate must lie inside"),
        (lambda: make_config(junk_quote_rate=float("nan")), "junk_quote_rate must lie inside"),
        (lambda: make_config(forward_move_rel=-0.1), "forward_move_rel must be non-negative"),
        (lambda: make_config(latency_seconds=0.0), "latency_seconds must be positive"),
        (lambda: make_config(jitter_seconds=-1.0), "jitter_seconds must be non-negative"),
        (lambda: make_config(cycles=0), "at least one cycle"),
        (lambda: make_config(interval_seconds=-1.0), "interval_seconds must be non-negative"),
    ],
)
def test_a_market_that_could_not_be_quoted_is_refused(
    build: Callable[[], SyntheticConfig], message: str
) -> None:
    """Plain ``ValueError``: a bad knob here is a wiring bug, not a market condition.

    Parametrised over calls rather than over ``(field, value)`` pairs, because a
    ``**{field: value}`` splat is exactly the shape that defeats mypy at every call site.
    """
    with pytest.raises(ValueError, match=message):
        build()


def test_an_expiry_without_parameters_is_refused() -> None:
    """The one invariant that spans two fields: the ladder of expiries and the surface must name
    the same set, or a slice would be generated from parameters nobody wrote."""
    with pytest.raises(ValueError, match="exactly one slice of true parameters"):
        make_config(expiries=(timedelta(days=7), *DEFAULT_EXPIRIES))


def test_a_repeated_expiry_is_refused() -> None:
    with pytest.raises(ValueError, match="must be distinct"):
        make_config(expiries=(timedelta(days=30), timedelta(days=30)))


def test_an_expiry_that_is_not_ahead_of_the_session_start_is_refused() -> None:
    with pytest.raises(ValueError, match="must be in the future"):
        make_config(
            expiries=(timedelta(0),),
            true_params={timedelta(0): DEFAULT_TRUE_PARAMS[timedelta(days=30)]},
        )


def test_an_expiry_the_session_would_outlive_is_refused_at_construction() -> None:
    """A mid-stream ``ExpiredInstrumentError`` would reach a consumer as a crash. The session's
    own length is known at construction, so an expiry the run outlives is knowable there too."""
    near = timedelta(hours=12)
    config = make_config(
        expiries=(near,),
        true_params={near: DEFAULT_TRUE_PARAMS[timedelta(days=30)]},
        cycles=10,
        interval_seconds=100_000.0,
    )

    with pytest.raises(ValueError, match="would outlive"):
        SyntheticProvider(make_conventions(), config, start=START)


def test_two_expiries_landing_on_one_instant_are_refused() -> None:
    """Distinct timedeltas are not distinct expiries: the venue's hour snaps both onto the same
    date, and the chain would then quote one instrument twice."""
    morning, evening = timedelta(days=30), timedelta(days=30, hours=6)
    config = make_config(
        expiries=(morning, evening),
        true_params={
            morning: DEFAULT_TRUE_PARAMS[timedelta(days=30)],
            evening: DEFAULT_TRUE_PARAMS[timedelta(days=90)],
        },
    )

    with pytest.raises(ValueError, match="same instant"):
        SyntheticProvider(make_conventions(), config, start=START)


def test_a_naive_start_is_refused() -> None:
    """Every subtraction in this engine mixes these instants with aware ones, and mixing the two
    kinds raises ``TypeError`` a long way from here."""
    with pytest.raises(ValueError, match="start must be timezone-aware"):
        SyntheticProvider(make_conventions(), make_config(), start=datetime(2026, 7, 27, 10, 30))


def test_the_default_start_is_the_wall_clock_and_carries_a_zone() -> None:
    """``datetime.now(UTC)``, never ``utcnow()``, which returns a naive value despite its name."""
    provider = SyntheticProvider(make_conventions(), make_config())

    assert provider.start.tzinfo is not None


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("a", float("nan"), "parameter a must be finite"),
        ("b", -0.1, "parameter b must be non-negative"),
        ("rho", 1.0, "parameter rho must be inside"),
        ("rho", -1.0, "parameter rho must be inside"),
        ("sigma", 0.0, "parameter sigma must be positive"),
        ("m", float("inf"), "parameter m must be finite"),
    ],
)
def test_a_slice_that_is_not_a_surface_is_refused(field: str, value: float, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        SVIParamsSpec(**(TRUE_SLICE | {field: value}))


def test_a_slice_that_would_price_at_zero_volatility_is_refused() -> None:
    """Stricter than the fitted ``SVIParams``, which admits a zero minimum because an optimiser
    walks through one. This is handwritten configuration and is never an intermediate state, so a
    collapsed slice here is only ever a mistake -- one that would surface as the shared kernel
    refusing a non-positive volatility, one quote at a time."""
    with pytest.raises(ValueError, match="strictly positive"):
        SVIParamsSpec(a=-0.05, b=0.05, rho=0.0, m=0.0, sigma=0.2)


def _as_fitted(spec: SVIParamsSpec) -> SVIParams:
    """The same five numbers in ``parametric_pricing``'s spelling, for the cross-context tests."""
    return SVIParams(a=spec.a, b=spec.b, rho=spec.rho, m=spec.m, sigma=spec.sigma)
