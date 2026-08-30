"""A known SVI surface, quoted like a venue quotes it: the generator the engine is measured on.

``ConstantProvider`` proves the port is usable. This one proves the *engine* is right, and the two
jobs are different enough to be two adapters. A chain priced from a single flat volatility has no
smile, no skew and no term structure, so a calibrator that fitted nothing but the level would look
perfect on it. Here the truth is five parameters per expiry and a whole surface -- and because
those parameters are known before the run starts, a fit can be checked against *what generated the
quotes* rather than against its own residuals.

**The premiums are real Black-76 prices of a real SVI surface.** Total variance
``w(k) = a + b (rho (k - m) + sqrt((k - m)^2 + sigma^2))`` is evaluated at the quote's own
log-forward-moneyness, annualised into a volatility by the tenor the venue's conventions imply,
and priced through the shared kernel. Everything downstream -- the inversion in Market Data's
sibling context, the fit, the published grid, the report -- therefore has a right answer to
recover, and any disagreement is a defect somewhere in the engine rather than in the feed.

**Then it is spoiled on purpose.** A clean surface tests arithmetic, not an engine. So the mid is
perturbed by a configured amount of volatility noise, the two sides are quoted around it, the
sizes wobble, the timestamps jitter, the forward walks between cycles, and a configured fraction
of the quotes come out as junk of a named kind -- crossed, absurdly wide, one-sided, sizeless or
an hour old. Those exist to make the admissibility rules of Design 4.3 *fire*: a flag that never
fires in any test is a flag nobody knows works.

**Reproducible, and that is a hard requirement rather than a convenience.** Every random draw comes
from one ``random.Random`` seeded from the configuration, drawn in a fixed order, and the whole
timeline is derived from a single ``start`` instant. Pass the same ``seed`` and the same ``start``
and the byte-for-byte identical stream comes out, which is what makes the end-to-end test of
Design 9 assertable at all.

That is the second of the two options ``docs/SEAMS.md`` names for this stage, and the seam it
records stays open: an adapter still cannot be handed the engine's ``Clock``, because
``ProviderFactory`` takes a ``MarketConfig`` and nothing else (ADR-022). This feed owns its own
timeline instead of reading the engine's, which makes *it* reproducible without making a whole
pipeline run under a ``ManualClock`` reproducible -- for that, ``start`` and the engine's clock
would have to be the same source, and that is a change to the factory signature, not to this file.
The module-level ``random`` functions are never used: they share one global generator, so any
other module seeding it -- a test, a library -- would silently change this stream.

**The SVI *type* is written out here; the SVI *mathematics* is imported.** ``parametric_pricing``
owns an ``SVIParams`` with the same five fields and it must not be imported: no context imports
another (import rule 6), and the reason is not bureaucratic. These five numbers are a
*specification of what to generate*, chosen by hand and fixed for a run; those five are the
*output of a fit*, carrying a validity band, a tenor and an optimiser's need to hold intermediate
iterates that are not yet surfaces. If the calibrator's type ever needed to change, this one would
not want the change -- which is the test ``CLAUDE.md`` sets for duplication, answered *it would
break the other side*.

The curve those numbers describe is a different question, and it gets the opposite answer.
``w(k)`` is mathematics, not a model choice: swapping SVI for SSVI or a stochastic-volatility
model means a different implementation arriving behind the ``Calibrator`` port, not this
expression coming to mean something else. So it lives once, in ``shared_kernel/domain/svi.py``,
and both spellings call it (ADR-026). That matters here more than anywhere: the surface this
adapter generates is what the calibrator is *measured against*, so two copies of one formula
would be one sign slip away from cancelling on both sides and passing the known-truth test with
the engine wrong -- which is the trap this stage was warned about.

**Nothing here is a threshold** in the sense ADR-012 governs. Cadence, admissibility and
acceptance are judgements the engine makes about a market and live in TOML; these are the
properties of an invented market, and the fixture that invents it is the right place for them.
"""

from __future__ import annotations

import asyncio
import math
import random
from collections.abc import AsyncIterator, Mapping
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from types import MappingProxyType
from typing import assert_never

from volengine.market_data.domain.admissibility import BASIS_POINTS_PER_UNIT
from volengine.market_data.domain.market_conventions import MarketConventions
from volengine.market_data.domain.option_quote import (
    InstrumentId,
    OptionKindD,
    QuoteObservation,
    QuoteUpdate,
)
from volengine.shared_kernel.domain import svi
from volengine.shared_kernel.domain.black76 import price
from volengine.shared_kernel.domain.instants import require_aware


@dataclass(frozen=True, slots=True)
class SVIParamsSpec:
    """The five raw SVI parameters of one expiry, as a *specification* of what to generate.

    Total variance in log-forward-moneyness, tenor-free by construction -- the tenor is already
    inside ``w``, which is exactly why an SVI slice can be held fixed while the session runs and
    the volatilities it implies rise as the expiry approaches.

    Its invariants are **stricter** than the fitted ``SVIParams`` one context over, and
    deliberately so. That type must admit a collapsed slice because an optimiser walks through
    them and the violation has to stay measurable; this one is handwritten configuration, it is
    never the intermediate state of anything, and a zero total variance here would be asked to
    price an option at zero volatility -- which the shared kernel refuses, one quote at a time,
    a long way from the line that chose the number.
    """

    a: float
    """Vertical level of the curve, in total-variance units. Finite."""

    b: float
    """Wing steepness, half the difference of the two asymptotic slopes. Non-negative, finite."""

    rho: float
    """Skew between the wings. Strictly inside ``(-1, 1)``; negative lifts the downside."""

    m: float
    """Where the smile's minimum sits, in log-forward-moneyness. Finite, and legitimately zero,
    so it must never be tested for truthiness."""

    sigma: float
    """Curvature at the bottom of the smile. Strictly positive and finite. A shape parameter of
    the hyperbola, not a volatility, despite the name the literature settled on."""

    def __post_init__(self) -> None:
        for name, value in (
            ("a", self.a),
            ("b", self.b),
            ("rho", self.rho),
            ("m", self.m),
            ("sigma", self.sigma),
        ):
            if not math.isfinite(value):
                raise ValueError(f"The SVI parameter {name} must be finite, got {value}")
        # Every guard joins the bad conditions with `or` after an explicit finiteness check: a
        # NaN compares False against everything, so `nan <= 0` would sail through an ordering
        # test that reads correct.
        if self.b < 0:
            raise ValueError(f"The SVI parameter b must be non-negative, got {self.b}")
        if abs(self.rho) >= 1:
            raise ValueError(f"The SVI parameter rho must be inside (-1, 1), got {self.rho}")
        if self.sigma <= 0:
            raise ValueError(f"The SVI parameter sigma must be positive, got {self.sigma}")
        if self.min_total_variance <= 0:
            raise ValueError(
                "The minimum total variance of a generated slice must be strictly positive, got "
                f"{self.min_total_variance}"
            )

    @property
    def min_total_variance(self) -> float:
        """Lowest value the curve attains, ``a + b * sigma * sqrt(1 - rho^2)``.

        The shared kernel's, not this module's: the closed form is the same fact wherever it is
        evaluated (ADR-026). Called during ``__post_init__`` **after** the guard on ``rho``, so
        the kernel's own domain check can never be the one that fires.
        """
        return svi.min_total_variance(a=self.a, b=self.b, rho=self.rho, sigma=self.sigma)

    def total_variance(self, k: float) -> float:
        """``w(k) = a + b * (rho * (k - m) + sqrt((k - m)^2 + sigma^2))``, from the kernel.

        The one place this generator and the calibrator it is measured against are *required*
        to agree, and since ADR-026 they agree by construction rather than by a test.
        """
        return svi.total_variance(k, a=self.a, b=self.b, rho=self.rho, m=self.m, sigma=self.sigma)

    def volatility(self, k: float, tenor_years: float) -> float:
        """``sqrt(w(k) / T)``: the annualised volatility this slice implies at that moneyness.

        Raises:
            ValueError: If the tenor is not positive and finite. A non-positive tenor has no
                volatility to imply, and dividing by it would produce an ``inf`` that every
                downstream constructor would then have to catch.
        """
        _require_positive_finite(tenor_years, "tenor")
        return math.sqrt(self.total_variance(k) / tenor_years)


class JunkKind(StrEnum):
    """The shapes of broken quote this feed produces on purpose, one per rule it must trip.

    Named and enumerated rather than improvised inside the generator, because "occasional junk"
    is only useful if a test can state *which* junk it expects and check that the corresponding
    admissibility rule reacted. A generator that produced an unnameable mess would exercise the
    filters and prove nothing about them.

    Values are spelled like the flags they are meant to raise, except ``ONE_SIDED``, which is
    the one kind that raises no flag at all -- an empty side has no mid, so there is no width to
    judge and no premium to invert. It drags ``coverage_ratio`` down instead, which is the other
    half of the quality block and has no other way of being exercised.
    """

    CROSSED = "CROSSED"
    WIDE_SPREAD = "WIDE_SPREAD"
    ONE_SIDED = "ONE_SIDED"
    LOW_SIZE = "LOW_SIZE"
    STALE = "STALE"


JUNK_KINDS: tuple[JunkKind, ...] = tuple(JunkKind)
"""Every kind, in declaration order, so the draw that picks one is reproducible."""

DEFAULT_FORWARD = 60_000.0
"""Initial forward, in the quote currency. BTC-shaped, because the first market is Deribit."""

DEFAULT_EXPIRIES = (timedelta(days=30), timedelta(days=90), timedelta(days=180))
"""Three tenors, so the published surface has a term structure a report can interpolate along
rather than a single slice everything is extrapolated from."""

DEFAULT_TRUE_PARAMS: Mapping[timedelta, SVIParamsSpec] = MappingProxyType(
    {
        timedelta(days=30): SVIParamsSpec(a=0.020, b=0.050, rho=-0.30, m=0.0, sigma=0.20),
        timedelta(days=90): SVIParamsSpec(a=0.055, b=0.090, rho=-0.25, m=0.0, sigma=0.25),
        timedelta(days=180): SVIParamsSpec(a=0.100, b=0.130, rho=-0.20, m=0.0, sigma=0.30),
    }
)
"""The surface the default configuration generates: a crypto-shaped smile at three tenors.

Chosen to be a market someone could plausibly be looking at -- around 60% at the money on the
near expiry, falling to 53% at six months, with the downside wing above the upside one as every
equity and crypto book has it. The three slices are ordered in total variance at every sampled
moneyness, so the surface is free of calendar arbitrage as well as of butterfly arbitrage; both
are asserted in the tests, against ``parametric_pricing``'s own diagnostics rather than against a
restatement of them here. A generator quietly producing an arbitrageable truth would make every
downstream acceptance test meaningless in a way nothing would report.
"""

DEFAULT_LOG_MONEYNESS_RANGE = (-0.25, 0.25)
"""The strike ladder's span, as ``ln(K / F)``. Wide enough to have wings, narrow enough that both
legs of every strike stay comfortably invertible."""

DEFAULT_STRIKES_PER_EXPIRY = 11
"""Strikes on the ladder. Enough for a five-parameter fit to be overdetermined rather than
interpolating, which is the difference between calibrating and solving."""

DEFAULT_SPREAD_BP = 200.0
"""Typical full bid-ask spread, in basis points of the mid premium. Two per cent: tight enough to
earn no ``WIDE_SPREAD`` under any threshold anyone would configure, wide enough to be a spread."""

DEFAULT_VOL_NOISE_BP = 50.0
"""Standard deviation of the noise added to each strike's volatility, in basis points of vol.

Half a volatility point. The quantity the known-truth test is stated against: the fit has to
recover the generating parameters *through* this, so it is the one number a calibrator must never
be tuned against -- see the trap in the plan. Zero turns the feed into an exact surface, which is
what the guard tests use to prove the noise is really there.
"""

DEFAULT_SIZE = 10.0
"""Typical size resting on each side, in contracts, before the per-quote wobble."""

DEFAULT_JUNK_QUOTE_RATE = 0.02
"""Fraction of quotes replaced by junk. Two per cent: frequent enough that a few minutes of
session exercises every kind, rare enough that the chain stays fittable."""

DEFAULT_FORWARD_MOVE_REL = 0.002
"""Standard deviation of the forward's log-move between cycles: twenty basis points a tick.

The forward walking is what makes this a *stream* rather than a still life. It moves the smile
against a fixed strike ladder, so each cycle presents genuinely different moneynesses, and it is
what gives the snapshot policy's material-move filter something to have an opinion about.
"""

LATENCY_SECONDS = 0.008
"""Central gap between ``ts_exchange`` and ``ts_local``, standing in for transport latency."""

JITTER_SECONDS = 0.05
"""Half-width of the uniform jitter on ``ts_exchange`` around the cycle's nominal instant.

A venue does not stamp a whole chain with one instant, and a feed that did would make every
per-quote age identical -- so the ages the admissibility rules measure would be an artefact of
the generator rather than a property of the data.
"""

DEFAULT_CYCLES = 20
"""How many times the chain is republished before the stream ends.

Finite, so a run over this provider terminates rather than idling forever. Longer than the
walking skeleton's five because this feed exists to be fitted, and a warm-started calibrator that
has only seen five snapshots has barely left its cold start.
"""

DEFAULT_INTERVAL_SECONDS = 1.0
"""Wall-clock gap between two cycles, and the step the synthetic timeline advances by."""

DEFAULT_SEED = 20_260_827
"""The seed. Any fixed integer would do; what matters is that it is written down."""

SIZE_JITTER_REL = 0.5
"""Half-width of the uniform wobble on each side's size, as a fraction of the configured size."""

SPREAD_JITTER_REL = 0.3
"""Half-width of the uniform wobble on each quote's relative spread."""

LATENCY_JITTER_REL = 0.5
"""Half-width of the uniform wobble on the transport latency. Below one, so ``ts_local`` stays
strictly after ``ts_exchange`` however the draw lands."""

MAX_SPREAD_REL = 1.9
"""Hard ceiling on any relative spread this module produces, junk included.

``bid = mid * (1 - spread / 2)``, so a relative spread of 2 puts the bid at zero and anything
beyond it puts the bid below zero, which ``QuoteObservation`` refuses outright. Junk is meant to
be flagged by the engine, never to be unconstructible.
"""

JUNK_SPREAD_FACTOR = 25.0
"""How much wider than configured a ``WIDE_SPREAD`` junk quote is quoted, before the ceiling."""

JUNK_STALE_SECONDS = 3600.0
"""How far back a ``STALE`` junk quote's exchange stamp is pushed.

An hour, which is past any ``max_age_seconds`` a deployment would configure for a live options
feed. The provider cannot read the thresholds -- they are the engine's judgement, not the venue's
-- so the backdate has to be large enough that the rule fires under all of them.
"""

MIN_VOL = 1e-4
"""Floor the noised volatility is clamped to before pricing.

Black-76 refuses a non-positive volatility, and additive noise on a low-volatility wing can cross
zero. Clamping keeps a legitimate configuration from failing one quote in ten thousand; the floor
is far below any volatility a market prints, so it never binds on realistic settings.
"""


@dataclass(frozen=True, slots=True)
class SyntheticConfig:
    """The invented market: what surface, quoted how badly, how often, and from which seed.

    Every field has a default, so ``SyntheticConfig()`` is a complete BTC-shaped market and a
    test bends the one knob it is about. The defaults are the module constants above, each with
    the reasoning for its value beside it.
    """

    forward0: float = DEFAULT_FORWARD
    """Forward at the first cycle. It walks from there; hence the zero in the name."""

    expiries: tuple[timedelta, ...] = DEFAULT_EXPIRIES
    """Times from ``start`` to each expiry, before the venue's expiry hour is applied.

    Kept beside ``true_params`` rather than derived from its keys because this is the *order* the
    chain is published in, and leaning on a mapping's insertion order for that would make the
    first snapshot of a session depend on how the mapping was written. The two must name exactly
    the same set, which ``__post_init__`` checks.
    """

    strikes_per_expiry: int = DEFAULT_STRIKES_PER_EXPIRY
    """Strikes on the ladder, spread evenly in log-moneyness across the range below."""

    log_moneyness_range: tuple[float, float] = DEFAULT_LOG_MONEYNESS_RANGE
    """Where the ladder starts and ends, as ``ln(K / F)`` against ``forward0``, as ``(lo, hi)``.

    Against the *initial* forward, and fixed for the run: a venue lists absolute strikes and does
    not relist them when the underlying moves. That is precisely what makes the moneyness of each
    quote drift as the forward walks, which is the drift a calibrator has to track.
    """

    true_params: Mapping[timedelta, SVIParamsSpec] = field(
        default_factory=lambda: DEFAULT_TRUE_PARAMS
    )
    """The generating surface, one slice per expiry. The answer a fit is checked against."""

    spread_bp: float = DEFAULT_SPREAD_BP
    """Typical full spread, in basis points of the mid premium."""

    vol_noise_bp: float = DEFAULT_VOL_NOISE_BP
    """Standard deviation of the volatility noise, in basis points of vol. Zero is exact."""

    size: float = DEFAULT_SIZE
    """Typical size on each side. Zero is legal and makes every quote trip ``LOW_SIZE``."""

    junk_quote_rate: float = DEFAULT_JUNK_QUOTE_RATE
    """Probability that a quote is replaced by junk of a randomly chosen kind. In ``[0, 1]``."""

    forward_move_rel: float = DEFAULT_FORWARD_MOVE_REL
    """Standard deviation of the forward's log-move per cycle. Zero pins the forward."""

    latency_seconds: float = LATENCY_SECONDS
    """Central transport latency. Strictly positive, so ``ts_local`` is never ``ts_exchange``."""

    jitter_seconds: float = JITTER_SECONDS
    """Half-width of the jitter on ``ts_exchange``. Zero stamps a whole cycle at one instant."""

    cycles: int = DEFAULT_CYCLES
    """How many times the whole chain is republished before the stream ends. At least one."""

    interval_seconds: float = DEFAULT_INTERVAL_SECONDS
    """Gap between cycles, both awaited and added to the synthetic timeline."""

    seed: int = DEFAULT_SEED
    """Seeds the one generator every draw comes from. Same seed and same start, same stream."""

    def __post_init__(self) -> None:
        _require_positive_finite(self.forward0, "forward0")
        if not self.expiries:
            raise ValueError("The synthetic chain needs at least one expiry")
        if len(set(self.expiries)) != len(self.expiries):
            raise ValueError(f"The expiries must be distinct, got {self.expiries}")
        if any(expiry <= timedelta(0) for expiry in self.expiries):
            raise ValueError(f"Every expiry must be in the future, got {self.expiries}")
        if set(self.expiries) != set(self.true_params):
            raise ValueError(
                "Every expiry needs exactly one slice of true parameters, got expiries "
                f"{sorted(self.expiries)} against {sorted(self.true_params)}"
            )
        if self.strikes_per_expiry < 2:
            raise ValueError(
                "The ladder needs at least two strikes to be a smile, got "
                f"{self.strikes_per_expiry}"
            )
        lo, hi = self.log_moneyness_range
        if not math.isfinite(lo) or not math.isfinite(hi) or lo >= hi:
            raise ValueError(
                f"The log_moneyness_range must be a finite (lo, hi) with lo < hi, got ({lo}, {hi})"
            )
        _require_positive_finite(self.spread_bp, "spread_bp")
        _require_non_negative_finite(self.vol_noise_bp, "vol_noise_bp")
        # Non-negative rather than positive: a zero size is a legal quote and is exactly what
        # raises `LOW_SIZE`, so refusing it here would make one admissibility rule unreachable.
        _require_non_negative_finite(self.size, "size")
        if not math.isfinite(self.junk_quote_rate) or not 0.0 <= self.junk_quote_rate <= 1.0:
            raise ValueError(
                f"The junk_quote_rate must lie inside [0, 1], got {self.junk_quote_rate}"
            )
        _require_non_negative_finite(self.forward_move_rel, "forward_move_rel")
        _require_positive_finite(self.latency_seconds, "latency_seconds")
        _require_non_negative_finite(self.jitter_seconds, "jitter_seconds")
        if self.cycles < 1:
            raise ValueError(f"The provider must publish at least one cycle, got {self.cycles}")
        _require_non_negative_finite(self.interval_seconds, "interval_seconds")

    @property
    def session_seconds(self) -> float:
        """Longest the synthetic timeline can reach past ``start``, jitter included."""
        return (self.cycles - 1) * self.interval_seconds + self.jitter_seconds


@dataclass(frozen=True, slots=True)
class _Slice:
    """One expiry of the built chain: when it expires, what it is worth, and at which strikes."""

    instant: datetime
    params: SVIParamsSpec
    strikes: tuple[float, ...]


class SyntheticProvider:
    """A ``MarketDataProvider`` that quotes a known SVI surface badly, and reproducibly.

    Satisfies the port structurally -- three methods, no inheritance, and nothing here the domain
    could import back. The chain's *identity* (expiries, strikes, order) is fixed in the
    constructor so that ``discover`` and ``stream`` cannot disagree; everything that varies -- the
    forward, the noise, the spread, the sizes, the stamps, the junk -- is drawn per cycle from the
    one seeded generator.
    """

    def __init__(
        self,
        conventions: MarketConventions,
        config: SyntheticConfig | None = None,
        start: datetime | None = None,
    ) -> None:
        """Build the chain against the venue's conventions and the instant the session starts at.

        Args:
            conventions: The market this feed pretends to be. Its ``underlying`` names every
                instrument -- ``QuoteChain`` refuses an update for any other -- its
                ``expiry_time_utc`` places the expiries, and its day count turns them into the
                tenors the premiums are computed at. Taken as an object rather than restated as
                an ``underlying`` field on ``SyntheticConfig``, for ADR-022's reason: two
                spellings of one identifier can disagree, and the copy that loses is the one
                every published event is stamped with.
            config: What surface to generate and how badly to quote it. Defaults to a complete
                BTC-shaped market.
            start: The instant the synthetic timeline begins at, aware. ``None`` reads the wall
                clock once, which is what a live-looking run wants; **passing an instant is what
                makes the stream reproducible bit for bit**, since every later stamp is derived
                from it. Not the engine's ``Clock`` port: ``ProviderFactory`` hands an adapter a
                ``MarketConfig`` and nothing else (ADR-022), and a venue stamps its own messages
                anyway.

        Raises:
            ValueError: On any argument that could not describe a chain, including a session long
                enough to outlive its own nearest expiry. Plain ``ValueError`` and not a
                ``MarketDataError``: these are construction bugs in the wiring, not market
                conditions anybody catches and recovers from.
        """
        self._conventions = conventions
        self._config = config if config is not None else SyntheticConfig()
        self._start = _resolved_start(start)
        self._rng = random.Random(self._config.seed)
        self._closed = False
        self._slices = _build_slices(self._conventions, self._config, self._start)
        self._instruments = tuple(
            InstrumentId(
                underlying=self._conventions.underlying,
                expiry=slice_.instant,
                strike=strike,
                kind=kind,
            )
            for slice_ in self._slices
            for strike in slice_.strikes
            for kind in _KINDS_OTM_FIRST[strike >= self._config.forward0]
        )

    @property
    def config(self) -> SyntheticConfig:
        """The market this feed invented, including the parameters a fit has to recover."""
        return self._config

    @property
    def start(self) -> datetime:
        """The instant the synthetic timeline begins at, resolved once at construction."""
        return self._start

    @property
    def instruments(self) -> tuple[InstrumentId, ...]:
        """Every instrument this feed quotes, in the order it publishes them."""
        return self._instruments

    async def discover(self) -> tuple[InstrumentId, ...]:
        """The full live set, never a delta, and fixed for the lifetime of the provider.

        Fixed because ``discover`` and ``stream`` must not disagree: a set recomputed against a
        later instant would place its expiries on different dates from the quotes already in the
        chain, and ``QuoteChain`` would report instruments it has never been quoted for as
        missing -- dragging coverage down for the whole session for a reason nobody can find in
        the data. A venue listing a new strike mid-session is ``RecordedProvider``'s problem.
        """
        return self._instruments

    def stream(self) -> AsyncIterator[QuoteUpdate]:
        """Open the feed. A plain ``def`` returning the iterator, as the port documents."""
        return self._stream()

    async def _stream(self) -> AsyncIterator[QuoteUpdate]:
        """Republish the chain ``cycles`` times, spoiling it afresh each time, then end.

        The draw order is fixed -- cycle, then expiry, then strike, then the two legs -- which is
        the whole of what reproducibility needs: the same seed replays the same sequence of
        variates, and the same ``start`` turns them into the same instants. ``close`` is checked
        between updates, which is what makes the promise in the port -- that closing ends the
        stream -- true rather than aspirational.

        **The generator is re-seeded here rather than in the constructor**, so opening the feed a
        second time replays the same session instead of continuing the first one. Reproducibility
        is then a property of the provider and not of the moment it was built, which is what a
        caller comparing two runs actually needs; a single provider driving two iterators at once
        is not a supported shape, and never was -- the port describes one long-lived stream.
        """
        self._rng = random.Random(self._config.seed)
        forward = self._config.forward0
        for cycle in range(self._config.cycles):
            if cycle > 0 and self._config.interval_seconds > 0:
                await asyncio.sleep(self._config.interval_seconds)
            ts_cycle = self._start + timedelta(seconds=cycle * self._config.interval_seconds)
            # Drawn unconditionally, and before the quotes: a lognormal step is exactly 1.0 when
            # the configured move is zero, so a branch here would only make the number of
            # variates depend on the configuration and the streams incomparable between runs.
            forward *= math.exp(self._rng.gauss(0.0, self._config.forward_move_rel))
            for slice_ in self._slices:
                tenor_years = self._conventions.tenor_years(slice_.instant, ts_cycle)
                for strike in slice_.strikes:
                    # One volatility per strike, shared by both legs: a market maker quotes a
                    # strike, not a leg, and arbitrageurs hold put-call parity tight. Perturbing
                    # each leg separately would break parity by the size of the noise and turn
                    # Market Data's forward cross-check into a measurement of this generator.
                    vol = self._quoted_volatility(slice_, strike, forward, tenor_years)
                    for kind in _KINDS_OTM_FIRST[strike >= self._config.forward0]:
                        if self._closed:
                            return
                        yield self._update(
                            instrument=InstrumentId(
                                underlying=self._conventions.underlying,
                                expiry=slice_.instant,
                                strike=strike,
                                kind=kind,
                            ),
                            premium=price(
                                forward=forward,
                                strike=strike,
                                tenor_years=tenor_years,
                                vol=vol,
                                is_call=kind is OptionKindD.CALL,
                            ),
                            forward=forward,
                            ts_cycle=ts_cycle,
                        )

    async def close(self) -> None:
        """Stop the stream at the next update. Idempotent, as the port requires."""
        self._closed = True

    def _quoted_volatility(
        self, slice_: _Slice, strike: float, forward: float, tenor_years: float
    ) -> float:
        """The true volatility at this strike, perturbed by the configured noise and floored.

        ``k`` is measured against the *current* forward, not the one the ladder was built from:
        that is the whole point of letting the forward walk, and computing it any other way here
        would build the very shared-convention error the plan's trap warns about -- one this feed
        and a calibrator could cancel between them while the engine was wrong.
        """
        k = math.log(strike / forward)
        noise = self._rng.gauss(0.0, self._config.vol_noise_bp / BASIS_POINTS_PER_UNIT)
        return max(MIN_VOL, slice_.params.volatility(k, tenor_years) + noise)

    def _update(
        self,
        instrument: InstrumentId,
        premium: float,
        forward: float,
        ts_cycle: datetime,
    ) -> QuoteUpdate:
        """Wrap one premium in the microstructure a venue would have put around it."""
        spread_rel = min(
            MAX_SPREAD_REL,
            (self._config.spread_bp / BASIS_POINTS_PER_UNIT) * self._jitter(SPREAD_JITTER_REL),
        )
        ts_exchange = ts_cycle + timedelta(
            seconds=self._rng.uniform(-self._config.jitter_seconds, self._config.jitter_seconds)
        )
        observation = QuoteObservation(
            bid=premium * (1.0 - spread_rel / 2.0),
            ask=premium * (1.0 + spread_rel / 2.0),
            bid_size=self._config.size * self._jitter(SIZE_JITTER_REL),
            ask_size=self._config.size * self._jitter(SIZE_JITTER_REL),
            ts_exchange=ts_exchange,
            ts_local=ts_exchange
            + timedelta(
                seconds=self._config.latency_seconds * self._jitter(LATENCY_JITTER_REL),
            ),
            # No `exchange_iv`. A synthetic feed publishing the number its own premiums were
            # built from would make `IV_DIVERGENCE` agree with itself by construction, and a
            # check that cannot fail is worse than an absent one because it reads like a check.
            # The rule is inert anyway until the chain inverts its own mids -- `docs/SEAMS.md`.
            exchange_iv=None,
        )
        # Drawn for every quote, so the junk decision costs the same variate whether or not it
        # fires -- otherwise raising the rate would reshuffle the entire stream after it.
        if self._rng.random() < self._config.junk_quote_rate:
            observation = _spoil(
                observation, self._rng.choice(JUNK_KINDS), premium=premium, spread_rel=spread_rel
            )
        return QuoteUpdate(instrument=instrument, observation=observation, underlying_price=forward)

    def _jitter(self, half_width: float) -> float:
        """A uniform multiplier around 1.0, used for every proportional wobble in this feed."""
        return self._rng.uniform(1.0 - half_width, 1.0 + half_width)


_KINDS_OTM_FIRST = {
    False: (OptionKindD.PUT, OptionKindD.CALL),
    True: (OptionKindD.CALL, OptionKindD.PUT),
}
"""Which leg of a strike is quoted first, keyed by whether the strike is at or above the forward.

Both legs of every strike are quoted: only the out-of-the-money one is fitted (ADR-017), but the
twin is what lets Market Data's put-call parity cross-check compute a second, independent forward.
The out-of-the-money leg goes first because the first update of a session is what the first
snapshot rests on, and that is where an inversion is best conditioned. At ``k = 0`` exactly the
call goes first, matching the tie-break the pricing ACL already makes -- there the two legs have
identical time value and identical vega, so having the same rule on both sides of the boundary is
what stops the choice from mattering by accident.
"""


def _spoil(
    observation: QuoteObservation, kind: JunkKind, premium: float, spread_rel: float
) -> QuoteObservation:
    """Turn one honest quote into junk of a named kind, keeping it constructible.

    Constructible matters: ``QuoteObservation`` refuses a negative premium or a negative size
    outright, and junk that could not be built would fail the run rather than the rule it is
    meant to trip. The point is to hand the engine a quote it must *flag*, not one it cannot
    represent.

    ``dataclasses.replace`` re-runs ``__post_init__``, so every kind below is still checked
    against the invariants of an observation before it leaves this module. The closing
    ``assert_never`` is what makes the list exhaustive: adding a member to ``JunkKind`` without a
    branch here is a ``mypy`` error rather than a quote that silently comes out honest.
    """
    if kind is JunkKind.CROSSED:
        # Swapped, not perturbed: a locked market (bid == ask) is legal and unflagged, so the
        # sides have to end up strictly the wrong way round.
        return replace(observation, bid=observation.ask, ask=observation.bid)
    if kind is JunkKind.WIDE_SPREAD:
        wide = min(MAX_SPREAD_REL, spread_rel * JUNK_SPREAD_FACTOR)
        return replace(
            observation, bid=premium * (1.0 - wide / 2.0), ask=premium * (1.0 + wide / 2.0)
        )
    if kind is JunkKind.ONE_SIDED:
        # `None`, never zero: an empty side is the absence of information and a zero bid is
        # information. Collapsing the two is the mistake this feed exists to catch elsewhere.
        return replace(observation, bid=None)
    if kind is JunkKind.LOW_SIZE:
        return replace(observation, bid_size=0.0, ask_size=0.0)
    if kind is JunkKind.STALE:
        return replace(
            observation,
            ts_exchange=observation.ts_exchange - timedelta(seconds=JUNK_STALE_SECONDS),
        )
    assert_never(kind)


def _resolved_start(start: datetime | None) -> datetime:
    """The session's first instant: the one given, or the wall clock read exactly once.

    ``datetime.now(UTC)``, never ``utcnow()``, which returns a naive value despite its name and
    would make every later subtraction raise ``TypeError``.
    """
    if start is None:
        return datetime.now(UTC)
    require_aware(start, "start")
    return start


def _build_slices(
    conventions: MarketConventions, config: SyntheticConfig, start: datetime
) -> tuple[_Slice, ...]:
    """Resolve the configured expiries into instants and hang a strike ladder off each.

    Sorted by expiry, and each ladder sorted by strike, for the reason the leg order exists: the
    snapshot policy emits on the first update it sees, so the first surface of a session rests on
    the nearest expiry's lowest strike. Leaving that to a mapping's iteration order would make it
    depend on nothing anybody chose.

    Raises:
        ValueError: If two configured expiries land on the same instant once the venue's expiry
            hour is applied, or if any of them would fall inside the session the configuration
            describes. The second is what stops a mid-stream ``ExpiredInstrumentError`` from
            reaching a consumer as a crash: an expiry the run outlives is a wiring mistake, and
            it is knowable at construction.
    """
    lo, hi = config.log_moneyness_range
    step = (hi - lo) / (config.strikes_per_expiry - 1)
    strikes = tuple(
        config.forward0 * math.exp(lo + step * index) for index in range(config.strikes_per_expiry)
    )
    session_end = start + timedelta(seconds=config.session_seconds)

    slices: list[_Slice] = []
    for offset in sorted(config.expiries):
        instant = conventions.expiry_instant((start + offset).date())
        if instant <= session_end:
            raise ValueError(
                f"The expiry {offset} lands at {instant}, which the session running to "
                f"{session_end} would outlive; there would be no tenor left to price it at"
            )
        slices.append(_Slice(instant=instant, params=config.true_params[offset], strikes=strikes))

    instants = [slice_.instant for slice_ in slices]
    if len(set(instants)) != len(instants):
        raise ValueError(
            f"Two expiries collapse onto the same instant at the venue's expiry hour: {instants}"
        )
    return tuple(slices)


def _require_positive_finite(value: float, what: str) -> None:
    """Reject anything that is not a usable positive number, NaN included.

    ``isfinite`` first and the bad cases joined with ``or``: ``float("nan") <= 0`` is ``False``,
    so a NaN walks straight through an ordering guard written the other way round.
    """
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"The {what} must be positive and finite, got {value}")


def _require_non_negative_finite(value: float, what: str) -> None:
    """Same guard, one step looser: zero is a legitimate setting for every caller of this one."""
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"The {what} must be non-negative and finite, got {value}")
