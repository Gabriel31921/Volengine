"""The second generator: a stochastic-volatility surface, and the four ways it could be wrong.

A closed-form option price computed by numerical integration fails silently. There is no exception
and no NaN -- a truncation that is too short, a branch cut crossed, a sign flipped in the
characteristic function all produce a plausible number, and a plausible number in a *generator* is
the worst kind, because everything downstream is then measured against a surface nobody quoted.
So the assertions here are chosen to be independent of the implementation rather than to restate
it:

* **The degenerate limit.** As the vol of vol goes to zero with ``v0 == theta``, Heston *is*
  Black-76 at ``sqrt(theta)``, and the shared kernel is an implementation of Black-76 that knows
  nothing about this module. That is a genuine oracle rather than a rearrangement.
* **A symmetry.** With zero correlation the log-price distribution is symmetric, so the smile must
  be even in ``k`` -- and it comes out even to the last bit, which no accidental sign convention
  survives.
* **A known constant.** The characteristic function evaluated at ``-i`` is the expectation of the
  forward under its own measure, which is one. It is the one number in the whole formula that is
  known exactly, and it pins the sign convention the Lewis integral depends on.
* **The truncation, with the guard that fails when it is disabled.** The exponential tail of the
  Heston characteristic function is fatter than a Gaussian, and how much fatter depends on the vol
  of vol. The two tests at the end compute the same integral by brute force over a deliberately
  excessive range -- once to show the module agrees with it, and once over the range a
  Gaussian-only truncation would have chosen, on the fat-tailed market where that choice is wrong
  by four orders of magnitude more than the rule's own error.

The last two use ``_characteristic``, which is private. The precedent is
``test_durrleman.py``, and the argument is the same: the quantity that has to be pinned is *inside*
the function, and checking it only through the price it produces is exactly the check a sign slip
survives.
"""

from __future__ import annotations

import math
from datetime import UTC, datetime, timedelta

import numpy as np
import pytest

from tests.market_data.builders import make_conventions
from volengine.market_data.adapters.heston import (
    TOTAL_STDEV_REACH,
    HestonParamsSpec,
    _characteristic,
)
from volengine.market_data.adapters.synthetic import (
    SyntheticConfig,
    SyntheticProvider,
    VolatilitySpec,
)
from volengine.shared_kernel.domain.black76 import implied_vol, price

FLAT = HestonParamsSpec(v0=0.16, kappa=2.0, theta=0.16, vol_of_vol=0.01, rho=0.0)
"""Deterministic variance, near enough: 40% volatility at every strike and every tenor.

``vol_of_vol`` cannot be zero -- the closed form divides by its square -- so the limit is
approached rather than evaluated at, and the tolerances below are stated at the scale that
approximation costs.
"""

SMILEY = HestonParamsSpec(v0=0.09, kappa=2.0, theta=0.16, vol_of_vol=0.6, rho=-0.7)
"""A crypto-shaped market: 30% now, reverting to 40%, with a hard downside skew."""

FAT_TAILED = HestonParamsSpec(v0=0.09, kappa=1.0, theta=0.09, vol_of_vol=1.5, rho=-0.5)
"""A vol of vol of 1.5, which is what makes the characteristic function's exponential tail matter.

Extreme, and chosen to be: the point of the truncation tests is that the rule holds at parameters
nobody would pick first. At the ``SMILEY`` above a Gaussian-only range happens to be good enough,
so a guard written there would pass whether or not the fix existed.
"""

TENORS = (7 / 365, 30 / 365, 0.25, 1.0, 2.0)
MONEYNESS = (-0.6, -0.25, 0.0, 0.25, 0.6)

RESOLVABLE = ((0.25, 0.25), (1.0, 0.6), (2.0, 0.6), (0.25, -0.25), (1.0, -0.6))
"""``(tenor, k)`` pairs whose out-of-the-money leg is worth more than the quadrature resolves.

A wing worth less than 1e-10 of the forward is refused rather than quoted, so the band test is
stated over pairs where there is a price to check. The refusal is a test of its own, below.
"""


def brute_force(params: HestonParamsSpec, k: float, tenor_years: float, upper: float) -> float:
    """The same integral by composite Simpson over an explicit range: an independent quadrature.

    Independent of the module's *rule* and of its *truncation*, which are the two things the tests
    using it are about. It shares the characteristic function, deliberately -- reimplementing that
    here would be testing a copy of the code against itself.
    """
    nodes = np.linspace(1e-12, upper, 400_001)
    argument = np.asarray(nodes - 0.5j, dtype=np.complex128)
    values = np.real(np.exp(-1j * nodes * k) * _characteristic(argument, params, tenor_years)) / (
        nodes**2 + 0.25
    )
    weights = np.ones(nodes.size)
    weights[1:-1:2] = 4.0
    weights[2:-1:2] = 2.0
    step = float(nodes[1] - nodes[0])
    return 1.0 - math.exp(k / 2.0) / math.pi * (step / 3.0 * float(np.dot(weights, values)))


# --- the invariants


@pytest.mark.parametrize("field", ["v0", "kappa", "theta", "vol_of_vol"])
def test_every_positive_parameter_is_refused_at_zero(field: str) -> None:
    """A zero variance, a zero reversion speed or a zero vol of vol is not a market."""
    with pytest.raises(ValueError, match=field):
        HestonParamsSpec(**{**_fields(), field: 0.0})


@pytest.mark.parametrize("field", ["v0", "kappa", "theta", "vol_of_vol", "rho"])
def test_a_nan_parameter_is_refused(field: str) -> None:
    """The trap this repository keeps meeting: ``nan <= 0`` is ``False``.

    A guard written as an ordering test alone accepts every one of these and the failure resurfaces
    as a NaN price nobody can attribute.
    """
    with pytest.raises(ValueError, match=field):
        HestonParamsSpec(**{**_fields(), field: float("nan")})


@pytest.mark.parametrize("rho", [1.0, -1.0, 1.5])
def test_a_correlation_outside_the_open_interval_is_refused(rho: float) -> None:
    """At ``|rho| = 1`` the two Brownian motions are the same one and the model degenerates."""
    with pytest.raises(ValueError, match="rho"):
        HestonParamsSpec(**{**_fields(), "rho": rho})


def test_a_violated_feller_condition_is_accepted() -> None:
    """Deliberately not enforced: it is about paths, and nothing here simulates one.

    ``2 kappa theta < sigma^2`` is the regime most real surfaces are fitted in, and refusing it
    would rule out the parameter sets most worth generating.
    """
    violating = HestonParamsSpec(v0=0.09, kappa=0.5, theta=0.04, vol_of_vol=1.2, rho=-0.5)

    assert 2 * violating.kappa * violating.theta < violating.vol_of_vol**2
    assert violating.volatility(0.0, 0.5) > 0.0


@pytest.mark.parametrize("tenor", [0.0, -1.0, float("nan")])
def test_a_tenor_that_is_not_a_length_of_time_is_refused(tenor: float) -> None:
    with pytest.raises(ValueError, match="tenor"):
        FLAT.volatility(0.0, tenor)


def test_an_infinite_moneyness_is_refused() -> None:
    """Before it becomes an ``exp`` overflow inside the quadrature."""
    with pytest.raises(ValueError, match="log-moneyness"):
        FLAT.call_price(float("inf"), 0.5)


# --- the oracle


@pytest.mark.parametrize("tenor", TENORS)
@pytest.mark.parametrize("k", MONEYNESS)
def test_a_vanishing_vol_of_vol_reproduces_black_76(tenor: float, k: float) -> None:
    """The headline oracle: with the variance held still, Heston *is* Black-76 at ``sqrt(theta)``.

    Independent in every sense that matters -- the shared kernel's closed form knows nothing about
    characteristic functions -- and it covers the whole band this module is documented for at once.
    """
    expected = price(forward=1.0, strike=math.exp(k), tenor_years=tenor, vol=0.4, is_call=True)

    assert FLAT.call_price(k, tenor) == pytest.approx(expected, abs=1e-5)


def test_the_flat_limit_is_flat_across_the_smile() -> None:
    """The same statement one level up: no vol of vol, no smile.

    Read in volatility rather than in price because that is what the feed quotes, and because a
    price tolerance means something different at every strike.
    """
    volatilities = [FLAT.volatility(k, 0.5) for k in MONEYNESS]

    assert volatilities == pytest.approx([0.4] * len(MONEYNESS), abs=1e-3)


def test_the_characteristic_function_is_one_at_minus_i() -> None:
    """``E[F_T / F_0] == 1``: the forward is a martingale, and this is the only exact number here.

    The sign convention of the whole Lewis integral hangs off it. Flip a sign in the exponent and
    the prices stay plausible while this comes out as anything but one.
    """
    at_minus_i = _characteristic(np.array([-1j], dtype=np.complex128), SMILEY, 0.75)

    assert complex(at_minus_i[0]) == pytest.approx(1.0 + 0.0j, abs=1e-12)


# --- the shape


def test_a_negative_correlation_lifts_the_downside_wing() -> None:
    """The skew, and its direction: variance rising as the price falls is what a book prints."""
    downside = SMILEY.volatility(-0.2, 0.5)
    upside = SMILEY.volatility(0.2, 0.5)

    assert downside > upside


def test_a_positive_correlation_lifts_the_other_one() -> None:
    """The guard on the test above: it is the correlation doing it, not a sloped implementation."""
    inverted = HestonParamsSpec(v0=0.09, kappa=2.0, theta=0.16, vol_of_vol=0.6, rho=0.7)

    assert inverted.volatility(-0.2, 0.5) < inverted.volatility(0.2, 0.5)


@pytest.mark.parametrize("k", [0.1, 0.2, 0.3])
def test_an_uncorrelated_smile_is_even_in_moneyness(k: float) -> None:
    """A symmetry that holds to the last bit, so any accidental asymmetry is visible.

    With zero correlation the distribution of ``ln(F_T / F_0)`` is symmetric, so the smile must be.
    ``abs=`` is stated because the difference being asserted is a difference of two numbers around
    0.4 and ``pytest.approx`` would otherwise pass on its relative bound alone.
    """
    symmetric = HestonParamsSpec(v0=0.16, kappa=2.0, theta=0.16, vol_of_vol=0.6, rho=0.0)

    assert symmetric.volatility(k, 0.5) == pytest.approx(symmetric.volatility(-k, 0.5), abs=1e-12)


def test_a_smile_curves_upwards_away_from_the_money() -> None:
    """Vol of vol makes a smile: the wings sit above the middle, which is SVI's ``b`` by another
    route."""
    symmetric = HestonParamsSpec(v0=0.16, kappa=2.0, theta=0.16, vol_of_vol=0.6, rho=0.0)

    assert symmetric.volatility(0.4, 0.5) > symmetric.volatility(0.0, 0.5)


def test_the_term_structure_walks_from_the_current_variance_to_the_long_run_one() -> None:
    """One model, a whole surface: the level moves with the tenor without anybody choosing it.

    This is the difference from an SVI slice that makes the second generator worth having -- the
    term structure is a consequence of the parameters rather than three sets of five numbers.
    """
    rising = HestonParamsSpec(v0=0.04, kappa=1.5, theta=0.16, vol_of_vol=0.4, rho=-0.6)

    near, far = rising.volatility(0.0, 0.02), rising.volatility(0.0, 2.0)

    assert near == pytest.approx(math.sqrt(0.04), abs=1e-2)
    assert far > near


def test_the_expected_variance_is_the_current_one_at_a_vanishing_tenor() -> None:
    """The ``0 / 0`` limit of ``(1 - exp(-kappa T)) / (kappa T)``, which is where it is worst."""
    assert SMILEY.expected_variance(1e-12) == pytest.approx(SMILEY.v0, rel=1e-9)


def test_the_expected_variance_reaches_the_long_run_one_eventually() -> None:
    assert SMILEY.expected_variance(500.0) == pytest.approx(SMILEY.theta, rel=1e-2)


@pytest.mark.parametrize(("tenor", "k"), RESOLVABLE)
def test_every_resolvable_price_lies_inside_the_no_arbitrage_band(tenor: float, k: float) -> None:
    """Strictly between intrinsic and the forward, or no volatility reproduces it."""
    premium = SMILEY.call_price(k, tenor)

    assert max(1.0 - math.exp(k), 0.0) < premium < 1.0


def test_a_wing_below_the_quadratures_resolution_is_refused_rather_than_quoted() -> None:
    """The deep-wing trap, in the one place this module can meet it.

    A one-week option a quarter of the way out of the money is worth around 1e-11 of the forward,
    which is the scale of the quadrature's own error: what comes back is noise, and it is sometimes
    negative. Inverting it would hand the feed a volatility with no digits in it. The refusal names
    the strike, so the answer is to narrow the chain rather than to trust the quote.
    """
    with pytest.raises(ValueError, match="below the"):
        SMILEY.volatility(0.25, 7 / 365)


def test_the_same_wing_one_month_out_is_quoted_normally() -> None:
    """The guard on the refusal above: it is the resolution refusing, not the wings.

    Without it, a floor set a thousand times too high would pass the previous test while silently
    deleting half of every generated chain.
    """
    assert SMILEY.volatility(0.25, 30 / 365) > 0.0


# --- the truncation


@pytest.mark.parametrize("params", [SMILEY, FAT_TAILED])
@pytest.mark.parametrize("tenor", [30 / 365, 1.0])
def test_the_price_agrees_with_a_brute_force_quadrature_over_a_wide_range(
    params: HestonParamsSpec, tenor: float
) -> None:
    """The module's rule against four hundred thousand Simpson points over an excessive range.

    What this pins is the *truncation* and the panel count, the two choices a fixed-node rule gets
    quietly wrong. Both parameter sets, because the fat-tailed one is where the truncation has work
    to do.
    """
    wide = 4_000.0 / math.sqrt(tenor)

    assert params.call_price(0.25, tenor) == pytest.approx(
        brute_force(params, 0.25, tenor, wide), abs=1e-9
    )


def test_a_range_sized_from_the_total_standard_deviation_alone_would_be_wrong() -> None:
    """The guard that fails when the fix is removed.

    Truncating at fourteen total standard deviations -- where the Gaussian body has fallen below
    ``exp(-98)``, and what an implementation written for a Black world would choose -- costs four
    orders of magnitude more than the rule's own error on a market whose vol of vol is 1.5.
    """
    tenor = 1.0
    total_stdev = math.sqrt(FAT_TAILED.expected_variance(tenor) * tenor)

    gaussian_only = brute_force(FAT_TAILED, 0.0, tenor, TOTAL_STDEV_REACH / total_stdev)

    assert abs(gaussian_only - FAT_TAILED.call_price(0.0, tenor)) > 1e-6


# --- quoted through the feed


def test_the_generator_satisfies_the_shape_the_synthetic_feed_asks_for() -> None:
    """Structural typing, checked at runtime as well as by mypy: one method, and no inheritance."""
    spec: VolatilitySpec = SMILEY

    assert spec.volatility(0.1, 0.5) > 0.0


async def test_the_feed_quotes_the_heston_surface_it_was_given() -> None:
    """The deliverable, end to end: a Heston market published as a chain of two-sided quotes.

    The mid of each quote is inverted back through the shared kernel and compared with what the
    model says at that strike, which is the same route the engine's own inversion takes. The noise
    is off, so the only difference left between the two numbers is the arithmetic in between.
    """
    conventions = make_conventions()
    start = datetime(2026, 8, 27, 12, 0, tzinfo=UTC)
    tenor = timedelta(days=30)
    provider = SyntheticProvider(
        conventions,
        SyntheticConfig(
            expiries=(tenor,),
            true_params={tenor: SMILEY},
            strikes_per_expiry=5,
            vol_noise_bp=0.0,
            junk_quote_rate=0.0,
            forward_move_rel=0.0,
            spread_bp=1.0,
            cycles=1,
            interval_seconds=0.0,
        ),
        start,
    )

    recovered: list[float] = []
    expected: list[float] = []
    async for update in provider.stream():
        observation = update.observation
        assert observation.bid is not None and observation.ask is not None
        mid = observation.mid
        assert mid is not None and update.underlying_price is not None
        forward = update.underlying_price
        strike = update.instrument.strike
        tenor_years = conventions.tenor_years(update.instrument.expiry, start)
        recovered.append(
            implied_vol(
                target_price=mid,
                forward=forward,
                strike=strike,
                tenor_years=tenor_years,
                is_call=update.instrument.kind.value == "CALL",
            )
        )
        expected.append(SMILEY.volatility(math.log(strike / forward), tenor_years))

    assert len(recovered) == 10
    assert recovered == pytest.approx(expected, abs=1e-6)


def _fields() -> dict[str, float]:
    """A valid parameter set as a mapping, so an invalid variant changes exactly one entry."""
    return {"v0": 0.09, "kappa": 2.0, "theta": 0.16, "vol_of_vol": 0.6, "rho": -0.7}
