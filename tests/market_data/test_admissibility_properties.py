"""The shape filters, quantified over random smiles instead of over handwritten price ladders.

``test_admissibility.py`` builds three or four prices by hand and checks that the rule fires or
does not. That is the right way to pin what each flag means, and it leaves one question open that
examples cannot answer: does the filter fire on *real* chains it should leave alone? Ingestion sees
thousands of slices an hour, every one of them a smile rather than a triple of numbers somebody
chose, and a convexity test that flagged ordinary markets would down-weight the whole book without
anything looking broken.

So the property here is stated across the boundary between what generates a chain and what judges
one: **price an arbitrage-free surface and the arbitrage filters must have nothing to say about
it.** The premiums are Black-76 prices of an SVI slice, and "arbitrage-free" is decided by
``parametric_pricing``'s own Durrleman diagnostic rather than by a second opinion written here --
Durrleman's condition *is* the statement that the call price is convex in strike, so if the filter
disagrees with it on a slice, exactly one of the two is wrong.

``tests/`` is subject to none of the import rules, which is what lets one module hold a producing
context's filter and a consuming context's diagnostic at once. Nothing in ``src/`` may do this, and
the point of the test is that nothing needs to: the two agree.
"""

from __future__ import annotations

import math
from random import Random

import numpy as np
from hypothesis import assume, given, settings
from hypothesis import strategies as st
from numpy.typing import NDArray

from volengine.market_data.domain.admissibility import (
    AdmissibilityThresholds,
    QuoteFlagD,
    SliceQuotes,
    flag_slice,
)
from volengine.market_data.domain.option_quote import OptionKindD
from volengine.parametric_pricing.domain.durrleman import butterfly_violation
from volengine.parametric_pricing.domain.svi_slice import SVIParams
from volengine.shared_kernel.domain.black76 import price

SLICES = st.builds(
    SVIParams,
    a=st.floats(min_value=5e-3, max_value=0.5),
    b=st.floats(min_value=5e-3, max_value=0.3),
    rho=st.floats(min_value=-0.95, max_value=0.95),
    m=st.floats(min_value=-0.3, max_value=0.3),
    sigma=st.floats(min_value=0.1, max_value=1.0),
)
"""A market-shaped slice, on the same box as ``test_svi_properties`` draws from."""

TENOR_YEARS = 1.0 / 12.0
"""One month. The tenor only scales the premiums here; the shape rules do not read it."""

FORWARD = 1.0
"""A normalised forward, so every premium is a fraction of it.

Not a crypto forward of sixty thousand, because the convexity rule compares a price against a
chord with an *absolute* tolerance: on premiums of order one, the tolerance below is a statement
about the last bits of the pricer, which is what it is meant to be.
"""

LADDER: tuple[float, ...] = tuple(np.linspace(-0.6, 0.6, 25))
"""Twenty-five strikes spanning +-60% in log-forward-moneyness, evenly spaced.

Dense enough that a butterfly violation has somewhere to show up between neighbours -- the rule
tests consecutive triples, so a breach narrower than the spacing is one no chain could see either.
"""

DENSE_MESH: NDArray[np.float64] = np.linspace(-1.5, 1.5, 601)
"""Where the density is checked: wider than the ladder, so a violation just off the last quoted
strike still disqualifies the draw instead of being priced into it unnoticed."""

THRESHOLDS = AdmissibilityThresholds(
    max_spread_rel=0.5,
    max_age_seconds=5.0,
    moneyness_range=(-5.0, 5.0),
    max_iv_divergence_bp=500.0,
    convexity_tolerance=1e-12,
    min_size=1.0,
)
"""Only ``convexity_tolerance`` is read by ``flag_slice``; the rest are along for the constructor.

A thousandth of a basis point of the forward: an allowance for the rounding of two exponentials
and an ``erfc``, and far too small to hide a violation anyone would trade.
"""

ARBITRAGEABLE = SVIParams(a=0.001, b=0.35, rho=-0.90, m=0.0, sigma=0.02)
"""A slice that really does price a negative density -- the same one the calibrator's own tests
use to prove its butterfly penalty bites. Two units deep in ``butterfly_violation``."""


def priced_ladder(params: SVIParams) -> SliceQuotes:
    """Both legs of every strike, priced from the slice through the shared kernel.

    Calls and puts alike, because ``flag_slice`` judges them as two separate curves that run in
    opposite directions: a sign convention inverted in one of them would leave the other clean.
    """
    quotes: list[tuple[float, OptionKindD, float]] = []
    for k in LADDER:
        strike = FORWARD * math.exp(k)
        vol = params.implied_vol(k, TENOR_YEARS)
        for kind, is_call in ((OptionKindD.CALL, True), (OptionKindD.PUT, False)):
            quotes.append((strike, kind, price(FORWARD, strike, TENOR_YEARS, vol, is_call)))
    return quotes


@settings(max_examples=150)
@given(params=SLICES)
def test_a_slice_with_a_non_negative_density_earns_no_shape_flag(params: SVIParams) -> None:
    """An arbitrage-free smile, priced honestly, must pass both shape rules.

    This is the half of the filter no example test reaches: that it stays quiet on the chains it
    will actually see. A rule that fired here would down-weight every well-behaved slice in the
    book, and the only symptom downstream would be a fit that had quietly stopped trusting its
    own data.
    """
    assume(butterfly_violation(params, DENSE_MESH) == 0.0)

    assert flag_slice(priced_ladder(params), THRESHOLDS) == ()


def test_a_slice_that_prices_a_negative_density_is_flagged_on_both_rules() -> None:
    """The guard on the property above: the filter is not simply silent.

    A test that only ever asserts an empty tuple would pass just as well against a ``flag_slice``
    that returned one unconditionally. The same ladder, priced from a slice whose density really
    does go negative, has to come back marked -- and it trips monotonicity as well as convexity,
    because a price curve that bends the wrong way far enough eventually runs uphill too.
    """
    flags = flag_slice(priced_ladder(ARBITRAGEABLE), THRESHOLDS)

    assert set(flags) == {QuoteFlagD.SLICE_MONOTONICITY, QuoteFlagD.SLICE_CONVEXITY}


@settings(max_examples=100)
@given(params=SLICES, permutation=st.randoms(use_true_random=False))
def test_the_verdict_does_not_depend_on_the_order_the_quotes_arrived_in(
    params: SVIParams, permutation: Random
) -> None:
    """A chain is assembled from a dictionary, so arrival order is arbitrary and must not matter.

    ``_prices_by_strike`` sorts before comparing neighbours for exactly this reason. Stated over
    random permutations rather than over one reversed list, because the failure mode of a sort
    that was dropped is a verdict that depends on which strike happened to tick last.
    """
    quotes = list(priced_ladder(params))
    shuffled = list(quotes)
    permutation.shuffle(shuffled)

    assert flag_slice(shuffled, THRESHOLDS) == flag_slice(quotes, THRESHOLDS)
