from __future__ import annotations

from collections.abc import Sequence
from statistics import fmean

import pytest

from volengine.market_data.domain.forward import (
    crosscheck_error,
    forward_from_parity,
    forward_from_provider,
)

FORWARD = 60_000.0
NEAR_THE_MONEY = (58_000.0, 59_000.0, 60_000.0, 61_000.0, 62_000.0)
UNUSABLE = [0.0, -1.0, float("nan"), float("inf"), float("-inf")]


# --- builders


def parity_pairs(
    forward: float = FORWARD,
    strikes: Sequence[float] = NEAR_THE_MONEY,
    discount: float = 1.0,
    put_mid: float = 1_000.0,
) -> list[tuple[float, float, float]]:
    """Pairs that satisfy C - P = D * (F - K) exactly, so parity must recover ``forward``.

    The put mid is fixed and the call derived from it: parity constrains the *difference*, so
    one leg is free and only the spread between them carries the forward.
    """
    return [(strike, put_mid + discount * (forward - strike), put_mid) for strike in strikes]


# --- forward_from_provider


def test_the_provider_price_is_the_forward() -> None:
    assert forward_from_provider(FORWARD) == FORWARD


@pytest.mark.parametrize("bad", UNUSABLE)
def test_provider_rejects_an_unusable_underlying_price(bad: float) -> None:
    """It becomes the denominator of every log(K / F) in the expiry, so a bad one poisons the
    whole slice rather than a single quote.
    """
    with pytest.raises(ValueError, match="provider underlying"):
        forward_from_provider(bad)


# --- forward_from_parity: the identity holds


def test_parity_recovers_a_known_forward() -> None:
    """The assertion the module exists for: F = K + (C - P) / D, inverted from synthetic
    quotes built with that same identity, must give back the F they were built from.
    """
    assert forward_from_parity(parity_pairs()) == pytest.approx(FORWARD, rel=1e-12)


def test_parity_recovers_a_known_forward_under_discounting() -> None:
    """Only the premium difference is discounted. Dividing the strike too -- one misplaced
    paren -- still returns a plausible-looking number, and this is what catches it.
    """
    discount = 0.98
    pairs = parity_pairs(discount=discount)
    assert forward_from_parity(pairs, discount=discount) == pytest.approx(FORWARD, rel=1e-12)


def test_ignoring_the_discount_factor_moves_the_forward() -> None:
    """Guards the test above: if the discount were unused, both calls would agree and the
    round-trip would pass while the factor did nothing.
    """
    pairs = parity_pairs(strikes=(61_000.0, 62_000.0, 63_000.0), discount=0.98)
    assert forward_from_parity(pairs, discount=1.0) != pytest.approx(FORWARD, rel=1e-6)


def test_a_single_pair_is_enough() -> None:
    """One strike quoted on both sides already determines the forward; the extra strikes buy
    robustness, not identification.
    """
    assert forward_from_parity(parity_pairs(strikes=(60_000.0,))) == pytest.approx(FORWARD)


def test_an_even_number_of_pairs_is_accepted() -> None:
    """The median of an even sample averages the two middle values; nothing about the rule
    requires an odd count, and a chain has whatever count it has.
    """
    assert forward_from_parity(parity_pairs(strikes=(59_000.0, 61_000.0))) == pytest.approx(FORWARD)


# --- forward_from_parity: why the median


def poisoned_pairs() -> list[tuple[float, float, float]]:
    """The at-the-money call carries a garbage mid -- a stale or crossed quote in one leg."""
    return [
        (strike, 50_000.0, put_mid) if strike == FORWARD else (strike, call_mid, put_mid)
        for strike, call_mid, put_mid in parity_pairs()
    ]


def test_the_median_survives_one_poisoned_pair() -> None:
    """The reason the aggregate is a median. Parity is exact in theory and noisy in practice,
    and a minority of broken pairs must not move the anchor of the whole surface.
    """
    assert forward_from_parity(poisoned_pairs()) == pytest.approx(FORWARD, rel=1e-12)


def test_the_poison_would_have_wrecked_a_mean() -> None:
    """Guards the test above: without this, a mean-based implementation could pass it on a
    poison too mild to matter, and the choice of median would go undefended.
    """
    implied = [strike + call_mid - put_mid for strike, call_mid, put_mid in poisoned_pairs()]
    assert fmean(implied) != pytest.approx(FORWARD, rel=0.1)


# --- forward_from_parity: what is an error rather than a bad quote


def test_parity_needs_something_to_infer_from() -> None:
    """No fallback: a default forward would be a guess wearing a measurement's clothes."""
    with pytest.raises(ValueError, match="at least one strike"):
        forward_from_parity([])


@pytest.mark.parametrize("bad", UNUSABLE)
def test_parity_rejects_an_unusable_discount(bad: float) -> None:
    with pytest.raises(ValueError, match="discount factor"):
        forward_from_parity(parity_pairs(), discount=bad)


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_parity_rejects_a_non_finite_call_premium(bad: float) -> None:
    """NaN passes every ordering guard, so it has to be caught by isfinite or it reaches the
    median and comes back out as a forward.
    """
    with pytest.raises(ValueError, match="must be finite"):
        forward_from_parity([(60_000.0, bad, 1_000.0)])


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_parity_rejects_a_non_finite_put_premium(bad: float) -> None:
    with pytest.raises(ValueError, match="must be finite"):
        forward_from_parity([(60_000.0, 1_000.0, bad)])


@pytest.mark.parametrize("bad", UNUSABLE)
def test_parity_rejects_an_unusable_strike(bad: float) -> None:
    with pytest.raises(ValueError, match="parity strike"):
        forward_from_parity([(bad, 2_000.0, 1_000.0)])


def test_a_zero_premium_is_a_price_not_a_missing_one() -> None:
    """`not 0.0` is True: testing a premium for truthiness rejects a legitimately worthless
    wing. Only non-finite premiums are errors here.
    """
    assert forward_from_parity([(60_000.0, 0.0, 0.0)]) == pytest.approx(60_000.0)


def test_a_chain_implying_a_negative_forward_is_rejected() -> None:
    """Malformed, not merely suspicious: log(K / F) has no value there, so there is no
    degraded answer to hand back.
    """
    with pytest.raises(ValueError, match="not a tradeable price"):
        forward_from_parity([(60_000.0, 0.0, 100_000.0)])


def test_a_chain_implying_a_zero_forward_is_rejected() -> None:
    with pytest.raises(ValueError, match="not a tradeable price"):
        forward_from_parity([(60_000.0, 0.0, 60_000.0)])


# --- crosscheck_error


def test_agreeing_routes_report_no_error() -> None:
    assert crosscheck_error(FORWARD, FORWARD) == 0.0


def test_the_error_is_relative_to_the_primary_forward() -> None:
    """A fixed absolute gap has to mean different things on a 5,000 index and a 60,000 coin,
    or no single TOML threshold can be written against it (ADR-012).
    """
    assert crosscheck_error(FORWARD, FORWARD * 1.001) == pytest.approx(0.001)


@pytest.mark.parametrize("secondary", [FORWARD * 1.01, FORWARD * 0.99])
def test_the_error_is_never_negative(secondary: float) -> None:
    """QualityBlock rejects a negative forward_crosscheck_error, so a signed residual would
    raise two layers away inside the ACL, with nothing at the raise site to explain it.
    """
    assert crosscheck_error(FORWARD, secondary) >= 0.0


def test_swapping_the_routes_barely_changes_the_magnitude() -> None:
    """Not exactly symmetric -- the primary is the denominator -- but close enough that which
    route is called primary does not decide whether a threshold trips.
    """
    lower, higher = FORWARD, FORWARD * 1.002
    assert crosscheck_error(lower, higher) == pytest.approx(
        crosscheck_error(higher, lower), rel=1e-2
    )


@pytest.mark.parametrize("bad", UNUSABLE)
def test_crosscheck_rejects_an_unusable_primary(bad: float) -> None:
    """It is the denominator: a zero would surface as an unattributable ZeroDivisionError."""
    with pytest.raises(ValueError, match="primary forward"):
        crosscheck_error(bad, FORWARD)


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_crosscheck_rejects_a_non_finite_secondary(bad: float) -> None:
    """A NaN would return a NaN, and the published contract accepts it as 'not negative'."""
    with pytest.raises(ValueError, match="secondary forward"):
        crosscheck_error(FORWARD, bad)


def test_a_secondary_of_zero_is_a_disagreement_not_an_error() -> None:
    """Only the denominator has to be positive. A zero secondary is a total disagreement, and
    reporting it as 100% is more useful than raising on it.
    """
    assert crosscheck_error(FORWARD, 0.0) == pytest.approx(1.0)
