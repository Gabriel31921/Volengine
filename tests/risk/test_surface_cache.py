"""The last-value cache the pipeline runs as its ``SurfaceProvider``.

Two behaviours carry the weight: that a newer surface replaces an older one rather than queueing
behind it, and that two producers on one market stay separable. The first is the bus's conflation
carried through to its consumer; the second is what makes the comparative report of Design 7.3
possible at all.
"""

from __future__ import annotations

from datetime import timedelta

import pytest

from tests.risk.builders import EXPIRIES, NOW, make_calibrated_surface
from volengine.risk.application.surface_cache import LastValueSurfaceProvider

MARKET = "BTC-DERIBIT"


def test_an_empty_cache_answers_none() -> None:
    """An ordinary answer, not an error: nothing has been published yet."""
    assert LastValueSurfaceProvider().latest(MARKET) is None


def test_an_unknown_market_answers_none() -> None:
    cache = LastValueSurfaceProvider()
    cache.accept(make_calibrated_surface())

    assert cache.latest("ETH-DERIBIT") is None


def test_an_accepted_surface_comes_back_as_this_contexts_own_model() -> None:
    """Never the DTO: the domain does not know the published language (rule 3)."""
    cache = LastValueSurfaceProvider()
    cache.accept(make_calibrated_surface())

    view = cache.latest(MARKET)

    assert view is not None
    assert view.total_variance[0][0] > 0


def test_a_newer_surface_replaces_the_older_one() -> None:
    """A stale surface has no value once a newer one exists (ADR-003)."""
    cache = LastValueSurfaceProvider()
    cache.accept(make_calibrated_surface(surface_id="first"))
    cache.accept(
        make_calibrated_surface(surface_id="second", ts_snapshot=NOW + timedelta(seconds=5))
    )

    view = cache.latest(MARKET)

    assert view is not None
    assert view.surface_id == "second"


def test_two_producers_on_one_market_are_kept_apart() -> None:
    """Otherwise the comparative report would have one surface to compare against itself."""
    cache = LastValueSurfaceProvider()
    cache.accept(make_calibrated_surface(producer_id="svi-scipy"))
    cache.accept(make_calibrated_surface(producer_id="mlp-torch"))

    assert cache.for_producer(MARKET, "svi-scipy") is not None
    assert cache.for_producer(MARKET, "mlp-torch") is not None


def test_a_named_producer_that_has_published_nothing_answers_none() -> None:
    cache = LastValueSurfaceProvider()
    cache.accept(make_calibrated_surface(producer_id="svi-scipy"))

    assert cache.for_producer(MARKET, "mlp-torch") is None


def test_latest_picks_the_producer_whose_data_is_newest() -> None:
    """Recency is a property of the market data behind a surface, not of when it arrived."""
    cache = LastValueSurfaceProvider()
    cache.accept(
        make_calibrated_surface(producer_id="mlp-torch", ts_snapshot=NOW + timedelta(seconds=9))
    )
    cache.accept(make_calibrated_surface(producer_id="svi-scipy", ts_snapshot=NOW))

    view = cache.latest(MARKET)

    assert view is not None
    assert view.producer_id == "mlp-torch"


def test_two_markets_are_held_separately() -> None:
    cache = LastValueSurfaceProvider()
    cache.accept(make_calibrated_surface(market_id="BTC-DERIBIT"))
    cache.accept(make_calibrated_surface(market_id="ETH-DERIBIT"))

    btc = cache.latest("BTC-DERIBIT")
    eth = cache.latest("ETH-DERIBIT")

    assert btc is not None and eth is not None
    assert btc.market_id == "BTC-DERIBIT"
    assert eth.market_id == "ETH-DERIBIT"


def test_a_surface_that_cannot_be_translated_is_refused_rather_than_dropped() -> None:
    """A cache that swallowed it would leave the report saying "no surface" with no reason why."""
    cache = LastValueSurfaceProvider()

    with pytest.raises(ValueError, match="first expiry"):
        cache.accept(make_calibrated_surface(ts_snapshot=EXPIRIES[0] + timedelta(days=1)))
