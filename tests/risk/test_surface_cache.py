"""The last-value cache the pipeline runs as its ``SurfaceProvider``.

Two behaviours carry the weight: that a newer surface replaces an older one rather than queueing
behind it, and that two producers on one market stay separable. The first is the bus's conflation
carried through to its consumer; the second is what makes the comparative report of Design 7.3
possible at all.
"""

from __future__ import annotations

from datetime import timedelta

import pytest

from tests.risk.builders import EXPIRIES, NOW, make_cache, make_calibrated_surface, make_view
from tests.support import RecordingMetrics
from volengine.contracts.calibrated_surface import SurfaceStatus
from volengine.platform.clock import ManualClock
from volengine.risk.application.surface_cache import ProducerSurfaces

MARKET = "BTC-DERIBIT"


def test_an_empty_cache_answers_none() -> None:
    """An ordinary answer, not an error: nothing has been published yet."""
    assert make_cache().latest(MARKET) is None


def test_an_unknown_market_answers_none() -> None:
    cache = make_cache()
    cache.accept(make_calibrated_surface())

    assert cache.latest("ETH-DERIBIT") is None


def test_an_accepted_surface_comes_back_as_this_contexts_own_model() -> None:
    """Never the DTO: the domain does not know the published language (rule 3)."""
    cache = make_cache()
    cache.accept(make_calibrated_surface())

    view = cache.latest(MARKET)

    assert view is not None
    assert view.total_variance[0][0] > 0


def test_a_newer_surface_replaces_the_older_one() -> None:
    """A stale surface has no value once a newer one exists (ADR-003)."""
    cache = make_cache()
    cache.accept(make_calibrated_surface(surface_id="first"))
    cache.accept(
        make_calibrated_surface(surface_id="second", ts_snapshot=NOW + timedelta(seconds=5))
    )

    view = cache.latest(MARKET)

    assert view is not None
    assert view.surface_id == "second"


def test_two_producers_on_one_market_are_kept_apart() -> None:
    """Otherwise the comparative report would have one surface to compare against itself."""
    cache = make_cache()
    cache.accept(make_calibrated_surface(producer_id="svi-scipy"))
    cache.accept(make_calibrated_surface(producer_id="mlp-torch"))

    assert cache.for_producer(MARKET, "svi-scipy") is not None
    assert cache.for_producer(MARKET, "mlp-torch") is not None


def test_a_named_producer_that_has_published_nothing_answers_none() -> None:
    cache = make_cache()
    cache.accept(make_calibrated_surface(producer_id="svi-scipy"))

    assert cache.for_producer(MARKET, "mlp-torch") is None


def test_latest_picks_the_producer_whose_data_is_newest() -> None:
    """Recency is a property of the market data behind a surface, not of when it arrived."""
    cache = make_cache()
    cache.accept(
        make_calibrated_surface(producer_id="mlp-torch", ts_snapshot=NOW + timedelta(seconds=9))
    )
    cache.accept(make_calibrated_surface(producer_id="svi-scipy", ts_snapshot=NOW))

    view = cache.latest(MARKET)

    assert view is not None
    assert view.producer_id == "mlp-torch"


def test_two_markets_are_held_separately() -> None:
    cache = make_cache()
    cache.accept(make_calibrated_surface(market_id="BTC-DERIBIT"))
    cache.accept(make_calibrated_surface(market_id="ETH-DERIBIT"))

    btc = cache.latest("BTC-DERIBIT")
    eth = cache.latest("ETH-DERIBIT")

    assert btc is not None and eth is not None
    assert btc.market_id == "BTC-DERIBIT"
    assert eth.market_id == "ETH-DERIBIT"


def test_a_surface_that_cannot_be_translated_is_refused_rather_than_dropped() -> None:
    """A cache that swallowed it would leave the report saying "no surface" with no reason why."""
    cache = make_cache()

    with pytest.raises(ValueError, match="first expiry"):
        cache.accept(make_calibrated_surface(ts_snapshot=EXPIRIES[0] + timedelta(days=1)))


# --- what the cache measures on arrival (Design 8.3)


def _timings(metrics: RecordingMetrics, name: str) -> list[tuple[float, dict[str, str]]]:
    return [(ms, tags) for timed, ms, tags in metrics.timings if timed == name]


def test_snapshot_to_surface_latency_is_timed_per_producer() -> None:
    metrics = RecordingMetrics()
    cache = make_cache(clock=ManualClock(NOW + timedelta(seconds=3)), metrics=metrics)

    cache.accept(
        make_calibrated_surface(
            producer_id="mlp-torch", ts_calibrated=NOW + timedelta(milliseconds=250)
        )
    )

    [(latency, tags)] = _timings(metrics, "risk.surface.snapshot_to_surface_ms")
    assert latency == pytest.approx(250.0)
    assert tags == {"market": MARKET, "producer": "mlp-torch"}


def test_delivery_lag_is_measured_from_the_fit_to_the_consumers_clock() -> None:
    """The conflation lag of the hop: the producer stamped it, and the consumer took it later."""
    metrics = RecordingMetrics()
    cache = make_cache(clock=ManualClock(NOW + timedelta(seconds=2)), metrics=metrics)

    cache.accept(make_calibrated_surface(ts_calibrated=NOW + timedelta(milliseconds=500)))

    [(lag, _)] = _timings(metrics, "risk.surface.delivery_lag_ms")
    assert lag == pytest.approx(1_500.0)


def test_a_republished_surface_is_counted_but_not_timed() -> None:
    """Its instants are the old fit's, so timing it would put a failure into a latency series."""
    metrics = RecordingMetrics()
    cache = make_cache(clock=ManualClock(NOW + timedelta(seconds=60)), metrics=metrics)

    cache.accept(make_calibrated_surface(status=SurfaceStatus.STALE_REPUBLISH))

    assert metrics.timings == []
    assert metrics.counters == [
        (
            "risk.surface.received",
            1,
            {"market": MARKET, "producer": "svi-scipy", "status": "STALE_REPUBLISH"},
        )
    ]


def test_a_fresh_surface_is_counted_with_its_status() -> None:
    """The guard on the test above: the same counter fires for a healthy surface, timed too."""
    metrics = RecordingMetrics()
    cache = make_cache(metrics=metrics)

    cache.accept(make_calibrated_surface(status=SurfaceStatus.OK))

    received = [tags for name, _, tags in metrics.counters if name == "risk.surface.received"]
    assert [tags["status"] for tags in received] == ["OK"]
    assert len(metrics.timings) == 2


def test_a_surface_that_fails_translation_is_not_measured() -> None:
    """A refused surface was never held; a latency for it would describe nothing Risk used."""
    metrics = RecordingMetrics()
    cache = make_cache(metrics=metrics)

    with pytest.raises(ValueError, match="first expiry"):
        cache.accept(make_calibrated_surface(ts_snapshot=EXPIRIES[0] + timedelta(days=1)))

    assert metrics.counters == []
    assert metrics.timings == []


# --- one producer's slot, seen through the port


def test_a_producer_slot_answers_with_its_own_producer_even_when_another_is_newer() -> None:
    """Otherwise a per-producer report would quietly value the other producer's surface."""
    cache = make_cache()
    cache.accept(make_calibrated_surface(producer_id="svi-scipy", ts_snapshot=NOW))
    cache.accept(
        make_calibrated_surface(producer_id="mlp-torch", ts_snapshot=NOW + timedelta(seconds=9))
    )

    view = ProducerSurfaces(cache, "svi-scipy").latest(MARKET)

    assert view is not None
    assert view.producer_id == "svi-scipy"


def test_a_producer_slot_that_was_never_filled_answers_none() -> None:
    cache = make_cache()
    cache.accept(make_calibrated_surface(producer_id="svi-scipy"))

    assert ProducerSurfaces(cache, "mlp-torch").latest(MARKET) is None


def test_a_producer_slot_must_name_its_producer() -> None:
    with pytest.raises(ValueError, match="name its producer"):
        ProducerSurfaces(make_cache(), "")


def test_the_cache_holds_the_same_view_the_acl_builds() -> None:
    """The measurement added in F3-E did not change what is cached."""
    cache = make_cache()
    cache.accept(make_calibrated_surface())

    assert cache.latest(MARKET) == make_view()
