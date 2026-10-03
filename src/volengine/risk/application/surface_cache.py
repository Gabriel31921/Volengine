"""The newest surface per market and producer, held in memory.

The implementation of ``SurfaceProvider`` the pipeline actually runs, and the composite Design
7.3 needs: one object serving every market and every producer, keyed by what arrives, so Risk
grows no registry and no per-market instance of anything.

**It lives in ``application/`` rather than in ``adapters/``, and that is rule 5 rather than a
filing accident.** Everything this class does is receive published DTOs, so putting it beside the
bus adapters would mean a second module in the context that knows the published language. It talks
to no technology: no socket, no file, no asyncio primitive. What feeds it is an adapter's problem
-- a subscription, a replay of a recording, a test calling :meth:`accept` directly -- and none of
those three needs to be told apart from here.

**Last value wins, and nothing is queued.** That is the bus's own semantics (ADR-003) carried
through to its consumer: a stale surface has no value once a newer one exists, and a report built
from the second-newest would answer a question about a market that has moved. Note what follows
for a republished stale surface -- it overwrites the fresh one it is standing in for, because it
*is* the newest thing the producer has said, and the report's freshness verdict is computed from
its timestamp rather than from how it got here.

**It is also where the latency of every producer is measured** (Design 8.3, F3-E). Every surface
of every producer passes through :meth:`LastValueSurfaceProvider.accept`, and this is the one
place in the engine that sees them all on arrival -- so "snapshot to surface, per producer" and
the conflation lag of the hop that delivered it are measured here, from the two instants the
contract carries and the consumer's own clock, rather than once per producer in two contexts that
would have to agree on a name.
"""

from __future__ import annotations

from datetime import timedelta

from volengine.contracts.calibrated_surface import CalibratedSurface, SurfaceStatus
from volengine.risk.application.acl import to_surface_view
from volengine.risk.domain.ports import Clock, MetricsSink
from volengine.risk.domain.surface_view import SurfaceView

MILLISECONDS_PER_SECOND = 1_000.0


class LastValueSurfaceProvider:
    """Every producer's most recent surface, one slot each.

    Satisfies ``SurfaceProvider`` structurally, so nothing here inherits from it and the domain
    never learns this class exists.

    Not thread-safe, and it does not need to be: it is written from the event loop that consumes
    the bus and read from the same loop when a report is computed.
    """

    def __init__(self, clock: Clock, metrics: MetricsSink) -> None:
        """Wire the cache to the clock it measures arrival against and the sink it reports to.

        Args:
            clock: Read once per accepted surface, for the delivery lag. The engine's own clock
                (ADR-004), so a replay measures recorded time.
            metrics: Where the per-producer latencies go.
        """
        self._clock = clock
        self._metrics = metrics
        self._by_market: dict[str, dict[str, SurfaceView]] = {}

    def accept(self, surface: CalibratedSurface) -> None:
        """Take a newly published surface, translating it once on the way in.

        Translating here rather than on each read is the point of the cache: a report reads the
        provider once per position and the conversion to total variance is a multiply per node, so
        doing it on arrival pays it once per publication instead of once per lookup.

        Raises:
            ValueError: If the surface cannot be expressed as a ``SurfaceView`` -- notably a front
                expiry that has run out between the fit and now. Deliberately not swallowed: a
                cache that quietly dropped surfaces would leave the report saying "no surface"
                with nothing anywhere to say why, which is the failure mode this whole context is
                built to make impossible.
        """
        view = to_surface_view(surface)
        self._observe(surface)
        self._by_market.setdefault(view.market_id, {})[view.producer_id] = view

    def _observe(self, surface: CalibratedSurface) -> None:
        """Count the arrival, and time the two hops behind it unless it is a republication.

        Three series, tagged by market and producer:

        * ``risk.surface.received``, with the ``status`` as a tag. The share of
          ``STALE_REPUBLISH`` among them is the failure rate of a producer as Risk lives it --
          ADR-006 turns every refused fit with a surface behind it into one of these.
        * ``risk.surface.snapshot_to_surface_ms``, ``ts_calibrated - ts_snapshot``: the snapshot
          to surface latency of Design 8.3. It includes the time the snapshot waited in a
          conflating mailbox for a busy calibrator, which is the point -- ``pricing.cycle_ms`` is
          the fit alone, and the gap between the two series is the queueing.
        * ``risk.surface.delivery_lag_ms``, ``now - ts_calibrated``: from the producer stamping
          its result to this consumer taking it. On one process that is the bus hop and the event
          loop's backlog, which is the conflation lag Design 8.3 names. Signed and not clamped,
          for the reason ``ComputeReportUseCase._stamp`` keeps a negative age visible.

        **A republished surface is counted and not timed.** ADR-006 republishes the last good
        surface with *both* of its original instants (``as_stale_republish`` replaces only the
        status), so its snapshot-to-surface latency is the old fit's again and its delivery lag
        is the age of that fit -- a staleness, and already measured as one by the report. Timing
        it would add a spike per failure to a latency series that should not move when a fit is
        refused.
        """
        tags = {"market": surface.market_id, "producer": surface.producer_id}
        self._metrics.counter("risk.surface.received", 1, status=surface.status.value, **tags)
        if surface.status is SurfaceStatus.STALE_REPUBLISH:
            return
        self._metrics.timing(
            "risk.surface.snapshot_to_surface_ms",
            _milliseconds(surface.ts_calibrated - surface.ts_snapshot),
            **tags,
        )
        self._metrics.timing(
            "risk.surface.delivery_lag_ms",
            _milliseconds(self._clock.now() - surface.ts_calibrated),
            **tags,
        )

    def latest(self, market_id: str) -> SurfaceView | None:
        """The newest surface held for this market, or ``None``.

        **When several producers have published, the one with the newest ``ts_snapshot`` wins**,
        and ties are broken by nothing -- the first in insertion order stays. That rule exists
        because the port asks a question with one answer, and it is the honest reading of "the
        most recent surface held for this market": recency is a property of the market data behind
        a surface, not of when it happened to arrive.

        It is also why the comparative report of Design 7.3 does **not** go through this method.
        Valuing one portfolio under two producers means asking for a *named* producer twice, and
        that is what :meth:`for_producer` is for. Leaving ``latest`` to pick a winner keeps the
        single-producer path -- the walking skeleton, the CLI's one-line report -- free of a
        choice nobody wanted to make.

        Returns ``None`` when nothing has been published for this market, which is the ordinary
        state at start-up and not an error: the port documents it as an answer, and it pairs with
        ``FreshnessDecision.REJECT`` to form the two honest ways of having nothing to say.
        """
        held = self._by_market.get(market_id)
        if not held:
            return None
        return max(held.values(), key=lambda view: view.ts_snapshot)

    def for_producer(self, market_id: str, producer_id: str) -> SurfaceView | None:
        """The newest surface from one named producer, or ``None``.

        The comparative report's entry point: the same portfolio valued off ``svi-scipy`` and off
        ``mlp-torch`` is two calls to this, and the difference between the two reports is the
        measurement Design 7.3 exists to take. Deliberately **not** part of the
        ``SurfaceProvider`` port -- the domain asks for "the surface for this market" and must not
        learn that producers are something a caller can enumerate, or the moment it can name them
        is the moment a rule can branch on one.
        """
        return self._by_market.get(market_id, {}).get(producer_id)


class ProducerSurfaces:
    """One named producer's slot in a shared cache, seen as a ``SurfaceProvider``.

    The binding Design 7.3 needs. :meth:`LastValueSurfaceProvider.latest` picks the newest surface
    across every producer, which is the right answer to the question the port asks and the wrong
    one for a report that must be *about* one producer: valuing one book under two producers
    means asking for a named one twice. That is ``for_producer``, deliberately not on the port --
    so this adapter pins the name and answers the port's question with it, and the use case on
    the other side never learns that producers can be named.

    Moved here from the composition root in F3-E, where it was private: the comparative report
    (``compare_producers.py``) needs the same binding and an application module cannot import from
    ``entrypoints/``.
    """

    def __init__(self, cache: LastValueSurfaceProvider, producer_id: str) -> None:
        """Pin one producer.

        Raises:
            ValueError: If ``producer_id`` is empty, which would pin a slot nothing can fill.
        """
        if not producer_id:
            raise ValueError("A producer slot must name its producer")
        self._cache = cache
        self._producer_id = producer_id

    def latest(self, market_id: str) -> SurfaceView | None:
        """The pinned producer's newest surface for this market, or ``None``."""
        return self._cache.for_producer(market_id, self._producer_id)


def _milliseconds(elapsed: timedelta) -> float:
    return elapsed.total_seconds() * MILLISECONDS_PER_SECOND
