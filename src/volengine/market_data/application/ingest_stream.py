"""The one loop in this context: consume a provider's stream and turn it into events.

**This is the asymmetric use case, and the exception to how every other one in the engine is
shaped.** Parametric Pricing, Neural Surface and Risk all react to something already on the bus,
so each of them is a plain handler -- one event in, one event out -- with the subscription loop
living in the composition root. Ingestion has no such luxury: its source is
``MarketDataProvider.stream()``, an ``AsyncIterator`` of ``QuoteUpdate``, and ``QuoteUpdate`` is
a domain object that may not cross a boundary (rule 3). Nobody outside this context can iterate
it without taking the chain's business rules with them, so the loop lives here.

What it emits is the published language and nothing else: ``SnapshotReady`` when the policy says
the chain is worth publishing, ``ChainCompositionChanged`` when the live instrument set moves.
It does **not** publish them. An ``AsyncIterator[Event]`` hands them to the composition root,
which owns the bus and the topic names -- so this module has no idea a bus exists, and a test
drives it with a list.

**The heartbeat has a seam, and it is here.** ``SnapshotPolicyConfig.max_quiet_seconds`` exists
so that a calm market is still heard from, but this loop only evaluates the policy when an update
arrives. A market that goes genuinely silent -- no ticks at all -- therefore emits nothing, which
is precisely the failure the heartbeat was written to prevent. Closing it needs a timer racing
the stream, which needs the ``sleep`` this context's ``Clock`` declares and the two-task
structure that belongs to ``entrypoints/pipeline.py`` (F1-07). It is stated here rather than
quietly left out: today the heartbeat covers a market that is quoting but not moving, which is
the common case, and not one whose feed has stopped, which is the dangerous one.
"""

from __future__ import annotations

from collections.abc import AsyncIterator

from volengine.contracts.events import Event
from volengine.market_data.application.acl import to_composition_changed, to_snapshot_ready
from volengine.market_data.application.build_snapshot import BuildSnapshotUseCase
from volengine.market_data.domain.option_quote import InstrumentId
from volengine.market_data.domain.ports import Clock, MarketDataProvider, MetricsSink
from volengine.market_data.domain.quote_chain import QuoteChain


class IngestStreamUseCase:
    """One market's ingestion: discover the universe, then fold every tick into the chain.

    One instance per market, sharing its ``QuoteChain`` with exactly one
    :class:`~volengine.market_data.application.build_snapshot.BuildSnapshotUseCase`. The sharing
    is the design: the chain is the aggregate, this object writes to it and that one reads from
    it, and putting the two behind one class would merge "record what arrived" with "decide what
    to publish", which is the split the domain spent ``QuoteChain.apply`` and
    ``QuoteChain.snapshot`` making.
    """

    def __init__(
        self,
        provider: MarketDataProvider,
        chain: QuoteChain,
        build_snapshot: BuildSnapshotUseCase,
        clock: Clock,
        metrics: MetricsSink,
        market_id: str,
    ) -> None:
        """Wire the loop to its collaborators.

        Args:
            provider: The venue adapter. Its lifetime is owned here -- see :meth:`run`.
            chain: The aggregate this loop writes into.
            build_snapshot: Consulted after every update. It answers ``None`` most of the time.
            clock: Stamps ``ChainCompositionChanged``. The snapshot cycle reads its own.
            metrics: Where the observations go.
            market_id: Which market this loop serves, for the composition event and the metric
                tags. Passed explicitly rather than read from the conventions because this object
                has no other reason to hold a ``MarketConventions``, and a collaborator injected
                to be asked one question is a dependency pretending to be a relationship.

        Raises:
            ValueError: If ``market_id`` is empty. It ends up inside a published event, where an
                empty identifier is a message nobody can route or attribute.
        """
        if not market_id:
            raise ValueError("The market id must not be empty")
        self._provider = provider
        self._chain = chain
        self._build_snapshot = build_snapshot
        self._clock = clock
        self._metrics = metrics
        self._market_id = market_id
        self._live: set[InstrumentId] = set()

    async def run(self) -> AsyncIterator[Event]:
        """Discover the universe, then stream until the provider ends or the task is cancelled.

        The sequence:

        1. **Discover**, and announce the result. The composition event goes out before a single
           quote does, because ``ChainCompositionChanged`` is what a consumer sizing fixed-shape
           buffers reacts to (ADR-009), and it would be useless arriving after the first snapshot
           it was supposed to describe.
        2. **Fold each update into the chain**, then ask the snapshot cycle whether that update
           made the chain worth publishing. Asking after every single tick is not as expensive as
           it looks: ``should_emit`` short-circuits on the cadence long before it touches a quote.
        3. **Announce an instrument that was born mid-session.** A strike listed between two
           discovery polls still trades, and ``QuoteChain.apply`` registers it on the spot rather
           than dropping its quotes to keep a tidy inventory. The chain therefore knows about it
           and no consumer does, until this loop says so.
        4. **Close the provider** when the loop ends, however it ends. The ``finally`` covers
           cancellation, which is the ordinary way a long-running task dies, and an exception from
           the venue, which is the interesting way.

        Yields:
            Published events, in the order they were decided. Nothing here knows what topic any of
            them belongs on; that is routing, and routing is the composition root's.

        Raises:
            ValueError: If the provider delivers an update for a different underlying.
                ``QuoteChain.apply`` raises it and this loop deliberately does not catch it: a
                chain fed another market's quotes would corrupt every slice-level rule silently,
                and the mistake is a wiring bug in the composition root -- a category that must
                fail loudly on its first occurrence rather than be counted and survived.
        """
        try:
            discovered = await self._provider.discover()
            self._live = set(discovered)
            self._chain.set_live_instruments(self._live)
            yield self._composition_event()

            async for update in self._provider.stream():
                self._chain.apply(update)
                self._metrics.counter("marketdata.update.applied", market=self._market_id)

                if update.instrument not in self._live:
                    self._live.add(update.instrument)
                    yield self._composition_event()

                snapshot = self._build_snapshot.build()
                if snapshot is not None:
                    yield to_snapshot_ready(snapshot)
        finally:
            await self._provider.close()

    def _composition_event(self) -> Event:
        """Announce the live set as it stands now.

        The whole set every time, never a delta (ADR-013, and the bus conflates: ADR-003). A
        consumer that missed the previous event cannot reconstruct the universe from increments,
        and one that receives only the latest of five is correct with no further work.
        """
        self._metrics.counter(
            "marketdata.composition.changed",
            market=self._market_id,
        )
        return to_composition_changed(
            market_id=self._market_id,
            ts=self._clock.now(),
            instruments=frozenset(self._live),
        )
