"""Decide whether the chain is worth publishing, and publish it if so.

The use case that owns the *cadence* of the whole engine: nothing downstream computes anything
until this object says a snapshot exists. It holds the two pieces of state the domain refused --
when a snapshot last went out, and how many have gone out -- and it does nothing else. Every
judgement it appears to make belongs to somebody else: ``SnapshotPolicy`` decides whether to
emit and whether the result is degraded, ``QuoteChain`` freezes the view, and ``acl.py``
translates it.

**Synchronous, and it reads the clock.** ``FreshnessPolicy``, ``SnapshotPolicy`` and every rule
in the domain take ``now`` as an argument precisely so that they stay pure; somebody has to read
a clock eventually, and this layer is where ADR-004 says it happens. Under a ``ManualClock`` the
whole cycle is therefore deterministic, which is what lets the tests below assert on an emission
decision with no waiting anywhere.
"""

from __future__ import annotations

import math
from datetime import datetime

from volengine.contracts.market_snapshot import MarketSnapshot
from volengine.market_data.application.acl import (
    build_snapshot_id,
    clock_skew_seconds,
    to_market_snapshot,
)
from volengine.market_data.domain.ports import Clock, MetricsSink
from volengine.market_data.domain.quote_chain import QuoteChain
from volengine.market_data.domain.snapshot_policy import SnapshotPolicy


class BuildSnapshotUseCase:
    """One market's snapshotting cycle: ask the policy, freeze the chain, translate, publish.

    One instance per market, owning that market's chain. Not thread-safe and not trying to be --
    ``QuoteChain`` is not either, and the concurrency model of this engine is one asyncio task per
    market with the numerical work pushed to executors (ADR-005).
    """

    def __init__(
        self,
        chain: QuoteChain,
        policy: SnapshotPolicy,
        clock: Clock,
        metrics: MetricsSink,
        max_skew_seconds: float,
    ) -> None:
        """Wire the cycle to its collaborators.

        Args:
            chain: The live chain for this market. Owned by this use case in the sense that
                nothing else may call ``reset_move_baseline`` on it -- the baseline is half of the
                emission decision, and two objects resetting it would make the movement filter
                measure an interval nobody chose.
            policy: Cadence, movement and quality thresholds. Stateless and shareable.
            clock: Where ``now`` comes from (ADR-004).
            metrics: Where the observations go. ``logging`` is banned below this layer.
            max_skew_seconds: How far the venue's clock may sit from ours before its stamps are
                disbelieved. Configuration (ADR-012); see ``acl.reconcile_exchange_instant``.

        Raises:
            ValueError: If ``max_skew_seconds`` is not positive and finite. Checked here as well
                as in the ACL so that a misconfigured deployment fails when it is *built* rather
                than on its first snapshot, which may be minutes into a session.
        """
        if not math.isfinite(max_skew_seconds) or max_skew_seconds <= 0:
            raise ValueError(
                f"The maximum clock skew must be positive and finite, got {max_skew_seconds}"
            )
        self._chain = chain
        self._policy = policy
        self._clock = clock
        self._metrics = metrics
        self._max_skew_seconds = max_skew_seconds
        self._last_emit: datetime | None = None
        self._sequence = 0

    @property
    def last_emit(self) -> datetime | None:
        """When a snapshot last went out, or ``None`` if none ever has.

        Exposed read-only because the heartbeat of ``SnapshotPolicyConfig.max_quiet_seconds`` is
        evaluated against it, and the pipeline of F1-07 -- which is the only thing that can drive
        a timer -- needs to know how long this market has been silent without being able to
        rewrite the answer.
        """
        return self._last_emit

    def build(self) -> MarketSnapshot | None:
        """Emit a snapshot if the policy says the chain is worth publishing, else ``None``.

        ``None`` is the ordinary answer and by far the most frequent one: the chain ticks several
        times a second and the cadence lets a fraction of those through. It is not an error, not a
        degraded state, and nothing upstream should log it.

        The sequence of decisions, in order, and why it is this order:

        1. **Read the clock once.** Every rule below is evaluated against one instant. Reading it
           twice would let the cadence and the freshness of the same snapshot disagree by however
           long the assembly took.
        2. **Ask the policy.** ``should_emit`` sees the stats, the movement since the baseline and
           the last emission. Nothing here second-guesses it.
        3. **Freeze the chain** only once the answer is yes -- ``snapshot()`` costs the same as
           ``stats()`` again, and there is no reason to pay it for a cycle that publishes nothing.
        4. **Refuse to publish an empty snapshot.** A view with no slices is legal -- every expiry
           may be waiting for its first forward -- and it is useless: a calibrator handed it can
           only fail, and the failure would be attributed to the calibrator. The baseline and
           ``last_emit`` are deliberately *not* advanced in that case, so the next update that
           produces a usable slice emits immediately instead of waiting out another cadence.
        5. **Translate, then advance the state.** In that order, so that a snapshot which fails
           its own DTO invariants leaves the use case exactly as it was rather than half-committed
           with a consumed sequence number and a reset baseline.

        Returns:
            The published snapshot, or ``None`` when this cycle has nothing worth sending.

        Raises:
            ValueError: If the frozen view cannot be expressed as a valid ``MarketSnapshot``.
                Deliberately not caught: that is a bug in the domain or in the translation, and
                swallowing it would publish silence for a market that is ticking perfectly well.
        """
        now = self._clock.now()
        stats = self._chain.stats(now)
        move = self._chain.max_relative_move_since_baseline()

        if not self._policy.should_emit(stats, move, self._last_emit, now):
            return None

        snapshot = self._chain.snapshot(now)
        if not snapshot.slices:
            self._metrics.counter("marketdata.snapshot.empty", market=snapshot.market_id)
            return None

        published = to_market_snapshot(
            snapshot=snapshot,
            snapshot_id=build_snapshot_id(snapshot.market_id, self._sequence),
            quality=self._policy.assess_quality(stats),
            max_skew_seconds=self._max_skew_seconds,
        )

        self._sequence += 1
        self._last_emit = now
        self._chain.reset_move_baseline()

        self._observe(published, skew=clock_skew_seconds(snapshot))
        return published

    def _observe(self, published: MarketSnapshot, skew: float) -> None:
        """Report what went out, so that a session can be judged without parsing log prose.

        The clock skew is gauged on **every** emission rather than only when it crosses the
        tolerance, which is the point of measuring it at all: a reconciliation that became visible
        only once it fired would leave a venue drifting towards the threshold invisible until the
        day it arrived, and the first symptom would be a timestamp changing meaning.
        """
        market = published.market_id
        self._metrics.counter("marketdata.snapshot.published", market=market)
        self._metrics.gauge("marketdata.clock_skew_seconds", skew, market=market)
        self._metrics.gauge(
            "marketdata.coverage_ratio", published.quality.coverage_ratio, market=market
        )
        self._metrics.gauge(
            "marketdata.max_age_seconds", published.quality.max_age_seconds, market=market
        )
        if published.quality.degraded:
            self._metrics.counter("marketdata.snapshot.degraded", market=market)
