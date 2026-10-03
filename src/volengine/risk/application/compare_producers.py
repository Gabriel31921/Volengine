"""The comparative report of Design 7.3: one book, two producers, and what differs.

``ComputeReportUseCase`` already says, of itself, that two instances pointed at two producers are
the comparative arrangement -- "the difference between the two reports is the measurement". This
use case is that sentence turned into one call: it runs the same report twice, through the same
policy, bumps and clock, against two named slots of one cache, pairs the lines, and adds the one
number neither report can produce alone -- the distance between the two surfaces themselves.

**Built from ``ComputeReportUseCase`` rather than beside it**, so nothing about valuation,
freshness or expiry handling exists twice. A comparison that priced positions its own way would
be measuring two code paths as well as two surfaces, and the difference between producers is only
worth reporting if it is the *only* difference.

**Run by the composition root whenever a market has two producers**
(``entrypoints/pipeline.py``, ``_comparisons_for``): the first configured calibrator is the
baseline, and the comparison is recomputed each time either producer publishes. Its output today
is the metric series; no writer prints a ``ComparativeReport`` yet, which ``docs/SEAMS.md`` (Risk)
records.
"""

from __future__ import annotations

from volengine.risk.application.compute_report import ComputeReportUseCase, ReportSettings
from volengine.risk.application.surface_cache import LastValueSurfaceProvider, ProducerSurfaces
from volengine.risk.domain.comparison import ComparativeReport, pair_lines, surface_distance
from volengine.risk.domain.freshness_policy import FreshnessPolicy
from volengine.risk.domain.portfolio import Portfolio
from volengine.risk.domain.ports import Clock, MetricsSink


class CompareProducersUseCase:
    """One market, one book, two producers, one comparison.

    The two producers are named at construction because a comparison is a standing arrangement
    -- "the neural surface, measured against the parametric one" -- and not a question asked with
    new names each time. ``baseline`` is the reference the differences are measured *from*.
    """

    def __init__(
        self,
        cache: LastValueSurfaceProvider,
        portfolio: Portfolio,
        policy: FreshnessPolicy,
        clock: Clock,
        metrics: MetricsSink,
        settings: ReportSettings,
        baseline_producer_id: str,
        challenger_producer_id: str,
    ) -> None:
        """Wire both reports to one cache, one book and one policy.

        Args:
            cache: The concrete cache, not the port. A comparison has to name producers, and
                naming them is exactly what the port refuses to offer (``for_producer``).
            portfolio: The book, valued identically under both.
            policy: One freshness policy for both. Two policies would let one producer pass on
                a surface the other would have been refused for, and the comparison would
                measure the thresholds.
            clock: Read by each report and therefore twice per comparison, which under a live
                clock gives two microseconds-apart ``ts_report`` values. That is deliberate: each
                report stays exactly what ``ComputeReportUseCase`` would have produced alone.
            metrics: Where both reports and the comparison report to. The two inner reports emit
                their own series as any report does, tagged by producer.
            settings: Bumps and discount, shared for the same reason as the policy.
            baseline_producer_id: The reference producer.
            challenger_producer_id: The producer measured against it.

        Raises:
            ValueError: If either id is empty, or if they are the same producer -- a comparison
                of a surface with itself is a wiring mistake that would report a perfect zero.
        """
        if baseline_producer_id == challenger_producer_id:
            raise ValueError(
                f"A comparison needs two different producers, got {baseline_producer_id!r} twice"
            )
        self._cache = cache
        self._metrics = metrics
        self._baseline_id = baseline_producer_id
        self._challenger_id = challenger_producer_id
        self._baseline, self._challenger = (
            ComputeReportUseCase(
                provider=ProducerSurfaces(cache, producer_id),
                portfolio=portfolio,
                policy=policy,
                clock=clock,
                metrics=metrics,
                settings=settings,
                expected_producer_id=producer_id,
            )
            for producer_id in (baseline_producer_id, challenger_producer_id)
        )

    def compute(self, market_id: str) -> ComparativeReport:
        """Value the book under both producers and compare.

        Returns:
            Always a report. When either side is refused -- no surface, a stale one, a book that
            has entirely expired -- the comparison carries both reports, no lines and no
            distance, and the refusal's message is in the report that refused.
        """
        baseline = self._baseline.compute(market_id)
        challenger = self._challenger.compute(market_id)
        tags = {
            "market": market_id,
            "baseline": self._baseline_id,
            "challenger": self._challenger_id,
        }

        comparison = ComparativeReport(
            market_id=market_id,
            baseline=baseline,
            challenger=challenger,
            lines=(),
            distance=None,
        )
        if not comparison.comparable:
            self._metrics.counter("risk.comparison.incomparable", 1, **tags)
            return comparison

        baseline_view = self._cache.for_producer(market_id, self._baseline_id)
        challenger_view = self._cache.for_producer(market_id, self._challenger_id)
        # Both are present: a report that is not REJECT was valued off a surface, and nothing
        # runs between the report and this read to evict it. Asserted rather than assumed, for
        # mypy and for the day something does.
        if baseline_view is None or challenger_view is None:
            raise RuntimeError("a comparable report was computed off a surface no longer held")

        distance = surface_distance(baseline_view, challenger_view)
        comparison = ComparativeReport(
            market_id=market_id,
            baseline=baseline,
            challenger=challenger,
            lines=pair_lines(baseline, challenger),
            distance=distance,
        )
        self._metrics.gauge("risk.comparison.value_diff", comparison.total_value_diff, **tags)
        if distance is None:
            self._metrics.counter("risk.comparison.no_overlap", 1, **tags)
        else:
            self._metrics.gauge("risk.comparison.distance_rms_vol_bp", distance.rms_vol_bp, **tags)
            self._metrics.gauge("risk.comparison.distance_max_vol_bp", distance.max_vol_bp, **tags)
        return comparison
