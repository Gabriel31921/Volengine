"""Value one book against one surface, or say out loud why there is nothing to value.

The end of the pipeline, and the only place in the engine where a business rule's entire input is
the difference between two instants. Everything upstream stamps time; here it is *compared*, and
the comparison decides whether the report has numbers in it at all (Design 7.2).

**The refusal is the loudest thing this context ever says, so it is a report and not an
exception.** A ``REJECT`` carries no positions, a message that names the market and the age, and
the timestamp of whatever stale surface it refused -- and it goes to the writer like any other
report. A use case that raised instead, or that returned ``None`` and let the caller decide, would
turn "there is no valid surface" into silence, which is precisely what an operator cannot tell
apart from a dead process.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime

from volengine.risk.domain.errors import ExpiredPositionError
from volengine.risk.domain.freshness_policy import FreshnessDecision, FreshnessPolicy
from volengine.risk.domain.portfolio import Portfolio
from volengine.risk.domain.ports import Clock, MetricsSink, SurfaceProvider
from volengine.risk.domain.risk_report import PositionRisk, RiskReport
from volengine.risk.domain.surface_view import SurfaceView
from volengine.risk.domain.valuation import BumpSpec, position_risk


@dataclass(frozen=True, slots=True)
class ReportSettings:
    """What a report needs beyond the surface and the book. Configuration (ADR-012).

    Two numbers and a portfolio's worth of policy live elsewhere; what is left here is the bump
    sizes -- which belong to ``BumpSpec`` -- and the discount, which is a market fact nobody in
    this engine currently computes.
    """

    bumps: BumpSpec
    """How far to move the forward and the volatility when differencing. No defaults anywhere."""

    discount: float = 1.0
    """``exp(-r * T)`` from expiry back to today. Positive and finite.

    **A parameter with an honest default rather than a field on the contract.** No producer in
    this engine computes a discount factor today, so a contract field would be one the ACL filled
    with ``1.0`` on every message -- a lie with a schema behind it. Undiscounted is exact on an
    inverse crypto book, where premium and settlement share a numeraire, and the factor is
    multiplicative on the value and all three greeks alike, so it cancels in every comparison
    Design 7.3 makes between two producers.
    """

    def __post_init__(self) -> None:
        if not math.isfinite(self.discount) or self.discount <= 0:
            raise ValueError(f"The discount must be positive and finite, got {self.discount}")


class ComputeReportUseCase:
    """One market, one book, one verdict.

    One instance per market and per producer in the comparative arrangement of Design 7.3: the
    same portfolio and the same policy, pointed at two providers, produce two reports whose
    difference is the measurement. Nothing in this class branches on which producer it is serving,
    which is what makes that difference mean something about the calibrators.
    """

    def __init__(
        self,
        provider: SurfaceProvider,
        portfolio: Portfolio,
        policy: FreshnessPolicy,
        clock: Clock,
        metrics: MetricsSink,
        settings: ReportSettings,
        expected_producer_id: str,
    ) -> None:
        """Wire the report to its collaborators.

        Args:
            provider: Where surfaces come from. A cache, a fixture, a stub answering ``None`` --
                this object cannot tell and must not learn.
            portfolio: The book. Non-empty by its own constructor, which is where a portfolio file
                that failed to load is caught.
            policy: The two freshness thresholds and the verdict that follows from them.
            clock: Read once per report, and the last place a wall clock could break ADR-004's
                promise that a recording replays to the same sequence of decisions.
            metrics: Where the observations go.
            settings: Bump sizes and the discount.
            expected_producer_id: Who this report is *expected* to come from.

                Used in exactly one situation and it is worth being precise about which:
                ``RiskReport`` requires a non-empty ``producer_id``, and when the provider holds
                no surface at all there is nobody to name. Every other report takes its producer
                from ``SurfaceView.producer_id``, which arrives attached to the surface actually
                handed back -- the only place it can be right when one call site sees two
                producers. So this is a label for the empty case, never a filter: nothing here
                compares it against what the provider returned, because a provider that answered
                with a different producer's surface has been wired that way on purpose.

        Raises:
            ValueError: If ``expected_producer_id`` is empty. It would make an unattributable
                report, and the failure would surface as a constructor error inside ``RiskReport``
                at the worst possible moment -- the first time a market went quiet.
        """
        if not expected_producer_id:
            raise ValueError("The expected producer id must not be empty")
        self._provider = provider
        self._portfolio = portfolio
        self._policy = policy
        self._clock = clock
        self._metrics = metrics
        self._settings = settings
        self._expected_producer_id = expected_producer_id

    def compute(self, market_id: str) -> RiskReport:
        """Value the book against the newest surface for this market.

        Returns:
            Always a report, never ``None`` and never an exception. Four outcomes:

            * **No surface at all.** ``REJECT``, with ``ts_snapshot`` absent -- because there
              genuinely is no instant, and inventing one would read as either a perfectly fresh
              surface or an absurdly stale one. Ordinary at start-up.
            * **A surface too old.** ``REJECT``, with the stale ``ts_snapshot`` present, which is
              the evidence for the refusal and belongs printed next to it.
            * **Every position expired.** ``REJECT``, because a report with no lines and a total
              of ``0.0`` is indistinguishable from a book that lost everything.
            * **Otherwise**, one line per position that could be valued, marked ``NORMAL`` or
              ``DEGRADED`` according to the age of the data behind it.
        """
        now = self._clock.now()
        view = self._provider.latest(market_id)

        if view is None:
            self._metrics.counter("risk.report.rejected", market=market_id, reason="no_surface")
            return RiskReport(
                market_id=market_id,
                producer_id=self._expected_producer_id,
                ts_snapshot=None,
                ts_report=now,
                freshness=FreshnessDecision.REJECT,
                positions=(),
                message=f"no surface has been published for {market_id}",
            )

        age = (now - view.ts_snapshot).total_seconds()
        decision = self._policy.evaluate(view.ts_snapshot, now)
        stamped = self._stamp(now, view)
        self._metrics.gauge(
            "risk.surface_age_seconds", age, market=market_id, producer=view.producer_id
        )
        self._metrics.counter(
            "risk.report.freshness",
            market=market_id,
            producer=view.producer_id,
            decision=decision.value,
        )

        if decision is FreshnessDecision.REJECT:
            return self._reject(
                view,
                stamped,
                f"no valid surface for {market_id}: the last snapshot is {age:.1f} seconds old",
            )

        lines = self._value(view)
        if not lines:
            return self._reject(
                view, stamped, f"every position in the book expired at or before {view.ts_snapshot}"
            )

        return RiskReport(
            market_id=market_id,
            producer_id=view.producer_id,
            ts_snapshot=view.ts_snapshot,
            ts_report=stamped,
            freshness=decision,
            positions=lines,
            message=(
                None
                if decision is FreshnessDecision.NORMAL
                else f"the surface is {age:.1f} seconds old"
            ),
        )

    def _stamp(self, now: datetime, view: SurfaceView) -> datetime:
        """When to say this report was produced, never earlier than the data it describes.

        **This is where two rules in this context's own domain disagree, and the disagreement is
        resolved here because nothing else can see both.** ``FreshnessPolicy.evaluate`` documents
        a negative age as ``NORMAL`` -- a venue clock running marginally ahead is routine, and
        rejecting the whole report over it would trade one harmless anomaly for a real outage --
        while ``RiskReport`` refuses to be stamped before the snapshot it describes, because a
        report that predates its own input is not a thing. Both are right about their own concern
        and they meet in this method.

        It is a real case rather than a hypothetical one, and by design: Market Data reconciles a
        venue clock only *outside* a configured tolerance, so a surface stamped a fraction of a
        second in the future is expected to reach here intact.

        Clamping to equality is the honest answer and the contract explicitly admits it -- the same
        equality that lets a ``ManualClock`` run the whole pipeline inside one tick (ADR-004). What
        it costs is microseconds of precision on a timestamp; what it buys is that the reported
        *age* stays negative in the metric, so the skew is still visible where it belongs instead
        of being hidden by a report that silently rounded it away.
        """
        return max(now, view.ts_snapshot)

    def _value(self, view: SurfaceView) -> tuple[PositionRisk, ...]:
        """One line per position, skipping the ones that have already expired.

        **An expired position is skipped rather than fatal**, and that is a judgement about what a
        portfolio file is. It is configuration, it outlives the contracts in it, and the first run
        after a roll will name an expiry that has passed -- so taking the whole report down for
        one dead leg would mean a book of forty positions goes unreported because one of them
        settled overnight. The skip is counted, so it is visible rather than silent.

        Nothing else is caught. A volatility the pricer refuses, a bump that cannot be taken, a
        strike the surface cannot place -- those are construction bugs upstream, and a report that
        swallowed them would publish a book that is quietly missing a line nobody asked it to drop.
        """
        lines: list[PositionRisk] = []
        for position in self._portfolio.positions:
            try:
                lines.append(
                    position_risk(
                        view=view,
                        position=position,
                        bumps=self._settings.bumps,
                        discount=self._settings.discount,
                    )
                )
            except ExpiredPositionError:
                self._metrics.counter(
                    "risk.position.expired",
                    market=view.market_id,
                    producer=view.producer_id,
                )
        return tuple(lines)

    def _reject(self, view: SurfaceView, now: datetime, message: str) -> RiskReport:
        """A refusal that still names the surface it refused.

        ``ts_snapshot`` is kept rather than blanked: the age *is* the reason, and a rejection that
        did not carry the instant behind it would leave an operator with a message and no way to
        check it.
        """
        self._metrics.counter(
            "risk.report.rejected", market=view.market_id, producer=view.producer_id
        )
        return RiskReport(
            market_id=view.market_id,
            producer_id=view.producer_id,
            ts_snapshot=view.ts_snapshot,
            ts_report=now,
            freshness=FreshnessDecision.REJECT,
            positions=(),
            message=message,
        )
