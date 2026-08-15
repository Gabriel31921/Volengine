"""When a snapshot is worth publishing, and how good the one we publish is.

The rule is **clock AND material change**. The cadence alone would republish the chain every
second whether or not anything happened, and every one of those snapshots costs a full
calibration downstream to reproduce a number nobody asked for again. Movement alone would let
a fast market emit hundreds of times a second. Neither half is sufficient; the conjunction is
the policy.

Everything here is a pure function wearing a class. ``SnapshotPolicy`` holds configuration and
nothing else -- no clock, no chain, no memory of the last decision. ``now`` and ``last_emit``
arrive as parameters, so the same inputs give the same answer on a live session and on a
replay of it (ADR-004), and a test needs no fixture beyond the values it asserts about.

Quality is assessed here too, and deliberately does *not* gate emission: a degraded snapshot
is published, marked as degraded. See ``should_emit`` for why turning that around would be a
mistake.

**Deliberately absent: any dependency on ``MarketConventions``.** Design 4.2 notes that in
equities the policy would consult the session calendar -- do not emit outside trading hours,
force one at the close. There is no calendar in this system (ADR-007: crypto trades
continuously, and the venue's expiry time is all the calendar the conventions carry), so the
policy would have nothing to ask it. Injecting a collaborator in order to never call it is a
worse lie than the missing seam: it reads as if the rule existed and is checked by nothing.
When a session-aware market arrives, the parameter arrives with it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum

from volengine.market_data.domain.quote_chain import ChainStats
from volengine.shared_kernel.domain.instants import require_aware


@dataclass(frozen=True, slots=True)
class SnapshotPolicyConfig:
    """Where the snapshotting policy draws its lines. Configuration, never constants (ADR-012).

    Loaded per market from TOML: a chain that ticks four times a minute and one that ticks four
    hundred times a second want different cadences, and that is an operational decision rather
    than a code change.
    """

    cadence_seconds: float
    """Minimum spacing between two snapshots. Positive and finite.

    A ceiling on frequency, not a promise of one: it says how often we may publish at most,
    never that we will.
    """

    material_move_threshold: float
    """Largest relative mid move, as a fraction, below which the chain counts as motionless.

    Non-negative and finite; ``0.0`` disables the filter, which turns the policy into a pure
    cadence timer. Zero is a legitimate deployment setting, consistently with ``min_size`` and
    ``convexity_tolerance`` in ``admissibility.py`` -- only negatives are impossible.
    """

    min_coverage_ratio: float
    """Fraction of known instruments that must carry an admissible quote. In ``[0, 1]``.

    Read by ``assess_quality`` only. It marks a snapshot, it never suppresses one.
    """

    max_quiet_seconds: float | None
    """Heartbeat: publish anyway after this long without one. ``None`` disables it.

    This exists because of what a strict AND does to a calm market. With no movement above the
    threshold, no snapshot is ever emitted -- and downstream, "nothing published" is
    indistinguishable from "the process died". Risk's ``FreshnessPolicy`` watches staleness
    climb, goes DEGRADED, then REJECT, and the end-of-day report says there was no valid
    surface, because the market was quiet. That is the worst failure mode this system can have:
    silent, and looking exactly like a real one.

    The heartbeat makes silence mean silence. Positive, finite, and never below
    ``cadence_seconds`` -- a heartbeat that fired more often than the cadence permits would be
    a second cadence contradicting the first.
    """

    def __post_init__(self) -> None:
        if not math.isfinite(self.cadence_seconds) or self.cadence_seconds <= 0:
            raise ValueError(
                f"The cadence_seconds must be positive and finite, got {self.cadence_seconds}"
            )

        # Zero is how a deployment switches the movement filter off, so only negatives are
        # rejected. NaN is caught by isfinite first: `nan < 0` is False and would pass.
        if not math.isfinite(self.material_move_threshold) or self.material_move_threshold < 0:
            raise ValueError(
                f"The material_move_threshold must be non-negative and finite, "
                f"got {self.material_move_threshold}"
            )

        if not 0 <= self.min_coverage_ratio <= 1:
            raise ValueError(
                f"The min_coverage_ratio must be in [0, 1], got {self.min_coverage_ratio}"
            )

        if self.max_quiet_seconds is not None:
            if not math.isfinite(self.max_quiet_seconds) or self.max_quiet_seconds <= 0:
                raise ValueError(
                    f"The max_quiet_seconds must be positive and finite, "
                    f"got {self.max_quiet_seconds}"
                )
            if self.max_quiet_seconds < self.cadence_seconds:
                raise ValueError(
                    f"The max_quiet_seconds must not be below the cadence, got "
                    f"{self.max_quiet_seconds} against a cadence of {self.cadence_seconds}"
                )


class DegradationReason(StrEnum):
    """Why a snapshot is worth less than it looks. One member, on purpose.

    The obvious candidates are already counted somewhere else. A stale chain makes each of its
    quotes carry ``QuoteFlagD.STALE``, a flagged quote is not admissible, and a chain of
    inadmissible quotes has a low admissible ratio -- so a ``STALE_CHAIN`` member would report
    the same fact a second time under a different name, and an operator reading two reasons
    would think two things went wrong.

    The enum stays open for a cause that is genuinely independent of coverage: a forward that
    stopped arriving, a venue in a declared maintenance window. Until one exists, adding
    members would only make the tally lie.

    Explicit string values, never ``auto()``: the ACL maps these onto the published
    ``QualityBlock``, and a rename must not silently change what crosses the boundary.
    """

    LOW_COVERAGE = "LOW_COVERAGE"


@dataclass(frozen=True, slots=True)
class QualityAssessment:
    """The verdict on one snapshot: every reason it is degraded, and nothing else.

    ``degraded`` is a property rather than a field, and that is not a stylistic preference. As
    a field it is derived data stored beside the data it derives from, so nothing stops
    ``QualityAssessment(degraded=False, reasons=(LOW_COVERAGE,))`` from existing: an object
    that constructs cleanly, compares equal to nothing suspicious, passes every test that does
    not happen to look at both halves, and lies to every consumer that trusts the flag. Derived
    from ``reasons`` on every read, the contradiction is unrepresentable.
    """

    reasons: tuple[DegradationReason, ...]

    @property
    def degraded(self) -> bool:
        """True when there is at least one reason. Truthiness of a tuple, which is safe."""
        return bool(self.reasons)


@dataclass(frozen=True, slots=True)
class SnapshotPolicy:
    """The decision to publish, and the judgement of what is published.

    Stateless by construction: it holds configuration and nothing that changes. The instant and
    the previous emission are arguments, which is what makes ``should_emit`` a total function
    of its inputs -- replayable, trivially testable, and safe to share between chains.
    """

    config: SnapshotPolicyConfig

    def should_emit(
        self,
        stats: ChainStats,
        max_relative_move: float,
        last_emit: datetime | None,
        now: datetime,
    ) -> bool:
        """Decide whether the chain as it stands is worth publishing.

        Args:
            stats: Coverage and freshness of the chain at ``now``.
            max_relative_move: Largest relative mid move since the last emission, as a
                fraction. An argument rather than a member of ``ChainStats`` because it is
                measured *against a baseline*, and ``ChainStats`` describes an instant: the
                same chain has a different move against a different reference, and storing it
                there would make the stats depend on when they were last read.
            last_emit: When a snapshot last went out, or ``None`` if none ever has.
            now: The instant being decided about. Timezone-aware.

        Raises:
            ValueError: If ``now`` or ``last_emit`` is naive. Both are subtracted from each
                other, and mixing naive with aware raises ``TypeError`` from inside the
                arithmetic, far from whoever produced the bad value.
        """
        require_aware(now, "now")
        if last_emit is not None:
            require_aware(last_emit, "last_emit")

        # `<= 0` rather than `not stats.coverage_ratio`: `not 0.0` is True, so the truthiness
        # version appears to work while being right for the wrong reason, and would reject a
        # legitimate zero anywhere else in this file.
        if stats.coverage_ratio <= 0:
            return False

        # `is None`, never `not last_emit`: datetime defines no __bool__ and is always truthy,
        # so the truthy form would silently never fire.
        if last_emit is None:
            return True

        # A backwards clock -- a ManualClock rewound in a replay -- yields a negative elapsed,
        # which is below any positive cadence and therefore does not emit. That is the intended
        # reading: time has not passed.
        elapsed = (now - last_emit).total_seconds()
        if elapsed < self.config.cadence_seconds:
            return False

        # The heartbeat outranks the movement test, and only that one: it is the escape hatch
        # for a market so calm that the AND would keep us silent forever. It never overrides
        # the cadence, which is why the config forbids it from being shorter.
        max_quiet = self.config.max_quiet_seconds
        if max_quiet is not None and elapsed >= max_quiet:
            return True

        # Quality is deliberately not consulted here. A degraded snapshot is published and
        # marked -- that is the entire reason `degraded` exists on the published QualityBlock.
        # Gating emission on min_coverage_ratio would convert an observable degradation into
        # silence, and silence is the one symptom downstream cannot tell from a dead process.
        #
        # `>=` so that a threshold of 0.0 means "filter off" and every cadence tick emits: a
        # move of exactly zero clears a threshold of exactly zero. A NaN move fails the
        # comparison and does not emit, which is the safe direction -- the heartbeat still
        # guarantees the chain is heard from.
        return max_relative_move >= self.config.material_move_threshold

    def assess_quality(self, stats: ChainStats) -> QualityAssessment:
        """Judge how much of the chain is genuinely usable, not merely present.

        Measured on the **admissible** ratio, ``n_quotes_admissible / n_instruments_known``,
        not on ``stats.coverage_ratio``. Design 4.2 defines degradation as less than X% of the
        chain having *fresh quotes with a reasonable spread*, and ``coverage_ratio`` only
        counts the ones a mid arrived for -- stale, crossed and absurdly wide quotes included.
        A chain where every instrument is quoted and every quote is a day old has full coverage
        and no usable data, and judging it by coverage would call it healthy.

        The ratio is computed here rather than added to ``ChainStats`` on purpose: the stats
        report counts, the policy passes judgement, and a ratio in the stats would be one more
        derived number to keep consistent with the counts beside it.
        """
        reasons: list[DegradationReason] = []

        # A chain with no known instruments is degraded outright, before any division: the
        # ratio is undefined, and an empty chain is exactly the case a coverage rule exists to
        # catch. Returning "healthy" for it would make an unconfigured market look fine.
        if stats.n_instruments_known <= 0:
            reasons.append(DegradationReason.LOW_COVERAGE)
        else:
            admissible_ratio = stats.n_quotes_admissible / stats.n_instruments_known
            if admissible_ratio < self.config.min_coverage_ratio:
                reasons.append(DegradationReason.LOW_COVERAGE)

        return QualityAssessment(reasons=tuple(reasons))
