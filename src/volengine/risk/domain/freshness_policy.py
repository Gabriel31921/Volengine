"""How old a surface may be before the report stops believing it, and what it says instead.

This is Design 7.2, the business rule the whole Risk context exists to demonstrate. Everything
else here -- the interpolation, the greeks, the book -- is arithmetic that any of a dozen
libraries could do. This module is the part that is *architecture made observable*: it takes two
timestamps that travelled across a bus inside an immutable contract and turns them into a verdict
somebody reads at the top of a report. Nothing else in the engine converts a field into a
sentence about whether the numbers below it can be trusted.

The rule is one subtraction and two thresholds, and the value is entirely in what each of the
three answers *means to whoever reads the report*:

``NORMAL`` -- the surface describes the market as it is now. The valuations below are as good as
the model gets, and a reader may act on them without qualification.

``DEGRADED`` -- the numbers are still published, and they are still the best answer available,
but the market data behind them is older than the desk agreed to treat as current. A reader is
being told to weight them accordingly: the report is not wrong, it is *late*, and how late is
exactly the gap the two thresholds bracket. Publishing under this label rather than suppressing
is the same judgement ADR-006 makes upstream -- an explicitly degraded number beats silence,
because silence is indistinguishable from a dead process.

``REJECT`` -- the report says out loud that **there is no valid surface**, and carries no
valuations at all. This is the point of the module. The alternative, and the failure it exists to
prevent, is a report that quietly prints yesterday's number in today's font: perfectly formatted,
internally consistent, and describing a market that has moved since. A blank with a reason on it
is recoverable by whoever reads it; a confident stale number is not, because nothing about it
looks wrong. ``RiskReport`` makes the consequence structural rather than optional -- a report
carrying ``REJECT`` cannot hold positions and must hold a message -- and ``risk/domain/errors.py``
records the other half of the same decision: this is a *verdict*, not an exception, because an
operator reading a traceback concludes the report generator broke, while an operator reading
"no valid surface, snapshot is 94s old" concludes the market data stopped, which is the truth.

**Staleness is measured from ``ts_snapshot``, never from when the surface was calculated**, and
that single choice is what lets this policy see a republished surface for what it is. ADR-006
says that when a calibration fails its acceptance criteria the last good surface is republished
with ``status=STALE_REPUBLISH`` and its **original** ``ts_snapshot``. So a market whose
calibration has been failing for two minutes hands Risk a contract that arrived seconds ago,
carries a fresh ``surface_id``, and is stamped with market data two minutes old -- and this
policy reads it as two minutes old, which is what it is. The calculation is recent; the market
data is not, and only one of those is a statement about whether the valuations are still true.
That is also why ``SurfaceView`` carries no status field: the staleness ADR-006 wants downstream
to notice is already fully visible in the one timestamp read here, and a second spelling of it
could only ever disagree with the first.

The thresholds are configuration loaded from TOML at the composition root (ADR-012), never
constants in this file, and the reason is that there is no single right pair of numbers. An
intraday hedging desk on a liquid crypto perp wants seconds, because a surface half a minute old
has already been overtaken by the tape. The same engine producing an end-of-day valuation over an
illiquid book wants minutes, because the alternative to a slightly old surface there is no
surface at all -- and a five-second rule would reject every report it ever produced. The
thresholds also differ per *market*: a chain that ticks four times a minute cannot be held to the
freshness of one that ticks four hundred times a second, and Market Data's own heartbeat
(``SnapshotPolicyConfig.max_quiet_seconds``) exists precisely so that a calm market keeps this
policy fed rather than sliding into ``REJECT`` for being quiet. None of that is a code change.

Deliberately absent: any clock, any state, and any memory of the previous verdict. ``now``
arrives as an argument, exactly as it does in ``SnapshotPolicy.should_emit``, so ``evaluate`` is
a total function of its inputs -- ADR-004's recorded session replays to the same three verdicts
in the same order, and a test states two instants and asserts an enum with no fixture in sight.
There is no hysteresis either: a surface oscillating around ``warn_seconds`` flips between
``NORMAL`` and ``DEGRADED`` on every report, and damping that would mean remembering the last
answer, which would make the policy's output depend on how often it happened to be called. What
the reader wants to know is how old the data is *now*, not how old it was when someone last asked.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum

from volengine.shared_kernel.domain.instants import require_aware


class FreshnessDecision(StrEnum):
    """The three things a report can say about the age of the surface behind it.

    ``NORMAL`` is "these numbers describe now". ``DEGRADED`` is "these numbers are the best
    available and they are late". ``REJECT`` is "there is no valid surface", and it is the only
    member that suppresses the valuations rather than qualifying them.

    Three members and no fourth. The tempting additions -- a ``NO_SURFACE`` for the case where the
    provider has published nothing yet, a ``REPUBLISHED`` for ADR-006's stale republish -- are
    both already covered. The first is ``REJECT`` with a different message: a report with nothing
    to value and a report with nothing fresh enough to value are the same report to whoever reads
    it, and splitting them would ask the reader to learn a distinction that changes no action. The
    second is not a decision at all but a cause, and it is already measured: a republished surface
    is simply old, and this enum reports age, not provenance.

    Explicit string values, never ``auto()``. These strings leave the domain through the ACL and
    land in a published report, a CSV column and a metrics tag, so they are a wire format: a
    member renamed with ``auto()`` behind it would silently change what a consumer parses and what
    a dashboard groups by, with nothing failing at the boundary to say so.
    """

    NORMAL = "NORMAL"
    DEGRADED = "DEGRADED"
    REJECT = "REJECT"


@dataclass(frozen=True, slots=True)
class FreshnessPolicy:
    """Where this deployment draws the two lines, and the verdict that follows from them.

    Configuration wearing a class, exactly like ``SnapshotPolicy`` upstream: it holds two numbers
    and no state whatsoever -- no clock, no surface, no record of what it last decided. That is
    what makes it safe to build once at the composition root and share across every market and
    every report, and what makes ``evaluate`` reproducible under ADR-004's replay.
    """

    warn_seconds: float
    """Age, in seconds, at which the report starts calling itself ``DEGRADED``. Positive, finite.

    The desk's answer to "how old is too old to present without a caveat". Strictly positive
    because zero would mark every report ever produced as degraded -- computing a report takes
    time, so the age is never exactly zero in a live session -- which would make the label mean
    nothing and train its readers to ignore it.
    """

    reject_seconds: float
    """Age, in seconds, beyond which there is no valid surface. Finite, and above ``warn_seconds``.

    Not "when the data becomes useless" but "when publishing it would mislead more than it
    informs". Strictly greater than ``warn_seconds`` rather than merely at least: equal thresholds
    would make ``DEGRADED`` exactly one instant wide -- reachable only by a report whose age
    landed on the boundary to the microsecond -- so the warning band would exist in the
    configuration and never once appear in a report. That is a policy nobody meant to write, and
    the likeliest way to write it by accident is a TOML file with one value pasted twice.

    Its positivity is not checked separately: it is above ``warn_seconds``, which is above zero.
    """

    def __post_init__(self) -> None:
        # Finiteness first, and the bad cases joined with `or`, for the reason this codebase keeps
        # rediscovering: `float("nan") <= 0` is False, so a NaN threshold walks straight through an
        # ordering guard written the other way round -- and then through `evaluate`, where every
        # comparison against it is False and the band it bounds silently ceases to exist. A NaN
        # `warn_seconds` deletes NORMAL, so a surface a second old is reported DEGRADED forever; a
        # NaN `reject_seconds` deletes DEGRADED, so the warning band never appears. Neither raises
        # anything anywhere, which is what makes the check here the only place to catch it. An
        # infinite `reject_seconds` fails the same guard on purpose: it is a policy that never
        # rejects anything, which is the one configuration this module exists to make impossible.
        if not math.isfinite(self.warn_seconds) or self.warn_seconds <= 0:
            raise ValueError(
                f"The warn_seconds must be positive and finite, got {self.warn_seconds}"
            )
        if not math.isfinite(self.reject_seconds):
            raise ValueError(f"The reject_seconds must be finite, got {self.reject_seconds}")
        # Safe as a bare comparison only because both values are already known to be finite.
        if self.reject_seconds <= self.warn_seconds:
            raise ValueError(
                f"The reject_seconds must be strictly above the warning threshold, got "
                f"{self.reject_seconds} against a warning at {self.warn_seconds}"
            )

    def evaluate(self, ts_snapshot: datetime, now: datetime) -> FreshnessDecision:
        """Judge a surface stamped ``ts_snapshot`` as of ``now``.

        The age is ``(now - ts_snapshot).total_seconds()``, and the bands are half-open at the
        bottom and closed at the top: ``age < warn_seconds`` is ``NORMAL``,
        ``warn_seconds <= age <= reject_seconds`` is ``DEGRADED``, and ``age > reject_seconds`` is
        ``REJECT``. Both boundaries therefore belong to ``DEGRADED``, which is the conservative
        reading in one direction and the generous one in the other: reaching the warning threshold
        exactly is already worth a caveat, while reaching the rejection threshold exactly is not
        yet worth throwing the report away. A boundary has to fall on one side, and these are the
        two sides that make the configured numbers read the way an operator says them out loud --
        "we warn at five seconds and we stop at thirty" means the warning has started at five and
        the stop has not happened at thirty.

        ``now`` is an argument rather than a reading of a clock, which is what keeps the verdict a
        pure function of its inputs: the ``Clock`` port lives in ``ports.py`` and the use case does
        the reading, so a recorded session replays to the same sequence of decisions (ADR-004).

        **A negative age is ``NORMAL``, not an error.** A surface stamped a second in the future is
        what a venue clock running marginally ahead produces, and it is routine rather than
        exotic: ``ChainSnapshot.ts_exchange`` is deliberately not clamped to ``ts_local`` upstream,
        so the skew survives all the way here by design and reconciling the two clocks is the ACL's
        job, not this policy's. Raising on it would take the entire risk report down for as long as
        a venue was a second fast -- trading one harmless anomaly for a real outage. It is the same
        lesson ``ReplayBuffer.prune`` states in ``neural_surface/``, where a sample from the future
        is kept and simply carries a negative age until the wall clock passes it, and this method
        must not contradict it. Note what is *not* claimed: a clock running backwards is a bug in
        the composition root, and the place to catch it is a comparison between successive readings
        of the clock, which is not something a stateless policy can see.

        Args:
            ts_snapshot: The instant the market data behind the surface describes -- never the
                instant the calibration ran. See the module docstring on ADR-006.
            now: The instant the report is being produced for. Timezone-aware.

        Raises:
            ValueError: If either instant is naive. They are subtracted from each other, and
                mixing a naive with an aware one raises ``TypeError`` from inside the arithmetic,
                far from whoever produced the bad value and with nothing naming the field.
        """
        require_aware(ts_snapshot, "ts_snapshot")
        require_aware(now, "now")

        age_seconds = (now - ts_snapshot).total_seconds()

        if age_seconds < self.warn_seconds:
            return FreshnessDecision.NORMAL
        if age_seconds <= self.reject_seconds:
            return FreshnessDecision.DEGRADED
        return FreshnessDecision.REJECT
