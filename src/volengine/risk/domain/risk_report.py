"""What this context finally says out loud: one portfolio, one surface, one verdict.

Two types. ``PositionRisk`` is a single valued line -- the position, the volatility that was
actually used to value it, and the numbers that came out. ``RiskReport`` is the whole statement:
which market and which producer it came from, which instant of market data it rests on, when it
was computed, how fresh that data was, and either the valued lines or the reason there are none.

**This is the output object of the Risk context, and it is shaped like one.** Its counterpart one
context over is ``CalibrationResult``: an answer that travels with the evidence the decision was
made on, so that whoever reads it can judge it without re-running the computation. The evidence
here is different in kind. A calibration reports how well it fitted; a risk report reports *what
it was allowed to say at all*, which is the freshness decision, the snapshot instant it was
measured against, and -- when the answer is a refusal -- a message naming the reason. Design 7.2
is the whole point of this context, and this type is where that rule becomes something an operator
can print.

**The rejection invariants are enforced in the constructor, deliberately.** A report carrying
``FreshnessDecision.REJECT`` holds no positions and must hold a message; a ``NORMAL`` or
``DEGRADED`` report holds at least one. Publishing a number under a ``REJECT`` label is the exact
failure Design 7.2 exists to prevent -- a silently old valuation wearing a warning nobody reads --
and a constructor prevents it more reliably than a use case does, because a use case is a place
where someone adds a branch under deadline and a constructor is a place where the object refuses
to exist. It is the same instinct as ``errors.py`` one module over: the refusal is a *value*,
which means the type system can be made to guarantee its shape.

**Nothing here is a contract.** Rule 3 forbids this layer from importing ``contracts/`` and rule 5
reserves every DTO for ``application/acl.py``, so there is no ``to_dict``, no ``from_dict`` and no
``schema_version`` below: nothing in this module ever crosses a process boundary. Turning a report
into output -- a console table, a CSV row, the two-producer comparison of Design 7.3 -- is the
``ReportWriter`` port's job, and every implementer of that port is an adapter. That keeps the
formatting vocabulary out of the domain entirely: this module knows what a valued position *is*
and has no opinion about what a column is.

Two departures from the sketch in ``Implementation.md``, both argued where they are made:
``total_value`` is a property rather than a field, and ``ts_snapshot`` is optional.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime

from volengine.risk.domain.freshness_policy import FreshnessDecision
from volengine.risk.domain.portfolio import Position
from volengine.shared_kernel.domain.instants import require_aware


def _require_finite(value: float, what: str) -> None:
    """Reject NaN and both infinities, as a plain ``ValueError``.

    The loosest guard in this module, and the right one for every quantity below that is
    legitimately negative or legitimately zero: a short position is worth a negative number, and a
    flattened leg has a delta of exactly zero. Anything stricter here would refuse an honest line
    of the report.

    A deliberate twin of the private helper in ``portfolio.py`` and in the pricing modules rather
    than an import of one of them: the leading underscore says that name is not part of any
    module's surface, and reaching across for it would tie this output object to a module it has
    no other reason to know. Two lines of duplication cost less than that edge.
    """
    if not math.isfinite(value):
        raise ValueError(f"The {what} must be finite, got {value}")


def _require_positive_finite(value: float, what: str) -> None:
    """Reject anything that is not a usable positive number.

    ``isfinite`` first, and the *bad* cases joined with ``or``, because ``float("nan") <= 0`` is
    ``False``: a NaN walks straight through any ordering guard written the other way round. That
    matters more here than almost anywhere, because this object is the last stop before the
    report is written: a NaN volatility that reaches this point has already survived the
    interpolation and the pricing, and past this constructor there is nobody left to catch it --
    it simply appears in the output as a line an operator has to reason about.
    """
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"The {what} must be positive and finite, got {value}")


@dataclass(frozen=True, slots=True)
class PositionRisk:
    """One position, valued, with the volatility it was valued at and the greeks that followed.

    The volatility is carried rather than left implicit, and that is the field that makes the
    report auditable. Everything else here is a consequence of it: given the surface, the position
    and this one number, the value can be recomputed by hand. Without it a reader who disagreed
    with a line would have to re-run the interpolation to find out whether the disagreement was
    about the surface or about the pricing, which is exactly the question a report should answer
    rather than pose.

    **Every number below is already scaled by the position's quantity.** They are portfolio
    quantities, not per-option ones: a delta of ``5.4`` on ten calls means the book moves by 5.4
    units of the underlying, not that each option has a delta of 5.4. This is stated plainly and
    repeated on each field because the other convention -- per-option greeks, multiplied by the
    holder at read time -- is just as common in practice, reads identically at a glance, and is
    wrong by a factor of the quantity in a way no invariant can detect. The sign follows from the
    same rule: a short leg has a negative delta, gamma and vega, because it is short them.
    """

    position: Position
    """The position these numbers describe, carried whole rather than by reference or by key.

    A report line that named its position by an identifier would be unreadable without the
    portfolio beside it, and the portfolio is configuration that may already have been reloaded by
    the time anyone reads the file. Carrying the immutable object costs nothing -- it is frozen,
    so there is no copy to keep in step -- and it makes the line self-describing: the strike, the
    side, the expiry and the quantity that produced every number here travel with them.
    """

    vol: float
    """The interpolated Black-76 volatility actually used, annualised as a decimal. Positive,
    finite.

    Read off the surface at this position's own ``(k, T)``, not at a node: it is what the bilinear
    interpolation returned, which is the number that was priced with and therefore the number
    worth reporting. A vol quoted at the nearest node instead would be a plausible-looking value
    that never entered any calculation.

    Strictly positive rather than non-negative. Every node of the grid carries a strictly positive
    total variance and the interpolation is a convex combination of nodes, so zero is unreachable
    by construction; a zero arriving here would mean the interpolation is broken, not that the
    market is calm, and the intrinsic-value limit it implies is not something this context prices.
    """

    value: float
    """Value of the whole position: ``quantity * option value``, in the quote currency. Finite,
    **any sign**.

    Negative is ordinary and must stay ordinary: a short option is a liability, and its line in
    the report is a negative number. So the guard is finiteness alone -- never a positivity test,
    and never a truthiness test, because ``not 0.0`` is ``True`` and a leg flattened intraday is
    worth exactly zero while still deserving its line in the report.

    Undiscounted unless a discount factor was passed to the valuation, which today it never is:
    the factor defaults to ``1.0`` because no producer computes one, and it scales this number and
    every greek below identically.
    """

    delta: float
    """Sensitivity of :attr:`value` to the forward, **already multiplied by quantity**. Finite,
    any sign.

    Computed by bumping the forward and revaluing *through the surface* -- the moneyness is
    recomputed at the bumped forward and the volatility re-interpolated there -- so it is a
    sticky-moneyness delta, not a sticky-strike one. That is a modelling choice with consequences
    for whoever hedges on it, and it is argued where it is made, in ``valuation.py``. Recorded
    here so that the number and its assumption are never separated.

    Zero is legal: a position with zero quantity, or one deep enough in either wing that the bump
    moves nothing at this precision. Finiteness is the only guard.
    """

    gamma: float
    """Second derivative of :attr:`value` in the forward, **already multiplied by quantity**.
    Finite, any sign.

    Positive for a long option and negative for a short one, which is why the sign is not
    constrained here: the constraint people have in mind -- gamma is positive -- is a statement
    about a *long* option, and encoding it would make a short book unrepresentable.

    A central second difference over an interpolated grid, so it inherits both the truncation
    error of the bump and the kinks the bilinear interpolation leaves at the nodes. It is the
    noisiest number in the report by some margin, and Design 7.4 wants exactly that comparison
    against Pricing's analytic-differentiation greeks rather than a smoothed-over version of it.
    """

    vega: float
    """Sensitivity of :attr:`value` to the volatility, **already multiplied by quantity**. Finite,
    any sign.

    Per unit of volatility, not per volatility point: a move of one point (``0.01``) changes the
    value by a hundredth of this number. Both conventions are in daily use and they differ by a
    factor of a hundred, so the unit is stated here rather than inferred from the magnitude.

    A bump of the single interpolated volatility this position sees, holding the forward fixed --
    not a parallel shift of the surface. On a book spread across expiries the sum of these numbers
    is therefore not the sensitivity to any one scenario, and adding them up is a decision for the
    reader with a scenario in mind, which is why this type reports the line and never the total.
    """

    def __post_init__(self) -> None:
        _require_positive_finite(self.vol, "implied volatility")
        _require_finite(self.value, "position value")
        _require_finite(self.delta, "delta")
        _require_finite(self.gamma, "gamma")
        _require_finite(self.vega, "vega")


@dataclass(frozen=True, slots=True)
class RiskReport:
    """The whole answer for one portfolio against one surface, verdict included.

    Either it carries valued positions, or it carries a refusal and says why. Those are the only
    two shapes this object admits, and the constructor is what makes that true -- see
    :meth:`__post_init__`. The freshness decision is not decoration attached to a number; it
    decides whether there is a number at all.
    """

    market_id: str
    """Which market these numbers price, in the engine's own naming. Non-empty.

    Carried because a report is meaningless without it and because the ``SurfaceProvider`` port is
    keyed by it: the same portfolio can be reported against several markets, and the identifier is
    what pairs a report with the surface it came from when both are sitting in an output
    directory.
    """

    producer_id: str
    """Which surface producer supplied the numbers -- the SVI calibrator, the neural one. Non-
    empty.

    This is the field that makes Design 7.3 possible. The comparative report is the same portfolio
    valued twice, and the two runs are distinguishable only by this string; every metric this
    context emits is tagged with it for the same reason. It belongs on the report rather than on
    each line because one report is one producer's answer: a line-level producer would describe a
    portfolio valued half by one model and half by another, which is not a comparison of anything.
    """

    ts_snapshot: datetime | None
    """The instant of market data these numbers rest on, timezone-aware. **May be absent.**

    A departure from ``Implementation.md``, which lists it as a plain ``datetime``, and the reason
    is that there are two distinct ways to have nothing to say. A surface that is too old has an
    instant -- an old one, which is precisely the evidence for the refusal and must be printed
    next to it. A market for which the provider held *no* surface at all, which is the ordinary
    state at start-up before the first calibration lands, has no instant, and there is nothing
    honest to put here. Making the field non-optional would force that second case to invent a
    timestamp: the report's own creation time, which would read as a perfectly fresh surface, or
    the epoch, which would read as an absurdly stale one. Both are fabrications about market data
    that never existed, and this type exists to refuse exactly that kind of fabrication.

    ``None`` is therefore an answer, not a missing value, and it pairs with ``SurfaceProvider
    .latest`` returning ``None``. Not when the surface exists: ADR-006 republishes a rejected
    calibration under its *original* snapshot instant, so a stale surface always brings its own
    timestamp along and the ``REJECT`` decision that follows is always explicable from this field.
    """

    ts_report: datetime
    """When this report was computed, timezone-aware. **Not** when the market data was observed.

    The distinction is the one ADR-006 draws upstream between ``ts_calibrated`` and
    ``ts_snapshot``, and it survives all the way down here for the same reason: a report produced
    a moment ago from data that is ten minutes old is a fresh report of a stale market, and
    collapsing the two instants would make that state unreportable. The freshness decision is
    computed from the gap between them, so both have to be present for the verdict to be
    auditable.

    When :attr:`ts_snapshot` is present, this may not precede it. Equality is allowed, and that is
    not slack: under a ``ManualClock`` (ADR-004) a test values a portfolio at the very instant the
    surface describes, and a strict inequality would make deterministic replay -- the whole point
    of that port -- fail on its most natural fixture.
    """

    freshness: FreshnessDecision
    """The verdict of the freshness policy on :attr:`ts_snapshot`. Design 7.2, made structural.

    ``NORMAL`` and ``DEGRADED`` reports carry numbers; the difference between them is that a
    ``DEGRADED`` one is asking to be read with suspicion. ``REJECT`` carries no numbers at all,
    and the invariants in :meth:`__post_init__` are what guarantee that -- the label and the
    content cannot come apart.
    """

    positions: tuple[PositionRisk, ...]
    """The valued lines, one per position of the portfolio, in the portfolio's own order.

    Empty if and only if :attr:`freshness` is ``REJECT``. In the other direction the emptiness is
    impossible rather than merely unusual: a ``Portfolio`` refuses to be empty, and every position
    in one either produces a line or raises ``ExpiredPositionError``, so a non-rejected report over
    zero positions would mean the use case silently dropped the entire book. Requiring at least one
    is what turns that into a failure at the point it happened instead of a total of ``0.0``
    indistinguishable from a portfolio that lost everything.

    Not netted and not aggregated: two lots of the same option are two lines, exactly as they are
    two positions in the portfolio. Netting them is a decision for whoever reads the report.
    """

    message: str | None
    """Free text explaining the verdict, in the report's own voice. Required when rejecting.

    On a ``REJECT`` this field *is* the report -- "no valid surface for BTC-DERIBIT: the last
    snapshot is 94 seconds old" -- and Design 7.2's requirement that the refusal be explicit is
    met here or nowhere. On a ``NORMAL`` or ``DEGRADED`` report it is optional and usually absent,
    though a ``DEGRADED`` one is the natural place to say how old the data actually is.

    Free text rather than a code, because it is written for a human reading a console or a CSV.
    Anything a machine has to branch on lives in :attr:`freshness`, which is an enum precisely so
    that nothing downstream is ever tempted to parse this string.
    """

    def __post_init__(self) -> None:
        """Enforce the identity fields, the ordering of the two instants, and the verdict rules.

        The aware-timestamp checks come first on purpose: comparing an aware datetime with a naive
        one raises ``TypeError``, so the ordering check below is only meaningful once both are
        known to carry a zone.

        The last two blocks are the business rule of Design 7.2 made structural. They are here,
        and not in the use case that builds reports, because a use case is a place where a branch
        gets added under deadline and a constructor is a place where the object simply refuses to
        exist. A ``REJECT`` carrying a valued position is the precise failure this context was
        built to demonstrate the absence of.
        """
        # Truthiness on strings is safe and idiomatic; on any number in this module it would not
        # be, since ``not 0.0`` is ``True`` and zero is a legal value everywhere it appears.
        if not self.market_id:
            raise ValueError("The market id must not be empty")
        if not self.producer_id:
            raise ValueError("The producer id must not be empty")

        require_aware(self.ts_report, "ts_report")
        if self.ts_snapshot is not None:
            require_aware(self.ts_snapshot, "ts_snapshot")
            if self.ts_report < self.ts_snapshot:
                raise ValueError(
                    "The report cannot be stamped before the snapshot it describes, got a report "
                    f"at {self.ts_report} and a snapshot at {self.ts_snapshot}"
                )

        if self.freshness is FreshnessDecision.REJECT:
            if self.positions:
                raise ValueError(
                    f"A rejected report must not carry any position, got {len(self.positions)}"
                )
            if not self.message:
                raise ValueError(
                    "A rejected report must carry a message saying why there is no valid surface"
                )
        elif not self.positions:
            raise ValueError(f"A {self.freshness.value} report must carry at least one position")

    @property
    def total_value(self) -> float:
        """Sum of the position values, in the quote currency. ``0.0`` when there are none.

        **Derived rather than stored, a departure from ``Implementation.md``'s sketch**, which
        lists it as a field. A stored total is a second source of truth for a number that is
        already fully determined by the lines above it, and the only way to keep the two honest
        would be an invariant that recomputes the sum and compares -- at which point the field is
        pure risk with no benefit, since the check costs exactly what the property costs. Worse,
        it would be a number an operator cannot audit: a total that disagrees with its own
        positions is a report nobody can act on, and there is no rule for deciding which half to
        believe. The same call ``TrainingSample.total_variance`` makes one context over, for the
        same reason.

        ``fsum`` rather than the builtin ``sum``, and that is not a flourish. A book is long and
        short at once, so this sum cancels by design: a portfolio worth a few thousand can be the
        difference between two legs worth millions, and pairwise floating-point addition loses
        precisely the low digits that survive the cancellation. ``fsum`` is exactly rounded, so
        the reported total does not depend on the order the positions happened to be listed in.

        Zero for a rejected report, which holds no positions -- and that is the one number this
        object will state without a surface behind it. It is safe only because the ``REJECT``
        label travels with it and the constructor guarantees they cannot be separated; read
        without :attr:`freshness`, a total of zero and a portfolio that lost everything look the
        same, which is why nothing in this context reports a total on its own.
        """
        return math.fsum(one.value for one in self.positions)
