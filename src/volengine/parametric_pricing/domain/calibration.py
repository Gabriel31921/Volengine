"""What one calibration cycle is given and what it hands back, in this context's own words.

Four types, in two pairs. ``CalibrationTask`` and its ``SliceTask`` are the problem statement:
the market, as numbers, with every convention already resolved. ``CalibrationResult`` and its
``SliceResult`` are the answer: the fitted parameters plus the evidence needed to decide whether
they may be published at all.

**None of these are contracts, and that is the point.** They look like ``SliceData`` and
``CalibratedSurface`` from a distance, and they must not become them: rule 3 forbids this layer
from importing ``contracts/`` at all, and rule 5 reserves every DTO for
``application/acl.py``. So there is no ``to_dict``, no ``from_dict`` and no ``schema_version``
anywhere below -- nothing here ever crosses a process boundary. The ACL translates an incoming
``MarketSnapshot`` into a ``CalibrationTask``, and an outgoing ``CalibrationResult`` into a
``CalibratedSurface``, and it is the single module that knows both vocabularies. The overlap
between the two is the architecture working rather than debt, exactly as ``OptionKindP`` next to
``OptionKind`` already is: a published datum and an optimiser's input are different things with
different invariants, and unifying them would put a shared type on the boundary and make every
context's vocabulary hostage to every other's.

**The task carries homogeneous numbers only.** No venue, no day count, no option side, no strike
in currency, no flags: log-moneyness, volatility, weights and a tenor. Everything
convention-dependent was resolved upstream (ADR-002), which is precisely what lets the scipy
calibrator of F2-05 be swapped for the JAX one of F3-A without either of them ever learning
which market it is fitting -- and what makes the two comparable at all, since they are handed
byte-identical inputs.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from itertools import pairwise
from typing import Protocol

from volengine.parametric_pricing.domain.svi_slice import SVIParams
from volengine.shared_kernel.domain.instants import require_aware


def _require_positive_finite(value: float, what: str) -> None:
    """Reject anything that is not a usable positive number, as a plain ``ValueError``.

    A deliberate twin of the private helper in ``black76.py`` rather than an import of it: a
    leading underscore says that name is not part of that module's surface, and reaching across
    for it would tie these value objects to a pricing module they have no other reason to know.
    Two lines of duplication cost less than that edge.

    The condition tests ``isfinite`` first and joins the *bad* cases with ``or``, because
    ``float("nan") <= 0`` is ``False``: a NaN walks straight through any ordering guard written
    the other way round, which is the mistake this codebase keeps rediscovering.
    """
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"The {what} must be positive and finite, got {value}")


def _require_non_negative_finite(value: float, what: str) -> None:
    """Same guard, one step looser: zero is admissible, NaN and infinity still are not."""
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"The {what} must be non-negative and finite, got {value}")


class _Expiring(Protocol):
    """Anything attached to one expiry: read-only, structural, private to this module.

    ``SliceTask`` and ``SliceResult`` are unrelated types with no common base -- one is an input
    and one is an output, and giving them a shared ancestor would be modelling a relationship
    that does not exist -- yet the collection invariant over both is literally the same rule.
    A structural ``Protocol`` lets :func:`_require_ordered_by_tenor` state it once without
    inventing that ancestor, which is the same reason every port in this project is a
    ``Protocol``: the contract is the shape, not the lineage.
    """

    @property
    def expiry(self) -> datetime: ...

    @property
    def tenor_years(self) -> float: ...


def _require_ordered_by_tenor(slices: Sequence[_Expiring], what: str) -> None:
    """A collection of expiries is addressable: non-empty, ordered, and free of duplicates.

    Structural rules, not modelling judgements -- the same three ``SVISurface`` enforces, for
    the same reasons. Anything walking the tenor axis assumes the order, and a duplicated expiry
    means two different answers to one question.

    ``pairwise``, never ``zip(xs, xs[1:], strict=True)``: consecutive pairs differ in length by
    one by design, so the ``strict`` flag this codebase requires everywhere else would be wrong
    here and its absence would look like an oversight.

    Uniqueness is checked separately from the ordering rather than deduced from it. Two distinct
    expiries can round to the same year fraction under a coarse day count, and one expiry
    appearing twice with tenors that differ by a rounding error would slip past a purely ordinal
    test.
    """
    if not slices:
        raise ValueError(f"A {what} must hold at least one slice")

    for near, far in pairwise(slices):
        if near.tenor_years >= far.tenor_years:
            raise ValueError(
                "The slices must be strictly increasing in tenor, got "
                f"{near.tenor_years} before {far.tenor_years}"
            )

    expiries = [one.expiry for one in slices]
    if len(set(expiries)) != len(expiries):
        raise ValueError(f"The slices must have unique expiries, got {expiries}")


@dataclass(frozen=True, slots=True)
class SliceTask:
    """One expiry, as the optimiser sees it: an axis, a target curve, and how much each point
    counts.

    Three parallel tuples rather than a tuple of quote objects, and the parallelism is the
    interface: a loss function consumes them as arrays, index by index, and any structure richer
    than that would have to be flattened at the top of every calibrator anyway. What was a quote
    upstream -- a strike, a side, a mid, a spread, a set of flags -- has already been collapsed
    into the only three numbers a fit in ``(k, w)`` space can use.

    **No minimum number of quotes, deliberately.** Five free parameters through three points is
    ill-conditioned, and a slice that thin genuinely cannot pin down a smile. But that is a
    modelling judgement, and ADR-008 already says how it is answered: regularise towards the
    neighbouring slice, or fix ``m`` and ``sigma`` below N quotes -- a threshold that is TOML
    configuration (ADR-012) and lives with the calibrator, not with the type. Refusing to build
    the object would instead mean refusing to *represent* a market that really does quote two
    strikes at that expiry, and would leave the use case unable to report why it declined to fit
    it. It is the same split ``SVIParams`` documents one module over: the value object admits,
    the metric judges.
    """

    expiry: datetime
    """Exact expiry instant, timezone-aware.

    Aware because it is subtracted from other timestamps and used as the key that matches this
    slice to its warm start across snapshots; mixing a naive datetime into that arithmetic raises
    ``TypeError`` far from here. Validated with the shared kernel's ``require_aware`` rather than
    a private copy of the same two lines.
    """

    tenor_years: float
    """Year fraction from the snapshot to expiry, under the market's day count. Positive, finite.

    Not recoverable from ``expiry`` alone: ACT/365 fixed and a 252-business-day count give
    different numbers for the same two instants, which is why the conversion happened upstream
    (ADR-002) and is carried rather than recomputed. It is what turns the fitted total variance
    ``w`` into an annualised volatility, and the units in which the errors below are stated.
    """

    forward: float
    """Forward price for this expiry, in the currency the strikes were quoted in. Positive,
    finite.

    Carried even though the fit never divides by it: ``log_moneyness`` is already measured
    against it, so the calibration itself needs nothing else. It is here because the result has
    to be turned back into strikes eventually -- by the grid builder, by the risk engine pricing
    a real position -- and the forward the axis was built from is the only one that makes that
    inversion consistent. Recovering it later from the snapshot would risk pairing a smile with a
    forward that has since moved.
    """

    log_moneyness: tuple[float, ...]
    """``k = ln(K / F)`` for each quote, finite and **strictly increasing**. Non-empty.

    Strictly, not merely non-decreasing, and that follows from how the slice is built: one
    volatility is inverted per strike, from the out-of-the-money twin -- the side of the strike
    with real premium and a real spread, rather than the deep in-the-money one whose price is
    almost all intrinsic. One option per strike means one ``k`` per strike, so two entries at the
    same ``k`` would be the same strike counted twice and would double its weight in the loss
    without anyone deciding to.

    Enforcing the order here rather than sorting on arrival keeps the tuples addressable by
    index: element ``i`` of all three fields describes one quote, and a sort that touched one
    tuple and not the others would silently pair a volatility with someone else's weight.
    """

    implied_vol: tuple[float, ...]
    """**Our own** Black-76 volatility for each quote, annualised as a decimal. Positive, finite.

    Inverted from the mid by ``black76.implied_vol`` with our forward and our tenor, never the
    venue's published IV (Design 4.5). An exchange's IV is the output of the exchange's own
    model, forward and expiry convention; fitting SVI to it would be fitting our model to
    someone else's, and every convention mismatch would enter the surface disguised as a smile
    feature. The venue's number travels as a comparison column and stops at the ACL.

    Positive rather than non-negative: a zero volatility is the degenerate limit where the option
    is worth exactly its intrinsic value, which is not a quote anyone can fit.
    """

    weights: tuple[float, ...]
    """Relative influence of each quote on the loss. Finite, non-negative, and summing above
    zero.

    **Already normalised and dimensionless by the time they arrive**, and the loss consumes them
    exactly as given (Design 5.3). They are built from vega and the relative spread -- a tight,
    high-vega quote near the money carries information about the smile that a wide, near-worthless
    wing quote does not -- but the ratio was formed upstream, so no calibrator re-derives it and
    the JAX and scipy implementations cannot silently weight the same market differently.

    Zero is legal for a single quote and is how a flagged quote is kept in the slice with no
    influence on the fit: the point stays visible in the residuals that are reported without
    steering the parameters. What is refused is the whole tuple summing to zero, which is not a
    weighting at all but an empty problem wearing the shape of a full one -- every calibrator
    would divide by that sum.
    """

    def __post_init__(self) -> None:
        require_aware(self.expiry, "expiry")
        _require_positive_finite(self.tenor_years, "tenor in years")
        _require_positive_finite(self.forward, "forward")

        if not (len(self.log_moneyness) == len(self.implied_vol) == len(self.weights)):
            raise ValueError(
                "The log-moneyness, implied volatility and weight tuples must have the same "
                f"length, got {len(self.log_moneyness)}, {len(self.implied_vol)} and "
                f"{len(self.weights)}"
            )
        if not self.log_moneyness:
            raise ValueError("A slice task must hold at least one quote")

        for index, k in enumerate(self.log_moneyness):
            if not math.isfinite(k):
                raise ValueError(f"The log-moneyness at index {index} must be finite, got {k}")
        for index, vol in enumerate(self.implied_vol):
            _require_positive_finite(vol, f"implied volatility at index {index}")
        for index, weight in enumerate(self.weights):
            _require_non_negative_finite(weight, f"weight at index {index}")

        # Checked per element above and in aggregate here, which are two different failures. A
        # single negative weight sails through a bare `sum(weights) > 0` whenever its neighbours
        # cover it, and it would quietly ask the optimiser to move *away* from that quote.
        total_weight = sum(self.weights)
        if total_weight <= 0:
            raise ValueError(f"The weights must sum to a positive number, got {total_weight}")

        for near, far in pairwise(self.log_moneyness):
            if near >= far:
                raise ValueError(
                    f"The log-moneyness must be strictly increasing, got {near} before {far}"
                )


@dataclass(frozen=True, slots=True)
class CalibrationTask:
    """Every slice of one market at one instant, plus the identity of the snapshot it came from.

    The identity fields are not decoration. ADR-006 republishes the last good surface with its
    **original** ``ts_snapshot`` when a fit is rejected, so downstream freshness is measured
    against the market data rather than against the calculation; that only works if the instant
    travels with the task from the very start and is never re-stamped along the way.
    """

    market_id: str
    """Which market this is, in the engine's own naming. Non-empty.

    Carried for routing and for the per-market warm start in ``CalibrationState``, never as an
    input to the mathematics: nothing below this line branches on it. That is the invariant that
    keeps the calibrator convention-free -- a market identifier the loss could read would be an
    open invitation to special-case one venue.
    """

    snapshot_id: str
    """Identity of the snapshot being fitted. Non-empty.

    What makes a published surface traceable back to the exact input that produced it, which is
    what a replay needs to reproduce a result rather than merely a similar one.
    """

    ts_snapshot: datetime
    """When the market data was observed, timezone-aware. **Not** when the fit ran.

    The distinction is the whole of ADR-006: a republished surface has a recent
    ``ts_calibrated`` and an old ``ts_snapshot``, and it is the second one that tells risk how
    stale the numbers really are.
    """

    slices: tuple[SliceTask, ...]
    """The expiries to fit, strictly increasing in ``tenor_years``, expiries unique. Non-empty.

    Ordered because the regularisation of ADR-008 pulls a thin slice towards its *neighbour*,
    and "neighbour" is only defined once the collection is sorted.
    """

    def __post_init__(self) -> None:
        # String emptiness, so plain truthiness is safe here -- unlike on any number in this
        # module, where `not 0.0` is True and a legitimate zero would be rejected.
        if not self.market_id:
            raise ValueError("The market id must not be empty")
        if not self.snapshot_id:
            raise ValueError("The snapshot id must not be empty")
        require_aware(self.ts_snapshot, "ts_snapshot")
        _require_ordered_by_tenor(self.slices, "calibration task")


@dataclass(frozen=True, slots=True)
class SliceResult:
    """One fitted expiry, with the evidence the acceptance rule is evaluated on.

    The parameters alone are not enough to decide whether a fit may be published. ADR-006 says
    an honest old surface beats a broken new one, and every field below the parameters exists so
    that "broken" can be answered from data instead of from hope.
    """

    expiry: datetime
    """The expiry these parameters describe, timezone-aware. The key that matches this result to
    its task, and to the warm start of the next cycle."""

    tenor_years: float
    """Year fraction used in the fit. Positive and finite.

    Repeated from the task rather than looked up, because it is what annualises ``w`` and the
    consumer of a result has no access to the task that produced it.
    """

    params: SVIParams
    """The five fitted raw SVI parameters, in total-variance space.

    Already guaranteed to *be* a surface by its own constructor -- finite, non-negative minimum
    variance, ``|rho| < 1`` -- and deliberately not guaranteed to be arbitrage-free: that is
    measured by ``durrleman.py`` and compared against a configured threshold, for the same
    reason the metrics below are numbers rather than exceptions.
    """

    rmse_vol_bp: float
    """Root-mean-square fit error, **in basis points of volatility**. Non-negative, finite.

    In vol rather than in price, and that choice is not cosmetic. A price error means something
    different at every strike: a whole currency unit is noise on a fat at-the-money premium and
    absurd in a wing where the option is worth a fraction of a tick, so a price-space RMSE is
    dominated by wherever the premiums happen to be largest. "12 basis points of vol" is the same
    statement everywhere on the slice, it is directly comparable to the bid-ask spread the fit is
    trying to beat -- a wide crypto wing trades a couple of vol points wide, so a fit inside 20
    bp is inside the noise -- and it is the unit the acceptance threshold is written in, in the
    TOML of ADR-012. One basis point of vol is 0.0001 in decimal terms.

    Weighted by the same weights the loss used, so this number is the loss's own view of the fit
    rather than an unweighted recount that would disagree with what was optimised.
    """

    max_err_vol_bp: float
    """Largest single-quote error, in the same units. Non-negative, finite, and never below
    ``rmse_vol_bp``.

    Published next to the RMSE because the two fail differently. A mean absorbs one badly missed
    quote across the whole slice, so a fit can look healthy in aggregate while being visibly
    wrong exactly where someone is about to price a position. The maximum is what makes that
    visible.

    The ordering invariant is arithmetic, not a policy: a maximum below its own root-mean-square
    is impossible for any set of errors, so a violation means the producer filled the two fields
    in the wrong order -- the kind of transposition that is invisible in review and obvious to a
    constructor.
    """

    n_quotes_used: int
    """How many quotes actually entered the fit. At least one.

    Not the number in the task: a quote whose mid admits no implied volatility at all is dropped
    by the inversion (``NoImpliedVolError``), so a result can rest on fewer points than were
    offered. It is the count that says whether a low RMSE means a good fit or five parameters
    passing exactly through three points, and it is what the regularisation threshold of ADR-008
    is compared against.
    """

    converged: bool
    """Whether the optimiser stopped because it found an optimum, rather than because it ran out
    of budget.

    First half of the acceptance rule's context. A fit that exhausted its iteration budget is a
    snapshot of a search in progress, and its RMSE may be under the threshold by luck of where
    the walk happened to stop.
    """

    at_bound: bool
    """Whether any parameter finished pinned against a practical bound.

    The second half, and the reason acceptance is not simply ``rmse_vol_bp < threshold``.
    ADR-006's rule is that RMSE **and** no parameter at a bound. A pinned parameter means the
    optimiser wanted to leave the admissible region and was stopped at its edge: what came back
    is a projection onto the boundary, not a fit, and the residual it reports is the residual of
    the projection. Such a slice can show a perfectly respectable RMSE while describing a smile
    whose wing is held up by the constraint rather than by the market -- publishing it as if it
    were healthy is exactly what "an honest old surface beats a broken new one" refuses.

    A flag rather than an exception, because a rejected calibration is an ordinary outcome of a
    working system: the use case publishes ``CalibrationFailed`` and republishes the last good
    surface as ``STALE_REPUBLISH``. See ``domain/errors.py`` on why that is not an error.
    """

    def __post_init__(self) -> None:
        require_aware(self.expiry, "expiry")
        _require_positive_finite(self.tenor_years, "tenor in years")
        _require_non_negative_finite(self.rmse_vol_bp, "RMSE in vol basis points")
        _require_non_negative_finite(self.max_err_vol_bp, "maximum error in vol basis points")

        # Safe as a bare comparison only because both values are already known to be finite; a
        # NaN would have compared False here and slipped through.
        if self.max_err_vol_bp < self.rmse_vol_bp:
            raise ValueError(
                "The maximum error cannot be below the RMSE, got a maximum of "
                f"{self.max_err_vol_bp} and an RMSE of {self.rmse_vol_bp}"
            )

        if self.n_quotes_used < 1:
            raise ValueError(
                f"A slice result must use at least one quote, got {self.n_quotes_used}"
            )


@dataclass(frozen=True, slots=True)
class CalibrationResult:
    """The whole surface as it came out of one calibration, with what it cost to get there.

    Deliberately not an ``SVISurface``: this is the optimiser's report, per-slice diagnostics
    included, and it is what the acceptance rule of ADR-006 is evaluated on. Only once that rule
    passes does the ACL turn the accepted slices into the published surface. Keeping the two
    types apart is what makes "calibrated but rejected" a representable state -- collapsing them
    would leave nowhere to put a fit that exists and must not be published.
    """

    slices: tuple[SliceResult, ...]
    """The fitted slices, strictly increasing in tenor, expiries unique. Non-empty.

    Every slice that was attempted, including the ones that failed their own acceptance. The
    judgement is the use case's, and it needs to see the rejects to report them.
    """

    n_iterations: int
    """Total optimiser iterations across the surface. Non-negative.

    Zero is legal and meaningful: a warm start that lands on the optimum from the previous
    snapshot, which is the normal case in a calm market and precisely what Design 5.6 is
    designed to produce. This field must therefore never be tested for truthiness -- ``not 0`` is
    ``True`` -- and the cheapest possible cycle is the one that would be misread as a failure.
    """

    duration_ms: float
    """Wall-clock time of the calibration, in milliseconds. Non-negative and finite.

    The number the streaming budget is measured against, and half of the comparison between the
    scipy baseline and the JAX implementation -- time, convergence, lines of code -- that
    Design 5.7 exists to make. It is reported, never enforced: a slow calibration is a metric,
    not an error.
    """

    def __post_init__(self) -> None:
        _require_ordered_by_tenor(self.slices, "calibration result")
        if self.n_iterations < 0:
            raise ValueError(f"The iteration count cannot be negative, got {self.n_iterations}")
        _require_non_negative_finite(self.duration_ms, "duration in milliseconds")
