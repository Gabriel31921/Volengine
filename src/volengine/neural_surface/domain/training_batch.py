"""What one fine-tuning step is fed, in this context's own words.

Two types. ``TrainingSample`` is a single observed point of the surface, reduced to the four
numbers a network can consume plus the instant it was seen at. ``TrainingBatch`` is one gradient
step's worth of them: the fresh quotes of the snapshot that triggered the update, followed by a
draw from the replay buffer, with the identity of the snapshot that occasioned the whole thing.

**Neither is a contract, and that is deliberate.** Rule 3 forbids this layer from importing
``contracts/`` at all and rule 5 reserves every DTO for ``application/acl.py``, so there is no
``to_dict``, no ``from_dict`` and no ``schema_version`` below -- nothing here ever crosses a
process boundary. The ACL turns an incoming ``MarketSnapshot`` into samples and is the only
module in this context that knows both vocabularies. That it resembles ``QuoteData`` from a
distance is the architecture working, not duplication to be tidied away: an observation published
on a bus and a training point are different things with different invariants.

**Points, not slices.** The obvious counterpart one context over, ``parametric_pricing``'s
``SliceTask``, is three parallel tuples grouped by expiry, because SVI fits one smile at a time
and the grouping *is* the problem statement. Here the samples are objects in a flat tuple and
nothing groups them: the network sees the whole surface at once, as scattered points in
``(k, T)``, and its parameters are shared across every expiry. That single difference is the
entire modelling stance that separates the two engines -- an MLP borrows strength across tenors
by construction, where SVI has to be told to (ADR-008's regularisation towards a neighbouring
slice). Consequently there is **no ordering invariant here at all**, not on moneyness and not on
tenor. A batch is a set of points, and the replay buffer interleaves tenors on purpose, because a
gradient step over points sorted by expiry would walk the surface in one direction and arrive
biased towards wherever it finished.

For the same reason duplicate ``(k, T)`` coordinates are legal. The same strike observed at two
instants is two observations, not one repeated: they carry different timestamps, they may carry
different volatilities, and the buffer is expected to hold both. Where ``SliceTask`` refuses a
repeated ``k`` -- there it would mean one strike counted twice inside a single fit -- refusing it
here would throw away exactly the history the replay mechanism exists to accumulate.

**Volatility is stored and total variance derived, never the reverse.** The network's *output* is
total variance (Design 6.2: monotonicity in ``T`` is the calendar condition, and it is far easier
to impose on the quantity that must be monotone than on the volatility beneath it). But a quote is
*observed* as a volatility -- that is what our Black-76 inversion returns -- and the conversion
needs the tenor, which is sitting in the same object. Keeping the observation as the field and the
target as a property leaves one source of truth and makes the two impossible to contradict; storing
both would invite a sample whose variance does not match its own volatility.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime

from volengine.shared_kernel.domain.instants import require_aware


def _require_finite(value: float, what: str) -> None:
    """Reject NaN and both infinities, as a plain ``ValueError``.

    The loosest of the three guards below, for the one quantity that is legitimately negative:
    log-moneyness is zero at the forward and runs negative below it.
    """
    if not math.isfinite(value):
        raise ValueError(f"The {what} must be finite, got {value}")


def _require_positive_finite(value: float, what: str) -> None:
    """Reject anything that is not a usable positive number.

    ``isfinite`` first, and the *bad* cases joined with ``or``, because ``float("nan") <= 0`` is
    ``False``: a NaN walks straight through any ordering guard written the other way round. It is
    the mistake this codebase keeps rediscovering, and a NaN reaching a training step is worse
    than most -- one bad sample poisons every weight in the network on the first backward pass,
    and nothing downstream can tell afterwards which point did it.
    """
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"The {what} must be positive and finite, got {value}")


def _require_non_negative_finite(value: float, what: str) -> None:
    """Same guard, one step looser: zero is admissible, NaN and infinity still are not."""
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"The {what} must be non-negative and finite, got {value}")


@dataclass(frozen=True, slots=True)
class TrainingSample:
    """One observed point of the surface, as the network is given it.

    Everything convention-dependent was resolved before this object existed (ADR-002): no venue,
    no day count, no option side, no strike in currency. What is left is a coordinate, a target, a
    weight and an age -- and the age is the only field the network itself never reads, kept
    because the replay buffer's retention policy is measured on it.
    """

    log_moneyness: float
    """``k = ln(K / F)`` for this quote. Finite, and of either sign.

    Zero is at-the-money forward and entirely ordinary, which is why the guard tests finiteness
    only and why nothing in this module may ever ask whether this value is truthy: ``not 0.0`` is
    ``True``, and the most informative point on the surface would be the one silently dropped.

    Measured against the forward rather than the spot so that a sample keeps its meaning as the
    underlying moves: the wing observed at ``k = -0.3`` this minute and the one observed there ten
    minutes ago describe the same part of the smile, which is the assumption the replay buffer's
    stratification rests on.
    """

    tenor_years: float
    """Year fraction from observation to expiry, under the market's day count. Positive, finite.

    An input to the network *and* the multiplier that turns a volatility into the total variance
    it is trained against, so it is carried rather than recomputed: two day counts give different
    numbers for the same two instants, and re-deriving it here would let a sample's target drift
    away from the number the ACL used when it built it.

    Strictly positive, because an expired option is not an observation of anything: total variance
    at zero tenor is zero for every volatility, so such a point would teach the network only that
    the origin exists.
    """

    implied_vol: float
    """**Our own** Black-76 volatility for this quote, annualised as a decimal. Positive, finite.

    Inverted from the mid with our forward and our tenor, never the venue's published number
    (Design 4.5). An exchange's IV is the output of the exchange's model, forward and expiry
    convention; training on it would fit our network to someone else's model, and every convention
    mismatch would enter the surface disguised as a feature of the smile. Worse here than for SVI:
    a parameterisation with five degrees of freedom cannot absorb an arbitrary distortion, and a
    network can, so it would learn the mismatch perfectly and never show it in a residual.

    Positive rather than non-negative: a zero volatility is the degenerate limit where the option
    is worth exactly its intrinsic value, and its total variance is zero -- a target no gradient
    step can move towards without dragging the whole neighbourhood down with it.
    """

    weight: float
    """Relative influence of this point on the loss. Finite and non-negative.

    Already dimensionless when it arrives, and the loss consumes it exactly as given: it is built
    upstream from vega and the relative spread, so no learner re-derives it and the neural and
    parametric engines cannot silently weight the same market differently. Comparability between
    the two is the whole reason this project has two of them.

    **Zero is legal**, and it is how a flagged quote is kept in the batch with no influence at all:
    the point remains visible in the residuals that get reported, and in the buffer's cell
    occupancy, without steering a single weight. What that costs is a guard written as ``< 0``
    rather than as a truthiness test, here and in the aggregate check one class down.
    """

    ts_observed: datetime
    """When the quote was observed, timezone-aware.

    The field the replay buffer's maximum age is measured on and the key its eviction orders by,
    which is the only reason it survives into the domain at all -- the network never sees it. Aware
    because that measurement is a subtraction against an instant supplied by the caller, and mixing
    a naive datetime into it raises ``TypeError`` a long way from whoever built the bad value.
    Validated with the shared kernel's ``require_aware`` rather than a private copy of the same
    two lines.

    Not the snapshot instant. A batch mixes points seen minutes apart on purpose, so a per-sample
    timestamp is the only one that can say how old any given point is.
    """

    def __post_init__(self) -> None:
        _require_finite(self.log_moneyness, "log-moneyness")
        _require_positive_finite(self.tenor_years, "tenor in years")
        _require_positive_finite(self.implied_vol, "implied volatility")
        _require_non_negative_finite(self.weight, "weight")
        require_aware(self.ts_observed, "ts_observed")

    @property
    def total_variance(self) -> float:
        """``implied_vol**2 * tenor_years``: the quantity the network actually predicts.

        Derived rather than stored, so the two can never disagree. Both factors are already known
        to be positive and finite by the time anyone can call this, so the product is too -- there
        is no guard here because there is nothing left for one to catch.

        This is the target of the loss (Design 6.2). Training in variance space rather than in
        volatility space is what makes the calendar condition a monotonicity in ``T``, a property
        an architecture can be built to guarantee outright, where "volatility rises with tenor" is
        not even the right statement of the condition.
        """
        return self.implied_vol * self.implied_vol * self.tenor_years


@dataclass(frozen=True, slots=True)
class TrainingBatch:
    """One update step's data: this snapshot's quotes, then a draw from the replay buffer.

    The unit ``SurfaceLearner.update`` consumes, and the reason that port can be a pure function.
    Everything the step needs -- the points, their weights, which of them are new, and the identity
    of the snapshot that occasioned it -- is in this one immutable object, so the same batch
    replayed later produces the same weights (ADR-004), and the learner holds no state between
    calls.
    """

    market_id: str
    """Which market this is, in the engine's own naming. Non-empty.

    Carried for routing and for tagging metrics, never as an input to the mathematics: nothing
    below this line branches on it, and a learner that could read it would be an open invitation to
    special-case one venue inside a loss function.
    """

    snapshot_id: str
    """Identity of the snapshot whose arrival triggered this step. Non-empty.

    What makes a published surface traceable back to the input that produced it. It identifies the
    *fresh* part of the batch only -- the replayed points came from earlier snapshots, by
    construction -- and that asymmetry is fine, because tracing an update means knowing which
    market event caused it, not enumerating everything the buffer happened to hand over.
    """

    ts_snapshot: datetime
    """When the market data was observed, timezone-aware. **Not** when the step ran.

    The instant that travels through to the published surface, so that downstream freshness is
    measured against the market rather than against the calculation -- the same distinction ADR-006
    draws for the parametric producer, and it only holds if the instant is carried from here and
    never re-stamped along the way.

    Deliberately unrelated to the samples' own timestamps, and not constrained against them. A
    replayed point is older than the snapshot, obviously; but a fresh quote can also carry a
    ``ts_observed`` a hair after ``ts_snapshot`` when a venue clock runs ahead, and refusing that
    here would turn a clock-skew problem, which belongs to the ACL, into a domain exception.
    """

    samples: tuple[TrainingSample, ...]
    """The points to train on, fresh first. Non-empty, with weights summing above zero.

    Non-empty because a gradient step over no data is a no-op that would still be counted, timed
    and published as an update -- ``EmptyBufferError`` exists so that the caller waits for the next
    snapshot rather than constructing a placeholder.

    No ordering rule beyond the fresh/replayed split below: not sorted by moneyness, not sorted by
    tenor, and duplicate coordinates admitted. See the module docstring -- the network consumes the
    surface as a point cloud, and every ordering the tuple could be given would be one the loss
    does not use.
    """

    n_fresh: int
    """How many leading samples came from this snapshot. ``0 <= n_fresh <= len(samples)``.

    The fresh/replayed split of Design 6.4, expressed as a count rather than as two tuples so that
    the batch stays one contiguous block the learner can hand to a single forward pass, while the
    provenance stays recoverable for the metrics that care. The ordering convention --
    ``samples[:n_fresh]`` are this snapshot's -- is the whole of its meaning, and it is a
    convention, so the ACL and the buffer are the two places responsible for honouring it.

    **Zero is legal and meaningful.** It is precisely the scheduled restart: every M hours the
    network is retrained from scratch on the entire buffer, with no fresh quotes at all, and the
    before-and-after comparison is the honest drift measurement of Design 6.4. This field must
    therefore never be tested for truthiness -- ``not 0`` is ``True`` -- and the cheapest way to
    lose the restart is to write ``if batch.n_fresh:`` somewhere.

    A value equal to ``len(samples)`` is equally legal and is the cold start of a session whose
    buffer is still filling: every point is new because there is no history yet.
    """

    def __post_init__(self) -> None:
        # String emptiness, so plain truthiness is safe here -- unlike on any number in this
        # module, where `not 0.0` is True and a legitimate zero would be rejected.
        if not self.market_id:
            raise ValueError("The market id must not be empty")
        if not self.snapshot_id:
            raise ValueError("The snapshot id must not be empty")
        require_aware(self.ts_snapshot, "ts_snapshot")

        if not self.samples:
            raise ValueError("A training batch must hold at least one sample")

        # Each weight is already known to be non-negative and finite -- `TrainingSample` refused
        # anything else -- so this is genuinely an aggregate check and not a second per-element
        # one. What it catches is a batch every one of whose points was flagged: a legal set of
        # weights that is not a weighting at all, only an empty problem wearing the shape of a
        # full one. Any weighted mean in the loss divides by this sum.
        total_weight = sum(sample.weight for sample in self.samples)
        if total_weight <= 0:
            raise ValueError(
                f"The sample weights must sum to a positive number, got {total_weight}"
            )

        if self.n_fresh < 0 or self.n_fresh > len(self.samples):
            raise ValueError(
                f"The fresh sample count must lie between 0 and {len(self.samples)}, "
                f"got {self.n_fresh}"
            )

    @property
    def fresh(self) -> tuple[TrainingSample, ...]:
        """The samples observed in this snapshot: ``samples[:n_fresh]``.

        Possibly empty, and that is the scheduled restart rather than an anomaly. Reported on
        separately from the rest because the residual against the *fresh* points is the honest
        answer to "does the model fit the market as it is now", while a residual over the whole
        batch is diluted by however much history the buffer happened to contribute.
        """
        return self.samples[: self.n_fresh]

    @property
    def replayed(self) -> tuple[TrainingSample, ...]:
        """The samples drawn from the replay buffer: ``samples[n_fresh:]``.

        The complement of :attr:`fresh` by construction -- the two partition ``samples`` exactly,
        with no gap and no overlap, because they are two halves of one slice at one index. They
        exist to mitigate catastrophic forgetting (Design 6.4): a step taught only by this
        snapshot's quotes, which cluster near the money and at the front expiry, would improve
        there and quietly unlearn the wings.
        """
        return self.samples[self.n_fresh :]
