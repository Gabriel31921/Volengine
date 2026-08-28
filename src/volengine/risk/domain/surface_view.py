"""The volatility surface as Risk holds it: two axes, a grid of total variance, and a calendar.

This is the only surface this context knows. Rule 3 forbids ``risk/domain/`` from importing
``contracts/`` at all, so the published ``CalibratedSurface`` never reaches this layer:
``application/acl.py`` receives it, converts it once, and hands the domain this object instead.
The awkward-looking consequence is the useful one -- every rule below can be tested against a
surface written by hand in a builder, with no bus, no calibrator and no producer anywhere in
sight, and the published vocabulary can be renamed tomorrow without a line of business logic
moving.

**Total variance is stored, not volatilities, and the conversion happens once in the ACL.** The
interpolation this whole context rests on is bilinear in total variance (Design 7.1), so keeping
``w = vol**2 * T`` at the nodes makes the hot path pure interpolation: a valuation reads four
numbers and multiplies. Storing vols instead would mean converting on every single lookup, twice
per greek bump, and would leave two representations of the same surface in memory with nothing
guaranteeing they agree. One conversion, at the boundary, leaves one source of truth; a
volatility is recovered where it is actually needed, by ``interpolation.implied_vol``.

**There is no status field, and the absence is an architectural point rather than an omission.**
The contract publishes ``SurfaceStatus``, including ``STALE_REPUBLISH`` for the case ADR-006
describes -- this cycle's calibration failed, so the previous surface goes out again. But ADR-006
republishes it with its **original** ``ts_snapshot``, precisely so that staleness is measured
against the market data rather than against the calculation. The one timestamp the freshness
policy of Design 7.2 already reads therefore says everything a status flag would have said, and
says it as a number that can be compared instead of a label that has to be interpreted. A fourth
spelling of a status enum in this context would add a second, redundant channel for the same
fact, and the first thing a second channel does is disagree with the first.

**The expiries and the forwards are what make a real position placeable.** A position is an
option with a contractual expiry on a calendar and a strike in currency; a surface is a function
of a year fraction and a log-moneyness. Turning the first into the second needs a daycount and a
forward, and ADR-002 forbids this context from owning either: ``ts_snapshot + tenor * 365 days``
is a daycount assumption wearing the clothes of arithmetic, right under ACT/365F and days out
under a business-day count, with nothing in the data to say which market it was. So the grid
states the answer at its own nodes -- an expiry instant and a forward price per tenor -- and
:meth:`SurfaceView.tenor_of` and :meth:`SurfaceView.forward_at` interpolate between them. Risk
recovers a tenor and a forward without ever learning which convention produced them.

What this module deliberately does **not** do is evaluate the surface. It resolves coordinates:
where on the grid does this position sit. What the grid says once you are there is
``interpolation.py``, next door, because that is a numerical method with limitations of its own
that have to be documented where they bite -- and keeping it out of the value object is what
allows a bicubic or a re-fitted slice to replace it later without touching the type every other
module in this context holds.
"""

from __future__ import annotations

import math
from bisect import bisect_right
from dataclasses import dataclass
from datetime import datetime
from itertools import pairwise

from volengine.risk.domain.errors import ExpiredPositionError
from volengine.shared_kernel.domain.instants import require_aware


def _require_positive_finite(value: float, what: str) -> None:
    """Reject anything that is not a usable positive number, as a plain ``ValueError``.

    Finiteness first, and the bad cases joined with ``or``, because ``float("nan") <= 0`` is
    ``False``: written the other way round a NaN walks straight through the ordering test. Here it
    would end up as a tenor the interpolation divides by, and a volatility of NaN then passes every
    downstream comparison as though it were healthy.
    """
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"The {what} must be positive and finite, got {value}")


@dataclass(frozen=True, slots=True)
class SurfaceView:
    """One calibrated surface, in the only form Risk ever sees it.

    A snapshot of a producer's answer, not a live object: nothing here calls back into a
    calibrator, and two views of the same market from two producers are independent values that
    can be held side by side and valued against the same portfolio. That is what the comparative
    report of Design 7.3 is made of, and it only works because this type carries the numbers
    rather than a reference to whoever computed them.

    The identity fields travel with the numbers for the same reason: a risk report has to be able
    to say *whose* surface it valued and *when* that surface saw the market, and a view that had
    to be paired with its provenance by the caller would eventually be paired with the wrong one.
    """

    market_id: str
    """The market this surface describes, in the engine's own naming. Non-empty.

    Carried for routing and for the report header, never as an input to any calculation below --
    nothing in this module or the next branches on it. It is deliberately **not** cross-checked
    against a portfolio's ``underlying``: the contract publishes a market identifier and no
    underlying, so the domain has nothing to compare them with, and pairing a book with the right
    market stays the use case's job.
    """

    producer_id: str
    """Which calibrator produced it: ``"svi-scipy"``, ``"svi-jax"``, ``"neural-torch"``. Non-empty.

    For attribution, for the metric tags, and for the comparative report -- and for nothing else.
    The moment any rule in this context reads it, the producers stop being interchangeable and the
    comparison the project exists to run stops measuring the calibrators.
    """

    surface_id: str
    """Identity of the calibration result this view was built from. Non-empty.

    What makes a reported number traceable back to the exact fit that produced it. The bus
    conflates and drops by design (ADR-003), so a report cannot be reconstructed from arrival
    order; it can be reconstructed from this.
    """

    ts_snapshot: datetime
    """Exchange instant of the market data behind the surface. Timezone-aware.

    **Not** when the fit ran, and the distinction is the whole of ADR-006: a republished surface
    carries a recent calibration time and an old snapshot time, and it is this one that says how
    stale the numbers really are. It is therefore the single input to the freshness policy of
    Design 7.2, and the origin every tenor below is measured from.

    Aware because staleness and tenors are both subtractions of datetimes, and mixing a naive one
    into that arithmetic raises ``TypeError`` far from whoever built the bad value.
    """

    log_moneyness: tuple[float, ...]
    """Moneyness axis, ``k = ln(K / F)``. Finite, strictly increasing, at least one node.

    Measured against the forward of each tenor, which is why :attr:`forwards` has to travel with
    it: an axis in ``k`` with no forward is a coordinate system with no origin, and a position
    holding a strike could not be placed on it.

    ``0.0`` is at-the-money forward and a perfectly ordinary value on this axis, so nothing may
    ever test these entries for truthiness -- ``not 0.0`` is ``True``.

    A single node is legal. It is a degenerate surface with no smile, which is exactly what a
    market quoting one strike per expiry produces, and the interpolation next door handles it by
    returning that column everywhere.
    """

    tenors: tuple[float, ...]
    """Tenor axis in years. Finite, strictly positive, strictly increasing, at least one node.

    Already resolved under the producing market's daycount (ADR-002); nothing here recomputes it
    from dates, and :meth:`tenor_of` exists precisely so that nothing has to.

    Strictly positive rather than non-negative: a tenor of zero is an expired option, which has no
    total variance to hold and is refused by :meth:`tenor_of` rather than represented here.
    """

    expiries: tuple[datetime, ...]
    """The expiry instant each tenor node stands for. Aware, one per tenor, strictly increasing,
    and all after :attr:`ts_snapshot`.

    Carried rather than derived, because deriving it is exactly what ADR-002 forbids downstream.
    These pairs are the grid's own statement of its daycount, at the only points where it can be
    stated exactly, and interpolating between them is how Risk answers "what year fraction is this
    contract's expiry" without ever naming a convention.

    The first one has to sit strictly after the snapshot for the axis to mean anything: a node at
    or before the instant the surface describes would be an expiry with negative time to run, and
    the extrapolation :meth:`tenor_of` performs below the first node -- from ``ts_snapshot`` at
    tenor zero -- would have no interval to work in. Ordering with no duplicates for the ordinary
    reason: two nodes at one instant are two answers to one question, and every lookup below
    assumes the order.
    """

    forwards: tuple[float, ...]
    """Forward price at each tenor node, in the currency the strikes are quoted in. Positive,
    finite, one per tenor.

    The other half of what makes a position placeable, and the number :attr:`log_moneyness` was
    measured against. A strike becomes ``k = ln(K / F(T))``, so a view without these could report
    a volatility at a moneyness nobody asked for.

    Recovering them downstream from a spot and a rate was never an option: the forward that makes
    the inversion consistent is the one the smile was fitted with, and any later reconstruction
    would pair today's smile with a forward that has since moved.
    """

    total_variance: tuple[tuple[float, ...], ...]
    """``w = vol**2 * T`` at every node, indexed ``[tenor][moneyness]``. Positive, finite,
    rectangular.

    One row per tenor, one entry per moneyness node. Rectangular by construction, because a ragged
    grid makes a bilinear lookup ambiguous exactly where the surface is thinnest.

    Strictly positive, which is what lets ``interpolation.implied_vol`` take a square root and
    return a usable volatility at every point of the grid *and* at every point between them: a
    convex combination of positive numbers is positive, so positivity survives the interpolation
    without a second guard.

    Deliberately **not** checked for convexity or for calendar monotonicity. No-arbitrage is a
    model criterion judged by the producers -- as a soft penalty in the loss and a hard gate before
    publishing (ADR-010) -- and a consumer that re-litigated it would be second-guessing a decision
    it has none of the evidence for. This type guarantees structural coherence and nothing more.
    """

    def __post_init__(self) -> None:
        # String emptiness, so plain truthiness is safe here -- and only here. On any number in
        # this module `not 0.0` is True and a legitimate at-the-money node would be rejected.
        for name, value in (
            ("market id", self.market_id),
            ("producer id", self.producer_id),
            ("surface id", self.surface_id),
        ):
            if not value:
                raise ValueError(f"The {name} must not be empty")
        require_aware(self.ts_snapshot, "ts_snapshot")

        if not self.log_moneyness:
            raise ValueError("The surface must have at least one moneyness node")
        if not self.tenors:
            raise ValueError("The surface must have at least one tenor")

        for index, k in enumerate(self.log_moneyness):
            if not math.isfinite(k):
                raise ValueError(f"The log-moneyness at index {index} must be finite, got {k}")
        # `pairwise`, never `zip(xs, xs[1:], strict=True)`: consecutive pairs differ in length by
        # one by design, so the strict flag this repo requires elsewhere would be wrong here.
        for near, far in pairwise(self.log_moneyness):
            if near >= far:
                raise ValueError(
                    f"The moneyness axis must be strictly increasing, got {near} before {far}"
                )

        for index, tenor in enumerate(self.tenors):
            _require_positive_finite(tenor, f"tenor at index {index}")
        for near_tenor, far_tenor in pairwise(self.tenors):
            if near_tenor >= far_tenor:
                raise ValueError(
                    "The tenor axis must be strictly increasing, got "
                    f"{near_tenor} before {far_tenor}"
                )

        if len(self.expiries) != len(self.tenors):
            raise ValueError(
                "There must be one expiry per tenor, got "
                f"{len(self.expiries)} expiries for {len(self.tenors)} tenors"
            )
        for index, expiry in enumerate(self.expiries):
            require_aware(expiry, f"expiry at index {index}")
        for near_expiry, far_expiry in pairwise(self.expiries):
            if near_expiry >= far_expiry:
                raise ValueError(
                    "The expiries must be strictly increasing, got "
                    f"{near_expiry} before {far_expiry}"
                )
        if self.expiries[0] <= self.ts_snapshot:
            raise ValueError(
                "The first expiry must be strictly after the snapshot instant, got an expiry of "
                f"{self.expiries[0]} against a snapshot at {self.ts_snapshot}"
            )

        if len(self.forwards) != len(self.tenors):
            raise ValueError(
                "There must be one forward per tenor, got "
                f"{len(self.forwards)} forwards for {len(self.tenors)} tenors"
            )
        for index, forward in enumerate(self.forwards):
            _require_positive_finite(forward, f"forward at index {index}")

        if len(self.total_variance) != len(self.tenors):
            raise ValueError(
                "There must be one row of total variance per tenor, got "
                f"{len(self.total_variance)} rows for {len(self.tenors)} tenors"
            )
        for row_index, row in enumerate(self.total_variance):
            if len(row) != len(self.log_moneyness):
                raise ValueError(
                    "Every row must hold one total variance per moneyness node, got "
                    f"{len(row)} values for {len(self.log_moneyness)} nodes at tenor "
                    f"{self.tenors[row_index]}"
                )
            for column_index, variance in enumerate(row):
                _require_positive_finite(
                    variance,
                    f"total variance at tenor {self.tenors[row_index]} and moneyness "
                    f"{self.log_moneyness[column_index]}",
                )

    def tenor_of(self, expiry: datetime) -> float:
        """The year fraction this expiry corresponds to, read off the grid's own calendar.

        **This is the payoff of carrying the expiries.** The grid states, at each of its nodes,
        that a particular instant is a particular year fraction -- and that statement already
        contains the producing market's daycount, whatever it is. Interpolating linearly in
        calendar time between two of those pairs therefore recovers a tenor for any expiry in
        between without this context ever learning whether the number came from ACT/365F, ACT/360
        or a 252-business-day count. That is ADR-002 honoured rather than merely obeyed: the
        convention is not hidden from Risk, it is genuinely never needed here.

        Linear in calendar time, specifically, and the approximation is worth naming. Under any
        daycount that is itself linear in calendar time -- ACT/365F and ACT/360 both are -- the
        interpolation is *exact*, not approximate, and it reproduces the nodes to the last bit. A
        business-day count is a step function instead, and between two nodes this returns a
        smoothed version of it; the error is bounded by the tenor of the weekend it smooths over,
        which is far below the interpolation error the vol grid already carries.

        Below the first node the same line is extended down to ``ts_snapshot``, which is tenor
        zero by definition -- an option expiring at the instant the surface describes has no time
        left. Extrapolating there rather than clamping to the first tenor is the only reading that
        keeps a front-week option priceable at all: a market whose nearest grid node is a month out
        still has options expiring on Friday, and clamping would value them as month-old ones.

        Beyond the last node the result is clamped to the last tenor. That is deliberately *not* an
        error, and it pairs with the flat extrapolation of ``interpolation.bilinear_total_variance``
        one module over: a long-dated position is reported with a documented weakness rather than
        dropped from the book. See ``errors.py`` on why only the expired case has nothing behind
        it.

        Args:
            expiry: The position's contractual expiry instant. Timezone-aware.

        Returns:
            The year fraction from ``ts_snapshot`` to ``expiry``, strictly positive.

        Raises:
            ValueError: If ``expiry`` is naive. Comparing it against the grid's aware nodes would
                otherwise raise ``TypeError`` from inside ``bisect``, three frames away from the
                caller that built it.
            ExpiredPositionError: If ``expiry`` is at or before ``ts_snapshot``. There is no
                honest number to return -- not zero, which would be a tenor the interpolation
                divides by, and not the intrinsic value, which is a settlement question rather
                than a valuation one. It is an ordinary runtime condition rather than a bug: a
                portfolio file is configuration and outlives its positions, so the first run after
                a roll will name an expiry that has passed.
        """
        require_aware(expiry, "expiry")
        if expiry <= self.ts_snapshot:
            raise ExpiredPositionError(
                f"The position expired at {expiry}, at or before the surface's snapshot instant "
                f"{self.ts_snapshot}, so it has no tenor on this grid"
            )

        # `bisect_right` on the expiries: the axis is already known to be strictly increasing, and
        # the alternative -- a linear scan -- would be the same answer written less clearly.
        index = bisect_right(self.expiries, expiry)
        if index == 0:
            # Between the snapshot (tenor zero) and the first node. The denominator cannot vanish:
            # the constructor requires `expiries[0] > ts_snapshot`.
            span = (self.expiries[0] - self.ts_snapshot).total_seconds()
            elapsed = (expiry - self.ts_snapshot).total_seconds()
            return self.tenors[0] * (elapsed / span)
        if index == len(self.expiries):
            return self.tenors[-1]

        near = index - 1
        span = (self.expiries[index] - self.expiries[near]).total_seconds()
        elapsed = (expiry - self.expiries[near]).total_seconds()
        # Written as `base + slope * weight` rather than as a weighted average of the two nodes so
        # that an expiry landing exactly on a node returns that node's tenor bit for bit: the
        # second term is exactly `0.0` there, and a tenor that drifted by an ulp would make the
        # single most common lookup in this context -- a position on a listed expiry -- inexact.
        return self.tenors[near] + (self.tenors[index] - self.tenors[near]) * (elapsed / span)

    def forward_at(self, tenor_years: float) -> float:
        """The forward for this tenor, interpolated at constant carry between the grid's nodes.

        **Linear in ``ln(F)``, not in ``F``, and the choice is a modelling statement rather than a
        numerical convenience.** A forward is ``F(T) = S * exp(r * T)`` for whatever carry ``r``
        the market is pricing -- funding, borrow and, in crypto, the perpetual basis -- so the
        logarithm is the quantity that is linear whenever the carry is constant. Interpolating
        between two nodes in log space therefore assumes exactly one thing: that the carry does not
        change between them, which is the mildest assumption available and the one a term structure
        of forwards is usually quoted under. Interpolating ``F`` itself would instead imply a carry
        that drifts continuously between every pair of nodes, in a direction nobody chose, and the
        implied instantaneous rate would jump at each node.

        Outside the tenor range the result is clamped flat, matching the flat extrapolation of the
        total variance next door. Extending the last segment's carry instead would compound a
        two-node slope out to arbitrary maturities, which turns a mild local assumption into an
        aggressive global one exactly where there is no data to check it against.

        A single-tenor grid returns its only forward everywhere, which is the honest answer: one
        node states a level and says nothing whatsoever about a carry.

        Args:
            tenor_years: Year fraction, strictly positive and finite. Normally the output of
                :meth:`tenor_of`, which cannot produce anything else.

        Returns:
            The forward price, strictly positive. Exact at the nodes.

        Raises:
            ValueError: If ``tenor_years`` is not positive and finite.
        """
        _require_positive_finite(tenor_years, "tenor in years")

        if tenor_years <= self.tenors[0]:
            return self.forwards[0]
        if tenor_years >= self.tenors[-1]:
            return self.forwards[-1]

        index = bisect_right(self.tenors, tenor_years)
        near = index - 1
        # The carry stated explicitly, because it is the assumption: `F(t) = F_near * exp(r * dt)`
        # with `r` the constant continuously-compounded rate that connects the two nodes. At
        # `tenor_years == tenors[near]` the exponent is exactly zero and `exp(0.0)` is exactly
        # `1.0`, so a lookup on a node returns that node's forward unchanged.
        carry = math.log(self.forwards[index] / self.forwards[near]) / (
            self.tenors[index] - self.tenors[near]
        )
        return self.forwards[near] * math.exp(carry * (tenor_years - self.tenors[near]))
