"""The memory of the network: which corners of the surface it is not allowed to forget.

A market ticks where the money is. Over any ten minutes of a live session the at-the-money
strikes of the front expiry produce hundreds of updates while a six-month 25-delta put produces
three, and a fine-tuning step over "whatever arrived since the last snapshot" is therefore a step
over the at-the-money strikes of the front expiry. Repeat that a few hundred times and the
network fits the region it keeps seeing and quietly forgets the rest -- the wings and the long
tenors, which is to say precisely the region nobody can price by eye and everybody needs a model
for. That is catastrophic forgetting, and Design 6.4 names it as the risk this module answers.

The answer is a **stratified** buffer. The ``(k, T)`` plane is cut into cells, every cell holds a
bounded number of recent observations, and a draw goes round the cells rather than round the
samples. A cell holding three wing quotes then contributes as much to a batch as a cell holding
three hundred at-the-money ticks. The stratification is not an optimisation and not a detail of
how PyTorch is fed: it is a **coverage policy**, a statement about what the model owes the parts
of the surface that trade thinly, and that is why it is domain state rather than something in an
adapter. Nothing here knows what a tensor is.

It is the one mutable object in this context, for the same reason ``QuoteChain`` is the one
mutable object in Market Data: everything else describes an instant, and this describes something
that accumulates across instants. Like the chain it is not thread-safe and does not pretend to
be -- one task owns one buffer, and concurrency between markets is concurrency between buffers.

Two things it deliberately does not do. It never reads a clock: ``prune`` is handed the instant,
because ADR-004's deterministic replay only works while every reading of time enters through a
port, and a buffer that called ``datetime.now`` would make a recorded session unreproducible from
the inside. And it never builds a ``TrainingBatch``. Assembling one -- this snapshot's fresh
quotes first, a draw from here after, with the ``n_fresh`` split that says which is which -- is
the use case's job in F1-06, because only the use case knows both halves. The buffer answers one
question, "what have we seen that is worth remembering", and stops there.
"""

from __future__ import annotations

import math
from bisect import bisect_right
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import datetime
from itertools import pairwise

import numpy as np

from volengine.neural_surface.domain.errors import EmptyBufferError
from volengine.neural_surface.domain.training_batch import TrainingSample
from volengine.shared_kernel.domain.instants import require_aware

CellIndex = tuple[int, int]
"""``(moneyness_index, tenor_index)``, both counted from zero along their own edge tuple.

Named because it is the key of two public mappings and the unit the whole coverage metric is
counted in; an unnamed ``tuple[int, int]`` in four signatures says nothing about which int is
which.
"""


def _require_edges(edges: tuple[float, ...], what: str, *, positive: bool) -> None:
    """A usable axis: at least one cell, every edge a real number, strictly increasing.

    Two edges is the minimum because ``n`` edges describe ``n - 1`` cells, and an axis with no
    cell has nothing to stratify. Strictly increasing rather than merely non-decreasing: two equal
    edges would describe an empty interval, a cell no sample can ever land in, which would drag
    ``cell_coverage`` permanently below one and make the metric of Design 6.5 unreadable.

    Finiteness is tested before the sign, and the bad cases are joined with ``or``, because
    ``float("nan") <= 0`` is ``False`` and a NaN edge walks straight through an ordering guard
    written the other way round -- and then through ``bisect_right``, which would place samples
    into cells at random.
    """
    if len(edges) < 2:
        raise ValueError(f"The {what} need at least two edges to bound one cell, got {len(edges)}")

    for index, edge in enumerate(edges):
        if not math.isfinite(edge):
            raise ValueError(f"The {what} must be finite, got {edge} at index {index}")
        # Safe as a bare comparison only because this edge is already known to be finite.
        if positive and edge <= 0:
            raise ValueError(f"The {what} must be positive, got {edge} at index {index}")

    # `pairwise`, never `zip(xs, xs[1:], strict=True)`: consecutive pairs differ in length by one
    # by design, so the strict flag this codebase requires everywhere else would be wrong here.
    for lower, upper in pairwise(edges):
        if lower >= upper:
            raise ValueError(f"The {what} must be strictly increasing, got {lower} before {upper}")


@dataclass(frozen=True, slots=True)
class StratificationSpec:
    """How the ``(k, T)`` plane is cut up, how much each cell remembers, and for how long.

    The whole coverage policy in one immutable value, separated from the buffer that enforces it
    so that it can be built from TOML at the composition root (ADR-012) and compared, logged or
    replayed on its own. Every number below is a modelling judgement about a particular market --
    where the wings begin, how far out the curve is quoted, how long a quote stays informative --
    and none of them belongs hard-coded next to the mutation logic.
    """

    moneyness_edges: tuple[float, ...]
    """Cell boundaries along ``k = ln(K / F)``. At least two, finite, strictly increasing.

    Any sign, and no requirement to straddle zero: the axis is a partition of the log-moneyness
    line, not a description of where the forward is. The outermost pair is not a range check --
    see :meth:`cell_of` -- so choosing them narrowly costs resolution in the wings, never data.
    """

    tenor_edges: tuple[float, ...]
    """Cell boundaries along the tenor in years. At least two, positive, finite, increasing.

    Positive because a ``TrainingSample`` cannot carry a non-positive tenor at all, so a
    non-positive edge would bound a cell nothing can ever reach. Uneven spacing is expected and
    healthy: a week and a month differ far more in behaviour than six months and seven do.
    """

    capacity_per_cell: int
    """How many observations one cell holds before it starts evicting. At least one.

    Per cell, never global, and that is the entire mechanism. A global cap on a buffer fed by a
    live chain is a cap that the at-the-money strikes win, because they arrive faster; the wing
    quotes would be evicted by sheer volume within minutes of the session opening. Bounding each
    cell separately makes the memory of a thinly traded corner independent of how loudly the rest
    of the surface is trading.

    It also bounds the whole object: at most ``capacity_per_cell * n_cells`` samples are ever
    held, for a process that is expected to run for hours.
    """

    max_age_seconds: float
    """How long an observation stays admissible. Positive and finite.

    An implied volatility from an hour ago is not a fact about the market now, and replaying it
    against today's quotes teaches the network to average two different regimes. This is the
    second bound on the buffer, orthogonal to the capacity one: capacity limits how much a busy
    cell remembers, age limits how long a quiet cell does. Enforced only when ``prune`` is called,
    because the domain has no clock of its own to notice the passing of time.
    """

    def __post_init__(self) -> None:
        _require_edges(self.moneyness_edges, "moneyness edges", positive=False)
        _require_edges(self.tenor_edges, "tenor edges", positive=True)

        if self.capacity_per_cell < 1:
            raise ValueError(
                f"The capacity per cell must be at least one, got {self.capacity_per_cell}"
            )
        if not math.isfinite(self.max_age_seconds) or self.max_age_seconds <= 0:
            raise ValueError(
                f"The maximum age must be positive and finite, got {self.max_age_seconds}"
            )

    @property
    def n_cells(self) -> int:
        """How many cells the two axes describe together. The denominator of the coverage metric.

        ``(len(moneyness_edges) - 1) * (len(tenor_edges) - 1)``: every moneyness band crossed with
        every tenor band. Computed rather than stored, so it cannot disagree with the edges.
        """
        return (len(self.moneyness_edges) - 1) * (len(self.tenor_edges) - 1)

    def cell_of(self, sample: TrainingSample) -> CellIndex:
        """Which cell a sample belongs to, **clamping** rather than rejecting the outliers.

        A sample beyond the outermost edge lands in the outermost cell, on both axes and in both
        directions, and this method never raises. The temptation is to treat the edges as a range
        check and drop what falls outside, and it would be exactly backwards: an observation
        further out than the configured wing is the rarest and most informative thing the buffer
        ever sees, the region the stratification was built to protect, and discarding it would
        make the buffer emptiest precisely where the model is weakest. Clamping costs resolution
        at the extremes -- everything beyond the last edge is remembered as one band -- and that
        is the right trade against remembering none of it.

        The intervals are half-open and closed at the top: a sample sitting exactly on an interior
        edge falls into the cell **above** it, the one that edge is the lower bound of, and a
        sample sitting exactly on the outermost upper edge falls into the last cell rather than
        off the axis. That convention comes free from ``bisect_right`` and the clamp, and it is
        stated here because "which side of the boundary" is the kind of question a test has to be
        able to answer from the docstring rather than from the arithmetic.
        """
        return (
            _band_of(self.moneyness_edges, sample.log_moneyness),
            _band_of(self.tenor_edges, sample.tenor_years),
        )


def _band_of(edges: tuple[float, ...], value: float) -> int:
    """Index of the band ``value`` falls in, clamped into ``[0, len(edges) - 2]``.

    ``bisect_right`` gives the number of edges at or below the value, so subtracting one turns it
    into a band index that is already correct for everything strictly inside the axis; the clamp
    is what folds the two open ends onto the outermost bands.
    """
    return max(0, min(len(edges) - 2, bisect_right(edges, value) - 1))


class ReplayBuffer:
    """Recent observations, held per cell, and the rules for remembering and recalling them.

    The aggregate root of this context's state. Every mutation goes through it -- nothing reaches
    inside the cells -- which is what makes "no cell exceeds its capacity, and nothing older than
    the last prune's horizon is held" a sentence that is true at every instant rather than a hope.

    The three policies it enforces are deliberately independent. Capacity is enforced on the way
    in, per cell, so a busy cell can only ever evict itself. Age is enforced when ``prune`` is
    handed an instant, because the domain has no clock. And balance is enforced on the way out, by
    a draw that goes round the cells instead of round the samples. None of the three can be
    satisfied by tuning another, which is why each is stated separately.
    """

    def __init__(self, spec: StratificationSpec) -> None:
        self._spec = spec
        self._cells: dict[CellIndex, list[TrainingSample]] = {}
        self._last_prune: datetime | None = None

    # --- writes

    def add(self, sample: TrainingSample) -> None:
        """Remember one observation, evicting the oldest of **its own cell** if that cell is full.

        Eviction is local by construction: the cell is chosen from the sample's coordinates and
        nothing outside it is ever touched. That is the whole point of stratifying. Under a global
        eviction rule a wing quote would be discarded by the arrival of unrelated at-the-money
        ticks -- the busiest region of the surface would decide what the quietest one is allowed
        to remember -- and the buffer would degenerate into the recency-biased sample it exists to
        replace.

        "Oldest" is by ``ts_observed``, not by arrival order, because the two genuinely differ:
        a chain can republish a stale quote, and a reconnection can deliver a backlog out of
        order. Among equal stamps the earliest inserted goes first, which keeps the method
        deterministic for a replay.

        The arriving sample always enters, even when it is older than everything already in the
        cell. Refusing it would need a second policy -- some notion of "too late to matter" that
        is really the age horizon in disguise -- and would make ``add`` a method whose effect the
        caller cannot predict. What it displaces is the oldest of what was already held; ``prune``
        is where staleness is judged, all in one place, against one instant.
        """
        cell = self._spec.cell_of(sample)
        held = self._cells.setdefault(cell, [])
        if len(held) >= self._spec.capacity_per_cell:
            oldest = min(range(len(held)), key=lambda index: held[index].ts_observed)
            del held[oldest]
        held.append(sample)

    def extend(self, samples: Iterable[TrainingSample]) -> None:
        """Remember a whole snapshot's worth of observations, in the order given.

        Not merely a convenience: the order matters when a snapshot delivers more samples for one
        cell than that cell can hold, and doing it one ``add`` at a time is what makes the
        outcome the same as if they had arrived one tick at a time. There is no batched fast path
        precisely because a faster one would have to answer the eviction question differently.
        """
        for sample in samples:
            self.add(sample)

    def prune(self, now: datetime) -> int:
        """Forget everything older than the age horizon, as of ``now``. Returns how many went.

        The instant arrives as an argument and the buffer never reads a clock, the same split as
        ``QuoteChain.apply`` and ``QuoteChain.snapshot``: ADR-004 puts time behind a port so that a
        recorded session replays identically, and a single ``datetime.now`` inside a domain object
        is enough to break that for the whole pipeline. It is what lets a test state an age in
        seconds and assert on a count.

        Strictly older: a sample whose age is exactly ``max_age_seconds`` is kept. The boundary has
        to fall somewhere and this is the side that makes "nothing older than the horizon is held"
        literally true.

        The count is returned rather than logged because the domain has no logger, and because it
        is a real signal: a prune that suddenly drops most of the buffer means the feed stopped
        without anyone noticing, which is not visible from the cell coverage alone.

        **A sample stamped after ``now`` is kept, not refused**, and it simply has a negative age
        until the wall clock passes it. That is not indulgence, it is the only reading that
        survives contact with a real venue: ``ChainSnapshot.ts_exchange`` is deliberately not
        clamped to ``ts_local``, so an exchange whose clock runs a second ahead delivers quotes
        from the future as a matter of routine, and a buffer that raised on them would take the
        training loop down for as long as the skew lasted. Reconciling the two clocks is the ACL's
        job; ageing what it hands over is this method's.

        Raises:
            ValueError: If ``now`` is naive, or if it precedes the instant of the previous prune.
                That comparison, and not one against the samples, is what a clock running
                backwards actually looks like: a bug in the composition root rather than a market
                condition, and one worth catching, because the ages it produces are negative,
                every sample looks fresh forever and the horizon silently stops being enforced.
                The check happens **before** a single sample is dropped, so a rejected prune
                leaves the buffer exactly as it was and does not move the reference instant.
        """
        require_aware(now, "now")

        if self._last_prune is not None and now < self._last_prune:
            raise ValueError(
                f"the clock ran backwards: pruning at {now} after a prune at {self._last_prune}"
            )
        self._last_prune = now

        horizon = self._spec.max_age_seconds
        dropped = 0
        for cell, held in list(self._cells.items()):
            kept = [
                sample for sample in held if (now - sample.ts_observed).total_seconds() <= horizon
            ]
            dropped += len(held) - len(kept)
            if kept:
                self._cells[cell] = kept
            else:
                # Emptied cells are removed rather than left behind, so that `occupancy` and
                # `cell_coverage` never have to distinguish "empty" from "absent".
                del self._cells[cell]
        return dropped

    # --- reads

    def sample(self, size: int, rng: np.random.Generator) -> tuple[TrainingSample, ...]:
        """Draw up to ``size`` observations, round-robin across the non-empty cells.

        This is the method the class exists for. The cells are put in a shuffled order and then
        walked in that order, one sample from each, and only once every non-empty cell has given
        one does any cell give a second. A cell holding three wing quotes therefore contributes as
        many points to the batch as a cell holding three hundred at-the-money ticks, until the
        thin cell runs out. A uniform draw over the same buffer would return the busy cell's
        contents almost exclusively -- which is the recency-and-liquidity bias of the raw feed,
        reproduced faithfully inside the very structure meant to correct it.

        The order is shuffled, and the members within each cell too, so that repeated draws do not
        keep handing the network the same points in the same sequence: without it the first sample
        of every cell would be its oldest, every batch, and the freshest observations in a busy
        cell would never be replayed at all.

        No held sample is returned twice within one draw -- each is taken from its cell and not put
        back -- so the result is a subset of what is held, never a resampling of it. Fewer than
        ``size`` items come back only when the buffer holds fewer than ``size`` in total.

        ``rng`` is injected rather than created here, and that is not ceremony:
        ``np.random.default_rng()`` inside this method would make two runs over the same recorded
        session produce different batches, hence different weights, hence a replay that cannot
        reproduce a published surface. ADR-004 forbids exactly that, and the injection is what
        makes the whole draw a pure function of ``(buffer state, size, rng state)``.

        Raises:
            ValueError: If ``size`` is below one. Asking for no samples is a caller bug -- an empty
                batch is not a training step, and ``TrainingBatch`` would refuse it anyway.
            EmptyBufferError: If nothing is held. Not a bug: the buffer is empty for exactly as
                long as the first snapshot takes to arrive, and a restart scheduled inside that
                window asks a reasonable question with no honest answer. The caller waits.
        """
        if size < 1:
            raise ValueError(f"A draw must ask for at least one sample, got {size}")
        if not self._cells:
            raise EmptyBufferError("The replay buffer holds no samples yet")

        cells = list(self._cells)
        order = [cells[int(position)] for position in rng.permutation(len(cells))]

        pools: dict[CellIndex, list[TrainingSample]] = {}
        for cell in order:
            held = self._cells[cell]
            pools[cell] = [held[int(position)] for position in rng.permutation(len(held))]

        drawn: list[TrainingSample] = []
        for depth in range(max(len(pool) for pool in pools.values())):
            for cell in order:
                pool = pools[cell]
                if depth < len(pool):
                    drawn.append(pool[depth])
                    if len(drawn) == size:
                        return tuple(drawn)
        return tuple(drawn)

    def snapshot(self) -> tuple[TrainingSample, ...]:
        """Everything held, in a stable order. What the scheduled restart of Design 6.4 trains on.

        A restart every M hours retrains from scratch on the whole buffer, and comparing the
        surface before with the surface after is the honest measure of how far continuous
        fine-tuning has drifted. That comparison is only meaningful if the training set is
        exactly reproducible, so the order here is by cell index and then by position within the
        cell -- a function of the contents alone, not of the sequence of adds and prunes that
        produced them.

        No draw, no shuffling, no age filter: it returns what is held. Whoever wants a horizon
        applied calls ``prune`` first, with the instant they mean.
        """
        return tuple(sample for cell in sorted(self._cells) for sample in self._cells[cell])

    def occupancy(self) -> Mapping[CellIndex, int]:
        """How many samples each **non-empty** cell holds, keyed ``(moneyness, tenor)``.

        Only the non-empty ones, because the empty cells are the complement and listing them would
        grow quadratically with a refinement of the edges while saying nothing new. A fresh
        dictionary rather than a view of the internals: a caller holding a live view would watch
        it change under the next ``add``, which is the same reason ``ChainSnapshot`` is
        disconnected from the chain that produced it.

        The diagnostic behind ``cell_coverage``: the fraction says how much of the surface is
        remembered at all, this says whether what is remembered is level or piled into one corner.
        """
        return {cell: len(held) for cell, held in self._cells.items()}

    def __len__(self) -> int:
        """Total samples held across every cell. Never above ``capacity_per_cell * n_cells``."""
        return sum(len(held) for held in self._cells.values())

    @property
    def cell_coverage(self) -> float:
        """Fraction of cells holding at least one sample, in ``[0, 1]``.

        The buffer's own quality signal, and one of the gauges Design 6.5 watches. It answers the
        question the sample count cannot: ten thousand samples in one cell is a buffer that has
        seen a lot of one strike, and a network fine-tuned on it will look excellent right up to
        the first move that reaches a corner nobody has quoted in an hour.

        Cells, not samples, deliberately -- a coverage weighted by occupancy would be dominated by
        the busy cells again, which is the bias the whole module is built against.
        """
        return len(self._cells) / self._spec.n_cells
