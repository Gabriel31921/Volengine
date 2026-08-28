"""What the Risk domain needs from the outside world, said in its own words.

These are the *driven* ports of this context: the domain declares the shape of the collaborators
it depends on, and the composition root supplies something that fits. Nothing below is imported
from anywhere -- not from the producers that make surfaces, not from ``platform/``, not from the
bus -- and that is the direction of the dependency arrow in a hexagon: the outside knows about
the inside, not the reverse.

**This context is the one that only consumes.** Market Data produces quotes, Parametric Pricing
and Neural Surface each produce a surface; Risk produces a report and reads everything else.
That asymmetry is why ``SurfaceProvider`` matters more here than any port matters in the two
producing contexts. A surface reaches Risk from a bus topic somebody else owns, in a language
somebody else published, on a schedule somebody else set -- and none of that is visible in this
file. What Risk states is a need: *give me the newest surface you have for this market, or say
you have none*. Whoever satisfies it -- the parametric producer, the neural one, or the
composite last-value cache that holds both and makes the comparative report of Design 7.3
possible -- is a wiring decision taken once, in ``entrypoints/``.

Everything below is a ``typing.Protocol``, so conformance is **structural**: an adapter is never
asked to inherit from anything, it simply has the right methods. Nothing here is
``@runtime_checkable``, deliberately. ``isinstance`` against a runtime-checkable Protocol only
compares *member names*, and the failure this context most needs to catch would sail straight
through such a check: an adapter whose ``latest`` hands back the published ``CalibratedSurface``
instead of this context's ``SurfaceView`` has a member with exactly the right name. That is the
leak the whole ACL exists to prevent -- the domain reading the published language directly, rule
3 broken in the one place it is hardest to see -- and a name check that blesses it is worse than
no check at all, because it reads like one. The real guard is ``mypy --strict`` at the
composition root, where the concrete object is assigned to the port, plus a call made through
the port in the tests.

``Clock`` and ``MetricsSink`` are duplicated on purpose across contexts rather than imported
from ``platform/``: a port describes a need, and a need belongs to whoever has it. This
context's ``Clock`` is visibly not Market Data's, and it is read for a reason neither producing
context has -- see its docstring.
"""

from __future__ import annotations

from datetime import datetime
from typing import Protocol

from volengine.risk.domain.risk_report import RiskReport
from volengine.risk.domain.surface_view import SurfaceView


class SurfaceProvider(Protocol):
    """The newest surface for a market, or nothing. **Defined by the reader, not the writer.**

    This is the central port of the context and the single most important thing in this file.
    Design 7.1 spells the inversion out -- *driven, defined by Risk, not imported from the
    producers* -- and everything this context is able to demonstrate rests on it. One report,
    one portfolio, one valuation routine, run against the SVI producer, against the neural one,
    or against a composite that holds the last surface from each: Risk never learns that any of
    those exist. The obvious alternative, a shared "surface source" interface published beside
    the producers, would have made the producers Risk's dependency and the comparison of Design
    7.3 a matter of Risk knowing how to ask two different objects the same question.

    **Identity travels in the answer, not on the port**, and this is where Risk parts company
    with its two sibling contexts. ``Calibrator`` and ``SurfaceLearner`` both carry a
    ``producer_id`` property, because each *is* one producer and has to name itself. A provider
    is not a producer: the composite of Design 7.3 serves several markets and several producers
    from one object, so asking it "who are you?" has no single answer. The question is answered
    by ``SurfaceView.producer_id``, which arrives attached to the surface actually handed back --
    the only place it can be right when one call site sees two producers.

    Implementations live in ``risk/adapters/``. This module knows nothing about bus topics,
    subscriptions, caches or eviction, and must not learn: an in-memory last-value cache fed by
    a subscription, a fixture loaded from a JSON file and a stub that always answers ``None`` are
    three implementations of the same three-line protocol.
    """

    def latest(self, market_id: str) -> SurfaceView | None:
        """The most recent surface held for ``market_id``, or ``None`` if there is none.

        Args:
            market_id: Which market to value against, as the published surface names it --
                ``"BTC-DERIBIT"``. A key rather than no argument at all because one provider
                object serves the whole engine: the multi-market run of F3-F wires a single
                composite and lets the key do the routing, so Risk never grows a registry, a
                lookup table or a per-market instance of anything. It is a plain ``str``, this
                context's whole knowledge of what a market is -- Risk does not parse it, does
                not split it into venue and underlying, and does not validate it, because every
                meaning it might carry belongs to Market Data.

        Returns:
            A ``SurfaceView``: this context's own model, in total variance on its own grid,
            built by ``application/acl.py`` from the incoming ``CalibratedSurface``. Never the
            DTO itself. The domain does not know the published language (rule 3), so the
            translation has to happen strictly outside this port, and typing the return value as
            the view is what makes that unavoidable rather than merely intended.

            **Or ``None``, which is an ordinary answer and not an error.** At start-up no
            producer has published anything yet, a market may simply have no producer wired to
            it, and a composite asked for a producer that has fallen silent since boot has
            nothing to give. None of those is exceptional, and none of them is a reason to raise:
            ``None`` pairs with ``FreshnessDecision.REJECT`` to form the two ways a report can
            honestly say there is nothing to value -- *no surface at all* and *a surface too old
            to use* -- which are genuinely different statements and are reported as such, one
            with ``ts_snapshot`` absent and one with it present. See ``domain/errors.py``: the
            deliberate absence of a "no surface" exception is the rule this return type
            implements.

        Raises:
            Nothing, in the ordinary course of business. An empty provider returns ``None``; a
            stale one returns the stale surface and lets ``FreshnessPolicy`` judge it, because
            an adapter that filtered by age would take Design 7.2's decision away from the layer
            that owns it, silently, with no metric left behind and nothing for the report to say.

        Synchronous, not ``async``, and not by omission. A provider is a *cache*: the surface is
        already in memory, put there by whatever subscribed to the bus, and reading it is a
        dictionary lookup. The network hop happened somewhere else, at a time nobody was waiting
        for it. Typing this ``async`` would push a coroutine through the valuation path -- and
        would invite an implementation to fetch a surface on demand, which is the one shape that
        would make a risk report's latency depend on a producer's availability.
        """
        ...


class ReportWriter(Protocol):
    """Somewhere for a finished report to go. One method, no return, no vocabulary.

    A driven port with a deliberately tiny surface, and the smallness is the whole design. A
    console writer for the walking skeleton, a CSV writer feeding the plots of F3-E, and the
    comparative report that puts one portfolio's values under two producers side by side, all
    have to be substitutable here without the domain ever learning what a column, a row, a width
    or a currency symbol is. Every method this protocol does not have is a formatting decision
    that stayed outside the domain: no ``header``, no ``format``, no ``flush``.

    Note what is missing in comparison with Market Data's ``MarketDataProvider``, which does
    declare a ``close``. A writer that owns a file handle certainly has a lifetime, and it is
    managed where the handle is opened -- in the adapter, by the composition root, as a context
    manager if it likes. Putting a ``close`` here would mean the domain deciding when an output
    file ends, which is a resource question that has nothing to do with valuing a book, and it
    would force a no-op on the console writer, the null writer and the in-memory writer the
    tests use.
    """

    def write(self, report: RiskReport) -> None:
        """Emit one finished report. Whatever "emit" means is entirely the adapter's business.

        Args:
            report: The report exactly as the domain built it, **including a rejected one**. A
                ``RiskReport`` carrying ``FreshnessDecision.REJECT``, no positions and a message
                is not an error path to be swallowed on the way out -- it is the loudest thing
                this context ever has to say (Design 7.2), and a writer that skipped it would
                turn "there is no valid surface" into silence, which is exactly what an operator
                cannot tell apart from a dead process.

        Returns:
            ``None``. Writing is a side effect and the domain wants nothing back: a writer that
            returned the rendered text, a row count or a status would tempt a use case into
            inspecting it, and the first thing anyone would inspect is the formatting this port
            exists to stay ignorant of.

        Raises:
            Whatever the destination genuinely fails with -- a full disk, a closed stream. That
            is an infrastructure failure, not a market condition, so it is deliberately not
            wrapped in a ``RiskError``: the hierarchy in ``domain/errors.py`` is for things the
            domain has an opinion about, and a broken output file is not one of them.

        Synchronous, not ``async``, even though this is the one port here that really does I/O.
        A report is written once per snapshot, at the cadence a human reads reports, and a writer
        that blocks is handed to the named thread pool of ADR-005 by the use case. Typing it
        ``async`` would put ``asyncio`` in the signature of every implementation and pull a
        runtime into the domain's vocabulary (rule 3) to buy nothing measurable.
        """
        ...


class Clock(Protocol):
    """Time, as this context needs it: read the current instant. Nothing else.

    Declared here rather than imported from ``platform/``, and the duplication is the design.
    ``platform.clock.SystemClock`` happens to meet this shape; neither module imports the other,
    in either direction, and the connection is made exactly once, in the composition root.

    **Everywhere else in this engine time is stamped; here it is compared**, and that is the
    argument for a port per context made concrete. Market Data stamps an arrival and waits for
    its cadence; Parametric Pricing and Neural Surface stamp a fit and time a cycle. Risk reads
    the clock to run the freshness policy of Design 7.2 -- ``now`` minus the surface's
    ``ts_snapshot``, against two thresholds, with three observable outcomes: a normal report, a
    report marked ``DEGRADED``, and a refusal that says out loud there is no valid surface. The
    reading is not decoration around a result, it *is* a business rule, and it is the only rule
    in the engine whose entire input is the difference between two instants.

    That makes this context the far end of ADR-004's claim. The ADR promises that a recorded
    file of exchange messages produces the same final risk report every time, and this port is
    the last place a wall clock could break that promise: a module here calling
    ``datetime.now()`` would make the verdict depend on when the replay was run rather than on
    what was recorded, and a ``DEGRADED`` label would appear or not according to how busy the
    machine was. With a ``SimulatedClock`` driven by the recording's own timestamps, the
    thresholds are crossed at the same points every run.

    **No ``sleep``**, where Market Data's ``Clock`` has one. A report is computed on demand or
    when a surface arrives; nothing here polls, and a context that never waits has no use for a
    method that waits. If Risk ever grows a periodic report it is added *here*, and only the
    implementations wired into *this* context have to grow it -- a single shared ``Clock`` would
    force every context's needs onto every context at once, which is how a shared kernel quietly
    becomes a god object.

    Note who does **not** get this port: the policy, the interpolation and the valuation.
    ``FreshnessPolicy.evaluate(ts_snapshot, now)`` takes the instant as an argument and never
    reads a clock itself, which is what lets both of its thresholds be pinned in a test with two
    literal datetimes and no clock at all; the greeks have no notion of when they ran. Time
    enters this context in one layer only -- the use case that reads it here and hands it down.
    """

    def now(self) -> datetime:
        """Current instant, always timezone-aware UTC.

        Aware, never naive: this value is subtracted from ``SurfaceView.ts_snapshot`` to measure
        staleness and stamped onto ``RiskReport.ts_report``, and subtracting a naive datetime
        from an aware one raises ``TypeError``. ``datetime.utcnow()`` returns a naive value
        despite its name and is banned everywhere in this repo.

        It may legitimately read **earlier** than the snapshot it is compared against: a venue
        clock running ahead makes a surface look stamped in the future, and the policy answers
        ``NORMAL`` rather than raising. An implementation is therefore not required to be
        monotonic with respect to anything the domain holds.
        """
        ...


class MetricsSink(Protocol):
    """Where this context's observations go, without it knowing where that is.

    ``logging`` is banned in the domain, and this is the replacement: this layer states *what*
    it observed and stops, leaving format, level and destination to the composition root.
    Declared by this context for the same reason as ``Clock``, and satisfied structurally by
    ``platform.metrics.LoggingMetricsSink`` and ``NullMetricsSink`` with no import in either
    direction.

    The series that matter here are not statements about a model, the way the producers' are.
    They are statements about the pipeline as its only consumer actually experiences it:

    * **The freshness distribution.** How much of a session ran ``NORMAL``, how much
      ``DEGRADED``, how often the report refused outright. Design 7.2 is a policy with three
      outcomes, and the shape of that split is the answer to "did the engine keep up", asked
      from the one place downstream enough to know.
    * **The staleness Risk actually observed**, which Design 8.3 names in exactly those words
      and does not name anywhere else. Upstream every context measures the latency of its own
      hop; this is the end-to-end age of the number a report was willing to publish, and it is
      the only measurement that includes every hop at once, including the ones nobody
      instrumented.
    * **The producer-to-producer value difference.** Design 7.3's comparative report as a time
      series: the same portfolio valued off two surfaces, the gap in total value and per
      position. A divergence that opens and closes over a session says more about two
      calibrators than any fit statistic either of them reports about itself.

    Tags are the dimensions a measurement is filtered by later -- market, producer, freshness
    decision, expiry -- and they are ``str`` so that any backend can carry them. ``producer_id``
    belongs on every one of them, and here more than anywhere: the comparative report values one
    portfolio twice, so an untagged series would average a parametric valuation together with a
    neural one and describe neither.
    """

    def gauge(self, name: str, value: float, **tags: str) -> None:
        """A value that goes up and down: the staleness of the surface a report was built on."""
        ...

    def counter(self, name: str, value: int = 1, **tags: str) -> None:
        """A value that only grows: how many reports were refused for want of a fresh surface."""
        ...

    def timing(self, name: str, ms: float, **tags: str) -> None:
        """A duration in milliseconds: surface received to report written."""
        ...
