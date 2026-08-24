"""What the Parametric Pricing domain needs from the outside world, said in its own words.

These are the *driven* ports of this context: the domain declares the shape of the
collaborators it depends on, and the composition root supplies something that fits. A scipy
least-squares fit (F2-05) and a JIT-compiled JAX one (F3-A) are two implementations of
``Calibrator``; neither is imported here, and neither ever will be. That is the direction of
the dependency arrow in a hexagon -- the outside knows about the inside, not the reverse --
and it is what makes the comparison between those two implementations meaningful at all: they
are interchangeable behind one signature, handed identical inputs, judged on the same metrics.

Everything below is a ``typing.Protocol``, so conformance is **structural**: an adapter is
never asked to inherit from anything, it simply has the right methods. Nothing here is
``@runtime_checkable``, deliberately. ``isinstance`` against a runtime-checkable Protocol only
compares *member names*, so a ``calibrate`` that returned the wrong type or took its arguments
in the wrong order would pass it happily. That is worse than no check at all, because it reads
like one. The real guard is ``mypy --strict`` at the composition root, where the concrete
object is assigned to the port, plus a call through the port in the tests.

``Clock`` and ``MetricsSink`` are duplicated on purpose across contexts rather than imported
from ``platform/``: a port describes a need, and a need belongs to whoever has it. This
context's ``Clock`` is visibly *not* Market Data's -- see its docstring -- which is the
divergence that duplication was chosen to allow.
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime
from typing import Protocol

from volengine.parametric_pricing.domain.calibration import CalibrationResult, CalibrationTask
from volengine.parametric_pricing.domain.svi_slice import SVIParams


class Calibrator(Protocol):
    """A fit: parameters in, parameters out. **A pure function wearing an object's clothes.**

    This is the central port of the context and the one place its most consequential decision
    is visible. ``calibrate`` performs no I/O, reads no clock, touches no network and -- above
    all -- **keeps no state between calls**. Everything it needs arrives in its arguments and
    everything it produces leaves in its return value. An implementation that cached the last
    result, counted its own invocations or looked up the time would satisfy the type checker
    and break every property below.

    Three things follow, and all three are the reason the port is shaped this way (Design 5.6):

    * **Replay is reproducible for free.** Feed a recorded sequence of tasks back through the
      same implementation and the same warm starts, and the surfaces come out identical, bit
      for bit -- no seeding ritual, no state to reset between runs. ADR-004 asks for that of the
      whole engine; here it costs nothing because there is nothing to reset.
    * **Two calibrators become comparable.** The scipy baseline and the JAX implementation are
      handed byte-identical inputs and their outputs differ only in the mathematics, which is
      exactly the experiment Design 5.7 exists to run. A stateful calibrator would carry its own
      history into the comparison and there would be no way to tell a better fit from a luckier
      one.
    * **Tests need no mocks.** A fit is checked by calling a function and looking at what came
      back, against a synthetic chain built from known SVI parameters (D-7).

    The state that a warm start obviously *is* still exists -- it lives in
    ``application/calibration_state.py``, which chains one cycle's output into the next cycle's
    ``previous``. That placement is the whole trick: the statefulness is real, it is simply kept
    outside the thing being compared and tested, in a module whose only job is to hold it.
    """

    @property
    def producer_id(self) -> str:
        """Stable identity of this implementation, such as ``"svi-scipy"`` or ``"svi-jax"``.

        What makes two calibrators distinguishable everywhere downstream: it names the topic the
        result is published on (``surface.{market_id}.{producer_id}``), it travels inside
        ``CalibratedSurface`` so a risk report can say which producer it trusted, and it is the
        key the comparative report of F3-E groups by. Two implementations running side by side
        on the same market is the normal case in this engine, not an exception, so an anonymous
        producer would leave every consumer unable to say what it was looking at.

        A read-only property rather than a plain attribute so that the protocol constrains
        neither how it is stored nor whether it is computed -- a frozen dataclass field, a class
        constant and a derived string all satisfy it.
        """
        ...

    def calibrate(
        self,
        previous: Mapping[datetime, SVIParams] | None,
        task: CalibrationTask,
    ) -> CalibrationResult:
        """Fit every slice of one task, optionally starting from the last accepted parameters.

        Args:
            previous: The warm start -- the parameters each expiry was last fitted to, keyed by
                that expiry. ``None`` is a cold start, which is the first cycle of a market and
                the recovery path after a failure, and an implementation must handle it rather
                than assume it never happens.

                Keyed by ``datetime``, never by position, because an option chain is not a fixed
                set: strikes and whole expiries are born and die between snapshots (ADR-013). A
                positional structure would pair a warm start with whichever tenor happened to
                occupy that index this time, which is not a wrong number anyone would notice --
                the fit would simply converge somewhere slightly odd. Keying by the expiry makes
                a missing entry mean exactly what it is: a tenor with no history yet.

                A ``Mapping`` rather than a ``dict``: the callee reads it and must not mutate it,
                and the state that owns it stays the only thing allowed to change it. Note the
                asymmetry that keeps this port honest -- it holds ``SVIParams``, this context's
                own vocabulary, and not the ``FreeParams`` an optimiser actually searches in. The
                reparameterisation is an implementation's business, and ``SVIParams.to_free`` is
                where each one crosses into it.

            task: The market to fit, as homogeneous numbers. See ``calibration.py``: no venue,
                no day count, no strike in currency, and no DTO -- the domain does not know the
                published language (rule 3), so a calibrator cannot depend on it either.

        Returns:
            One ``SliceResult`` per slice attempted, with the fit metrics and the ``converged``
            and ``at_bound`` flags beside the parameters.

            **Including the slices that fitted badly.** Deciding whether a result may be
            published is not this port's job: the acceptance rule of ADR-006 -- RMSE under
            threshold and no parameter at a bound -- belongs to the use case, which compares
            against thresholds that are TOML configuration (ADR-012) and which republishes the
            last good surface as ``STALE_REPUBLISH`` when the new one is refused. An
            implementation that raised on a poor fit, or that silently dropped the slice, would
            take that decision away from the layer that owns it and leave nothing to report.

        Raises:
            Nothing, in the ordinary course of business. A bad fit is a return value, not an
            exception; see ``domain/errors.py``, which explains that boundary and is the module
            this one must not contradict. An implementation may still fail -- an optimiser that
            cannot be constructed at all -- and such a failure should surface as a
            ``CalibrationError``.

        Synchronous, not ``async``, and not by omission. Calibration is CPU-bound, so the use
        case hands it to the named thread pool of ADR-005 and awaits the executor rather than
        the fit. Typing it ``async`` would invite an implementation to reach for I/O inside --
        the one thing the purity above forbids -- and would put a coroutine in the hot loop of a
        function that never waits for anything.
        """
        ...


class Clock(Protocol):
    """Time, as this context needs it: read the current instant. Nothing else.

    Declared here rather than imported from ``platform/``, and the duplication is the design.
    ``platform.clock.SystemClock`` happens to meet this shape; neither module imports the other,
    in either direction, and the connection is made exactly once, in the composition root.

    **One method, where Market Data's has two**, and that difference is the argument for
    duplicating the port made concrete. Ingestion drives its own cadence and therefore has to
    wait; calibration is driven by snapshots arriving on the bus and never sleeps -- it reacts.
    Sharing one ``Clock`` across the engine would force ``sleep`` onto a context with no use for
    it, and the next context's needs onto both, which is how a shared kernel quietly becomes a
    god object. If this context ever grows a periodic cold restart (Design 5.6), the method it
    needs is added *here*, and only the implementations wired into *this* context have to care.

    Note who does **not** get this port: the calibrator. ``Calibrator.calibrate`` takes no clock
    and is the better for it. Time enters this context only where a fit has to be stamped --
    ``ts_calibrated`` on the way out, the duration reported beside it -- which is the use case,
    not the mathematics. Reading it as a port at all is what ADR-004 requires so a recorded
    session replays exactly: a module calling ``datetime.now()`` directly is a module whose
    output cannot be reproduced.
    """

    def now(self) -> datetime:
        """Current instant, always timezone-aware UTC.

        Aware, never naive: this value is compared against ``CalibrationTask.ts_snapshot`` to
        stamp and to time a cycle, and subtracting a naive datetime from an aware one raises
        ``TypeError``. ``datetime.utcnow()`` returns a naive value despite its name and is
        banned everywhere in this repo.
        """
        ...


class MetricsSink(Protocol):
    """Where this context's observations go, without it knowing where that is.

    ``logging`` is banned in the domain, and this is the replacement: this layer states *what*
    it observed and stops, leaving format, level and destination to the composition root.
    Declared by this context for the same reason as ``Clock``, and satisfied structurally by
    ``platform.metrics.LoggingMetricsSink`` and ``NullMetricsSink`` with no import in either
    direction.

    The series that matter here are the experiment itself. Fit quality in basis points of vol,
    how long a cycle took, how often a surface was refused and republished stale, how far the
    butterfly and calendar diagnostics ran from zero -- Design 5.7 compares two calibrators on
    exactly those numbers, and a comparison assembled by parsing log prose after the fact is not
    a comparison anyone should trust. Structured from day one.

    Tags are the dimensions a measurement is filtered by later -- market, producer, expiry,
    reason -- and they are ``str`` so that any backend can carry them. ``producer_id`` belongs
    on every one of them: two calibrators publishing the same metric name under the same tags
    would silently average into a series that describes neither.
    """

    def gauge(self, name: str, value: float, **tags: str) -> None:
        """A value that goes up and down: the RMSE in vol basis points of the last fit."""
        ...

    def counter(self, name: str, value: int = 1, **tags: str) -> None:
        """A value that only grows: how many fits were refused for a parameter at a bound."""
        ...

    def timing(self, name: str, ms: float, **tags: str) -> None:
        """A duration in milliseconds: snapshot received to surface published."""
        ...
