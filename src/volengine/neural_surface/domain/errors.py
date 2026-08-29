"""Domain errors for the Neural Surface context.

One hierarchy per context, rooted here. A caller that catches ``NeuralSurfaceError`` catches
everything this context can fail with and nothing another context can fail with -- a guarantee
that holds only while every failure raised from ``neural_surface/`` inherits from this class.

Validation failures inside value objects deliberately stay plain ``ValueError``: a
``TrainingSample`` with a negative tenor, or a mesh whose points run backwards, is a programming
error at construction time, not a market condition anyone catches and recovers from. This
hierarchy is for the second kind.

**The absence here is the whole of ADR-010.** A surface that fails the hard gate does *not*
raise. Arbitrage on the mesh is measured, reported as an ``ArbitrageReport``, and answered by
declining to publish and republishing the last good surface -- the same division of labour
ADR-006 already draws for the parametric calibrator. Refusing a surface is the gate working, not
the system breaking, and an exception would put a routine outcome on the failure path where it
would be counted as an incident instead of as the quality signal it actually is.

What is left are the two situations where there is no honest number to return at all: a model
whose output is not a surface, and a buffer with nothing in it. Neither is an arbitrage
judgement, and keeping them out of the report is what stops a diverged network from being
recorded as a clean one -- ``nan < 0`` is ``False``, so a NaN that reached the report would pass
every tolerance in the engine.
"""

from __future__ import annotations


class NeuralSurfaceError(Exception):
    """Base of every failure this context raises. Not raised directly."""


class SurfaceEvaluationError(NeuralSurfaceError):
    """Raised when a learned surface returns something that is not a surface.

    The wrong shape, a non-finite value, or a total variance that is not strictly positive. Any
    of the three means the model has stopped describing a market: total variance is the square of
    a volatility times a tenor, so zero is the degenerate limit where all uncertainty has
    collapsed and negative has no reading at all.

    This is a **diverged model, not an arbitrageable one**, and the distinction is the reason the
    error exists rather than being folded into the arbitrage report. A single bad gradient step
    can leave every weight NaN, and a NaN propagates silently through the gate: ``nan < 0`` is
    ``False``, so a mesh full of them reports a butterfly violation of exactly zero and a surface
    with no numbers in it would be published as the cleanest one all session. Raising puts that
    case on the path it belongs to -- nothing published, a counter incremented, the previous
    surface republished -- without ever letting it near a metric series that is supposed to mean
    "how close to arbitrage-free is this fit".

    Raised by ``learned_surface.evaluate_total_variance``, which every consumer of a surface goes
    through for exactly this reason: validating in one place is what makes the guarantee worth
    anything, and an adapter that evaluated its own model directly could hand a caller values no
    one had looked at.
    """


class NoImpliedVolError(NeuralSurfaceError):
    """Raised when a quoted premium admits no implied volatility at all.

    The price sits at or outside the open interval between the discounted intrinsic value and the
    no-arbitrage ceiling, so no volatility reproduces it and there is nothing for the inversion to
    find. Routine rather than exceptional: a mid taken from a crossed or stale book falls below
    intrinsic regularly, and a deep in-the-money quote sits close enough to a bound that one tick
    of noise crosses it.

    The caller drops the quote from the batch; it does not repair it. That is why this is an
    exception rather than a ``None`` return -- a quote that cannot be turned into a training point
    has to be excluded by name, and folding it into an optional would put "no price" and "a price
    of nothing" behind one check.

    **A deliberate twin of ``parametric_pricing``'s error of the same name**, for the same reason
    ``pricing.py`` in this package is a twin of its ``black76``: no context imports another
    (rule 6), and one hierarchy per context is what makes ``except NeuralSurfaceError`` mean
    "everything this context can fail with and nothing another can".
    """


class EmptyBufferError(NeuralSurfaceError):
    """Raised when a training batch is asked of a replay buffer that holds nothing.

    Not a bug. The buffer is empty for exactly as long as the first snapshot takes to arrive, and
    a scheduled restart that fires in that window asks a perfectly reasonable question the buffer
    cannot answer yet. There is no empty batch to hand back either: ``TrainingBatch`` requires at
    least one sample, because a fine-tuning step over no data is a no-op that would still be
    counted, timed and published as an update.

    The caller waits for the next snapshot. It does not construct a placeholder.
    """
