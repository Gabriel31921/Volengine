"""Domain errors for the Parametric Pricing context.

One hierarchy per context, rooted here. A caller that catches ``CalibrationError`` catches
everything this context can fail with and nothing another context can fail with -- but that
guarantee only holds while every failure raised from ``parametric_pricing/`` inherits from
this class. A bare ``RuntimeError`` escaping the context silently defeats it.

Validation failures inside value objects deliberately stay plain ``ValueError``: an ``SVIParams``
with ``rho = 1.4`` is a programming error at construction time -- the free-parameter mapping
exists precisely so that no optimiser can propose one -- not a market condition anyone catches
and recovers from. This hierarchy is for the second kind.

The hierarchy is deliberately small, and one absence explains the shape of the whole context.
**A fit that is not good enough to publish does not raise.** It is not an exception: the use
case compares the metrics against the acceptance thresholds and publishes ``CalibrationFailed``
plus the previous surface as ``STALE_REPUBLISH`` (ADR-006). A rejected calibration is an
ordinary, expected outcome of a working system -- exceptions are for the cases where there is
no honest number to return at all, and that is the single case below.
"""

from __future__ import annotations


class CalibrationError(Exception):
    """Base of every failure this context raises. Not raised directly."""


class NoImpliedVolError(CalibrationError):
    """Raised when a target price admits no implied volatility at all.

    Black-76 is strictly increasing in volatility, from the intrinsic value at ``vol -> 0`` to
    the undiscounted forward (calls) or strike (puts) at ``vol -> infinity``. A price outside
    that open interval is not reproduced by *any* volatility, so there is nothing to return:
    not a large number, not zero, not NaN. The inversion states that it failed instead.

    Not an anomaly. A mid built from a crossed or stale book falls below intrinsic routinely,
    and deep-in-the-money quotes sit close enough to the bound that a tick of noise crosses it.
    The caller drops that quote from the slice; it does not repair it. Which is the same
    division of labour Market Data already follows -- ingestion flags, the calibrator weights
    or excludes -- applied one step further down: an uninvertible quote carries no information
    about volatility, so it never reaches the loss.

    Deliberately *not* raised for slow convergence. The inversion brackets the root and falls
    back to bisection, which halves the interval every iteration on a monotone function, so
    "the solver did not converge" is unreachable rather than unhandled, and an error class for
    it would be documentation of something that cannot happen.
    """
