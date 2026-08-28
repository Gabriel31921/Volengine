"""Domain errors for the Risk context.

One hierarchy per context, rooted here. A caller that catches ``RiskError`` catches everything
this context can fail with and nothing another context can fail with -- a guarantee that holds
only while every failure raised from ``risk/`` inherits from this class.

Validation failures inside value objects deliberately stay plain ``ValueError``: a ``Position``
with a negative strike, or a ``FreshnessPolicy`` whose reject threshold sits below its warning
one, is a configuration or programming error at construction time, not a market condition anyone
catches and recovers from. This hierarchy is for the second kind.

**The absence here is the star business rule of the context.** Neither "there is no surface" nor
"the surface is too old to use" raises anything. They are a ``RiskReport`` carrying
``FreshnessDecision.REJECT`` and a message saying so out loud, which is precisely Design 7.2: the
timestamps of a contract turned into observable behaviour. An exception would put the most
important thing this context has to say on the failure path, where an operator would read it as
an incident in the report generator rather than as the report's own verdict on the market data.
The rule downstream is the same one ADR-006 draws upstream -- an honest statement that the numbers
are missing beats a silently old number -- and it only works if the statement is a value someone
can print.

What is left is the one situation where a position cannot be valued at all, no matter how fresh
the surface is.
"""

from __future__ import annotations


class RiskError(Exception):
    """Base of every failure this context raises. Not raised directly."""


class ExpiredPositionError(RiskError):
    """Raised when a position's expiry is at or before the instant the surface describes.

    An option that has already expired has no time value to interpolate and no implied volatility
    to look up: the surface's tenor axis starts after ``ts_snapshot`` by construction, and
    extrapolating backwards through zero would produce a total variance of zero or less, which is
    not a market anyone can price. There is no number to return -- not the intrinsic value, which
    is a settlement question rather than a valuation one, and certainly not zero.

    **Not an anomaly, which is why it is an error class and not an assertion.** A portfolio is
    configuration, it outlives its positions, and a file written last month will name an expiry
    that has since passed on the very first run after the roll. The caller drops the position from
    the report or refreshes the file; it does not repair the date.

    Deliberately *not* raised for a position merely far outside the grid. A strike ten times the
    forward, or an expiry beyond the last tenor node, is extrapolated flat and reported, because
    those are answers with a known and documented weakness rather than absences. Only the expired
    case has nothing at all behind it.
    """
