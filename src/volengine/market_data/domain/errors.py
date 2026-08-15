"""Domain errors for the Market Data context.

One hierarchy per context, rooted here. A caller that catches ``MarketDataError`` catches
everything this context can fail with and nothing another context can fail with -- but that
is only true while every failure raised from ``market_data/`` inherits from this class. A
bare ``RuntimeError`` escaping the context silently defeats the guarantee.

Validation failures inside value objects deliberately stay plain ``ValueError``: an invalid
strike is a programming error at construction time, not a market condition the application is
expected to catch and recover from. This hierarchy is for the second kind.
"""

from __future__ import annotations


class MarketDataError(Exception):
    """Base of every failure this context raises. Not raised directly."""


class ExpiredInstrumentError(MarketDataError):
    """Raised when a tenor is asked for an expiry that is not strictly in the future.

    Not an anomaly: some venues keep expired instruments in the chain until they settle, so the
    chain will meet these routinely. It is an error only in the sense that there is no honest
    year fraction to return -- zero and negative tenors break the pricing formulas and are
    rejected by ``SliceData`` anyway. The caller drops the instrument; it does not repair it.
    """
