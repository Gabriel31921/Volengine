"""The outside world for risk: book sources and report sinks.

Implementations of the ports the domain declares -- where a portfolio is read from, where a
valuation is written to. Nothing imports this package except ``entrypoints/``.

The asymmetry with the other contexts is worth noticing: risk has no exchange connection and
no numerical library of its own. Everything it needs about the market arrives as a contract
on the bus, which is precisely the property that makes it a fair judge of two calibrators it
cannot distinguish.
"""

from __future__ import annotations
