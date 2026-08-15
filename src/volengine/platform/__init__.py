"""Technical machinery shared by every context: bus, clock, metrics, executors.

Infrastructure, not a fifth bounded context. Nothing here knows what a volatility surface is,
and nothing here may grow business rules -- the day a rule about markets appears in this
package, it belongs to a context instead.

Imports are ``contracts/`` plus stdlib. The bus needs the event types in order to carry them;
it never needs a domain object, because a domain object is precisely what is not allowed on
it.

The relationship with the contexts is inverted on purpose. Each context declares its own
``Clock`` and ``MetricsSink`` ports as ``typing.Protocol``, and the implementations here
satisfy them **structurally** -- no import, no inheritance, no shared base class. That is what
lets a context be tested against a hand-written fake, and lets this package be replaced
wholesale without touching a single context.
"""

from __future__ import annotations
