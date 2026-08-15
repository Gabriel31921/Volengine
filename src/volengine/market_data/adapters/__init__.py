"""The outside world, on this context's terms: exchange feeds and recordings.

Implementations of the ports the domain declares -- websocket and REST clients, a replay
provider reading a recorded session, the venue's symbol grammar. Everything venue-specific
stops here: an adapter *uses* the market conventions, it never *defines* them (ADR-002),
because conventions buried next to parsing code are cheap to write and impossible to audit.

Nothing imports this package except ``entrypoints/``. That is what lets the same use case run
against Deribit and against a recording without knowing the difference, which is in turn what
makes a replay deterministic (ADR-004) instead of merely repeatable.
"""

from __future__ import annotations
