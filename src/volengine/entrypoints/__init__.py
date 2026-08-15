"""The composition root: the only place allowed to import everything.

Here the adapters are instantiated, the ports are wired to them, the bus is created and the
contexts are started. Every import rule binding the rest of the codebase is suspended in this
package alone -- which is exactly why it must stay thin. Logic that ends up here is logic no
context owns and no test can reach without booting the whole system.

This is also where configuration is loaded. Cadences, admissibility bounds, acceptance
criteria and freshness limits are TOML data read at startup (ADR-012), never constants buried
in the code that uses them: a threshold buried in a domain module cannot be varied per market,
cannot be swept in an experiment, and cannot be reviewed without reading code.
"""

from __future__ import annotations
