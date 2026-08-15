"""Use cases of ingestion, and the anticorruption layer.

Orchestration only: drive the feed, apply updates to the chain, decide when a snapshot is due
and hand it to the bus. The rules live in ``domain/``; the wiring lives in ``entrypoints/``.
Imports are the domain plus ``contracts/``, never ``adapters/`` -- this layer talks to the
outside strictly through the ports the domain declares.

``acl.py`` is the anticorruption layer: the single place allowed to translate between the
internal ``ChainSnapshot`` and the published ``MarketSnapshot``. Concentrating that mapping
in one file is what makes the boundary auditable -- a change in the published schema shows up
as a diff in one module rather than scattered across the context. If translation starts
happening anywhere else, the boundary has already leaked.
"""

from __future__ import annotations
