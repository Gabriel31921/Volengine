"""Business rules of ingestion: the chain, the conventions, the quality judgement.

Imports are stdlib, ``shared_kernel/`` and numpy. Never asyncio, never an exchange client,
never ``contracts/`` -- the third one is stronger than plain DDD and is the point: this layer
does not know the published language. ``QuoteChain.snapshot()`` returns an internal
``ChainSnapshot``, and ``application/acl.py`` translates it into ``MarketSnapshot``. That
makes the published vocabulary replaceable without touching a single business rule, and makes
the constraint checkable by import-linter instead of by review.

No ``logging`` either: observability goes through the ``MetricsSink`` port, so this layer
states what it observes and never decides where that ends up.

Errors raised from here are plain ``ValueError`` for invalid construction and
``MarketDataError`` subclasses for conditions the application is expected to handle.
"""

from __future__ import annotations
