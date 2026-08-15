"""Market Data context: turn an exchange feed into snapshots worth calibrating.

Owns the live option chain, the market's conventions, and the judgement about whether what
arrived is usable. Publishes ``MarketSnapshot`` on ``SnapshotReady`` and announces changes in
the tradable universe on ``ChainCompositionChanged``. It consumes nothing from other
contexts: it is the source of the pipeline.

Its two hard responsibilities:

**Resolve conventions before the boundary** (ADR-002). Daycount, exact expiry instant,
numeraire and forward construction are decided here, so ``tenor_years`` and ``forward`` leave
as plain numbers. A calibrator must never learn which market produced its input -- otherwise
a difference between the BTC surface and the SPX surface becomes impossible to attribute to
the market rather than to our daycount.

**Flag, never filter.** Staleness, wide spreads, crossed markets, extreme moneyness and
slice-level monotonicity or convexity are all recorded as flags on the way out. Whether a
flagged quote is down-weighted or dropped is the calibrator's call, because hard no-arbitrage
filtering is a model criterion, not a data one. Ingestion that silently discarded quotes
would make the two calibrators incomparable, since neither could tell what it never saw.
"""

from __future__ import annotations
