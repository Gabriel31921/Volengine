"""How ``deribit-btc-2026-09-18.jsonl`` was produced, as the command that produces it.

The committed fixture is a real Deribit session recorded through the engine's own machinery --
``DeribitProvider`` behind ``RecordingProvider`` inside ``build_pipeline`` -- with one thing in
front of it: a filter that narrows the universe to a chosen set of expiries before the chain sees
it. Two reasons, both about size. A full BTC chain is some nine hundred instruments ticking at
around 450 updates a second, which is 14 MB a minute -- not a file for plain git -- and, as of
F3-C, more than the ingestion loop keeps up with in real time (``docs/SEAMS.md``), so the receipt
stamps of a full-chain recording lag the venue by tens of seconds and a fixture made from one
would carry that artefact into every staleness number downstream. Narrowed to two expiries the
loop keeps up, and the stamps mean what ``ts_local`` says they mean.

The narrowing is a decorator over the provider, exactly like the recording tap, so the session is
still an ordinary run: discovery, the composition event, the snapshot policy, a calibration and a
report all happen. Only the calibrator differs from ``examples/deribit-live.toml`` -- ``flat-vol``
in place of ``svi-scipy`` -- because a fit that holds the GIL for seconds on every snapshot is the
other thing that stalls the loop that stamps ``ts_local``, and a recording is about what the chain
saw, not about the fit.

Run from the repository root, with the ``deribit`` extra installed::

    uv run python -m tests.fixtures.record_deribit_fixture \\
        tests/fixtures/deribit-btc-2026-09-18.jsonl

It records ``DURATION_SECONDS`` of the expiries in ``EXPIRIES`` and prints the size. The venue's
chain moves every day, so a re-recording is a *new* fixture: different instruments, different
prices, a different date in the name.
"""

from __future__ import annotations

import asyncio
import sys
from collections.abc import AsyncIterator
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

from volengine.entrypoints.config import DERIBIT_PROVIDER, load_config
from volengine.entrypoints.pipeline import build_pipeline, default_adapters, with_recording
from volengine.market_data.domain.option_quote import InstrumentId, QuoteUpdate
from volengine.market_data.domain.ports import MarketDataProvider
from volengine.platform.bus import InProcessConflatingBus
from volengine.platform.clock import SystemClock
from volengine.platform.metrics import NullMetricsSink

CONFIG = Path("examples/deribit-live.toml")
EXPIRIES = frozenset(
    {
        datetime(2026, 9, 25, 8, 0, tzinfo=UTC),
        datetime(2026, 12, 25, 8, 0, tzinfo=UTC),
    }
)
"""The nearest quarterly and the year-end one: 132 and 118 instruments on the day, a liquid body
and thin wings on each, and a tenor axis long enough for a position to sit inside it."""

DURATION_SECONDS = 30.0
CALIBRATOR = "flat-vol"


class NarrowedProvider:
    """Another provider, seen through a filter on the expiry: the universe a fixture can hold."""

    def __init__(self, inner: MarketDataProvider, expiries: frozenset[datetime]) -> None:
        self._inner = inner
        self._expiries = expiries

    async def discover(self) -> tuple[InstrumentId, ...]:
        return tuple(one for one in await self._inner.discover() if one.expiry in self._expiries)

    def stream(self) -> AsyncIterator[QuoteUpdate]:
        return self._filtered(self._inner.stream())

    async def close(self) -> None:
        await self._inner.close()

    async def _filtered(self, updates: AsyncIterator[QuoteUpdate]) -> AsyncIterator[QuoteUpdate]:
        async for update in updates:
            if update.instrument.expiry in self._expiries:
                yield update


def main(target: Path) -> None:
    config = load_config(CONFIG)
    config = replace(config, calibration=replace(config.calibration, calibrators=(CALIBRATOR,)))
    adapters = default_adapters()
    deribit = adapters.providers[DERIBIT_PROVIDER]
    narrowed = replace(
        adapters,
        providers={
            **adapters.providers,
            DERIBIT_PROVIDER: lambda market: NarrowedProvider(deribit(market), EXPIRIES),
        },
    )
    pipeline = build_pipeline(
        config,
        with_recording(narrowed, target),
        SystemClock(),
        InProcessConflatingBus(NullMetricsSink()),
        NullMetricsSink(),
    )
    asyncio.run(pipeline.run(duration_seconds=DURATION_SECONDS))
    lines = sum(1 for _ in target.open(encoding="utf-8")) - 1
    print(f"{target}: {lines} quotes, {target.stat().st_size / 1e6:.2f} MB")


if __name__ == "__main__":
    main(Path(sys.argv[1]))
