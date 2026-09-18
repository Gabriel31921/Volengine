# Test fixtures

## `deribit-btc-2026-09-18.jsonl` — the golden fixture, in its committable form

Thirty seconds of the real Deribit BTC option chain, recorded on 2026-09-18 from 10:38:45 UTC,
narrowed to two expiries — the nearest quarterly (`25SEP26`, 132 instruments) and the year-end one
(`25DEC26`, 118) — through this engine's own machinery: `DeribitProvider` behind
`RecordingProvider` inside `build_pipeline`, with a filter on the expiry in front of the chain. It
is the file `tests/entrypoints/test_deribit_fixture.py` replays on every test run.

**Produced by** `tests/fixtures/record_deribit_fixture.py`, verbatim:

```
uv sync --extra deribit
uv run python -m tests.fixtures.record_deribit_fixture tests/fixtures/deribit-btc-2026-09-18.jsonl
```

The script is the provenance: the expiries, the duration, the calibrator it ran with (`flat-vol`)
and the reason for each are in its docstring. Nothing was edited afterwards. What was **narrowed**:
the universe, to two of the day's twelve expiries, before the chain — the provider still subscribed
to and decoded all 928 instruments, so the receipt stamps are those of a session carrying the whole
venue's traffic. What was **not** narrowed: the duration was recorded as 30 s, not trimmed from a
longer file.

| | |
|---|---|
| header instruments | 250 |
| quote lines | 6212 |
| span | 28.1 s of session (the first quote lands ~1.5 s after the run starts) |
| size | 2.11 MB |
| receipt lag `ts_local − ts_exchange` | median 1.55 s, p95 2.54 s, max 2.67 s — flat over the session |
| two-sided quotes | 84.8 % of the universe at every snapshot |
| forward cross-check error (ADR-002) | 1.3 × 10⁻⁴ to 2.0 × 10⁻⁴ across 29 one-second snapshots |

Why not the full hour the plan names (`Plan.md` F3-C): a full-chain minute is 14 MB and, as of
F3-C, the ingestion loop consumes a full chain at about the rate the venue produces it
(`docs/SEAMS.md`), so a full-chain recording's stamps lag the venue by tens of seconds — a
fixture made from one would carry that artefact into every staleness number downstream. The
narrowed session keeps the loop comfortably ahead. Where the full hour lives is an owner decision,
recorded as open in `docs/SEAMS.md`.

The venue's chain moves every day, so a re-recording is a *new* fixture: different instruments,
different prices, a different date in the name, and the numbers in `examples/deribit-live.toml`
and in the E2E's bounds re-measured against it.
