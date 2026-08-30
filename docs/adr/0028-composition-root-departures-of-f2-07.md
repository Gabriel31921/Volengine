# ADR-028: Composition-root departures of F2-07

**Status:** Accepted · 2026-08 · taken while wiring F2's adapters into `entrypoints/`

## Context

ADR-020, ADR-021, ADR-022, ADR-023, ADR-025 and ADR-027 collect the departures of F1-05, F1-06,
F1-07, F1-08, F2-03 and F2-05. F2-07 produced its own and gets its own record on the same terms:
`CLAUDE.md` says a phase's departures are collected in an ADR, and that a new stage's departures
get a new one rather than an edit to an old one — an edit would misdate the decision and make that
record's own heading false.

F2-07 is the stage that puts F2's three adapters behind configured names: `SyntheticProvider`,
`ScipyCalibrator` and `CsvReportWriter`. Two of them carry settings of their own, and F2-05's and
F2-03's records both left those settings homeless on purpose — "F2-07 owns the TOML" — with the
gaps written into `docs/SEAMS.md`. Giving them that home is what produced the departures below,
and the first of them supersedes a sentence of ADR-022.

---

## `config.py` imports two `*/adapters/` modules, and one sentence of ADR-022 no longer holds

**Superseded:** ADR-022, *"`default_adapters()` is the only function in the engine allowed to
import from a `*/adapters/` package"*. That sentence stands for everything it was written to
protect and fails on its literal wording. `entrypoints/config.py` now imports `SyntheticConfig` and
`SVIParamsSpec` from `market_data/adapters/synthetic.py`, and `FitSettings` from
`parametric_pricing/adapters/scipy_calibrator.py`.

The alternative was mirrored `…Config` dataclasses in `entrypoints/` — a `SyntheticFeedConfig` and
a `FitConfig` holding the same twenty-one numbers, validated again, handed to the factories to be
copied field by field into the adapters' own types. That is precisely the arrangement ADR-022's
*second* departure refused for the thresholds: `Implementation.md` listed a parallel mirror
(`ConventionsConfig`, `AdmissibilityConfig`) and `config.py` reads the TOML straight into
`MarketConventions` and `AdmissibilityThresholds` instead, because a second declaration of the same
numbers carries a second copy of the same guards, free to drift from the first. Nothing about that
argument changes when the type that owns the numbers happens to live in an adapter. Taking the
mirror here would have meant holding two contradictory positions inside one module, decided by
which package a settings type happens to sit in rather than by anything about the numbers.

So the two ADR-022 sentences were in tension and the weaker one gives way. What survives is every
property the original was defending, and it is worth naming them, because "only one function
imports adapters" was always a proxy for these:

- **`default_adapters()` is still the only function in the engine that *constructs* an adapter.**
  The registry is unchanged: a configuration file names a string, and the mapping from string to
  object is made in exactly one place.
- **A configuration file still cannot reach an object.** `provider`, `calibrators` and `writer` are
  names; nothing in the TOML is an import path, a class or a dotted reference.
- **No module below `entrypoints/` gained an import.** The layering contract that matters is
  import rule 8 — `entrypoints/` is the composition root and may import anything — and rules 1–7
  are untouched. `lint-imports` reports eight contracts kept.

What is genuinely lost is the mechanical simplicity of the old claim: "grep for `adapters` in
`entrypoints/` and you will find one function" is no longer true, and a reviewer now has to read
*what* is imported. Two settings types and no adapter class is the line, and it is stated here
rather than left as a habit.

The pressure that produced this is ADR-012's, not convenience. Those numbers are the Huber scale,
the butterfly penalty, the ridge, the generating surface and the seed — thresholds and market
parameters a deployment retunes without touching Python, which is what ADR-012 asks to arrive as
data. Somewhere between the TOML and the adapter, a validated object has to be built. The only
question was whether it is built once or twice.

## The two settings sections are optional, and complete when present

`[market.synthetic]` and `[calibration.fit]` may be absent, and then the adapter's own defaults
apply. Present, every key is required.

Both halves are decisions rather than conveniences. **Optional**, because a file that runs
`flat-vol` and the constant feed — `examples/walking-skeleton.toml` is one — would otherwise have
to spell twenty-one numbers no adapter in that run ever reads, which is the same "configuration
nothing loads" that F2-05 refused to invent in the first place. **Complete when present**, because
a partial table would need `config.py` to hold a default for every key the file omitted, which is
the mirror type this record has just refused, arriving one field at a time.

The cost is honest and recorded in `docs/SEAMS.md`: for a deployment that writes no section, the
defaults are still constants in code. ADR-012's requirement is that a number a deployment wants to
change can be changed from a file, and that is met; its stronger reading — that no default may
exist anywhere — was never met by any adapter in this repository and is not met here.

## A provider's settings are a named typed field, one per adapter

`MarketConfig.synthetic: SyntheticSettings | None`, named after the one adapter that reads it, with
`SYNTHETIC_PROVIDER` spelled once in `config.py` and imported by `pipeline.default_adapters()` as
the registry key. A `[market.synthetic]` beside any other provider is **refused**, not ignored.

The alternative was a generic `Mapping[str, Any]` passed through to the factory. It was rejected
because it moves parsing into `*/adapters/`, and with it the one property this loader exists for:
every failure is a `ConfigError` naming the table and the key it came from. A bag would also make
the mis-pairing above unnoticeable — a section nobody reads is the failure mode that costs an
afternoon.

The consequence is that `MarketConfig` grows one optional field per configurable provider; F3-C's
Deribit adapter will add the second. That is a line of code and a docstring, and it stays
type-checked. `SyntheticSettings` itself is a pair — the adapter's `SyntheticConfig` plus a `start`
the adapter takes as a constructor argument — because the origin is not part of the generated
market, and pairing them keeps `MarketConfig` to one field for one provider.

## `Pipeline.run` gained `duration_seconds`, and `run` gained `--duration`

`Implementation.md`'s F2-07 entry shows `volengine run --market BTC-SYNTH --calibrators svi-scipy
--duration 60`; `Plan.md`'s F2-07 line does not mention it, and F1-07 fixed the CLI's shape without
it. It is implemented, as a second stopping rule beside the report goal: whichever completes first
ends the run, and neither drains work in flight.

It is timed on the **injected `Clock`**, not on `asyncio.sleep`, for ADR-004's reason and with the
heartbeat's known consequence — under a `ManualClock` the timer advances simulated time on its own,
which is what makes a bounded run assertable in microseconds instead of in seconds. A duration
measured on the wall clock would make a replay end when the machine felt like it.

It is on `run` alone. `report` already stops at `--count`, and a second stopping rule there would
be two answers to one question.

## Consequences

- **ADR-022 is not edited.** One of its sentences is superseded by this record and the rest of it
  stands; `docs/adr/README.md` marks the row accordingly. That is the mechanism the index
  describes: a record is immutable, and a later one says what changed.
- **`instructions/Implementation.md:496` carries the same now-outdated sentence** — "the claim
  above is unaffected — `default_adapters()` is still the only function allowed to import from a
  `*/adapters/` package". It is git-excluded and is not amended here. `CLAUDE.md` already says the
  ADRs win where they and that document disagree, and this is one of those places.
- Each departure is also argued in the docstring that owns it — `entrypoints/config.py`,
  `entrypoints/pipeline.py`, `entrypoints/cli.py` — and this record is where `CLAUDE.md` says they
  must also be collected.
- **F3-C inherits the shape.** A Deribit adapter with settings adds a second named field to
  `MarketConfig` and a second import to `config.py`, on the terms set out above; what it must not
  do is turn the two fields into one untyped bag because there are now two.
- **F3-A inherits the same question for the JAX calibrator.** If it takes tuning, it takes it
  through a `[calibration.…]` table of its own rather than by widening `[calibration.fit]`, which
  names one adapter's settings and not "the calibrator settings".
