# ADR-022: Composition-root departures of F1-07

**Status:** Accepted · 2026-08 · taken while building `entrypoints/`

## Context

ADR-020 collects the departures from `Implementation.md` found while building the domain layer,
and is headed as such: *taken while building F1-05*. `entrypoints/` is a different layer built in
a different phase, and it produced three departures of its own. They needed somewhere to live, and
ADR-020 is not it — a record here is immutable (`docs/adr/README.md`), so a phase's departures are
collected once, when that phase is built, and never appended to afterwards. Editing ADR-020 to
carry F1-07's decisions would also make its own heading false, which is the failure mode the
immutability rule exists to prevent.

This is therefore the third of the collecting records, on the same terms as the other two: each
item below is also stated in the docstring that owns it, and none is large enough to earn a file of
its own.

---

## `build_pipeline` takes a fifth argument, `adapters`

`Implementation.md` spells it `build_pipeline(config, clock, bus, metrics) -> Pipeline` and calls
the module **the only place in the code where concrete adapters are instantiated**. Both halves are
kept, and the registry is what keeps them: `Adapters` maps a configured name — `provider =
"constant"` — onto the callable that builds the object, `default_adapters()` is the only function
in the engine allowed to import from a `*/adapters/` package, and `build_pipeline` receives the
registry rather than reaching for it.

The reason is that the graph must be testable before an adapter exists. F1-08 is what supplies the
first three, so a `build_pipeline` that called `default_adapters()` itself would have arrived
untestable and stayed that way for exactly as long as it took someone to notice. Passing the
registry in makes the whole composition — routing, publication, executors, the heartbeat — provable
against fakes now, and changes nothing about where the real ones are named.

It also moves the failure to start-up. A configuration naming an adapter nobody registered raises
`ConfigError` with the name it could not find, instead of composing a pipeline with a context
missing — which downstream is indistinguishable from a market that is not moving.

## `config.py` reuses the contexts' own threshold types

`Implementation.md` lists a parallel mirror — `ConventionsConfig`, `AdmissibilityConfig` — beside
the types the use cases are actually built from. `config.py` reads the TOML straight into
`MarketConventions`, `AdmissibilityThresholds`, `SnapshotPolicyConfig`, `GridSpec`, `Weighting`,
`Acceptance`, `FreshnessPolicy`, `BumpSpec` and `ReportSettings` instead.

A mirror type is a second declaration of the same numbers carrying a second copy of the same
guards, free to drift from the first — and the drift would be silent, because both sides would
still parse. The module parses; it does not restate. That is also why the TOML keys are spelled
exactly like the fields they fill: one vocabulary, and no translation table to keep honest.

This is not a breach of rule 8. `entrypoints/` is the composition root and may import anything; the
constraint that matters is the opposite one, that no *context* learns about configuration, and
reusing the domain's own types is what keeps the dependency pointing that way.

The consequence is that **`MarketConfig` exposes `market_id` and `underlying` as properties, not
fields**, delegating to `conventions`. `MarketConventions` already carries both. Two spellings of
one identifier can disagree, and the copy that would lose is the one every published event is
stamped with.

Each construction is wrapped so the failure is a `ConfigError` naming the table it came from, with
the domain's `ValueError` chained as `__cause__`: a traceback out of `risk/domain/` for a typo in a
file sends an operator to read the wrong code, but the rule that was broken is still the right
message.

## The composition root splits the book by underlying, and this fixed a real bug

`Implementation.md` gives `RiskConfig` one `portfolio` and says nothing about which market values
it. Handing every market the whole book prices BTC legs off the ETH smile and **prints a number
rather than raising**: `position_risk` never compares `Position.underlying` against the surface it
is valued on, because `CalibratedSurface` publishes a `market_id` and no underlying at all — a seam
that is open on purpose and recorded as such.

Pairing a book with the right market is therefore the caller's obligation, and since it is
unenforceable by any type in `risk/`, it has to be discharged where both sides are visible at once.
That is the composition root and nowhere else, so `_book_for(market, portfolio)` gives each market
the positions written on its own underlying, and the obligation is written down as code instead of
as a convention.

Two configurations are refused rather than run:

- **A market with nothing to value.** `Portfolio` requires at least one position, and a market
  configured with an empty book is either a typo in `underlying` or a market nobody meant to run.
- **A position on an underlying no market quotes** (`_require_every_position_is_quoted`). This is
  the half `_book_for` cannot catch: with two markets configured, an orphan position leaves both
  with a legal non-empty book of their own and is simply never valued by anybody. Silently unvalued
  risk is the one outcome the Risk context exists to prevent, and it is the only failure of the
  three that is invisible in the output.

## Consequences

- `Implementation.md` is behind the code in these three places too, and the code is right. It is
  now ADR-020 **and this record** that `CLAUDE.md` points at: ADR-020 for the domain layer,
  ADR-022 for the composition root.
- The pairing of book to market is enforced at start-up and only there. Nothing structural stops a
  future caller of `position_risk` from handing over a whole book again — that seam stays open in
  `docs/SEAMS.md`, and closing it means putting an underlying on the published surface.
- Reusing the contexts' types means a domain invariant tightening is felt by the configuration file
  immediately, with no mirror to update. That is the intent; it also means a threshold rename is a
  wire-format change to the TOML, which is the cost accepted.
