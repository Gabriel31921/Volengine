# ADR-025: `SyntheticProvider`'s departures from `Implementation.md` in F2-03

**Status:** Accepted · 2026-08 · taken while building the synthetic feed

## Context

ADR-020, ADR-021, ADR-022 and ADR-023 collect the departures of F1-05, F1-06, F1-07 and F1-08.
F2-03 is the first stage of phase 2 and it produced its own, so it gets its own record on the same
terms: `CLAUDE.md` says a phase's departures are collected in an ADR and that a new phase's
departures get a new one rather than an edit to an old one, which would misdate a decision and
make that record's own heading false.

`Implementation.md:677-687` sketches the stage as a `SyntheticConfig` of eight fields and a
`SyntheticProvider` with one method, under a paragraph of prose describing what the feed must do.
The prose and the signature do not agree: everything below is the code following the prose.

---

## `SyntheticConfig` has no `underlying`

The sketch's first field is `underlying: str`. The provider's constructor takes a
`MarketConventions`, which already carries one, so the field would be a second spelling of a
single identifier — and two spellings of one identifier can disagree.

The copy that loses is the one every published event is stamped with. `QuoteChain` refuses an
update for any instrument outside the market it was built for, so a `SyntheticConfig` naming
`ETH` against conventions naming `BTC` would produce a feed that is rejected quote by quote, one
layer downstream, with a message about an unknown instrument rather than about a misconfigured
generator. Deleting the field makes that state unrepresentable.

This is ADR-022's argument applied one layer down: that record removed the parallel
`ConventionsConfig`/`AdmissibilityConfig` mirror for the same reason — one vocabulary, no
translation table to keep honest. It is also the standing rule that a parameter no code path
reads gets deleted rather than documented.

## Eight fields the prose demands and the signature omits

The sketch asks for "a known SVI surface → Black-76 prices over the forward → microstructure
noise: bid/ask, sizes, timestamp jitter and occasional junk quotes", with a fixed seed. Every one
of those clauses is a number, and none of the numbers is in the sketch. They arrive in three
groups:

- **The ladder needs a width.** `strikes_per_expiry` says how many strikes; nothing says where
  they go. `log_moneyness_range` places them, in `ln(K / F)` against the *initial* forward and
  fixed for the run, because a venue lists absolute strikes and does not relist them when the
  underlying moves — which is exactly what makes each quote's moneyness drift as the forward
  walks, and that drift is what a calibrator has to track.
- **The spoiling needs its amplitudes.** `vol_noise_bp` (the perturbation on the mid, in vol),
  `size` (what sits on each side; zero is legal and is what makes `LOW_SIZE` reachable),
  `latency_seconds` and `jitter_seconds` (the double exchange/local stamp, and the spread of it),
  and `forward_move_rel` (the forward walks between cycles, or the whole session is one static
  chain). A generator with no amplitude for a kind of noise cannot produce that kind of noise.
- **The stream needs a shape and an end.** `cycles` and `interval_seconds`. `stream()` is an
  `AsyncIterator`, and one that never terminates cannot be awaited by a test; the interval is
  both slept on and added to the synthetic timeline, which is what makes the timeline advance.

Every field also gained a default, which the sketch has none of, so that `SyntheticConfig()` is a
complete BTC-shaped market and a test bends the one knob it is about.

**None of this is TOML.** ADR-012 puts cadence, admissibility, acceptance and freshness in a file
an operator edits, because those are judgements this engine makes about a real market. These are
the properties of an invented one, and the fixture that invents it is the right place for them.
The distinction matters for F2-07, which owns the registration: what an operator will be handed
there is a way to pick a *scenario*, not fifteen knobs.

## The constructor takes `conventions` and `start`

Neither appears in the sketch, which shows `SyntheticProvider` with `stream()` alone.

`conventions` is what makes the generated chain a *market*: the expiry hour the venue settles at,
the underlying every instrument is named from, and the tenor convention that turns an expiry into
the `tenor_years` the pricing is done in. Convention-dependent quantities are computed before the
boundary, and this is the object that owns them.

`start` is the reproducibility requirement, and it is the more interesting of the two.
Reproducibility here is not a convenience: it is what makes the end-to-end test of Design 9
assertable at all, and a feed that stamps its quotes off the wall clock is not reproducible no
matter how well seeded its noise is. The engine's own `Clock` cannot be used, because ADR-022 has
`ProviderFactory` take a `MarketConfig` and nothing else. So the timeline is derived from a single
`start` instant: passing one makes the whole session a function of `(seed, start)` and reproducible
bit for bit, and omitting it reads the wall clock once, which is what a live-looking run wants.

`docs/SEAMS.md` had recorded two options for this stage and the code took a third. The seam is
therefore **narrowed, not closed**: the provider is reproducible standalone, but a whole pipeline
run under `ManualClock` is still impossible, because closing that needs a change to the factory
signature rather than to this adapter.

## Consequences

- `Implementation.md` is not amended. A document and an ADR disagreeing is what this record exists
  to resolve, and `CLAUDE.md` says the ADR wins.
- Each departure is also argued in the docstring that owns it, in
  `market_data/adapters/synthetic.py`; this record is where `CLAUDE.md` says they must also be
  collected.
- `SVIParamsSpec` is not a departure. The sketch names the type in `true_params` without defining
  it, and what it is — a *specification of what to generate*, with invariants stricter than the
  fitted `SVIParams` one context over — is argued in the module and, for the mathematics the two
  share, in ADR-026.
- F2-07 inherits the open question this leaves: which of these fifteen fields an operator may set
  from TOML, and under what name. Until then the adapter is reachable from tests and not from a
  configuration file, which is why it is absent from `default_adapters()`.
