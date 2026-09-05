# Known seams

Gaps that are **open on purpose**, each written into the docstring that owns it.

This is not an ADR list and deliberately not in `docs/adr/`. An ADR is immutable: it records a
decision that was taken. These are conditions that are still true, and the file shrinks as they are
closed. Read it before "fixing" something here — every item was seen, weighed and left, and several
close naturally in a later phase.

## Market Data

- **`ChainSnapshot.ts_exchange` is not clamped to `ts_local`**, so a venue clock running ahead would
  make `CalibratedSurface` reject `ts_calibrated < ts_snapshot`. The ACL reconciles them
  (ADR-021); the domain type still permits the state.
- **`IV_DIVERGENCE` never fires.** `flag_quote` receives `own_iv=None` from the chain, and will
  until Market Data grows its own mid inversion in F2 — a wrapper over the shared kernel since
  ADR-014.
- **A recording holds the normalised stream, not the venue's bytes.** `RecordingProvider` taps
  `QuoteUpdate`s, so everything upstream of the port — the symbol grammar, the expiry resolution,
  the numeraire decision — is *inside* the recording rather than reproduced from it, and no replay
  exercises the parser its file came from. Deliberate, and it is what lets one recorder serve every
  provider: recording frames instead would mean one recorder per venue and a replay re-running the
  code most likely to have changed since. It does mean F3-C's golden fixture proves the engine and
  not the Deribit grammar; a second sink beside this one is what would close it.
- **`RecordingSink` writes and flushes on the event loop, one line per quote.** That is what makes
  a session killed by a signal replayable up to the line it died on, and it is blocking file I/O in
  the middle of ingestion: an hour of Deribit is millions of lines, and the flush is the first thing
  to reconsider when the recorder is measured against a real feed. Buffering, or a writer task
  behind a queue, both cost the property the flush buys.
- **The Heston generator resolves a premium only to about 1e-11 of the forward.** The Lewis
  integral is accurate in *absolute* terms, so a deep out-of-the-money option at a short tenor —
  a one-week strike a quarter of the way out is worth around 1e-11 — comes back as the quadrature's
  own rounding, occasionally negative. `heston.PRICE_RESOLUTION` refuses to invert below that
  rather than quote a volatility with no digits in it, which turns the wing into a loud failure
  and not a silent one. Closing it means a different formulation for the wings (a control variate
  against Black-76 was tried and does not help at a realistic vol of vol), not a smaller tolerance.
- **The Heston generator has no TOML home.** `[[market.synthetic.slice]]` builds `SVIParamsSpec`,
  so a Heston market is reachable from Python and from the tests and not from a configuration
  file. Wiring it needs a table of its own and a rule for which of the two generators a file may
  name — a question F3-B did not have to answer, because the deliverable was a second generator
  and not a second way to configure one.

## Parametric Pricing

- **`SVIParams` admits `min_total_variance == 0` while `durrleman_g` refuses it.** `g` divides by
  `w`, and both `inf` and `nan` survive `max(0, -min(g))` as a clean zero, which would report a
  collapsed slice as arbitrage-free. The soft penalty of F2-05 does trip on it, at iterates the
  optimiser walks through: `adapters/scipy_calibrator._residuals` answers the raise with a barrier
  residual rather than letting one bad step kill a whole surface. The asymmetry itself stands.
- **The scipy calibrator's tuning is optional configuration.** `FitSettings` — Huber scale,
  butterfly penalty, ridge, penalty mesh, pinning threshold, evaluation budget — is read from
  `[calibration.fit]` since F2-07, but the section may be absent, and then the adapter's own
  defaults apply. That is deliberate — a file running `flat-vol` alone must not have to state
  seven numbers it never reads — and it does leave those defaults as constants in code for anyone
  who does not write the section. The bounds behind `at_bound` are still *not* part of it
  (ADR-021).
- **`test_durrleman.py` imports the private `_curve_and_derivatives`**, with no precedent in the
  repo. A sign slip in `w'` still yields a plausible `g`, so the derivative has to be pinned against
  an independent computation rather than only through the function that consumes it.
- **The padded reservation cannot grow while the engine runs** (ADR-009). `PadShape` *is* the
  compiled signature, so widening it is a new compilation and therefore a restart. A task with more
  expiries or more strikes than reserved is refused by `adapters/padding.pad` with a
  `CalibrationError` naming the number it would need, and the use case republishes the last good
  surface. Handling `ChainCompositionChanged` so the reservation follows a growing chain is F3-C's,
  which is where that event is first produced.
- **The JAX cold cycle is "after a failure" and never "periodic".** Design 5.6 asks for both, and a
  `Calibrator` reads no clock and keeps no state (the port forbids it), so a pure function cannot
  know that an interval has elapsed. The half that is expressible is implemented: a warm start that
  comes back unconverged or pinned is retried from the multi-start. The periodic half belongs to
  the use case that owns `CalibrationState`, and nothing there schedules it yet.
- **The JAX calibrator has no TOML home and is not in `default_adapters`.** `JaxFitSettings` and
  `PadShape` are constructor arguments with working defaults, and `svi-jax` is reachable from
  Python and from the contract harness but not from a configuration file. Wiring it means a factory
  that imports an optional extra lazily and a settings section beside `[calibration.fit]`; F3-A
  stayed inside the two modules `Implementation.md` names for it. Until then a deployment runs the
  scipy baseline.
- **`adapters/jax_greeks.py` has no consumer in the engine, deliberately.** Greeks do not travel in
  the contract (ADR-001, Design 2.2) and Risk computes its own by bumping (Design 7.4), so the AD
  greeks of Design 5.8 are a module an analyst calls and the pipeline does not. Nothing imports it,
  and that is the correct amount rather than an oversight.

## Neural Surface

- **`fit_metrics` costs `n²` evaluations to measure `n` residuals.** It evaluates the learned
  surface on the full outer product of its two axes and reads the diagonal. The fix is a
  paired-evaluation method on `LearnedSurface`, not a reshape.
- **A restart interval longer than the replay buffer's `max_age_seconds`** finds a buffer holding
  only what arrived in between, because the prune runs first.
- **A snapshot delivered later than that horizon still trains the step**, but its points are pruned
  one cycle later. That is the price of pruning before the draw (ADR-021).
- **`SurfaceLearner.update` returns a bare `LearnedSurface`** where `Calibrator.calibrate` returns a
  metrics-carrying `CalibrationResult` (ADR-019). The use case therefore recomputes the fresh-batch
  RMSE of §6.5 by re-evaluating the surface, and "these K steps diverged" has no channel short of
  `SurfaceEvaluationError`.
- **`implied_vol_grid` raises `SurfaceEvaluationError` when a legal but subnormal tenor overflows
  the division**, which blames the model for what is arguably the caller's axis.

## Risk

- **`Position.underlying` is never cross-checked against the surface**, because `CalibratedSurface`
  publishes a `market_id` and no underlying. Pairing a book with the right market is the caller's
  job, and since F1-07 the caller is `entrypoints.pipeline._book_for` (ADR-022), which splits the
  configured book by underlying and gives each market only its own share. Nothing structural stops a future
  caller from handing over the whole book again.
- **`FreshnessPolicy.evaluate` takes two bare instants**, so nothing structurally stops a caller
  passing `ts_calibrated`. The ADR-006 argument that it must be `ts_snapshot` is defended in the
  docstring only.
- **`PositionRisk` has no invariant tying `value` or the greeks to `position.quantity`** — it cannot
  know the option's own worth — so the quantity-scaling convention is upheld by `valuation.py`
  alone.
- **Bilinear interpolation does not inherit no-arbitrage** between nodes. Past the last tenor a flat
  total variance means the implied vol decays as `1/sqrt(T)`: a two-year option off a one-year grid
  is priced at about 71% of the one-year vol.
- **`RiskConfig.output_path` is unvalidated until a writer opens it.** F2-07 gave the CSV writer
  its TOML home — `writer = "csv"` with an `output_path` beside it, both refused at start-up if
  the second is missing — but the path is read as written, and only the writer that uses one ever
  looks at it. A `console` configuration carrying an `output_path` is accepted and ignored, which
  is the opposite of how a `[market.synthetic]` beside another provider is treated. The asymmetry
  is deliberate and thin: a path is not a block of thresholds, and refusing it would need
  `config.py` to know which writers take one.
- **Risk's numerical greeks carry the grid's kinks** as well as the bump's truncation error. The
  at-the-money gamma of the test surface is about twice the analytic value — a fact Design §7.4
  wants visible, not a defect.

## Entrypoints

- **An adapter cannot be handed the engine's `Clock`.** `ProviderFactory` receives a
  `MarketConfig` and nothing else (ADR-022), so `ConstantProvider` stamps its quotes off the wall
  clock instead. Harmless here — a venue stamps its own messages and `max_skew_seconds`
  reconciles them — and it stays harmless as long as the deterministic replay of ADR-004 arrives
  as `RecordedProvider` in F3-B, replaying recorded instants rather than asking a synthetic feed
  to read a different clock. That is how it arrived: `pipeline.with_replay` hands the clock to the
  one adapter whose job is to move it, through a closure, and `ProviderFactory` is unchanged. F2-07 took the second of the two answers this seam offered for the
  synthetic feed: `[market.synthetic]` carries the seed *and* a `start`, so the stream is
  reproducible from a file without any factory learning about the engine's clock. What is still
  not reproducible from a file is a whole *run* under a `ManualClock` — the feed's timeline and
  the engine's clock are two sources, and only the first one is in the TOML.
- **A `ManualClock` run is reproducible only while nothing advances it.** F2-08 pins both sources
  from the test and gets a byte-identical report out of two runs
  (`tests/entrypoints/test_determinism.py`), and it does so on a clock that never moves: with no
  heartbeat, the cadence is met once, and the session is one snapshot, one fit, one report
  whatever the thread pool does. Configure `max_quiet_seconds` and the heartbeat becomes the
  thing moving time, turning as fast as the event loop lets it — so *how much* simulated time has
  passed when a fit comes back off its pool is a scheduling detail. No test may assert on it, and
  `tests/entrypoints/test_degradation.py` deliberately asserts only what grows more true with
  time. **F3-B closes the half of this that a file can close**: `volengine replay` puts both
  sources in the recording — the quotes and the instants the engine read them at — so two replays
  of one file write the same bytes with nothing pinned from a test
  (`tests/entrypoints/test_replay.py`). What stays open is the *live* run: a `SystemClock` session
  is reproducible only by recording it first, and a `ManualClock` one only by a test holding both
  ends, because a TOML file still has no way to name the engine's clock. **And conflation is
  reproduced rather than removed**: a replay delivers its quotes as fast as the consumer takes
  them, so a snapshot published while the previous fit is still on its pool is overwritten in a
  one-slot mailbox (ADR-003), and how many reports a long recording produces is still a function
  of how fast the machine fits. The test above is byte-exact because its session is one snapshot,
  one fit, one report — the same construction `test_determinism.py` relies on. Closing it for a
  long recording means a replay that awaits each handler before pulling the next quote, which is a
  second path through `Pipeline` and would no longer be reproducing the engine that ran.
- **A replay runs without the heartbeat, and `cli._without_heartbeat` is where that is decided.**
  Under a `SimulatedClock` nothing but the recording moves time, so the timer racing the stream
  either never fires or spins — and whether it fired at all would depend on how the event loop
  interleaved two tasks, which is the non-determinism the command exists to remove. The cost is
  real and unmeasured: the snapshots the heartbeat emitted during the original session are not
  reproduced by its replay, so a recording of a market that went quiet replays as a shorter
  sequence of snapshots than it produced. Closing it means recording the heartbeat's own occasions
  as events in the file, which is a second line kind and a second thing to keep in step.
- **A provider's settings are a named field, one per adapter.** `MarketConfig.synthetic` names the
  one adapter it configures, and a second provider with settings gets a second field rather than a
  shared untyped bag. That keeps every value validated where the error can name its table, and it
  does mean the type grows by one optional field per configurable adapter — the alternative, a
  `Mapping[str, Any]` handed to the factory, was refused because it moves parsing into
  `*/adapters/` and takes the table name out of the message.
- **Neural Surface is not wired into the pipeline.** `build_pipeline` runs Market Data to
  Parametric Pricing to Risk; `TrainOnSnapshot` is built and tested but nothing constructs one,
  and `AppConfig` has no section for the replay buffer, the arbitrage mesh, the gate thresholds,
  the restart schedule or the seed it would need. Deliberate: its learner is torch, an optional
  extra that arrives in F3-C, and five thresholds nothing in this build can exercise would be
  five numbers chosen by guesswork. `--calibrators` therefore selects among parametric producers
  only, which is why its help line does not repeat Design 8.1's `svi,neural` example.
- **The shipped examples' position expiries are fixed instants (2027-06-25) that will rot.** Both
  `examples/walking-skeleton.toml` and `examples/synthetic-svi.toml` carry one. TOML has no "N
  months from now" literal, while the feeds' tenors are relative to start-up, so a file can only
  pin an absolute date inside the window those tenors currently cover. Past it, valuation refuses with `ExpiredInstrumentError` instead of interpolating
  a value — a loud failure, not a silent one, but a maintenance date nothing enforces. Each file's
  own comment says so and the tests guarding them deliberately assert only that the config loads
  and names registered adapters, never the date. Closing this for good means either a config field
  expressed as an offset from start-up (which the composition root would resolve against
  `SystemClock`) or accepting the periodic bump as the cost of a literal example file.

## Cross-cutting

- **`_require_positive_finite` is written out in five domain modules.** The same argument as
  ADR-014 one level down, deliberately left for its own change. `require_aware` lived in six before
  it earned a name.
- **Rule 5 of the import table has no mechanical guard.** import-linter sees imports, and "only
  `*/application/acl.py` builds or consumes `contracts/` DTOs" is about construction: every use
  case imports `contracts.events` legitimately, because ADR-016 has it return events. The other
  seven rules became contracts in F1-09; this one stays a review rule, and the
  `[tool.importlinter]` comment in `pyproject.toml` says so beside the seven that did not.
- **The library bans are deny lists, not allow lists.** Rules 1, 2, 3 and 7 are `forbidden`
  contracts naming `jax`, `torch`, `scipy` and `numpy`, so they catch the dependencies the design
  argued about and would not notice a brand-new third-party import appearing in the domain.
- **The vectorised JAX Black-76 cannot live in the shared kernel** (rule 1, ADR-011). It is a
  genuine reimplementation -- batched and differentiable -- and has lived in
  `parametric_pricing/adapters/jax_black76.py` since F3-A, tested against the kernel as its oracle,
  deep wing included. The duplication itself stands and is the only copy of the formula left.
