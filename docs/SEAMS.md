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

## Parametric Pricing

- **`SVIParams` admits `min_total_variance == 0` while `durrleman_g` refuses it.** `g` divides by
  `w`, and both `inf` and `nan` survive `max(0, -min(g))` as a clean zero, which would report a
  collapsed slice as arbitrage-free. The soft penalty of F2-05 does trip on it, at iterates the
  optimiser walks through: `adapters/scipy_calibrator._residuals` answers the raise with a barrier
  residual rather than letting one bad step kill a whole surface. The asymmetry itself stands.
- **The scipy calibrator's tuning is not configuration yet.** `FitSettings` — Huber scale,
  butterfly penalty, ridge, penalty mesh, pinning threshold, evaluation budget — arrives through
  the constructor with defaults, and `entrypoints/config.py` has no section that feeds it. F2-07
  owns the TOML; a section written here would have been thresholds no file reads, which is the
  guesswork ADR-012 exists to prevent. The bounds behind `at_bound` are deliberately *not* part of
  it (ADR-021).
- **`test_durrleman.py` imports the private `_curve_and_derivatives`**, with no precedent in the
  repo. A sign slip in `w'` still yields a plausible `g`, so the derivative has to be pinned against
  an independent computation rather than only through the function that consumes it.

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
- **`CsvReportWriter` has no TOML home.** `RiskConfig.writer` names an adapter but carries no
  output path, and `WriterFactory` receives the risk configuration and nothing else, so
  `default_adapters()` registers `console` alone and the CSV writer is constructed by tests only.
  F2-07 owns the wiring, on the same terms as `FitSettings` and `SyntheticConfig`: a path field
  added before there is a factory to read it would be configuration nothing loads (ADR-012).
- **Risk's numerical greeks carry the grid's kinks** as well as the bump's truncation error. The
  at-the-money gamma of the test surface is about twice the analytic value — a fact Design §7.4
  wants visible, not a defect.

## Entrypoints

- **An adapter cannot be handed the engine's `Clock`.** `ProviderFactory` receives a
  `MarketConfig` and nothing else (ADR-022), so `ConstantProvider` stamps its quotes off the wall
  clock instead. Harmless here — a venue stamps its own messages and `max_skew_seconds`
  reconciles them — and it stays harmless as long as the deterministic replay of ADR-004 arrives
  as `RecordedProvider` in F3-B, replaying recorded instants rather than asking a synthetic feed
  to read a different clock. A synthetic *generator* that must be reproducible (F2-03) is the case
  that would reopen this: it needs either a fourth argument on the factories or a seed in its own
  configuration section.
- **Neural Surface is not wired into the pipeline.** `build_pipeline` runs Market Data to
  Parametric Pricing to Risk; `TrainOnSnapshot` is built and tested but nothing constructs one,
  and `AppConfig` has no section for the replay buffer, the arbitrage mesh, the gate thresholds,
  the restart schedule or the seed it would need. Deliberate: its learner is torch, an optional
  extra that arrives in F3-C, and five thresholds nothing in this build can exercise would be
  five numbers chosen by guesswork. `--calibrators` therefore selects among parametric producers
  only, which is why its help line does not repeat Design 8.1's `svi,neural` example.
- **`examples/walking-skeleton.toml`'s position expiry is a fixed instant (2027-06-25) that will
  rot.** TOML has no "N months from now" literal, while `ConstantProvider`'s three tenors are
  relative to start-up, so the file can only pin an absolute date inside the window those tenors
  currently cover. Past it, valuation refuses with `ExpiredInstrumentError` instead of interpolating
  a value — a loud failure, not a silent one, but a maintenance date nothing enforces. The file's
  own comment says so and the test guarding it deliberately asserts only that the config loads and
  names registered adapters, never the date. Closing this for good means either a config field
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
- **The vectorised JAX Black-76 of F3-A cannot live in the shared kernel** (rule 1, ADR-011). It is
  a genuine reimplementation and will be tested against the kernel as its oracle.
