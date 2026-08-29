# Known seams

Gaps that are **open on purpose**, each written into the docstring that owns it.

This is not an ADR list and deliberately not in `docs/adr/`. An ADR is immutable: it records a
decision that was taken. These are conditions that are still true, and the file shrinks as they are
closed. Read it before "fixing" something here — every item was seen, weighed and left, and several
close naturally in a later phase.

## Market Data

- **`max_quiet_seconds` does not cover a dead feed.** `IngestStreamUseCase` evaluates the snapshot
  policy only when an update *arrives*, so the heartbeat covers a market that is quoting but not
  moving, and not one whose feed has stopped. Closing it needs a timer racing the stream, which is
  the two-task structure of F1-07.
- **`ChainSnapshot.ts_exchange` is not clamped to `ts_local`**, so a venue clock running ahead would
  make `CalibratedSurface` reject `ts_calibrated < ts_snapshot`. The ACL reconciles them
  (ADR-021); the domain type still permits the state.
- **`IV_DIVERGENCE` never fires.** `flag_quote` receives `own_iv=None` from the chain, and will
  until Market Data grows its own mid inversion in F2 — a wrapper over the shared kernel since
  ADR-014.

## Parametric Pricing

- **`SVIParams` admits `min_total_variance == 0` while `durrleman_g` refuses it.** `g` divides by
  `w`, and both `inf` and `nan` survive `max(0, -min(g))` as a clean zero, which would report a
  collapsed slice as arbitrage-free. Revisit in F2-05 only if the soft penalty ever trips on it.
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
  publishes a `market_id` and no underlying. Pairing a book with the right market is the use case's
  job.
- **`FreshnessPolicy.evaluate` takes two bare instants**, so nothing structurally stops a caller
  passing `ts_calibrated`. The ADR-006 argument that it must be `ts_snapshot` is defended in the
  docstring only.
- **`PositionRisk` has no invariant tying `value` or the greeks to `position.quantity`** — it cannot
  know the option's own worth — so the quantity-scaling convention is upheld by `valuation.py`
  alone.
- **Bilinear interpolation does not inherit no-arbitrage** between nodes. Past the last tenor a flat
  total variance means the implied vol decays as `1/sqrt(T)`: a two-year option off a one-year grid
  is priced at about 71% of the one-year vol.
- **Risk's numerical greeks carry the grid's kinks** as well as the bump's truncation error. The
  at-the-money gamma of the test surface is about twice the analytic value — a fact Design §7.4
  wants visible, not a defect.

## Cross-cutting

- **`_require_positive_finite` is written out in five domain modules.** The same argument as
  ADR-014 one level down, deliberately left for its own change. `require_aware` lived in six before
  it earned a name.
- **The vectorised JAX Black-76 of F3-A cannot live in the shared kernel** (rule 1, ADR-011). It is
  a genuine reimplementation and will be tested against the kernel as its oracle.
