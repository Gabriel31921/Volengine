# ADR-020: Domain API departures from `Implementation.md`

**Status:** Accepted · 2026-08 · taken while building F1-05, all four contexts

## Context

`Implementation.md` specifies the domain layer module by module, down to signatures. Building it
found places where the specified signature could not carry its own weight. Each departure below was
deliberate and is defended in the docstring that owns it; this record exists so that the *set* of
them is visible in one place, and so that nobody "restores" one to match the plan.

The larger shape decisions taken in the same period have their own records: ADR-014 (Black-76 in
the shared kernel), ADR-015 (`VolGrid`), ADR-016 (handlers, not loops), ADR-017 (the OTM twin),
ADR-018 (weights), ADR-019 (the learner mirrors the calibrator).

---

## `market_data/`

- **`QuoteChain.set_live_instruments(full_set)`** instead of an added/removed delta. ADR-013's
  argument — a composition event carries state, not a diff — applies to the method as much as to
  the event.
- **`max_relative_move_since_baseline()` + `reset_move_baseline()`** instead of
  `max_relative_move_since(ts)`, which would need an unbounded history to answer.
- **`QuoteChain.apply(update)` takes no `now`.** Judgement happens in `snapshot()`; applying an
  update is bookkeeping and needs no clock.
- **`flag_surface` is declared and returns `()`.** The calendar check needs interpolation and
  belongs to `durrleman.calendar_violation`, which is another context's F2-04.
- **`SnapshotPolicyConfig.max_quiet_seconds` added** as a heartbeat. A strict clock-AND-movement
  rule keeps a calm market silent, and downstream cannot tell silence from a dead process.

## `parametric_pricing/`

- **`OptionKindP` is declared in `black76.py`**, a spelling of "option side" of its own next to
  `OptionKindD`, `OptionKindN`, `OptionKindR` and the contract's `OptionKind`. Rule 6 forbids
  importing another context's and rule 3 the contract's; ADR-014 keeps the vocabulary out of the
  shared kernel too.
- **No `PricingEngine` port**, though Design §5.1 lists one. Its only purpose is the AD greeks of
  §5.8, which are JAX and arrive in F3. A port with no implementer and no consumer is speculation.
- **Pricing's `Clock` declares only `now()`, no `sleep()`.** This context reacts to snapshots on the
  bus and never waits. That divergence is the concrete argument for declaring a port per context
  instead of sharing one.
- **`VOL_TOLERANCE` is a public module constant and explicitly not TOML config.** ADR-012 is for
  business thresholds; this is a property of the root-finder, which is also the first admission
  test of ADR-014 and why it went to the kernel alongside the solver.
- **`implied_vol` never raises on non-convergence.** It brackets and bisects, so termination is a
  property of the construction. An iteration budget sized above the bisection worst case is a guard
  against a coding error, not a give-up point.

## `neural_surface/`

- **`TrainingSample` / `TrainingBatch` live in `training_batch.py`**, mirroring `calibration.py`,
  rather than inside `replay_buffer.py`.
- **`TrainingSample` stores `implied_vol` and derives `total_variance`.** One source of truth: the
  network's *output* is variance (§6.2), but a quote is *observed* as a vol.
- **`ArbitrageReport.exceeds(butterfly_tol, calendar_tol)`** puts the rule in the domain and the
  thresholds in the argument list — ADR-010 and ADR-012 satisfied at the same time.
- **`ReplayBuffer.prune(now)` checks the `Clock` for monotonicity against the previous prune**, not
  `now` against the held samples. The second reading would fire on the venue skew the `ts_exchange`
  seam leaves open and hard-fail the training loop, while what it meant to catch is a backwards
  clock in the composition root.
- **The hard gate differences numerically** where `parametric_pricing/durrleman.py` is analytic,
  and owns the resulting truncation error: an MLP has no closed form, and its autodiff lives inside
  a framework rule 3 bars.

## `risk/`

- **`discount` is a parameter defaulting to `1.0`, not a contract field.** No producer computes a
  discount factor today, so a field the ACL would fill with `1.0` is a lie with a schema — and the
  factor is multiplicative, so it cancels in every comparison of §7.3.
- **`SurfaceView` holds total variance, not vols**, converted once in the ACL, because the
  interpolation is bilinear in `w`.
- **`SurfaceView` has no status field.** ADR-006 republishes a stale surface with its *original*
  `ts_snapshot`, so staleness is already visible in the one timestamp the freshness policy reads. A
  second channel for one fact only ever disagrees with the first.
- **`forward_at` is linear in `ln(F)`** — constant carry between nodes — clamped flat outside.
- **`RiskReport.total_value` is a property using `math.fsum`, not a field.** One source of truth,
  and a book cancels long against short by design, so pairwise addition would make the total depend
  on the order of the lines.
- **`RiskReport.ts_snapshot` is `datetime | None`**, because "the provider had nothing" genuinely
  has no instant, unlike "the surface is old".
- **The REJECT invariants are enforced in `__post_init__`** — no positions, and a message. A
  constructor is where an object refuses to exist; a use case is where a branch gets added under
  deadline.
- **`BumpSpec` has no defaults** (ADR-012), and bounds `forward_rel` strictly below 1, or the
  down-bumped forward reaches zero and `ln(K/F)` dies three frames down.
- **Vega divides by the actual span `vol_up - vol_down`**, not `2e`, so a clamped down-bump reports
  a quotient over the interval it was really taken on.
- **The vol floor is a fraction of the vol**, not an absolute number, because an absolute one can
  sit above the up-bump and return a vega with the wrong sign.
- **Greeks are sticky-moneyness**, named and measured: on the test surface a call 15% out of the
  money is 0.360 against 0.380 sticky-strike.
- **`SurfaceProvider` carries no `producer_id`**, unlike `Calibrator` and `SurfaceLearner`. A
  provider is not a producer, the composite of §7.3 serves several, and identity travels in the
  answer as `SurfaceView.producer_id`.

## Consequences

- `Implementation.md` is now behind the code in these places, and the code is right. Read this
  record before treating a signature in that document as authoritative.
- Every item above is also stated in the docstring of the module that owns it, which is where an
  agent working in that file will meet it.
