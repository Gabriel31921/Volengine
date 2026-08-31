# ADR-029: `JaxCalibrator`'s departures from the plan in F3-A

**Status:** Accepted · 2026-08 · taken while building the second calibrator

## Context

ADR-020, ADR-021, ADR-022, ADR-023, ADR-025, ADR-027 and ADR-028 collect the departures of the
phases that produced them. F3-A produces its own and gets its own record on the same terms:
`CLAUDE.md` says a phase's departures are collected in an ADR, and that a later phase's departures
get a later record rather than an edit to an earlier one, which would misdate a decision and make
that record's own heading false.

Two of the items below are not departures at all in the usual sense. `instructions/STATE.md` and
ADR-027 explicitly *handed this stage* the question of what `n_iterations` reports, and ADR-024
explicitly anticipated the CI change; they are recorded here because a delegated decision that is
taken and never written down is indistinguishable from one nobody noticed.

The rest have one shape in common. `Design.md §5.5–5.8`, `Plan.md:342-360` and
`Implementation.md:979-1000` describe this stage in terms of *what the fit must achieve* — a fixed
shape with a mask, one compilation at start-up, Adam warm and L-BFGS cold, greeks by AD — and every
one of those claims holds in the code. What moved is the machinery underneath, in the four places
where a gradient method does not tolerate a choice a trust-region method tolerated. A calibrator is
not a formula; it is a formula plus the geometry the optimiser walks over, and that half could not
have been settled by a document written before either producer existed.

---

## `optax.lbfgs`, not `jaxopt`

`Design.md §5.6`, `Plan.md:346` and `Implementation.md:989` all name **jaxopt** for the cold
cycle's multi-start L-BFGS, and `Plan.md:85` (D-4) lists `jaxopt` among the dependencies to add.
The code uses `optax.lbfgs` with `optax.scale_by_zoom_linesearch`.

jaxopt is deprecated: its solvers were folded into optax, and it is not maintained against current
JAX releases. It is also **not in this repository's optional extra** — `pyproject.toml:14` declares
`jax = ["jax>=0.11.0", "optax>=0.2"]` and never carried jaxopt, which means the disagreement
predates this stage and was already resolved in the committed configuration. `CLAUDE.md` says the
repository's own artefacts win over the planning documents where they disagree; the extra is one of
those artefacts and this record is where the discrepancy stops being silent.

Nothing the design asked for changes. The cold cycle is still a deterministic multi-start
quasi-Newton search with a linesearch, one instance per slice under `vmap`, and the *reason* the
design gave for it — escaping the basin a warm start is stuck in — is what
`test_a_warm_start_from_a_stale_basin_is_retried_cold` asserts.

## The search space is the minimum total variance, not `a`

`Design.md §5.4` describes the reparametrisation as softplus for `b` and `sigma`, tanh for `rho`,
and leaves `a` free; `domain/svi_slice.FreeParams` implements exactly that and the scipy baseline
searches in it. This adapter searches `(w_min, b, rho, m, sigma)` instead, with
`w_min = softplus(x0)` the *minimum* of the curve and `a = w_min - b·sigma·sqrt(1 - rho²)`
recovered from it — the shared kernel's own closed form for that minimum, inverted.

The reason is that `FreeParams` leaves a region of R⁵ where the slice is not a surface: the one SVI
constraint the reparametrisation cannot make structural is a non-negative minimum total variance,
because it couples `a` with the other four. ADR-027 answers that in the baseline with `BARRIER_BP`,
a large finite residual, and argues correctly that a trust region is *supposed* to walk through
inadmissible points and have its step rejected. A gradient method has no step rejection. Adam steps
into the region, `jnp.sqrt` of a negative variance is a NaN, and — because the whole surface is one
batched call — that NaN reaches every parameter of every slice in the same update. The failure is
silent in the worst way: the fit finishes, the numbers are shaped like an answer, and none of them
came from the market.

In the space actually used, **every point of R⁵ decodes to a valid `SVIParams`**. There is no
barrier, no clamp on the way out, and no iterate at which the model cannot be evaluated. The port
already sanctions this — it carries `SVIParams` across the boundary and says in as many words that
the reparametrisation is an implementation's business — and what crosses the boundary is
identical for both producers, which is the only property the comparison of Design 5.7 needs.

Two consequences had to be paid for explicitly, and both are argued in the docstrings that own
them:

- **The ridge is measured in the baseline's coordinates** (`_drift`). A softplus preimage
  compresses small values logarithmically — at a total variance of 0.03 its derivative is about
  thirty — so `ridge_bp`, a tie-breaker in the baseline's coordinates, would be a hundred and fifty
  basis points per unit of variance in this adapter's and would drag every fit back towards its own
  starting level. The level's drift is therefore measured decoded, in total-variance units, and the
  other four coordinates are the baseline's already. Rescaling the *search* coordinate instead was
  tried and rejected: it fixes the ridge and wrecks the conditioning, because the level's gradient
  then dwarfs the other four and the linesearch spends itself on one direction.
- **The practical bounds are imported from `scipy_calibrator`, not restated.** ADR-027 argues that
  a bound is not a threshold a deployment tunes but the *ruler* `at_bound` is read against; two
  producers judged by two rulers would be two acceptance rules wearing one name, which is half of
  ADR-006 deleted without anyone editing ADR-006. The search here is unconstrained, so the test is
  "at or past the bound" where the baseline's is "stopped at it" — the same statement about the
  same region: the optimiser wanted to leave the region a publishable fit lives in.

## The loss is the baseline's residual vector, not `§5.5`'s literal expression

`Design.md:320-321`, `Plan.md:344` and ADR-009 all state the masked loss as
`where(mask, err², 0).sum() / mask.sum()`. `_objective` computes

    sum(where(mask, huber(sqrt(n · weight_i) · err_i), 0))  +  penalty block  +  ridge block

and recovers the interpretable quantity in `_root_bp` as `sqrt(cost / n)`.

The difference is arithmetic rearrangement in two places and a real addition in a third.

- **The division by `mask.sum()` is folded into the weights and taken back out at the end.**
  ADR-018's weights arrive normalised to one, so `sum(weight_i · err_i²)` *is* the masked mean of
  §5.5; multiplying by `n` puts each residual back on the scale of an individual error, and
  `_root_bp` divides by `n` again to read the objective as an RMSE in basis points. The minimum is
  the same in every one of these forms. What is not the same is what a *tuning constant* means:
  without the `n`, one `huber_scale_bp`, one `ridge_bp` and one `durrleman_penalty_bp` would be
  different trade-offs on a twelve-strike slice and on a forty-strike one, and different again
  between this producer and the baseline. ADR-027 made the same argument for the same factor on the
  scipy side; §5.5's expression predates both.
- **`err²` is `huber(err)`** — Design §5.3 asks for Huber and §5.5's sketch of the mask does not
  mention it. There is no conflict; the two sentences are about different things.
- **Huber applies to the butterfly and ridge blocks too**, which neither document states either
  way, and it is load-bearing rather than tidy. `scipy.optimize.least_squares(loss="huber")`
  transforms *every* residual it is given, so the baseline has always charged a deep butterfly
  violation linearly. Squared, a penalty of ten thousand basis points per unit of violation
  dominates every quote on the slice as soon as the true smile has curvature the wings cannot hold,
  and the fit walks away from the market to buy a fraction of a percent of density — measured at
  five basis points of RMSE where the baseline reached two tenths. The two producers minimise one
  function, residual for residual, up to a constant factor of two that does not move a minimum, and
  that is the condition under which Design 5.7 compares two *methods* rather than two problems.

The mask itself is untouched and is asserted in both of ADR-009's forms: values under it may be
perturbed anywhere within a thousand units without moving the objective or either search, and
`padding.py` guarantees what no `where` can — that the padded cells are *finite*, since the
gradient of a `where` multiplies the untaken branch by zero and `0 · nan` is `nan`.

## Single precision, with every reported number rebuilt in float64

The fit runs in `float32`, JAX's default, and this adapter does not enable `jax_enable_x64`:
that flag is process-wide global state mutated by an import, which is not an action an optional
adapter may take on behalf of a whole engine. Everything else in this repository is float64, so the
seam is real and is confined rather than hidden.

It is confined by recomputing every *reported* number on the host: `_to_params` decodes the five
parameters in float64, and `_metrics` computes the RMSE, the maximum error and the quote count
through the domain's own `SVIParams.implied_vol`. The baseline's metrics are computed the same way
from the same code, so a difference between the two producers' RMSEs is a difference between the
fits and not between two dtypes. It also protects an invariant: `SliceResult` refuses
`max_err_vol_bp < rmse_vol_bp`, and single precision can round those two the wrong way round on a
slice where every error is the same size.

What single precision costs is about seven significant digits, three orders of magnitude below the
uncertainty of any fit, and one boundary worth stating: in `jax_black76` a premium below roughly
1e-38 underflows, which on a two-week option is a strike some twelve standard deviations out and
worth 1e-32. The wing that matters is intact — the `erfc` form of the normal CDF is what keeps it,
in this precision as in any other — and the tolerances in the tests are written at float32's scale
rather than float64's, with an explicit `abs=` on every deep-wing assertion.

## `n_iterations` carries objective evaluations, summed over the slices fitted

ADR-027 changed this field's unit on the scipy side to residual evaluations and closed with: *"F3-A
inherits the comparison. A JAX calibrator has a real iteration count, so Design 5.7's
scipy-versus-JAX table is only meaningful if both producers report the same quantity. Whichever is
chosen there, it is a decision that stage has to take deliberately."* `instructions/STATE.md` says
the same. This is that decision.

**Objective evaluations**, not Adam steps: one per hot step, and one per linesearch evaluation in
the cold cycle, where several go into a single L-BFGS iteration. Summed over slices, which is the
same population the baseline sums `nfev` over, and excluding the padded lanes, whose searches are
an artefact of the fixed shape — charging for them would make the reported cost depend on how much
margin the grid was reserved with rather than on the market.

Iterations were the more flattering number and would have compared a step against an evaluation:
this producer's hot cycle spends about two hundred evaluations where the baseline's warm fit spends
three, and reporting "one hundred iterations" against "three evaluations" would have hidden the
factor rather than shown it. The field is a measurement, and the measurement is only worth having
if it is the same quantity on both sides.

## Two modules more than the plan names, and two mechanisms it could not have named

`Implementation.md:981` names `jax_calibrator.py` and `padding.py`. Four modules ship:
`jax_black76.py` and `jax_greeks.py` are the other two, and `Plan.md`'s own acceptance list for
this stage is what requires them — the AD greeks of Design §5.8, and the vectorised Black-76 the
plan itself calls "the one remaining copy of the formula". Neither belongs inside a calibrator that
prices nothing: the fit works in `(k, w)` space on volatilities Market Data already inverted.

Two mechanisms inside the search are in no document because neither is visible until a fit is run:

- **A search reports the best point it visited, not its last iterate** (`_keep_best`). Adam's step
  size is set by its own moment ratio and not by the magnitude of the gradient, so at an optimum —
  which is exactly where a warm start begins in a calm market — it keeps stepping about a learning
  rate away and settles wherever the tolerance catches it. Without this, a hot cycle handed
  yesterday's converged parameters returns a *worse* fit than it was given, every snapshot, and the
  surface drifts with no market behind the movement. L-BFGS with a linesearch is monotone and does
  not need it, which is why both cycles can answer the same question.
- **Convergence is a run of `PATIENCE` quiet steps, not one** (`_quiet_after`). A quasi-Newton
  method takes small steps while it is still building curvature information and a gradient method
  takes small steps whenever it crosses a narrow valley; either produces a single step whose change
  in RMSE is below any sensible tolerance while the fit is still tens of basis points away. Three
  in a row costs two extra evaluations on a converged slice and is the difference between a cold
  cycle that lands on the market and one that stops in the first ditch.

The same empirical pass moved `tolerance_bp` to a thousandth of a basis point. It is tighter than
it looks like it needs to be, and the reason is the one above: at a twentieth of a basis point both
searches stop while still crawling and the fit lands ten to twenty basis points out, which the
acceptance rule then rejects for a reason that has nothing to do with the market.

## A chain wider than the reservation is refused, not truncated

`padding.pad` raises `CalibrationError` naming the number the grid would need. Truncating would
drop quotes the venue really published, silently, and the surface would still be released as
healthy; refusing makes the use case republish the last good one (ADR-006). ADR-009 foresees the
other half — handling `ChainCompositionChanged` so the reservation follows a growing chain — and
that event is first produced in F3-C. The gap is in `docs/SEAMS.md`.

## Consequences

- **`Plan.md`, `Implementation.md` and `Design.md` are not amended.** A document and an ADR
  disagreeing is what this record exists to resolve, and `CLAUDE.md` says the ADR wins. In
  particular `Design.md §5.5`'s expression for the loss and §5.6's `jaxopt` are superseded here,
  not corrected there; §5.5's *decision* — fixed shape, boolean mask, one compilation — stands
  untouched, and ADR-009 is not superseded by any of this.
- Each departure is also argued in the docstring that owns it, in
  `parametric_pricing/adapters/{jax_calibrator,padding,jax_black76,jax_greeks}.py`; this record is
  where `CLAUDE.md` says they must also be collected.
- **`ci.yml` grew a matrix leg rather than a second workflow**, which is the shape ADR-024 said it
  would take when a calibrator first needed an optional extra. The bare leg is the standing promise
  that the engine runs on numpy and scipy alone — the JAX test modules are not collected there and
  the contract harness runs one implementer fewer — and the `jax` leg is the only place this code
  is exercised at all. Both must pass. One side effect is worth naming: the import-linter contract
  forbidding `jax` in the domain layer is only genuinely enforced on the leg where jax is
  installed, and until now there was no such leg.
- **The contract harness gained an implementer and not a line of test body**, which was its own
  stated bar for what "interchangeable" means. It is conditional on the extra, so the set is one
  shorter on a machine without it — a statement the harness makes for itself.
- **The benchmark of Design 5.7 exists and does not say what the design assumed.** On a
  twenty-one-strike slice on CPU the baseline's trust-region step is far cheaper per unit of
  progress (6 ms and 19 evaluations against 38 ms and 584); the two hot cycles are level (about
  1.4 ms each); and on a twelve-expiry chain the JAX cost is flat where the baseline's grows with
  the number of slices (821 ms against 71 ms cold, 18.8 ms against 18.4 ms warm). Fit quality is
  comparable throughout — hundredths of a basis point on clean quotes. The honest summary is that
  the padding buys shape-independence and a GPU path, not a CPU speed-up at this size, and the
  numbers are in `tests/parametric_pricing/test_jax_vs_scipy.py` where they are re-measured on
  every run rather than quoted from here.
- **`svi-jax` is not registered in `default_adapters` and has no TOML section.** F3-A stayed inside
  the modules its own task names; wiring it needs a lazily-importing factory and a
  `[calibration.…]` table of its own, on the terms ADR-028 sets out. It is in `docs/SEAMS.md`, with
  the rest of what this stage deliberately left open.
