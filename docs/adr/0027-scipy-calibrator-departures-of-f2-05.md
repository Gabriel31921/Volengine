# ADR-027: `ScipyCalibrator`'s departures from the plan in F2-05

**Status:** Accepted · 2026-08 · taken while building the first real calibrator

## Context

ADR-020, ADR-021, ADR-022, ADR-023 and ADR-025 collect the departures of F1-05, F1-06, F1-07,
F1-08 and F2-03. F2-05 produced its own and gets its own record on the same terms: `CLAUDE.md`
says a phase's departures are collected in an ADR, and that a new stage's departures get a new one
rather than an edit to an old one, which would misdate a decision and make that record's own
heading false.

The shape of the disagreement is different from ADR-025's, and worth naming. `Plan.md:275-289` and
`Implementation.md:812-822` describe F2-05 as a list of *ingredients* — `least_squares` over the
free variables, a vega-weighted volatility error, ADR-018's weights, Huber, a soft Durrleman
penalty, and either a pull towards a neighbouring slice or pinning for thin ones — under a single
signature that is the `Calibrator` port itself. Nothing in the code contradicts that list. What the
code adds are five mechanisms the list does not name, each of which exists because the fit does not
survive real data without it, plus one field of a domain type that now carries a different quantity
than its own docstring described.

An ingredient list is not wrong for being incomplete, so none of this is a correction of the plan.
It is the record of what "empirical tuning against F2-03" — the plan's own words for this stage —
actually turned out to require, which is precisely the thing a document written before the fit
existed could not contain.

---

## A Tikhonov ridge towards the starting point

`FitSettings.ridge_bp`, five basis points per unit of free coordinate, applied only to the
coordinates actually being searched.

The failure it answers is invisible until it happens. **A market with no smile has no `rho`, no
`m` and no `sigma`.** As `b` goes to zero the curve is `w(k) = a` whatever the other three say, the
cost surface becomes a plateau in those three directions, and the optimiser drifts along it until
it reaches the side of the box — which `at_bound` then reports honestly, and ADR-006 refuses a fit
whose RMSE was perfectly healthy. Measured on the flat chain with twenty basis points of noise:
five seeds in six without a ridge, one in twenty at `ridge_bp = 1.0`, none at the default, while
the parameters recovered from a clean smile move in the sixth decimal and the fitted RMSE stays
under a thousandth of a basis point.

It is a tie-breaker and deliberately not a prior: a whole unit of free coordinate costs what five
basis points of vol error costs, so a fit that is genuinely determined moves as far as it likes.
Its second effect is the one a streaming engine wants anyway — under a warm start the tie breaks
towards *yesterday's* parameters, so consecutive surfaces do not jump between equally good answers
on floating-point noise.

Zero is legal and turns it off, which is how the test that shows what it prevents is written.

## Residuals scaled by `sqrt(n * weight_i)`

ADR-018's weights arrive normalised to one, so `sum(w_i * e_i^2)` is a weighted *mean* square.
Without the `n` the individual residuals shrink as the chain widens, and
`FitSettings.huber_scale_bp` — "an error of this many basis points stops being ordinary noise" —
would silently be a different robustness rule on a twelve-strike slice and on a forty-strike one.
The scaling puts each residual back on the scale of an individual error, which is the only reading
under which the setting means what its name says.

The penalty block is divided by `sqrt(nodes)` for the same reason in the other direction:
refining the mesh should measure a violation more finely, not charge more for it.

Neither factor changes the minimum. Both are what make the *tuning* constants portable, which is
what F2-07 needs before it can put them in a TOML file that one operator writes for several
markets.

## A cold retry of a failed warm start

Design 5.6 gives multi-start to the JAX cold cycle and the plan repeats it there; this adapter has
no multi-start. What it does instead is the other half of Design 5.6's own rule — a cold cycle
"periodic or after a failure" — at the scale a scipy baseline can afford: if a warm-started fit
comes back unconverged or pinned, it is retried once from the cold guess read off the quotes, and
`_preferred` keeps the better of the two.

The cost is bounded and lands where it should. A cycle that was going to be accepted pays nothing;
a cycle that was going to be *rejected* pays one more fit, which is the cheapest possible answer to
the specific way a warm start fails — the previous cycle's parameters sitting in a basin the market
has since left, from which the search runs out of budget or walks into a bound and reports exactly
that.

`_preferred` prefers a converged, unpinned fit over a lower RMSE, which is the same argument
ADR-006 makes one layer up: the residual of a projection onto a bound is not comparable to the
residual of a fit, and comparing them on RMSE alone would pick the projection whenever the bound
happened to sit near the data.

## A finite barrier instead of a propagating exception

Every point of R^5 has to produce finite residuals, and two do not on their own: an `a` deep enough
to make the minimum total variance negative, which `SVIParams` refuses outright, and a slice whose
total variance touches zero somewhere on the mesh, which `durrleman_g` refuses because `g` divides
by `w` — the asymmetry recorded in `docs/SEAMS.md`. Both are answered with `BARRIER_BP`, a residual
so large that a trust-region step into that region is always rejected.

Three properties of that choice are load-bearing:

- **Finite, not infinite.** `least_squares` refuses a non-finite residual outright, and a NaN is
  the failure this repository keeps rediscovering: it does not raise, it makes the fit stop
  somewhere arbitrary.
- **Returned, not raised.** The alternative is one inadmissible iterate of one slice killing a
  whole surface — a `CalibrationError` where the honest answer is a rejected step. The optimiser
  is *supposed* to walk through inadmissible points; that is what a search is.
- **Sloped, not flat.** The barrier grows as `a` falls, so the finite-difference Jacobian at an
  inadmissible point still points back towards the admissible set instead of facing a wall with no
  visible far side.

The one place an exception does escape is the point the optimiser *returns*: if that cannot be
turned back into `SVIParams` the adapter raises `CalibrationError`, because a silent fallback there
would publish a slice nobody fitted. It is unreachable in principle — the barrier makes any
inadmissible point cost more than the admissible `x0` the search began at, and `trf` never returns
a point worse than the one it started from — and it is reported rather than papered over.

## The loss is injected; the box is not

`FitSettings` — Huber scale, penalty weight, penalty mesh, ridge, pinning threshold, evaluation
budget — arrives through the constructor, because every one of those numbers is empirical and will
move against real Deribit data. The *bounds* are module constants, and that asymmetry is the
decision.

`at_bound` is one of the two halves of ADR-006's acceptance rule, and ADR-021 says that half is not
negotiable. A bound is not a threshold an operator tunes against their market: it is the ruler the
acceptance check is read against, so a deployment free to widen a bound until nothing is ever
pinned is a deployment that eventually does — and the effect is not a looser rule, it is half of
ADR-006 deleted silently, with the surfaces still published under a rule that reads unchanged in
the TOML. The bounds are chosen to be *practical* in Design 5.4's sense: never binding on a real
slice (`b = 4` is not a smile, `|m| = 2` is outside any quoted band), and unmistakable when they
bind.

`B_START_MIN` is on the other side of the same line and is not a bound at all: `softplus'`
underflows to zero for a very negative `b_raw`, so a fit started exactly at `b = 0` has a zero
Jacobian column for the wings and can never grow them, whatever the market says. It is a floor on
the *starting point*, applied to a warm start for the same reason as to a cold one, and the
optimiser is free to walk straight back down — which it does on a genuinely flat chain.

The seam that follows is that `FitSettings` has no TOML home until F2-07, which owns wiring.
Inventing a section here that no file reads would be the guesswork ADR-012 exists to prevent, and
it is recorded in `docs/SEAMS.md` rather than left to be rediscovered.

## Thin slices are pinned, not pulled

`Implementation.md:820-822` offers ADR-008's two answers to risk 3 of Design 11 — a penalty towards
the neighbouring slice's parameters, or pinning `m` and `sigma` — and the code takes the second.
Pinning is the one that keeps `calibrate` a per-slice function: a pull towards a neighbour is a
coupling this adapter would have to invent, against ADR-008's own decision to fit each expiry
independently. Below `min_quotes_for_free_shape` quotes the level, wings and skew are still fitted
and the two shape parameters keep whatever the start gave them.

`_at_bound` then tests only the *searched* coordinates. A pinned `m` sitting on a bound never moved
there, and reporting it as pinned would make ADR-006 refuse a fit for a reason the rule does not
name: it refuses a fit that was **stopped** at the edge of the admissible region, not one that was
held at a value chosen before the search began.

## `duration_ms` is zero, and deliberately not measured

Following ADR-023, for the reason that record already gives: reading a clock is the one thing the
`Calibrator` port forbids outright, and it would make two runs over the same recording differ. The
number a consumer sees is the use case's own measurement of the call, taken from the injected
`Clock`, so nothing is lost.

## `n_quotes_used` and `max_err_vol_bp` count the quotes that entered the fit

Both are computed over the strictly positive-weight quotes. A zero weight is how the ACL keeps a
flagged quote *visible* without letting it steer the parameters, so the model's error against one
is the error against something we deliberately ignored — and on a slice where every wing quote was
flagged, that number would dominate the maximum and mean nothing. The domain's own wording is
"how many quotes actually entered the fit", and this is that count.

`FlatVolCalibrator` reports both over every quote in the task. The divergence is real and is left
standing here rather than repaired in passing: it is F1-08's row to settle, and it is recorded as
debt.

## `n_iterations` carries residual evaluations, and the domain's docstring is amended

`least_squares` publishes no iteration count. It publishes `nfev`, which is also the honest measure
of what a fit cost — a finite-difference Jacobian is about six evaluations per step — and it is the
quantity the comparison of Design 5.7 will be read against, since the evaluation budget is what
`converged` reports the exhaustion of.

That substitution changes what the field means. `CalibrationResult.n_iterations` documented "total
optimiser iterations across the surface" and offered a meaningful zero: "a warm start that lands on
the optimum from the previous snapshot". For this producer that zero is unreachable, because
`nfev >= 1` always — the residual is evaluated at `x0` before anything is decided.

The docstring is amended rather than left to be contradicted by its only real implementation. A
docstring is not an ADR: it describes what is true now, and this record is what makes the change
of meaning traceable. What the amendment keeps is everything that is actually a contract of the
type — non-negative, never tested for truthiness (`not 0` is `True`, and the cheapest possible
cycle is the one that would be misread as a failure), and zero still legal, which is what
`FlatVolCalibrator` reports because it searches nothing. What it drops is the claim that the unit
is iterations for every producer.

## Consequences

- `Plan.md` and `Implementation.md` are not amended. A document and an ADR disagreeing is what this
  record exists to resolve, and `CLAUDE.md` says the ADR wins. `Plan.md`'s v0.4 note points here
  rather than restating any of it.
- Each departure is also argued in the docstring that owns it, in
  `parametric_pricing/adapters/scipy_calibrator.py`; this record is where `CLAUDE.md` says they
  must also be collected.
- `parametric_pricing/domain/calibration.py`'s `n_iterations` docstring changes with this record
  and cites it. No behaviour changes: the invariant and the constructor check are untouched.
- **F3-A inherits the comparison.** A JAX calibrator has a real iteration count, so Design 5.7's
  scipy-versus-JAX table is only meaningful if both producers report the same quantity. Whichever
  is chosen there, it is a decision that stage has to take deliberately rather than by filling a
  field that no longer names its own unit.
- **F2-08 inherits one caution.** The contract tests that run two calibrators over the same task
  must not assume `n_quotes_used` counts the same population for both, until F1-08's debt row is
  settled one way or the other.
- The bounds stay out of `FitSettings` when F2-07 gives it a TOML section. That is the one item
  here that a later stage could quietly undo, which is why it is stated as a decision and not only
  as a constant.
