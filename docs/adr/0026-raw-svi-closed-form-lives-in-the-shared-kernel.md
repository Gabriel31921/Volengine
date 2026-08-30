# ADR-026: The raw SVI closed form lives in the shared kernel

**Status:** Accepted · 2026-08 · project decision, not in Design.md

## Context

F2-03's synthetic feed had to evaluate a known SVI surface in order to price it, and it wrote the
raw SVI form out to do so:

```
w(k) = a + b * (rho * (k - m) + sqrt((k - m)^2 + sigma^2))
```

That made three copies in `src/`. `parametric_pricing/domain/svi_slice.py` had it in
`SVIParams.total_variance`, `parametric_pricing/domain/durrleman.py` had it again inside
`_curve_and_derivatives`, and `market_data/adapters/synthetic.py` now had a third in
`SVIParamsSpec`. ADR-014 draws the line this crosses: duplication across contexts is intentional
**for models** and not for mathematics, and it admits a module to the shared kernel only if it
passes three tests at once — a fact and not a policy, no vocabulary, and frozen by nature.

The synthetic module argued that its `SVIParamsSpec` and the fitted `SVIParams` must stay separate
types. That is true, and it is a different claim: it is an argument about two value objects, and
it says nothing about the curve they both describe.

The question left open was whether raw SVI is a *model choice* — in which case each context is
entitled to its own — or mathematics, in which case ADR-014 already decided it.

## Decision

**It is mathematics. `w(k)` moves to `shared_kernel/domain/svi.py`, and every caller evaluates it
there.**

Choosing SVI over SSVI or a stochastic-volatility model is unquestionably a policy, and this
project already expresses it as one: ADR-008 records the choice, and the `Calibrator` port is the
abstraction it lives behind. Changing it means a different implementation arriving behind that
port — one model out, another in — and the new one writes down whatever form *it* is. It does not
mean this expression comes to mean something else. `w(k) = a + b(rho(k - m) + sqrt(...))` is what
the name "raw SVI" denotes; if a later milestone fits SSVI, this stays exactly as true as it was,
which is criterion 3 being satisfied rather than violated. Mathematics does not fork.

The other two tests pass as plainly. It is a fact: given five numbers and a moneyness there is one
right answer, and nothing in it is tunable, so nothing belongs in the TOML of ADR-012. It carries
no vocabulary: five floats in, one float out, exactly as `black76.price` takes `is_call: bool`
rather than any context's enum.

**What is admitted is the curve and its minimum, and nothing else.** Three things stay outside,
each for a reason:

- **The two value objects.** They fail the second test outright, and must. `SVIParams` is the
  output of a fit and has to admit a collapsed slice, because an optimiser walks through them and
  the violation must stay measurable; `SVIParamsSpec` is handwritten configuration, is never the
  intermediate state of anything, and is strictly stricter. Promoting either would make one
  context's invariants binding on the other's, which is the anti-pattern the shared kernel is most
  at risk of becoming.
- **The annualisation `sqrt(w / T)`.** What each caller owns there is the guard, not the formula:
  a non-positive tenor is a construction bug whose `ValueError` should name that context's own
  parameter. Moving it would relocate a message.
- **The derivatives and the vectorised evaluation.** Both want numpy, and rule 1 confines the
  shared kernel to the standard library. `durrleman.py` keeps `w'` and `w''`, which are different
  formulas rather than another spelling of the same one, and takes `w` itself from
  `SVIParams.total_variance`. `SVIParams.total_variance`'s array branch stays a numpy expression
  — evaluating a dense grid one Python call at a time is not an option inside an optimiser's loss
  — and is the elementwise image of the kernel's scalar form, pinned to it by
  `test_total_variance_agrees_between_scalar_and_array_input`, whose scalar side is now the
  kernel's own answer.

## Why this duplication was more dangerous than Black-76's

ADR-014 was written after the same Black-76 had been typed out four times, and the danger it
names is that a numerical fix or an improvement lands in one copy and not the others — two
producers then weight the same market differently while every test in both contexts keeps passing.

The SVI copies were worse than that, because of *where* they sat. Market Data's generator produces
the surface that Parametric Pricing's calibrator is measured against. A sign slip in one copy does
not show up as a disagreement; it **cancels** against a fit performed with the same slip and
passes the known-truth test with the engine wrong. `Plan.md` warned about exactly this trap for
F2-03 — "this generator and the F2-05 calibrator are both ours, so a shared error that cancels ...
passes the known-truth test with the system broken" — and two copies of one formula on the two
sides of the measurement is the most direct way to spring it.

## Consequences

- `shared_kernel/domain/svi.py` is the second member of the shared kernel, with
  `total_variance(k, a, b, rho, m, sigma)` and `min_total_variance(a, b, rho, sigma)`. Rule 1 is
  asserted for it by its own AST test, as it is for `black76.py`.
- Four call sites now read the same curve: `SVIParams.total_variance` (scalar branch) and
  `SVIParams.min_total_variance`, `SVIParamsSpec.total_variance` and
  `SVIParamsSpec.min_total_variance`. `durrleman._curve_and_derivatives` reads it through the
  first of those.
- `test_the_two_spellings_of_svi_describe_the_same_curve` now passes by construction. It stays,
  for the reason its sibling in `tests/neural_surface/test_acl.py` stayed after Black-76 was
  extracted: the risk is not that someone breaks a copy, it is that someone re-inlines one, and
  that edit is invisible from either side of a measurement that cancels.
- `durrleman` recomputes `r` once per grid point that it used to share with `w`. One square root
  against the optimiser's own arithmetic, and not where that loop's time goes.
- The two value objects stay duplicated, deliberately, and `tests/` remains the only place they
  are allowed to meet.
- ADR-014's admission criteria did the work here rather than a judgement call, which is the second
  piece of evidence that they are the right ones. The kernel is now two modules; each new
  candidate still has to pass all three.
