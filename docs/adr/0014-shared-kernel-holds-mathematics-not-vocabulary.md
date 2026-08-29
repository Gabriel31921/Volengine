# ADR-014: The shared kernel holds mathematics, not vocabulary

**Status:** Accepted · 2026-08 · project decision, not in Design.md

## Context

By the end of F1-06 the closed-form Black-76 had been written out four times: in
`parametric_pricing/domain/black76.py`, in `neural_surface/domain/pricing.py` (a near-literal
384-line copy, whose own docstring called it "the fourth copy, and the one that hurts"), in
`risk/domain/pricing.py` (the price half only), and with Market Data's inversion for
`IV_DIVERGENCE` still to come in F2. Each copy carried the same normal CDF, the same `d1`/`d2`,
the same validation guard, and — in two of them — the same bracketed Newton inversion.

Every copy was defended in its own docstring by the same argument: rule 6 forbids one context
importing another, so the alternative to writing the formula again is a shared pricing library on
the boundary, which is the canonical-model anti-pattern that bounded contexts exist to prevent.

The argument conflates two different things. DDD's rule against sharing is about **models** —
`QuoteObservation` and `QuoteData` must stay separate because they *mean* different things and
would evolve apart under one owner's compromise. It is a statement about semantics, and a closed
form has no semantics to diverge: `d1 = (log(F/K) + w/2)/sqrt(w)` is the same equation whoever
asks. Mathematics does not fork.

What the duplication actually cost was never the typing. It was that a numerical fix lands in one
copy and not the others. The `erfc` spelling of the normal CDF is written down under recurring
traps in `CLAUDE.md` *because it kept reappearing*, once per copy — empirical evidence from this
repo that this particular duplication is the dangerous kind. And Design §6.5 compares two producers
fitted to one market while holding the weights constant; both weight by vega, so two
implementations of one vega is precisely how that comparison stops meaning anything, not through a
visible bug but through one of them being improved.

## Decision

The closed form moves to `shared_kernel/domain/black76.py` — `norm_cdf`, `norm_pdf`, `intrinsic`,
`ceiling`, `price`, `vega`, `implied_vol`, `VOL_TOLERANCE`, `PriceNotInvertibleError` — and each
context keeps a thin wrapper owning its enum, its error hierarchy, and a docstring saying what a
wrong answer means *there*.

The Shared Kernel is DDD's sanctioned exception to "no context imports another", bought at the
price that changing it requires every owner's agreement. Admission requires **all three** of:

1. **It is a fact, not a policy.** A right answer independent of who is asking. No threshold, no
   business rule, nothing an operator could reasonably want to tune — so nothing that belongs in
   the TOML of ADR-012.
2. **It carries no vocabulary.** Primitives in, primitives out. Concretely: the side of the
   contract arrives as `is_call: bool` and never as an enum, so `OptionKindD`, `OptionKindP`,
   `OptionKindN`, `OptionKindR` and the published `OptionKind` all keep their own spelling and each
   context maps in one line at the call site. Sharing the arithmetic must not become sharing the
   model.
3. **It is frozen by nature.** It changes only if it was wrong, and then every caller wants the fix.

`require_aware` (already extracted, after living in six modules) passes all three. Black-76 passes
all three. `SnapshotPolicy`, `Forward`, `SurfaceView` and every quote model fail the first outright.

`implied_vol` raises `PriceNotInvertibleError`, which belongs to no context; each wrapper catches it
and re-raises its own `NoImpliedVolError` from it, so that a caller draping `except CalibrationError`
around a slice still catches the one genuine market outcome and still does not catch a construction
bug.

## Alternatives considered

**Keep the copies.** Defensible on the letter of rule 6 and indefensible on the evidence: the same
numerical trap had already reappeared once per copy before any of this code priced a real quote.

**A `common/` or `utils/` package.** "Common" is a name with no admission criterion, so it becomes
a landfill. The Shared Kernel is named after a pattern with a defined cost, and the cost is what
stops it growing.

**Share the enum too.** It is identical text in four places, but vocabulary is exactly what a
bounded context owns, and the members are not a promise: Risk's describes a *position* somebody
holds, Pricing's a *quote* being inverted, and if Risk ever books a perpetual that member has no
business in a calibrator's language. Three lines of duplication cost less than one shared type on
the boundary.

**Share `implied_vol` only where duplicated, leaving Risk's `price` alone.** Rejected as
inconsistent: Risk's copy is the one whose failure mode is worst — a wing that prices at zero
*stays* at zero under a bump, so an unhedged tail reports as no risk at all.

## Consequences

- One implementation, one place to fix, one oracle. All 1592 existing tests passed unchanged, which
  is the evidence that this was a refactor and not a redesign; 18 tests were added for the kernel's
  own surface.
- Rule 1 (`shared_kernel/` imports stdlib only) still holds and is now asserted by a test that
  parses the module's imports, because this is the first module in the kernel with a plausible
  reason to reach for numpy — and F3 will make that temptation real.
- **The vectorised JAX Black-76 of F3-A cannot live here** (rule 1, ADR-011). That duplication
  remains, and the extraction improves it rather than removing it: a differentiable batched
  implementation is a genuine reimplementation, and it will be tested *against* the kernel, which
  turns an accidental copy into a deliberate one with a reference.
- Market Data's F2 inversion for `IV_DIVERGENCE` is now a wrapper, not a fifth copy.
- **One thing is genuinely lost:** two independent spellings of one formula that agreed to the last
  bit were a real cross-check, and there is now one spelling to be wrong. Accepted because the
  failure that check guarded against is far rarer than the one duplication causes.
- The cross-context agreement test in `tests/neural_surface/test_acl.py` now passes by construction
  for the formula. It is kept, because the failure it guards against only moved: it fails if anyone
  inlines the formula back into a context, and it still covers the ACL's out-of-the-money rule and
  spread discount, which remain duplicated prose.
- `_require_positive_finite` is still written out in five domain modules. Extracting it is the same
  argument one level down and is deliberately left for its own change.
