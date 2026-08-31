# Architecture Decision Records

One file per decision, recording **why** the system is the way it is — the thing the code
itself can never tell you.

An ADR is immutable. If a decision is reversed, its record is not edited: a new ADR is written
that supersedes it, and the old one is marked `Superseded`. The value is in the history, not
in the current state.

| # | Decision | Status |
|---|---|---|
| [001](0001-grid-not-evaluable-object.md) | The published surface is a grid, not an evaluable object | Accepted |
| [002](0002-market-conventions-as-domain-concept.md) | MarketConventions is a first-class domain concept | Accepted |
| [003](0003-in-process-conflating-event-bus.md) | In-process event bus with conflation | Accepted |
| [004](0004-clock-port-and-deterministic-replay.md) | Clock port from day one, and deterministic replay | Accepted |
| [005](0005-thread-pool-per-calibrator.md) | One thread pool per calibrator | Accepted |
| [006](0006-republish-stale-surface-on-failure.md) | An honest old surface beats a broken new one | Accepted |
| [007](0007-european-index-options-first.md) | European index options before American single names | Accepted |
| [008](0008-raw-svi-per-slice.md) | Raw SVI per slice in v1 | Accepted |
| [009](0009-fixed-shape-padding-and-mask.md) | Fixed shape by padding plus a boolean mask | Accepted |
| [010](0010-three-tier-neural-constraints.md) | Three tiers of constraints on the neural surface | Accepted |
| [011](0011-grid-as-nested-tuples.md) | The published grid is nested tuples of floats | Accepted |
| [012](0012-thresholds-as-toml-configuration.md) | Thresholds are configuration data, not constants | Accepted |
| [013](0013-composition-events-carry-state-not-deltas.md) | Composition events carry state, not deltas | Accepted |
| [014](0014-shared-kernel-holds-mathematics-not-vocabulary.md) | The shared kernel holds mathematics, not vocabulary | Accepted |
| [015](0015-volgrid-carries-expiries-and-forwards.md) | `VolGrid` carries expiries and forwards | Accepted |
| [016](0016-use-cases-are-handlers-not-loops.md) | Use cases are handlers, not loops | Accepted |
| [017](0017-acl-inverts-the-out-of-the-money-twin.md) | The ACL inverts the out-of-the-money twin | Accepted |
| [018](0018-quote-weights-are-vega-over-a-spread-discount.md) | Quote weights are vega over a spread discount | Accepted |
| [019](0019-the-two-producers-are-one-function-twice.md) | The two producers are one function twice | Accepted |
| [020](0020-domain-api-departures-from-implementation-md.md) | Domain API departures from `Implementation.md` | Accepted |
| [021](0021-acl-and-event-decisions-of-f1-06.md) | ACL and event decisions of F1-06 | Accepted |
| [022](0022-composition-root-departures-of-f1-07.md) | Composition-root departures of F1-07 | Accepted · one sentence superseded by [028](0028-composition-root-departures-of-f2-07.md) |
| [023](0023-flat-vol-departure-of-f1-08.md) | `FlatVolCalibrator`'s departure from `Plan.md` in F1-08 | Accepted |
| [024](0024-one-gate-script-run-by-developers-agents-and-ci.md) | One gate script, run by developers, agents and CI | Accepted |
| [025](0025-synthetic-provider-departures-of-f2-03.md) | `SyntheticProvider`'s departures from `Implementation.md` in F2-03 | Accepted |
| [026](0026-raw-svi-closed-form-lives-in-the-shared-kernel.md) | The raw SVI closed form lives in the shared kernel | Accepted |
| [027](0027-scipy-calibrator-departures-of-f2-05.md) | `ScipyCalibrator`'s departures from the plan in F2-05 | Accepted |
| [028](0028-composition-root-departures-of-f2-07.md) | Composition-root departures of F2-07 | Accepted |
| [029](0029-jax-calibrator-departures-of-f3-a.md) | `JaxCalibrator`'s departures from the plan in F3-A | Accepted |

ADR-001 to ADR-010 are extracted from the design document. ADR-011 onward were taken during
implementation and exist only here.

ADR-020, ADR-021, ADR-022, ADR-023, ADR-025, ADR-027, ADR-028 and ADR-029 are the one departure
from "one file, one decision": each collects the small API decisions of a build phase — the domain
layer, the ACLs and use cases, the composition root, the walking skeleton, the synthetic feed, the
first real calibrator, the wiring of F2, the JAX calibrator — which are individually too slight for
a record of their own and collectively too load-bearing to lose. Every item in them
is also stated in the docstring of the module that owns it. They are written once, when that phase
is built, and never appended to afterwards: a later phase's departures get a later record, because
these files are immutable like every other one.

**Gaps left open on purpose are not ADRs.** They live in `docs/SEAMS.md`, because a record here is
immutable and a seam is a condition that is still true until someone closes it.
