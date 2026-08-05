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

ADR-001 to ADR-010 are extracted from the design document. ADR-011 and ADR-012 were taken
during implementation planning and exist only here.
