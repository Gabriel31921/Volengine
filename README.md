# volengine -- [documentation](https://gabriel31921.github.io/Volengine/)

A real-time, multi-market implied volatility surface calibration engine built with
Hexagonal Architecture and Domain-Driven Design (DDD).

It ingests option chains (streaming or batch, crypto or equities), calibrates the surface
using two competing engines—parametric (SVI) and neural (MLP)—and produces portfolio risk
reports with explicit guarantees of freshness and arbitrage-free consistency.

> **Status: Phase 1 complete, Phase 2 under way.** The whole architecture is built and the
> pipeline runs end to end on trivial adapters; what phase 2 adds is the mathematics that is not
> knowable from the design document — the real calibrator first. This is a working README; the
> narrative one, including the context map, arrives in F3-G.

## The Four Contexts

| Context              | Role                                                | Publishes           |
| -------------------- | --------------------------------------------------- | ------------------- |
| `market_data`        | Ingests and normalizes option chains                | `MarketSnapshot`    |
| `parametric_pricing` | Calibrates SVI per expiry (SciPy, later JAX)        | `CalibratedSurface` |
| `neural_surface`     | Learns the surface using a neural network (PyTorch) | `CalibratedSurface` |
| `risk`               | Values portfolios and enforces the freshness policy | `RiskReport`        |

The contexts communicate **exclusively** through immutable DTOs defined in `contracts/`. No tensors,
no exchange-specific types, and no domain objects cross context boundaries. The parametric and neural
engines are **competing contexts**: they consume the same upstream data, expose the same output
contract, but rely on fundamentally different internal models.

## Project Structure

```text
src/volengine/
├── shared_kernel/domain/   value objects, and the mathematics every context shares
├── contracts/              published language: DTOs shared across boundaries
├── platform/               in-process bus, clock, metrics, executors
├── market_data/            domain · application · adapters
├── parametric_pricing/     domain · application · adapters
├── neural_surface/         domain · application · adapters
├── risk/                   domain · application · adapters
└── entrypoints/            cli.py, pipeline.py, config.py
```

Each context follows the same hexagonal structure: `domain/` (pure business logic with no dependencies),
`application/` (use cases and anti-corruption layers), and `adapters/` (port implementations).

The shared kernel is the one sanctioned exception to "no context imports another", and it stays
small on purpose: value objects, aware-timestamp validation, and the two closed forms that answer
*yes, always* to "would every context want this change" — Black-76 (ADR-014) and the raw SVI curve
(ADR-026). Mathematics lives once; models are duplicated deliberately.

## What runs today

The walking skeleton is wired and runnable — Market Data → Parametric Pricing → Risk, over the
in-process bus, from a TOML file:

```bash
uv run volengine report --config examples/walking-skeleton.toml --count 2
```

It prints risk reports carrying back the volatility the feed was priced at, across all three
wired contexts. `volengine run` drives the same graph without a stopping rule; `record` and
`replay` refuse with exit 1 until F3-B.

The adapters behind it are deliberately trivial — a constant chain, a flat-volatility calibrator,
a console writer — because their job is to prove the graph, not the mathematics. A second feed,
`SyntheticProvider`, generates a *known* SVI surface priced through Black-76 and then spoils it
(spread, sizes, timestamp jitter, junk quotes keyed to the admissibility rules), which is what
gives the calibrator a right answer to be measured against. Neural Surface is implemented but
deliberately not wired yet; see `docs/SEAMS.md`.

## Development

```bash
uv sync --group dev          # development environment (without JAX or PyTorch)
uv sync --extra jax          # adds the JAX calibrator (F3-A)

uv run pytest                # tests
uv run ruff check .          # lint
uv run lint-imports          # import rules between layers
uv run mypy                  # type checking

bash scripts/verify.sh       # all of the above, in the order CI runs them
```

`scripts/verify.sh` is the single definition of "healthy": the developer, the review agents and
`.github/workflows/ci.yml` all run that one script, so there is no second list to keep in step
(ADR-024).

## Documentation

**`docs/adr/`** holds the architectural decision records — 26 of them, one decision per file,
immutable. They are the answer to *why* the system is the way it is, and they are authoritative:
where a record and a planning document disagree, the record wins. `docs/adr/README.md` is the
index.

**`docs/SEAMS.md`** holds the gaps left open on purpose. Every item there was seen, weighed and
left, and several close naturally in a later phase — read it before "fixing" one.

**`instructions/` is kept outside Git**, and holds the planning documents: `Design.md` (contexts
and contracts), `Plan.md` (phases), `Implementation.md` (signatures) and `STATE.md` (current build
state). Docstrings throughout the code cite them — `Design §7.2`, `Implementation.md`'s sketch of
a type — so a citation that cannot be followed from a clone is pointing at one of those four. The
ADRs are written so that the reasoning survives without them: where the code departs from a
planning document, the departure is argued in the docstring that owns it *and* collected in an
ADR (020, 021, 022, 023, 025). `CLAUDE.md`, the standing rules this repository is written to, is
outside Git for the same reason; its import rules are executable and live in `pyproject.toml`'s
`[tool.importlinter]` contracts, which `tests/architecture/` runs.
