# volengine -- [documentation](https://gabriel31921.github.io/Volengine/)

A real-time, multi-market implied volatility surface calibration engine built with
Hexagonal Architecture and Domain-Driven Design (DDD).

It ingests option chains (streaming or batch, crypto or equities), calibrates the surface
using two competing engines—parametric (SVI) and neural (MLP)—and produces portfolio risk
reports with explicit guarantees of freshness and arbitrage-free consistency.

> **Status: Phases 1 and 2 complete, Phase 3 four blocks in (F3-A to F3-D).** The pipeline runs
> end to end on synthetic data and on the live Deribit BTC chain, with a SciPy SVI calibrator
> wired in. A JAX SVI calibrator and a PyTorch MLP learner are built and tested behind the same
> contract but not yet reachable from a configuration file; F3-E wires them in alongside the
> comparative report. This is a working README; the narrative one, including the context map,
> arrives in F3-G.

## The Four Contexts

| Context              | Role                                                | Publishes           |
| -------------------- | --------------------------------------------------- | ------------------- |
| `market_data`        | Ingests and normalizes option chains                | `MarketSnapshot`    |
| `parametric_pricing` | Calibrates SVI per expiry (SciPy or JAX)            | `CalibratedSurface` |
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
wired contexts. `volengine run` drives the same graph without a stopping rule, and since F3-B
`record` and `replay` drive it too:

```bash
uv run volengine record --config examples/walking-skeleton.toml --recording session.jsonl
uv run volengine replay --config examples/walking-skeleton.toml --recording session.jsonl
```

`record` is an ordinary session with a tap on it, writing the normalised quote stream to JSON
Lines; `replay` runs the whole engine off that file alone, on a clock the recording moves. Two
replays of one recording write the same report byte for byte, which is what ADR-004 asked for —
with the one caveat about conflation on longer sessions that `docs/SEAMS.md` records.

The adapters behind it are deliberately trivial — a constant chain, a flat-volatility calibrator,
a console writer — because their job is to prove the graph, not the mathematics. The second shipped
configuration runs the same graph on the adapters that do:

```bash
uv run volengine report --config examples/synthetic-svi.toml --count 2
```

`SyntheticProvider` generates a *known* SVI surface priced through Black-76 and then spoils it
(spread, sizes, timestamp jitter, junk quotes keyed to the admissibility rules); `ScipyCalibrator`
fits raw SVI back out of it with `least_squares`; and the report carries a volatility that can be
checked against the parameters the file itself declares. `--duration <seconds>` bounds a `run`,
`--metrics` logs every metric the contexts emit, and `writer = "csv"` with an `output_path` writes
one row per valued position instead of printing.

The third configuration opens a socket, to the live Deribit BTC options chain:

```bash
uv sync --extra deribit
uv run volengine run --config examples/deribit-live.toml --duration 120 --metrics
```

`DeribitProvider` discovers the chain over REST, streams `ticker.*.100ms` over a websocket,
reconnects with backoff, and rediscovers the universe periodically, because a Deribit expiry dies
at 08:00 UTC every day. Inverse (BTC-settled) premiums are converted into the strike's currency
before they cross the boundary, so the calibrator never learns which market they came from. No API
key is needed. Every threshold in that file was measured against a real recording, and the
measurement is written beside the value it defends. Thirty seconds of that recording, narrowed to
two expiries, is committed as `tests/fixtures/deribit-btc-2026-09-18.jsonl`, and every test run
replays it through the whole engine.

### Built, not yet wired

Two more producers are implemented and tested behind the same `CalibratedSurface` contract, but no
TOML can select them yet:

- **`svi-jax`** (`uv sync --extra jax`): the same SVI objective as the SciPy baseline, jitted, with
  a warm Adam cycle and a cold multi-start L-BFGS one over a fixed padded shape (ADR-009, ADR-029).
  The module also has forward-mode and reverse-mode greeks, which sit outside the contract.
- **`mlp-torch`** (`uv sync --extra neural`): an MLP over log-forward-moneyness and tenor, trained
  with soft Durrleman and calendar penalties and checked by a hard no-arbitrage gate before
  publication. It fine-tunes warm from snapshot to snapshot, carrying its optimiser state across.

A contract test harness runs every producer through the same assertions. Wiring both into the
pipeline is F3-E's job; `docs/SEAMS.md` records why it waits. `market_data/adapters/heston.py` is
also built, a Heston generator for synthetic chains with a real smile, and it is not reachable from
a configuration either.

## Development

```bash
uv sync --group dev          # development environment (without JAX or PyTorch)
uv sync --extra jax          # adds the JAX calibrator (F3-A)
uv sync --extra neural       # adds the PyTorch learner, CPU build (F3-D)
uv sync --extra deribit      # adds the live Deribit feed (F3-C)

uv run pytest                # tests
uv run ruff check .          # lint
uv run lint-imports          # import rules between layers
uv run mypy                  # type checking

bash scripts/verify.sh       # all of the above, in the order CI runs them
```

`scripts/verify.sh` is the single definition of "healthy": the developer, the review agents and
`.github/workflows/ci.yml` all run that one script, so there is no second list to keep in step
(ADR-024). CI runs it three times: once bare, once with `--extra jax` and once with
`--extra neural`. The Deribit tests need no extra, because the adapter imports its network
libraries lazily.

## Documentation

**`docs/adr/`** holds the architectural decision records — 29 of them, one decision per file,
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
ADR (020, 021, 022, 023, 025, 027, 028, 029). `CLAUDE.md`, the standing rules this repository is written to, is
outside Git for the same reason; its import rules are executable and live in `pyproject.toml`'s
`[tool.importlinter]` contracts, which `tests/architecture/` runs.
