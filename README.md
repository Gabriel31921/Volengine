# volengine -- [documentation](https://gabriel31921.github.io/Volengine/)

A real-time, multi-market implied volatility surface calibration engine built with
Hexagonal Architecture and Domain-Driven Design (DDD).

It ingests option chains (streaming or batch, crypto or equities), calibrates the surface
using two competing engines—parametric (SVI) and neural (MLP)—and produces portfolio risk
reports with explicit guarantees of freshness and arbitrage-free consistency.

> **Status: Phase 1 (overall architecture), under construction.** This is a working README;
> the narrative README, including the context map, will arrive in F3-G.

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
├── shared_kernel/domain/   trivial value objects (Strike, Tenor, Moneyness)
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

## Development

```bash
uv sync --group dev          # development environment (without JAX or PyTorch)
uv sync --extra jax          # adds the JAX calibrator (F3-A)

uv run pytest                # tests
uv run ruff check .          # lint
uv run lint-imports          # import rules between layers
uv run mypy                  # type checking
```

## Documentation

`instructions/` (kept outside Git) contains the system design, the phased roadmap, and the
implementation plan. Architectural decisions are documented in `docs/adr/`.
