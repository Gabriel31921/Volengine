"""The narrative deliverables of Design §12, as code that regenerates them (F3-G).

Two benchmarks and the pieces they are built from:

* :mod:`benchmarks.scipy_vs_jax` -- the two SVI engines on one market (Design §5.7: time,
  convergence, lines of code).
* :mod:`benchmarks.parametric_vs_neural` -- the SVI fit against the network (Design §12: RMSE,
  violations, drift).
* :mod:`benchmarks.charts` -- every chart drawn from a ``CsvMetricsSink`` file and nothing else.

Each benchmark runs the engine exactly as an operator does -- a shipped ``examples/*.toml`` through
``entrypoints.pipeline.build_pipeline`` on the real registry -- with the metrics sink pointed at a
file, and then reads that file back. **The numbers in the results are the engine's own
measurements**, not a second instrumentation written for the occasion: a benchmark that timed the
calibrators its own way would be measuring code the engine never runs.

**This package sits beside ``entrypoints/``, not inside a context**, and plays the same role: it
may import anything, because comparing two contexts is exactly what no context is allowed to do
(rule 6). It is outside ``src/`` because it is not part of the installed engine -- nothing in
``volengine`` imports it -- and outside ``scripts/`` because that directory is git-excluded
workstation tooling (ADR-024). Its tests live in ``tests/benchmarks/``, and mypy checks it through
them.

Run from the repository root::

    uv run --extra jax python -m benchmarks.scipy_vs_jax
    uv run --extra neural python -m benchmarks.parametric_vs_neural
    uv run python -m benchmarks.charts metrics.csv --out-dir charts/

The first two write a Markdown report, its charts and the raw metrics file into
``benchmarks/results/``. The numbers are this machine's on the day they were taken; the report
header says which machine and which day.
"""
