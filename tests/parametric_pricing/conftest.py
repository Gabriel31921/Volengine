"""Collection rules for this context's tests. Fixtures live nowhere: there are none.

``jax`` and ``optax`` are an optional extra (ADR-024): the CI leg that syncs the dev group alone
has neither, and neither does a developer who has not asked for them. The modules below import the
JAX adapters at module level, which is right -- an import inside a test function hides a dependency
-- so the whole file is skipped from *collection* when the extra is absent, rather than skipped
test by test after an import has already failed.

``collect_ignore_glob`` is the pytest hook for exactly that, which is why this lives in a
``conftest.py`` and not in a builder module: ``CLAUDE.md`` reserves ``conftest`` for the fixtures
and hooks pytest injects, and reserves plain modules for everything a test could have imported
itself.

``tests/parametric_pricing/test_calibrator_contract.py`` is deliberately **not** in the list. It
runs against every calibrator this build can construct, and that set is one item shorter without
the extra -- which is a statement the contract test makes for itself.
"""

from __future__ import annotations

from importlib.util import find_spec

collect_ignore_glob: list[str] = [] if find_spec("jax") is not None else ["test_jax_*.py"]
"""The JAX-only modules, ignored entirely when the extra is not installed."""
