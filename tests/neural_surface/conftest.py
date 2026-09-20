"""Collection rules for this context's tests. Fixtures live nowhere: there are none.

``torch`` is an optional extra (the ``neural`` group, ADR-024): the CI leg that syncs the dev group
alone does not have it, and neither does a developer who has not asked for it. The modules below
import the torch adapter at module level, which is right -- an import inside a test function hides
a dependency -- so the whole file is skipped from *collection* when the extra is absent, rather
than skipped test by test after an import has already failed.

``collect_ignore_glob`` is the pytest hook for exactly that, which is why this lives in a
``conftest.py`` and not in a builder module: ``CLAUDE.md`` reserves ``conftest`` for the fixtures
and hooks pytest injects. The same arrangement as ``tests/parametric_pricing/conftest.py`` for the
JAX modules, for the same reason.
"""

from __future__ import annotations

from importlib.util import find_spec

collect_ignore_glob: list[str] = [] if find_spec("torch") is not None else ["test_torch_*.py"]
"""The torch-only modules, ignored entirely when the extra is not installed."""
