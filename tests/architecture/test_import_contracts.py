"""CLAUDE.md's import rules, run as a test (F1-09).

The rules themselves are declarative and live in `[tool.importlinter]` in `pyproject.toml`;
this module is what makes them fail the suite rather than only a separate `lint-imports` step.

These tests assert architecture, not behaviour. They are permanent: their value is in the
imports F2 and F3 have not written yet, not in the ones already on disk.
"""

from __future__ import annotations

from pathlib import Path

from importlinter.api import read_configuration
from importlinter.cli import lint_imports

PYPROJECT = Path(__file__).resolve().parents[2] / "pyproject.toml"

#: The rules of CLAUDE.md that must stay declared, by contract name. Checked as a subset, so a
#: later phase may add contracts freely and may not quietly drop one to make the linter pass.
REQUIRED_CONTRACTS = frozenset(
    {
        "The stack is layered: entrypoints on top, shared kernel at the bottom (rules 7, 8)",
        "Inside a context: adapters > application > domain (rule 4)",
        "No context imports another context (rule 6)",
        "The domain does not know the contracts (rule 3)",
        "The domain runs on stdlib, the shared kernel and numpy (rule 3)",
        "The contracts carry no array library (rule 2, ADR-011)",
        "The shared kernel is stdlib only (rule 1, ADR-014)",
        "Platform carries contracts and stdlib (rule 7)",
    }
)

_BROKEN_CONFIG = """
[tool.importlinter]
root_package = "volengine"

[[tool.importlinter.contracts]]
name = "A rule the composition root is known to break"
type = "forbidden"
source_modules = ["volengine.entrypoints"]
forbidden_modules = ["volengine.contracts"]
"""


def test_every_import_contract_is_kept() -> None:
    assert lint_imports(config_filename=str(PYPROJECT), no_cache=True) == 0


def test_a_broken_contract_is_reported(tmp_path: Path) -> None:
    """Guard for the test above, which would pass vacuously if the runner never returned 1.

    `entrypoints` imports `contracts` — it is the composition root and rule 8 lets it — so a
    contract forbidding that import must come back broken.
    """
    config = tmp_path / "pyproject.toml"
    config.write_text(_BROKEN_CONFIG, encoding="utf-8")

    assert lint_imports(config_filename=str(config), no_cache=True) == 1


def test_no_rule_has_been_dropped_from_the_configuration() -> None:
    """The linter can only check the contracts it is given, so the list itself is guarded."""
    configured = read_configuration(str(PYPROJECT))["contracts_options"]
    declared = {contract["name"] for contract in configured}

    assert declared >= REQUIRED_CONTRACTS
