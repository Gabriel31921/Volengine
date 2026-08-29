"""A published contract is data, not an object with behaviour (ADR-001, Design 2.2).

`tests/contracts/` asserts this one DTO at a time — that `QuoteData` carries no greeks, that a
`json.dumps` hop survives. This is the structural version, and it applies to the DTOs nobody has
written yet: it reads `contracts/` as source and refuses any method that is not construction-time
validation or serialisation.

Why source and not `dir()`: a `@dataclass` grows `__init__`, `__eq__` and friends at runtime, so
introspection cannot tell an authored method from a generated one. The AST can.
"""

from __future__ import annotations

import ast
from pathlib import Path

CONTRACTS_DIR = Path(__file__).resolve().parents[2] / "src" / "volengine" / "contracts"

#: `__post_init__` validates on construction, `to_dict`/`from_dict` cross the bus (ADR-003).
#: Anything else — an `implied_vol(K, T)`, a greek, a cached property — is behaviour, and belongs
#: to the consumer that chose its own interpolation method.
ALLOWED_METHODS = frozenset({"__post_init__", "to_dict", "from_dict"})


def _contract_modules() -> list[ast.Module]:
    return [
        ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for path in sorted(CONTRACTS_DIR.glob("*.py"))
        if path.name != "__init__.py"
    ]


def _classes() -> list[ast.ClassDef]:
    return [
        node
        for module in _contract_modules()
        for node in ast.walk(module)
        if isinstance(node, ast.ClassDef)
    ]


def test_no_contract_class_carries_behaviour() -> None:
    offenders = [
        f"{klass.name}.{member.name}"
        for klass in _classes()
        for member in klass.body
        if isinstance(member, ast.FunctionDef | ast.AsyncFunctionDef)
        and member.name not in ALLOWED_METHODS
    ]

    assert offenders == [], f"behaviour in contracts/: {offenders}"


def test_the_scan_actually_reaches_the_published_dtos() -> None:
    """Guard for the test above, which an empty glob would pass in silence."""
    scanned = {klass.name for klass in _classes()}

    assert {"MarketSnapshot", "CalibratedSurface", "VolGrid", "SnapshotReady"} <= scanned
