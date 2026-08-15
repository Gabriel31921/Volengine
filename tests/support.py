"""Helpers shared by every test package.

Deliberately not ``conftest.py``: conftest is where pytest looks for fixtures and hooks it
*injects*, and importing from it is discouraged precisely because it is loaded by collection
magic rather than by an import statement. This repo's builders are plain functions that tests
call directly, so they live in ordinary modules that say where they came from.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any


def replace_field[T](instance: T, field: str, value: Any) -> T:
    """``dataclasses.replace`` with the field name chosen at runtime.

    The parametrised rejection tests pick which field to poison from the parameter list, so the
    keyword is a string known only while the test runs. Splatting that dict into ``replace``
    defeats mypy: it sees ``**dict[str, float]`` and reports every *other* field of the
    dataclass as receiving a ``float``. Scattering ``# type: ignore`` across twenty tests to
    silence that would also silence the real mistakes those tests exist to catch, so the
    dynamism is confined to this one call instead.
    """
    return replace(instance, **{field: value})  # type: ignore[type-var]
