"""Lines of code, counted so that a docstring-heavy codebase is not flattered or punished.

Design §5.7 lists lines of code among the scipy-vs-JAX criteria. This repository argues its
decisions in docstrings, often at greater length than the code they sit on, so a physical line
count measures prose as much as engineering. Both numbers are reported: **physical lines**, and
**code lines** -- lines carrying at least one token that is not a comment, and not part of a bare
string statement (a module, class or function docstring, or an attribute's docstring).
"""

from __future__ import annotations

import ast
import io
import tokenize
from dataclasses import dataclass
from pathlib import Path

_NON_CODE = frozenset(
    {
        tokenize.COMMENT,
        tokenize.NL,
        tokenize.NEWLINE,
        tokenize.INDENT,
        tokenize.DEDENT,
        tokenize.ENCODING,
        tokenize.ENDMARKER,
    }
)


@dataclass(frozen=True, slots=True)
class LineCount:
    """Physical and code lines of one or more modules."""

    physical: int
    code: int

    def __post_init__(self) -> None:
        if self.code < 0 or self.physical < self.code:
            raise ValueError(f"Inconsistent counts: {self.code} code of {self.physical} physical")

    def __add__(self, other: LineCount) -> LineCount:
        return LineCount(self.physical + other.physical, self.code + other.code)


def count_source(source: str) -> LineCount:
    """Count one module's lines.

    Raises:
        SyntaxError: If ``source`` is not Python.
    """
    prose: set[int] = set()
    for node in ast.walk(ast.parse(source)):
        if (
            isinstance(node, ast.Expr)
            and isinstance(node.value, ast.Constant)
            and isinstance(node.value.value, str)
        ):
            end = node.end_lineno if node.end_lineno is not None else node.lineno
            prose.update(range(node.lineno, end + 1))

    code: set[int] = set()
    for token in tokenize.generate_tokens(io.StringIO(source).readline):
        if token.type in _NON_CODE:
            continue
        for line in range(token.start[0], token.end[0] + 1):
            if line not in prose:
                code.add(line)
    return LineCount(physical=len(source.splitlines()), code=len(code))


def count_files(paths: tuple[Path, ...]) -> LineCount:
    """The sum over several modules."""
    totals = LineCount(0, 0)
    for path in paths:
        totals = totals + count_source(path.read_text(encoding="utf-8"))
    return totals
