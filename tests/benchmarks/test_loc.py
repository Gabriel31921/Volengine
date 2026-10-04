"""Lines of code: physical lines, and the lines that carry code."""

from __future__ import annotations

from pathlib import Path

import pytest
from benchmarks.loc import LineCount, count_files, count_source

SOURCE = '''"""Module docstring,
over two lines."""

import math  # a trailing comment does not make a line prose

# A comment line.


def f(x):
    """A function docstring."""
    total = (
        x
        + 1
    )
    label = "a string assigned is code"
    return math.sqrt(total)
'''


def test_physical_lines_count_every_line() -> None:
    assert count_source(SOURCE).physical == len(SOURCE.splitlines())


def test_code_lines_exclude_docstrings_comments_and_blank_lines() -> None:
    # import, def, the four lines of the parenthesised expression, label, return.
    assert count_source(SOURCE).code == 8


def test_an_attribute_docstring_is_prose() -> None:
    source = 'X = 1\n"""What X means."""\n'

    assert count_source(source) == LineCount(physical=2, code=1)


def test_counts_over_several_files_add_up(tmp_path: Path) -> None:
    first, second = tmp_path / "a.py", tmp_path / "b.py"
    first.write_text("a = 1\n", encoding="utf-8")
    second.write_text("b = 2\n\n", encoding="utf-8")

    assert count_files((first, second)) == LineCount(physical=3, code=2)


def test_more_code_than_physical_lines_is_refused() -> None:
    with pytest.raises(ValueError, match="Inconsistent"):
        LineCount(physical=1, code=2)
