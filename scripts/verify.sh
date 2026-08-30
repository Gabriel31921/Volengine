#!/usr/bin/env bash
# Single source of truth for "is this repository healthy right now".
#
# Three callers run this same list and no other: the developer before a commit, the reviewer and
# implementer agents (`.claude/agents/`), and `.github/workflows/ci.yml`. That is the whole point
# of the file — a CI workflow restating these five commands would be a second list free to drift
# from this one, and the drift would be invisible until the day the two disagree (ADR-024).
#
# It is the one committed file under `scripts/`; the rest of the directory is workstation-local
# agent tooling and stays git-excluded.
set -euo pipefail

# Announce each command before running it. `set -e` aborts silently otherwise, and in a CI log the
# last thing printed is the only clue to which gate failed.
run() {
    echo "==> $*"
    "$@"
}

run uv run pytest -q
run uv run ruff check .
run uv run ruff format --check .
run uv run mypy
run uv run lint-imports
