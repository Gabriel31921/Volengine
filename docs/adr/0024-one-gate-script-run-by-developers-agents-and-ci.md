# ADR-024: One gate script, run by developers, agents and CI

**Status:** Accepted · 2026-08 · taken while closing F1-09

## Context

`Plan.md` requires F1-09's architecture tests to be "executable from F1 and in CI forever" and
`Implementation.md` says of them "they enter CI and do not leave". Neither half was true.
`.github/workflows/` held one workflow, `docs.yml`, which builds and deploys the pdoc site; nothing
anywhere ran `pytest`, `mypy` or `lint-imports`.

The whole gate was `scripts/verify.sh`, and `scripts/` was git-excluded in `e686822` because the
directory also holds `setup_agents.sh` and `gate-commit.sh` — workstation-local agent scaffolding
that has no business in the tree. The exclusion was written for those two and swallowed the gate
along with them. A fresh clone therefore had no way to run what F1-09 built, and
`pyproject.toml`'s own comment claiming `lint-imports` "runs from `scripts/verify.sh` and from CI"
was false on both counts.

This is not a debt item. It is a decision about the shape of the project — whether the definition of
"healthy" is a committed artefact of the repository or a habit of the machine the work happens on —
and both halves of it were still open.

---

## The gate is a committed script, and CI calls it

`scripts/*` is excluded with a single negation, `!scripts/verify.sh`. The two agent scripts stay
local; the gate is committed. `.github/workflows/ci.yml` installs the locked dev environment and
runs `bash scripts/verify.sh`, on every push to `main`, every pull request, and on demand.

The alternative was to leave `scripts/` excluded entirely and restate the five commands as five
named steps in the workflow. That reads better in the GitHub UI — a failing step is named in the
run summary rather than found in a log — and it was rejected anyway, because it produces two lists
of what "healthy" means, in two files, with nothing keeping them equal. They would agree on the day
they were written and drift silently afterwards: the day a sixth check is added locally and not in
CI is the day CI stops being the authority it exists to be. The repository already refuses this
shape of duplication for mathematics (ADR-014); a definition of done deserves the same treatment,
and it is the more dangerous of the two, because a drifted gate reports success.

The legibility that costs is bought back inside the script: `verify.sh` echoes each command before
running it, so the last line of a failed CI log names the gate that failed. `set -e` aborts
silently otherwise.

What the workflow keeps for itself is only what is genuinely about CI and not about health:
least-privilege `contents: read`, cancellation of superseded runs (unlike the docs deployment,
which must not be interrupted mid-deploy), and `uv sync --frozen`, so a `pyproject.toml` edited
without re-locking fails in CI instead of resolving to an environment nobody has run.

## Consequences

- The gate is single-sourced. Adding a check means editing `verify.sh`, and it reaches CI, the
  reviewer agent and the commit hook in the same commit — there is no second place to remember.
- `pyproject.toml`'s comment on the import-linter contracts becomes true as written.
- CI installs the dev group only. `jax` and `torch` are optional extras arriving in F3-A and F3-C;
  when a calibrator needs them, this workflow grows a matrix leg rather than a second workflow, and
  `verify.sh` stays the thing each leg runs.
- The two remaining scripts under `scripts/` are still invisible to a clone, which is correct: they
  configure agents on a developer's machine and say nothing about whether the code is sound.
- `docs.yml` is untouched. Building the documentation and judging the code are separate concerns
  with different permissions and different concurrency rules, and merging them would give the
  test job the Pages deployment token it must not have.
