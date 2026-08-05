# ADR-011: The published grid is nested tuples of floats

**Status:** Accepted · 2026-07 · project decision, not in Design.md

## Context

`VolGrid.vols` is a matrix of implied vols. numpy would be the ergonomic choice for the
consumer, which will do arithmetic on it. But `contracts/` is the published language, and
non-functional objective 2 of the design states that the domain is framework-agnostic and
that no tensor crosses a boundary.

## Decision

`tuple[tuple[float, ...], ...]`, with the axes as `tuple[float, ...]`. `contracts/` imports
no numpy at all, and an import-linter contract enforces it. The conversion to `np.asarray`
happens inside each consumer's ACL, which is already the only place allowed to touch DTOs.

## Alternatives considered

**A read-only `np.ndarray`.** No conversion needed and direct numerical ergonomics. But numpy
enters `contracts/`, immutability rests on `flags.writeable = False` — a convention rather
than a language guarantee — and `to_dict` still needs `tolist()` to be serializable. The rule
"contracts imports stdlib plus shared_kernel" would stop being mechanically checkable, which
is most of its value.

## Consequences

- The DTO is frozen, hashable and serializable with no conversion step.
- Conversion costs one pass over roughly a thousand floats at 1 Hz. Irrelevant.
- Validation of the grid (rectangular, positive, strictly increasing axes) is pure Python and
  needs no dependency.
- Consumers that want vectorized maths pay one `np.asarray` at the ACL, which is the correct
  place for it to happen anyway.
