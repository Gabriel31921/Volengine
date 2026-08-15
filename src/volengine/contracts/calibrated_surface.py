"""Published calibration contract: the surface both calibrators emit.

The output side of the boundary, and the one that decides how replaceable a producer is. SVI
on JAX and an MLP on PyTorch publish *this same type*, with nothing in it that would let a
consumer tell them apart -- if the contract favoured either one, the comparison the project
exists to run would be rigged from the start.

Hence a grid of already-evaluated vols rather than an evaluable object (ADR-001). An
``implied_vol(K, T)`` method would be more faithful and exact between nodes, but it drags the
producer's behaviour across the boundary: it cannot be serialized, cannot cross a process,
cannot be written to a recording and read back tomorrow. Accuracy between nodes is traded for
a contract that survives a ``json.dumps`` hop, and the mesh is sized so the interpolation
error stays far below the market's bid-ask noise.

No greeks either. A greek depends on a differentiation convention, a bump size and the
consumer's own model; publishing one would ship the producer's assumptions to everyone
downstream as though they were facts. Risk computes its own.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from itertools import pairwise
from typing import Any, Final

from volengine.shared_kernel.domain.instants import require_aware

SCHEMA_VERSION: Final[int] = 1


class SurfaceStatus(StrEnum):
    """How much this surface can be trusted, stated rather than inferred.

    The consumer must be able to distinguish three situations a bare grid of numbers makes
    look identical: a clean fit, a fit that met the acceptance criteria only barely, and a
    previous surface republished because this cycle's calibration failed.

    The member values are the wire format and must not change when a member is renamed.
    """

    OK = "OK"
    """Calibration succeeded and met every acceptance criterion."""

    DEGRADED = "DEGRADED"
    """Published, but something was wrong: a degraded input snapshot, or a marginal fit.

    Usable, and explicitly labelled so a consumer can widen its own margins or refuse it.
    """

    STALE_REPUBLISH = "STALE_REPUBLISH"
    """This cycle's calibration failed, so the previous surface is being republished (ADR-006).

    ``ts_calibrated`` therefore refers to an earlier fit than this event. Republishing beats
    silence: a consumer cannot distinguish a missing message from a quiet market, but it can
    act on a surface that says out loud that it is old.
    """


@dataclass(frozen=True, slots=True)
class VolGrid:
    """The surface as pure data: two axes and a matrix of implied vols.

    Nested tuples rather than a numpy array (ADR-011), which is the whole reason this package
    can forbid numpy. An array in the contract would hand every consumer a dependency on the
    producer's numerical stack, would not survive serialization without a codec, and -- being
    mutable -- would let a consumer corrupt a surface another consumer is still reading.
    Tuples are hashable, frozen and JSON-native, and the conversion cost is paid once in the
    consumer's ACL rather than on every read.
    """

    log_moneyness: tuple[float, ...]
    """Moneyness axis, ``log(K / F)``. Strictly increasing and finite, at least one node.

    Log-moneyness rather than strikes because it makes slices of different expiries and
    different forwards directly comparable, and puts at-the-money forward at exactly ``0.0``
    -- a legitimate value, so never test these for truthiness.
    """

    tenors: tuple[float, ...]
    """Tenor axis in years. Strictly increasing and finite, at least one node.

    Already resolved under the producing market's daycount (ADR-002); nothing downstream
    recomputes it from dates.
    """

    vols: tuple[tuple[float, ...], ...]
    """Implied vols in absolute terms (``0.65`` is 65%), indexed ``vols[tenor][moneyness]``.

    One row per tenor, one entry per moneyness node, every value strictly positive and finite.
    Rectangular by construction: a ragged grid would make interpolation ambiguous exactly
    where the surface is thinnest.

    Deliberately **not** checked for monotonicity or convexity. No-arbitrage is a model
    criterion, not a data one: it belongs to the calibrators, as a soft penalty in the loss
    and a hard gate before publishing (ADR-010). This contract guarantees structural
    coherence and nothing more.
    """

    def __post_init__(self) -> None:
        tenors_n = len(self.tenors)
        log_moneyness_n = len(self.log_moneyness)
        if tenors_n == 0:
            raise ValueError("The grid must have at least one tenor")
        if log_moneyness_n == 0:
            raise ValueError("The grid must have at least one moneyness node")
        if not all(math.isfinite(k) for k in self.log_moneyness):
            raise ValueError(f"The moneyness axis must be finite, got {self.log_moneyness}")
        if not all(math.isfinite(t) for t in self.tenors):
            raise ValueError(f"The tenor axis must be finite, got {self.tenors}")
        if not all(a < b for a, b in pairwise(self.log_moneyness)):
            raise ValueError("The moneyness axis must be a strictly increasing sequence")
        if not all(a < b for a, b in pairwise(self.tenors)):
            raise ValueError("The tenor axis must be a strictly increasing sequence")
        if tenors_n != len(self.vols):
            raise ValueError(
                "There should be one smile per tenor, got "
                f"{len(self.vols)} smiles for {tenors_n} tenors"
            )
        for i, smile in enumerate(self.vols):
            if len(smile) != log_moneyness_n:
                raise ValueError(
                    "Every smile must have one vol per moneyness node, got "
                    f"{len(smile)} vols for {log_moneyness_n} nodes at tenor {self.tenors[i]}"
                )
            for j, vol in enumerate(smile):
                if vol <= 0 or not math.isfinite(vol):
                    raise ValueError(
                        f"The volatility must be positive and finite, got {vol} at "
                        f"tenor {self.tenors[i]} and moneyness {self.log_moneyness[j]}"
                    )

        # No monotonicity or convexity check on the vols on purpose: no-arbitrage is a model
        # criterion, not a data one. It belongs to the calibrators, as a soft penalty in the
        # loss and as a hard gate before publishing (ADR-010). This contract only guarantees
        # that the structure is coherent.

    def to_dict(self) -> dict[str, Any]:
        return {
            "log_moneyness": list(self.log_moneyness),
            "tenors": list(self.tenors),
            "vols": [list(smile) for smile in self.vols],
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> VolGrid:
        return cls(
            log_moneyness=tuple(raw["log_moneyness"]),
            tenors=tuple(raw["tenors"]),
            vols=tuple(tuple(smile) for smile in raw["vols"]),
        )


@dataclass(frozen=True, slots=True)
class FitMetrics:
    """What the fit cost and how well it worked -- the basis for comparing two calibrators.

    Published in the contract rather than kept as internal telemetry, because "is the neural
    surface competitive with SVI?" is a question about accuracy *and* latency together, and a
    consumer that saw only the surface could not answer it. These are the numbers the
    comparison is actually decided on, so they travel with the result.
    """

    rmse_vol_bp: float
    """Root-mean-square fit error in vol basis points. Non-negative and finite.

    Basis points of volatility, not of price: ``1 bp`` is ``0.0001`` in absolute vol. Errors
    are expressed in vol space so they are comparable across strikes, where the same price
    error means wildly different things.
    """

    max_err_vol_bp: float
    """Worst single-quote error, same units. Never smaller than ``rmse_vol_bp``.

    The RMSE hides the wings, which is exactly where a calibrator fails first and where a
    consumer's risk is most sensitive.
    """

    n_quotes_used: int
    """Quotes that actually entered the fit. Strictly positive -- a fit on nothing is not a fit.

    Smaller than the snapshot's total when the calibrator excluded flagged quotes: ingestion
    flags, the calibrator weights or excludes, and this is where that decision becomes visible.
    """

    n_iterations: int
    """Optimiser iterations. Non-negative -- zero is legitimate when a warm start already
    converged, which is the normal case for a well-behaved market between snapshots."""

    duration_ms: float
    """Wall-clock time of the fit in milliseconds. Non-negative and finite.

    Zero is legitimate for a sub-microsecond calibration that rounds down; only negatives are
    impossible. Measures the fit alone, not ingestion or publication.
    """

    def __post_init__(self) -> None:
        # Zero is a legitimate value for every error and cost here: a noiseless synthetic
        # chain fits perfectly, a warm start can converge in zero extra iterations, and a
        # sub-microsecond calibration rounds to 0.0 ms. Only negatives are impossible.
        if self.rmse_vol_bp < 0 or not math.isfinite(self.rmse_vol_bp):
            raise ValueError(f"The RMSE must be non-negative and finite, got {self.rmse_vol_bp}")
        if self.max_err_vol_bp < 0 or not math.isfinite(self.max_err_vol_bp):
            raise ValueError(
                f"The maximum vol bp error must be non-negative and finite, "
                f"got {self.max_err_vol_bp}"
            )
        if self.rmse_vol_bp > self.max_err_vol_bp:
            raise ValueError(
                "The RMSE error can't be greater than the maximum error, they are, "
                f"respectively: {self.rmse_vol_bp}, {self.max_err_vol_bp}"
            )
        if self.n_quotes_used <= 0:
            raise ValueError(
                f"The number of quotes used must be positive, got {self.n_quotes_used}"
            )
        if self.n_iterations < 0:
            raise ValueError(
                f"The number of iterations must be non-negative, got {self.n_iterations}"
            )
        if self.duration_ms < 0 or not math.isfinite(self.duration_ms):
            raise ValueError(
                f"The fitting duration (ms) must be non-negative and finite, got {self.duration_ms}"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "rmse_vol_bp": self.rmse_vol_bp,
            "max_err_vol_bp": self.max_err_vol_bp,
            "n_quotes_used": self.n_quotes_used,
            "n_iterations": self.n_iterations,
            "duration_ms": self.duration_ms,
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> FitMetrics:
        return cls(
            rmse_vol_bp=raw["rmse_vol_bp"],
            max_err_vol_bp=raw["max_err_vol_bp"],
            n_quotes_used=raw["n_quotes_used"],
            n_iterations=raw["n_iterations"],
            duration_ms=raw["duration_ms"],
        )


@dataclass(frozen=True, slots=True)
class CalibratedSurface:
    """One calibration result: the surface, what it cost, and what produced it.

    The payload of ``SurfaceCalibrated`` and the only thing Risk ever receives about a
    volatility surface.
    """

    surface_id: str
    """Identity of this calibration result. Non-empty."""

    source_snapshot_id: str
    """The ``MarketSnapshot`` this was fitted to. Non-empty.

    The link that makes the whole pipeline attributable. The bus conflates and drops by
    design (ADR-003), so arrival order proves nothing: two surfaces from different producers
    are only comparable when this field says they saw the same market.
    """

    market_id: str
    """The market the source snapshot came from. Non-empty."""

    ts_snapshot: datetime
    """Exchange instant of the source snapshot. Timezone-aware.

    The instant the surface *describes*, which is what staleness is measured against.
    """

    ts_calibrated: datetime
    """When this fit finished. Timezone-aware, never earlier than ``ts_snapshot``.

    Equality is allowed: under a ``ManualClock`` the whole pipeline runs inside one tick, and
    forbidding it would make deterministic replay impossible (ADR-004). Under
    ``STALE_REPUBLISH`` this refers to the earlier fit being republished, not to now.
    """

    producer_id: str
    """Which calibrator produced it: ``"svi-scipy"``, ``"svi-jax"``, ``"neural-torch"``.

    For attribution and metrics only. **Nothing downstream may branch on it** -- the moment
    Risk treats one producer differently, the two stop being interchangeable and the
    comparison stops measuring the calibrators.
    """

    grid: VolGrid
    """The surface itself -- see :class:`VolGrid`."""

    fit: FitMetrics
    """Accuracy and cost of this fit -- see :class:`FitMetrics`."""

    status: SurfaceStatus
    """Trust level of this result -- see :class:`SurfaceStatus`."""

    producer_meta: Mapping[str, float] | None
    """Producer-specific extras: raw SVI parameters, a final loss, a learning rate. ``None``
    when there are none.

    An explicit escape hatch, and the one field in this contract with no shared meaning: its
    keys differ per producer and may change without a schema bump. Risk must not read it. It
    exists so diagnostics and the F3-E plots can reach producer internals without those
    internals leaking into the part of the contract everyone depends on.
    """

    schema_version: int = SCHEMA_VERSION
    """Wire format version, checked by :meth:`from_dict`."""

    def __post_init__(self) -> None:
        for name, value in (
            ("surface_id", self.surface_id),
            ("source_snapshot_id", self.source_snapshot_id),
            ("market_id", self.market_id),
            ("producer_id", self.producer_id),
        ):
            if not value:
                raise ValueError(f"The {name} must not be empty")
        require_aware(self.ts_snapshot, "ts_snapshot")
        require_aware(self.ts_calibrated, "ts_calibrated")
        # Equality is allowed: under a ManualClock the whole pipeline runs within one tick.
        if self.ts_calibrated < self.ts_snapshot:
            raise ValueError(
                "The surface can't be calibrated before the market was observed, snapshot "
                f"and calibration times are: {self.ts_snapshot}, {self.ts_calibrated}"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "surface_id": self.surface_id,
            "source_snapshot_id": self.source_snapshot_id,
            "market_id": self.market_id,
            "ts_snapshot": self.ts_snapshot.isoformat(),
            "ts_calibrated": self.ts_calibrated.isoformat(),
            "producer_id": self.producer_id,
            "grid": self.grid.to_dict(),
            "fit": self.fit.to_dict(),
            "status": self.status.value,
            "producer_meta": None if self.producer_meta is None else dict(self.producer_meta),
            "schema_version": self.schema_version,
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> CalibratedSurface:
        """Rebuild a surface from its primitive form, refusing a version we cannot read.

        Raises:
            ValueError: If ``schema_version`` is newer than this build supports. Reading an
                unknown format under an old interpretation would publish plausible nonsense,
                which is worse than failing.
        """
        version = raw.get("schema_version", 1)
        if version > SCHEMA_VERSION:
            raise ValueError(
                f"cannot read schema version {version}, this build supports {SCHEMA_VERSION}"
            )
        meta = raw["producer_meta"]
        return cls(
            surface_id=raw["surface_id"],
            source_snapshot_id=raw["source_snapshot_id"],
            market_id=raw["market_id"],
            ts_snapshot=datetime.fromisoformat(raw["ts_snapshot"]),
            ts_calibrated=datetime.fromisoformat(raw["ts_calibrated"]),
            producer_id=raw["producer_id"],
            grid=VolGrid.from_dict(raw["grid"]),
            fit=FitMetrics.from_dict(raw["fit"]),
            status=SurfaceStatus(raw["status"]),
            producer_meta=None if meta is None else dict(meta),
            schema_version=version,
        )
