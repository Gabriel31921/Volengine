"""The four things that can happen, as the bus carries them.

Events are envelopes, not documents. Each one wraps a payload from the sibling modules in
this package and adds nothing of its own except, where the payload is not a DTO already, the
minimum needed to act on it. That is why they hold no logic and almost no validation: an
event that reasoned about its payload would put business rules on the transport.

They are what makes the contexts genuinely decoupled. A producer publishes to a topic without
knowing who listens, or whether anyone does; a consumer subscribes without knowing what
produced what it receives. Nothing here may be relied on for delivery: the bus conflates,
each mailbox holds one event, and a slow consumer loses intermediate messages by design
(ADR-003). Anything that must be correlated is correlated by id -- ``source_snapshot_id`` --
never by arrival order.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from volengine.contracts.calibrated_surface import CalibratedSurface
from volengine.contracts.market_snapshot import MarketSnapshot


@dataclass(frozen=True, slots=True)
class SnapshotReady:
    """Market Data has assembled a snapshot worth calibrating.

    The trigger of the pipeline. Both calibrators subscribe to it and both receive the same
    payload, which is the precondition for comparing them at all.
    """

    snapshot: MarketSnapshot
    """The snapshot. Its ``quality`` block says how much it can be trusted."""


@dataclass(frozen=True, slots=True)
class SurfaceCalibrated:
    """A calibrator has produced a surface.

    Emitted identically by SVI and by the neural producer -- a consumer inspecting this event
    cannot tell which one sent it, other than by an id it is not allowed to branch on. Also
    emitted when a previous surface is republished after a failure (ADR-006), in which case
    ``surface.status`` is ``STALE_REPUBLISH``.
    """

    surface: CalibratedSurface
    """The surface, its fit metrics and its trust level."""


@dataclass(frozen=True, slots=True)
class ChainCompositionChanged:
    """The set of live instruments in a market has changed.

    New expiries listed, strikes added as spot moves, contracts expired. Separate from
    ``SnapshotReady`` because it changes what the *shape* of future snapshots will be, and a
    consumer may need to react before the next one arrives -- notably the JAX calibrator,
    whose padded fixed shapes exist precisely so a changing instrument count does not
    retrigger a JIT compile (ADR-009).
    """

    market_id: str
    """Which market's universe changed."""

    ts: datetime
    """When the change was observed. Timezone-aware."""

    instruments: tuple[str, ...]
    """The complete live set after the change, not a delta.

    A full replacement because the bus conflates: a consumer that missed the previous event
    could not reconstruct the current universe from a sequence of deltas, and would silently
    diverge. With the whole set, a single received event is enough to be correct.
    """


@dataclass(frozen=True, slots=True)
class CalibrationFailed:
    """A calibrator could not produce a surface for a snapshot.

    A first-class outcome, not an error channel. A failure is a data point in the comparison
    -- a calibrator that fails on hard chains is worse than one that does not, even when its
    successful fits are more accurate -- so it is published rather than logged and forgotten.
    """

    market_id: str
    """Which market's snapshot could not be calibrated."""

    source_snapshot_id: str
    """The snapshot that defeated it, so the failure is reproducible from a recording."""

    producer_id: str
    """Which calibrator failed. The one place this id is genuinely meant to be read."""

    reason: str
    """Human-readable cause. For diagnosis only -- never parsed, never branched on."""

    ts: datetime
    """When the failure occurred. Timezone-aware."""


type Event = SnapshotReady | SurfaceCalibrated | ChainCompositionChanged | CalibrationFailed
"""Everything the bus can carry.

A closed union on purpose: it lets the type checker prove a consumer's handling is exhaustive,
and it makes adding a fifth event a deliberate, reviewable act rather than an accident.
"""
