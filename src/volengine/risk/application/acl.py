"""The one module in Risk that speaks both languages (rule 5).

One direction only, and the asymmetry is the context. Market Data, Parametric Pricing and Neural
Surface each publish something; Risk publishes nothing onto the bus. A ``RiskReport`` leaves
through ``ReportWriter``, which is a port whose entire vocabulary is the domain's own -- so there
is no outbound translation here, and the absence is worth stating rather than looking like an
oversight.

**The conversion is volatilities to total variance, and it happens exactly once.** The
interpolation this whole context rests on is bilinear in ``w = vol**2 * T`` (Design 7.1), so
converting at the boundary makes every later lookup pure arithmetic: read four numbers, multiply.
Storing the published volatilities instead would mean a square and a multiply on every single
lookup -- twice per greek bump, five bumps per position -- and would leave two representations of
one surface in memory with nothing keeping them in agreement.

**What is deliberately not translated: the status.** ``CalibratedSurface`` carries a
``SurfaceStatus``, ``STALE_REPUBLISH`` included, and ``SurfaceView`` has no field for it. ADR-006
republishes a stale surface under its **original** ``ts_snapshot``, so the one timestamp the
freshness policy already reads says everything the label would have said, as a number that can be
compared rather than a word that has to be interpreted. A second channel for one fact only ever
disagrees with the first.
"""

from __future__ import annotations

from volengine.contracts.calibrated_surface import CalibratedSurface
from volengine.risk.domain.surface_view import SurfaceView


def to_surface_view(surface: CalibratedSurface) -> SurfaceView:
    """Translate a published surface into the only form this context knows.

    A field-by-field copy apart from the variance conversion, and one thing worth noticing about
    what it copies: ``producer_id`` travels *in the view* rather than being attached by whoever
    called this. A composite provider serves several producers from one object (Design 7.3), so
    the identity has to arrive with the surface or it will eventually be paired with the wrong one.

    Args:
        surface: The published result, exactly as it came off the bus.

    Returns:
        This context's own model: total variance on a moneyness-by-tenor grid, with the expiries
        and forwards that make a real position placeable on it.

    Raises:
        ValueError: If the published grid does not satisfy ``SurfaceView``'s own invariants.
            Two of them are genuinely stronger than the contract's and can fire here on a
            perfectly valid DTO:

            * ``SurfaceView`` requires every expiry to sit **strictly after** ``ts_snapshot``,
              while ``VolGrid`` only requires the expiries to be increasing. A surface whose front
              node has expired between the fit and this translation is refused, which is right --
              a node with negative time to run has no total variance -- and it is a market
              condition rather than a bug, so the use case is where it is handled.
            * The tenors must be strictly positive, which the contract also does not demand.

            Deliberately not caught and converted here. This module translates; deciding what to
            do about a surface that cannot be translated is the use case's, and it has a report to
            put the answer in.
    """
    return SurfaceView(
        market_id=surface.market_id,
        producer_id=surface.producer_id,
        surface_id=surface.surface_id,
        ts_snapshot=surface.ts_snapshot,
        log_moneyness=surface.grid.log_moneyness,
        tenors=surface.grid.tenors,
        expiries=surface.grid.expiries,
        forwards=surface.grid.forwards,
        total_variance=tuple(
            tuple(vol * vol * tenor for vol in smile)
            for tenor, smile in zip(surface.grid.tenors, surface.grid.vols, strict=True)
        ),
    )
