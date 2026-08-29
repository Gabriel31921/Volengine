"""The newest surface per market and producer, held in memory.

The implementation of ``SurfaceProvider`` the pipeline actually runs, and the composite Design
7.3 needs: one object serving every market and every producer, keyed by what arrives, so Risk
grows no registry and no per-market instance of anything.

**It lives in ``application/`` rather than in ``adapters/``, and that is rule 5 rather than a
filing accident.** Everything this class does is receive published DTOs, so putting it beside the
bus adapters would mean a second module in the context that knows the published language. It talks
to no technology: no socket, no file, no asyncio primitive. What feeds it is an adapter's problem
-- a subscription, a replay of a recording, a test calling :meth:`accept` directly -- and none of
those three needs to be told apart from here.

**Last value wins, and nothing is queued.** That is the bus's own semantics (ADR-003) carried
through to its consumer: a stale surface has no value once a newer one exists, and a report built
from the second-newest would answer a question about a market that has moved. Note what follows
for a republished stale surface -- it overwrites the fresh one it is standing in for, because it
*is* the newest thing the producer has said, and the report's freshness verdict is computed from
its timestamp rather than from how it got here.
"""

from __future__ import annotations

from volengine.contracts.calibrated_surface import CalibratedSurface
from volengine.risk.application.acl import to_surface_view
from volengine.risk.domain.surface_view import SurfaceView


class LastValueSurfaceProvider:
    """Every producer's most recent surface, one slot each.

    Satisfies ``SurfaceProvider`` structurally, so nothing here inherits from it and the domain
    never learns this class exists.

    Not thread-safe, and it does not need to be: it is written from the event loop that consumes
    the bus and read from the same loop when a report is computed.
    """

    def __init__(self) -> None:
        self._by_market: dict[str, dict[str, SurfaceView]] = {}

    def accept(self, surface: CalibratedSurface) -> None:
        """Take a newly published surface, translating it once on the way in.

        Translating here rather than on each read is the point of the cache: a report reads the
        provider once per position and the conversion to total variance is a multiply per node, so
        doing it on arrival pays it once per publication instead of once per lookup.

        Raises:
            ValueError: If the surface cannot be expressed as a ``SurfaceView`` -- notably a front
                expiry that has run out between the fit and now. Deliberately not swallowed: a
                cache that quietly dropped surfaces would leave the report saying "no surface"
                with nothing anywhere to say why, which is the failure mode this whole context is
                built to make impossible.
        """
        view = to_surface_view(surface)
        self._by_market.setdefault(view.market_id, {})[view.producer_id] = view

    def latest(self, market_id: str) -> SurfaceView | None:
        """The newest surface held for this market, or ``None``.

        **When several producers have published, the one with the newest ``ts_snapshot`` wins**,
        and ties are broken by nothing -- the first in insertion order stays. That rule exists
        because the port asks a question with one answer, and it is the honest reading of "the
        most recent surface held for this market": recency is a property of the market data behind
        a surface, not of when it happened to arrive.

        It is also why the comparative report of Design 7.3 does **not** go through this method.
        Valuing one portfolio under two producers means asking for a *named* producer twice, and
        that is what :meth:`for_producer` is for. Leaving ``latest`` to pick a winner keeps the
        single-producer path -- the walking skeleton, the CLI's one-line report -- free of a
        choice nobody wanted to make.

        Returns ``None`` when nothing has been published for this market, which is the ordinary
        state at start-up and not an error: the port documents it as an answer, and it pairs with
        ``FreshnessDecision.REJECT`` to form the two honest ways of having nothing to say.
        """
        held = self._by_market.get(market_id)
        if not held:
            return None
        return max(held.values(), key=lambda view: view.ts_snapshot)

    def for_producer(self, market_id: str, producer_id: str) -> SurfaceView | None:
        """The newest surface from one named producer, or ``None``.

        The comparative report's entry point: the same portfolio valued off ``svi-scipy`` and off
        ``mlp-torch`` is two calls to this, and the difference between the two reports is the
        measurement Design 7.3 exists to take. Deliberately **not** part of the
        ``SurfaceProvider`` port -- the domain asks for "the surface for this market" and must not
        learn that producers are something a caller can enumerate, or the moment it can name them
        is the moment a rule can branch on one.
        """
        return self._by_market.get(market_id, {}).get(producer_id)
