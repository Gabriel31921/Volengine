"""The subscription loop, written once and shared by every consumer of the bus.

Three use cases in this engine react to events -- calibrate, train, report -- and all three are
synchronous handlers: one event in, the events it caused out. Something still has to sit on a
subscription and call them, and this is that something.

**Why the loop is here rather than inside each use case.** A use case that owned its own loop would
be an object mixing two jobs: waiting for I/O and doing arithmetic. Three costs follow, and they
are the reason the split is worth a module of its own:

* **Tests.** Asserting "a snapshot over the RMSE threshold publishes a failure" against a loop
  means an event loop, a fake bus, a publish and then a wait for "it has processed now" -- and that
  last step has no non-racy spelling. Against a handler it is a function call.
* **ADR-004.** Replaying a recording is a ``for`` statement over recorded events, feeding the same
  handlers. With the loop inside, a replay would have to simulate the bus.
* **ADR-005.** The heavy work is CPU-bound and belongs on a named thread pool. Whoever owns the
  loop is whoever calls ``run_in_executor``, so keeping it out here lets the composition root put a
  handler on a pool without the handler ever learning that threads exist. It does exactly that, in
  its own closure -- a helper here could not, because publishing what a handler returned is the
  composition root's job and a generic wrapper would have to discard it.

**It knows nothing about what it carries.** The handler is a coroutine function taking an ``Event``,
so this class is as useful to Risk as to either producer, and it has no idea which. What it does
know is the one thing every consumer of a conflating bus has to: that events are dropped by design
(ADR-003), so it counts what it processed rather than assuming it saw everything.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable

from volengine.contracts.events import Event
from volengine.platform.bus import Subscription
from volengine.platform.metrics import MetricsSink

type EventHandler = Callable[[Event], Awaitable[None]]
"""What a runner drives: a coroutine taking one event and returning nothing.

Asynchronous even though every use case behind it is synchronous, and that is the seam ADR-005
needs: the composition root wraps a synchronous handler in an ``async def`` that awaits an executor,
and the runner is unchanged. Typing it synchronous would force the pool decision back up here,
where the runner would have to know which handlers are expensive.
"""


class BusRunner:
    """One subscription, one handler, until cancelled.

    Deliberately tiny and deliberately not configurable. Everything a real consumer needs to decide
    -- which topic, which handler, whether the work goes to a pool -- is decided by the composition
    root and arrives in the constructor, so there is nothing here to get wrong per context.
    """

    def __init__(
        self,
        subscription: Subscription,
        handler: EventHandler,
        metrics: MetricsSink,
        name: str,
    ) -> None:
        """Wire the loop.

        Args:
            subscription: The mailbox to read. Opened by the composition root, because naming a
                topic is routing and routing is not this class's business.
            handler: What to do with each event.
            metrics: Where the counts go.
            name: What to tag them with -- ``"svi-scipy"``, ``"risk"``. A label rather than an
                identity: two runners with one name would sum into one series, which is a wiring
                mistake rather than something to defend against here.

        Raises:
            ValueError: If ``name`` is empty, which would make every metric it emits
                unattributable.
        """
        if not name:
            raise ValueError("A runner must be named, so its metrics can be attributed")
        self._subscription = subscription
        self._handler = handler
        self._metrics = metrics
        self._name = name

    async def run(self) -> None:
        """Receive and handle, forever, until the task is cancelled.

        **An exception from the handler is counted and swallowed, and the loop goes on.** That is
        the opposite of the choice ``IngestStreamUseCase`` makes, where a bad update is allowed to
        kill the stream, and the difference is what each failure means. There, an update for the
        wrong underlying is a wiring bug in the composition root: it will recur on every message,
        and failing loudly on the first is the only way anyone finds out. Here, the handler has
        already converted every outcome it has an opinion about into an event -- a refused
        calibration, a rejected report -- so an exception that reaches this line is something
        nobody anticipated, about *one* event. Letting it stop the loop would take a market off the
        air permanently in response to a single message the bus is about to overwrite anyway.

        ``CancelledError`` is not caught, and the omission is deliberate rather than an oversight:
        it is how the composition root shuts a runner down, it is not an error, and swallowing it
        would produce a task that cannot be stopped. It inherits from ``BaseException`` in modern
        Python, so ``except Exception`` already lets it through -- stated here because that is a
        detail readers are entitled to see rather than remember.
        """
        while True:
            event = await self._subscription.receive()
            try:
                await self._handler(event)
            except Exception:
                self._metrics.counter("runner.handler_failed", subscriber=self._name)
            else:
                self._metrics.counter("runner.handled", subscriber=self._name)
