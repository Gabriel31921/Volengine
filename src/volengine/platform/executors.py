"""Thread pools that keep two calibrators from contending for the same queue (ADR-005).

Everything in this engine runs on one asyncio event loop, and a calibration is the one thing
that would block it for a meaningful time. Pushing that work to threads keeps ingestion
responsive; giving each producer its own pool keeps their latencies attributable, which is
the measurement the whole project turns on.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor


class NamedExecutors:
    """One thread pool per producer (ADR-005).

    JAX and PyTorch release the GIL while doing numerical work, so separate pools give real
    parallelism. A single shared pool would queue a neural fine-tuning step ahead of a
    parametric calibration and make each producer's latency impossible to attribute.

    One worker per pool on purpose: calibrations of the same producer must stay sequential,
    because the use case chains warm-start state from one snapshot to the next. Two of them
    racing would interleave that state. Parallelism here is *across* producers, not within.

    Keyed by producer and not by market, which is ADR-005 read literally: with two markets fitted
    by one producer, both queue on that producer's one worker, each with its own calibration state
    -- the state is per market, the thread is per producer. ``entrypoints/pipeline.py`` looks a
    pool up on every fit rather than holding one, because :meth:`shutdown` forgets the pools and
    the next lookup is what builds a fresh one for the next run.
    ``tests/entrypoints/test_pipeline.py`` asserts both the parallelism across producers and the
    sharing across markets (F3-F).
    """

    def __init__(self, max_workers_per_pool: int = 1) -> None:
        self._max_workers = max_workers_per_pool
        self._pools: dict[str, ThreadPoolExecutor] = {}

    def for_producer(self, producer_id: str) -> ThreadPoolExecutor:
        """The pool of this producer, created on first use.

        Threads are named after the producer so a stack trace or a profile says which
        calibrator was running.
        """
        pool = self._pools.get(producer_id)
        if pool is None:
            pool = ThreadPoolExecutor(
                max_workers=self._max_workers,
                thread_name_prefix=f"volengine-{producer_id}",
            )
            self._pools[producer_id] = pool
        return pool

    def shutdown(self, wait: bool = True) -> None:
        """Close every pool and forget them.

        Args:
            wait: Let running calibrations finish first. Leave it true on a normal stop --
                a fit killed mid-flight leaves the producer's warm-start state torn between
                two snapshots, and the next start would resume from it.
        """
        for pool in self._pools.values():
            pool.shutdown(wait=wait)
        self._pools.clear()
