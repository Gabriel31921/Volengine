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

    Only the abstraction is used in phase 1; it is wired for real in F3-F.
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
        for pool in self._pools.values():
            pool.shutdown(wait=wait)
        self._pools.clear()
