from __future__ import annotations

import threading

from volengine.platform.executors import NamedExecutors


def test_the_same_producer_always_gets_the_same_pool() -> None:
    executors = NamedExecutors()

    assert executors.for_producer("svi-jax") is executors.for_producer("svi-jax")

    executors.shutdown()


def test_different_producers_get_different_pools() -> None:
    """ADR-005: this separation is the whole point. A shared pool would let one calibrator
    delay the other and make their latencies impossible to attribute.
    """
    executors = NamedExecutors()

    assert executors.for_producer("svi-jax") is not executors.for_producer("neural-torch")

    executors.shutdown()


def test_threads_are_named_after_their_producer() -> None:
    """So a stack trace or a profile says which calibrator was running."""
    executors = NamedExecutors()

    thread = executors.for_producer("svi-jax").submit(threading.current_thread).result()

    assert thread.name.startswith("volengine-svi-jax")

    executors.shutdown()


def test_shutdown_releases_every_pool() -> None:
    executors = NamedExecutors()
    pool = executors.for_producer("svi-scipy")

    executors.shutdown()

    assert executors.for_producer("svi-scipy") is not pool
