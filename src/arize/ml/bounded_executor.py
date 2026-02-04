"""Bounded thread pool executor with queue size limits."""

from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from threading import BoundedSemaphore


class BoundedExecutor:
    """BoundedExecutor behaves as a ThreadPoolExecutor which will block on calls to submit().

    Blocks once the limit given as "bound" work items are queued for execution.

    :param bound: Integer - the maximum number of items in the work queue
    :param max_workers: Integer - the size of the thread pool
    """

    def __init__(self, bound: int, max_workers: int) -> None:
        """Initialize the bounded executor.

        Args:
            bound: Maximum number of items in the work queue.
            max_workers: Size of the thread pool.
        """
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.semaphore = BoundedSemaphore(bound + max_workers)

    def submit(
        self, fn: Callable[..., object], *args: object, **kwargs: object
    ) -> object:
        """Submit a callable to be executed with bounded concurrency.

        This method blocks if the work queue is full (at the bound limit) until
        space becomes available. Compatible with concurrent.futures.Executor.submit().

        Args:
            fn: The callable to execute.
            *args: Positional arguments to pass to the callable.
            **kwargs: Keyword arguments to pass to the callable.

        Returns:
            concurrent.futures.Future: A Future representing the pending execution.

        Raises:
            Exception: Any exception raised during submission is re-raised after
                releasing the semaphore.
        """
        self.semaphore.acquire()
        try:
            future = self.executor.submit(fn, *args, **kwargs)
        except Exception:
            self.semaphore.release()
            raise
        else:
            future.add_done_callback(lambda _: self.semaphore.release())
            return future

    def shutdown(self, wait: bool = True) -> None:
        """Shutdown the executor, optionally waiting for pending tasks to complete.

        Compatible with concurrent.futures.Executor.shutdown().

        Args:
            wait: If True, blocks until all pending tasks complete. If False,
                returns immediately without waiting. Defaults to True.
        """
        self.executor.shutdown(wait)
