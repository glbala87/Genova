"""Async batch queue with dynamic batching and priority support.

Collects incoming inference requests and groups them into efficient batches
before forwarding them to the model.  This reduces per-request overhead and
improves GPU utilisation.

Usage::

    from genova.api.batch_queue import BatchQueue, DynamicBatcher

    batcher = DynamicBatcher(engine, max_batch_size=32, max_wait_ms=50)
    await batcher.start()

    # From an async request handler:
    future = batcher.submit({"sequences": ["ACGT"], "task": "embed"})
    result = await future

    await batcher.stop()
"""

from __future__ import annotations

import asyncio
import time
import uuid
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any, Callable, Dict, List, Optional

from loguru import logger


# ---------------------------------------------------------------------------
# Priority levels
# ---------------------------------------------------------------------------

class Priority(IntEnum):
    """Request priority levels (lower value = higher priority)."""
    CRITICAL = 0
    HIGH = 1
    NORMAL = 2
    LOW = 3


# ---------------------------------------------------------------------------
# Request / response wrappers
# ---------------------------------------------------------------------------

@dataclass
class _QueueItem:
    """An enqueued inference request."""
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    request: Dict[str, Any] = field(default_factory=dict)
    priority: Priority = Priority.NORMAL
    future: asyncio.Future = field(default_factory=lambda: asyncio.get_event_loop().create_future())
    enqueued_at: float = field(default_factory=time.monotonic)

    def __lt__(self, other: "_QueueItem") -> bool:
        """Priority queue ordering: lower priority value first, then FIFO."""
        if self.priority != other.priority:
            return self.priority < other.priority
        return self.enqueued_at < other.enqueued_at


# ---------------------------------------------------------------------------
# BatchQueue
# ---------------------------------------------------------------------------

class BatchQueue:
    """Priority queue that collects requests into batches.

    Parameters
    ----------
    max_batch_size : int
        Maximum number of requests in a batch.
    max_wait_ms : float
        Maximum milliseconds to wait before dispatching a partial batch.
    process_fn : callable, optional
        Async function that processes a batch of request dicts and returns a
        list of results (one per request).  If not set, :meth:`process_batch`
        must be overridden or set later.
    """

    def __init__(
        self,
        max_batch_size: int = 32,
        max_wait_ms: float = 50.0,
        process_fn: Optional[Callable] = None,
    ) -> None:
        self.max_batch_size = max_batch_size
        self.max_wait_ms = max_wait_ms
        self._process_fn = process_fn
        self._queue: asyncio.PriorityQueue = asyncio.PriorityQueue()
        self._running = False
        self._worker_task: Optional[asyncio.Task] = None

        # Metrics
        self.total_submitted: int = 0
        self.total_processed: int = 0
        self.total_batches: int = 0
        self.total_timeouts: int = 0

    # -- public API ----------------------------------------------------------

    def submit(
        self,
        request: Dict[str, Any],
        priority: Priority = Priority.NORMAL,
        timeout: Optional[float] = None,
    ) -> asyncio.Future:
        """Submit a request and return a Future for the result.

        Parameters
        ----------
        request : dict
            The inference request payload.
        priority : Priority
            Request priority (default NORMAL).
        timeout : float, optional
            Per-request timeout in seconds.  If the request is not processed
            within this time the future is cancelled.

        Returns
        -------
        asyncio.Future
            Resolves to the inference result.
        """
        try:
            loop = asyncio.get_running_loop()
            future: asyncio.Future = loop.create_future()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            future = loop.create_future()

        item = _QueueItem(
            request=request,
            priority=priority,
            future=future,
        )
        self._queue.put_nowait(item)
        self.total_submitted += 1

        if timeout is not None:
            asyncio.ensure_future(self._apply_timeout(item, timeout))

        return future

    async def _apply_timeout(self, item: _QueueItem, timeout: float) -> None:
        """Cancel the future if it hasn't resolved within *timeout* seconds."""
        await asyncio.sleep(timeout)
        if not item.future.done():
            item.future.cancel()
            self.total_timeouts += 1

    async def process_batch(self, batch: List[Dict[str, Any]]) -> List[Any]:
        """Process a batch of requests.

        Override this or provide *process_fn* at construction time.
        """
        if self._process_fn is not None:
            result = self._process_fn(batch)
            if asyncio.iscoroutine(result):
                return await result
            return result
        raise NotImplementedError("Provide process_fn or override process_batch")

    # -- lifecycle -----------------------------------------------------------

    async def start(self) -> None:
        """Start the background batch worker."""
        if self._running:
            return
        self._running = True
        self._worker_task = asyncio.ensure_future(self._worker_loop())
        logger.info(
            "BatchQueue started (batch_size={}, wait_ms={})",
            self.max_batch_size,
            self.max_wait_ms,
        )

    async def stop(self) -> None:
        """Stop the background worker and drain remaining items."""
        self._running = False
        if self._worker_task is not None:
            self._worker_task.cancel()
            try:
                await self._worker_task
            except asyncio.CancelledError:
                pass
        # Drain remaining
        await self._drain()
        logger.info("BatchQueue stopped.")

    # -- internal ------------------------------------------------------------

    async def _worker_loop(self) -> None:
        """Main loop: collect items and process in batches."""
        while self._running:
            items: List[_QueueItem] = []
            deadline = time.monotonic() + self.max_wait_ms / 1000.0

            # Collect up to max_batch_size items
            while len(items) < self.max_batch_size:
                remaining = max(deadline - time.monotonic(), 0)
                if remaining <= 0 and items:
                    break
                try:
                    item = await asyncio.wait_for(
                        self._queue.get(), timeout=max(remaining, 0.001)
                    )
                    if not item.future.cancelled():
                        items.append(item)
                except asyncio.TimeoutError:
                    break

            if not items:
                continue

            # Sort by priority (already sorted by PriorityQueue, but ensure)
            items.sort()

            # Process
            requests = [it.request for it in items]
            try:
                results = await self.process_batch(requests)
                for item, result in zip(items, results):
                    if not item.future.done():
                        item.future.set_result(result)
                self.total_processed += len(items)
                self.total_batches += 1
            except Exception as exc:
                for item in items:
                    if not item.future.done():
                        item.future.set_exception(exc)
                logger.error("Batch processing failed: {}", exc)

    async def _drain(self) -> None:
        """Process any remaining items in the queue."""
        items: List[_QueueItem] = []
        while not self._queue.empty():
            try:
                item = self._queue.get_nowait()
                if not item.future.cancelled():
                    items.append(item)
            except asyncio.QueueEmpty:
                break

        if items:
            requests = [it.request for it in items]
            try:
                results = await self.process_batch(requests)
                for item, result in zip(items, results):
                    if not item.future.done():
                        item.future.set_result(result)
            except Exception as exc:
                for item in items:
                    if not item.future.done():
                        item.future.set_exception(exc)


# ---------------------------------------------------------------------------
# DynamicBatcher
# ---------------------------------------------------------------------------

class DynamicBatcher:
    """High-level dynamic batcher wrapping :class:`BatchQueue` and an
    :class:`~genova.api.inference.InferenceEngine`.

    Parameters
    ----------
    engine : InferenceEngine
        Loaded inference engine.
    max_batch_size : int
        Maximum batch size (default 32).
    max_wait_ms : float
        Maximum wait before dispatching (default 50 ms).
    """

    def __init__(
        self,
        engine: Any = None,
        max_batch_size: int = 32,
        max_wait_ms: float = 50.0,
    ) -> None:
        self.engine = engine
        self._queue = BatchQueue(
            max_batch_size=max_batch_size,
            max_wait_ms=max_wait_ms,
            process_fn=self._process,
        )

    def configure(
        self,
        max_batch_size: Optional[int] = None,
        max_wait_ms: Optional[float] = None,
    ) -> None:
        """Update batching parameters at runtime.

        Parameters
        ----------
        max_batch_size : int, optional
            New maximum batch size.
        max_wait_ms : float, optional
            New maximum wait time in milliseconds.
        """
        if max_batch_size is not None:
            self._queue.max_batch_size = max_batch_size
        if max_wait_ms is not None:
            self._queue.max_wait_ms = max_wait_ms
        logger.info(
            "DynamicBatcher reconfigured: batch_size={}, wait_ms={}",
            self._queue.max_batch_size,
            self._queue.max_wait_ms,
        )

    async def start(self) -> None:
        """Start the background batcher."""
        await self._queue.start()

    async def stop(self) -> None:
        """Stop the background batcher."""
        await self._queue.stop()

    def submit(
        self,
        request: Dict[str, Any],
        priority: Priority = Priority.NORMAL,
        timeout: Optional[float] = None,
    ) -> asyncio.Future:
        """Submit an inference request.

        Parameters
        ----------
        request : dict
            Must contain ``"task"`` (one of ``"embed"``, ``"expression"``,
            ``"methylation"``, ``"variant"``) and ``"sequences"`` (list of
            DNA strings).
        priority : Priority
            Request priority.
        timeout : float, optional
            Per-request timeout in seconds.

        Returns
        -------
        asyncio.Future
            Resolves to the prediction result.
        """
        return self._queue.submit(request, priority=priority, timeout=timeout)

    def _process(self, batch: List[Dict[str, Any]]) -> List[Any]:
        """Dispatch a batch of requests to the engine.

        Groups by task type and processes each group.
        """
        results: List[Any] = [None] * len(batch)

        # Group by task
        task_groups: Dict[str, List[int]] = {}
        for idx, req in enumerate(batch):
            task = req.get("task", "embed")
            task_groups.setdefault(task, []).append(idx)

        for task, indices in task_groups.items():
            all_seqs: List[str] = []
            for idx in indices:
                seqs = batch[idx].get("sequences", [])
                all_seqs.extend(seqs)

            if not all_seqs:
                for idx in indices:
                    results[idx] = []
                continue

            try:
                if task == "variant":
                    mid = len(all_seqs) // 2
                    preds = self.engine.predict_variant(all_seqs[:mid], all_seqs[mid:])
                elif task == "expression":
                    preds = [p.tolist() for p in self.engine.predict_expression(all_seqs)]
                elif task == "methylation":
                    preds = [p.tolist() for p in self.engine.predict_methylation(all_seqs)]
                else:
                    preds = [e.tolist() for e in self.engine.embed(all_seqs)]

                # Distribute results back
                offset = 0
                for idx in indices:
                    n = len(batch[idx].get("sequences", []))
                    results[idx] = preds[offset: offset + n]
                    offset += n
            except Exception as exc:
                for idx in indices:
                    results[idx] = {"error": str(exc)}

        return results

    @property
    def stats(self) -> Dict[str, Any]:
        """Return batching statistics."""
        return {
            "total_submitted": self._queue.total_submitted,
            "total_processed": self._queue.total_processed,
            "total_batches": self._queue.total_batches,
            "total_timeouts": self._queue.total_timeouts,
            "pending": self._queue._queue.qsize(),
            "max_batch_size": self._queue.max_batch_size,
            "max_wait_ms": self._queue.max_wait_ms,
        }
