"""
Concurrency-controlled processing queue for call analysis pipelines.

Uses an asyncio.Queue + asyncio.Semaphore to ensure at most
MAX_CONCURRENT_PIPELINES sessions are analysed simultaneously.
All state is in-process — no external broker required.
"""

import asyncio
import logging
import os

logger = logging.getLogger(__name__)

_MAX_CONCURRENT = int(os.getenv("MAX_CONCURRENT_PIPELINES", "3"))


class ProcessingQueue:
    def __init__(self, max_concurrent: int = _MAX_CONCURRENT) -> None:
        self._queue: asyncio.Queue[str] = asyncio.Queue()
        self._semaphore = asyncio.Semaphore(max_concurrent)
        self._processing: set[str] = set()
        self._drain_task: asyncio.Task | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def enqueue(self, session_id: str) -> None:
        """Add a session to the tail of the queue and ensure the drain loop is running."""
        self._queue.put_nowait(session_id)
        logger.info(f"[queue] enqueued session {session_id} (depth={self._queue.qsize()})")
        self._ensure_drain_running()

    def queue_stats(self) -> dict:
        return {
            "queued": self._queue.qsize(),
            "processing": len(self._processing),
        }

    def start(self) -> None:
        """Call once at application startup to begin draining."""
        self._ensure_drain_running()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _ensure_drain_running(self) -> None:
        if self._drain_task is None or self._drain_task.done():
            self._drain_task = asyncio.create_task(self._drain_loop())

    async def _drain_loop(self) -> None:
        """Continuously pop sessions from the queue and run the pipeline."""
        # Import here to avoid circular imports at module load time
        from app.pipelines.analysis_pipeline import run_analysis_pipeline

        while True:
            try:
                session_id = await asyncio.wait_for(self._queue.get(), timeout=30.0)
            except asyncio.TimeoutError:
                # No work arrived in 30 s — exit loop; it will restart on next enqueue.
                logger.debug("[queue] drain loop idle timeout, stopping")
                return

            # Acquire semaphore before starting the pipeline task
            await self._semaphore.acquire()
            self._processing.add(session_id)
            logger.info(
                f"[queue] starting pipeline for {session_id} "
                f"(active={len(self._processing)}, queued={self._queue.qsize()})"
            )

            async def _run(sid: str) -> None:
                try:
                    await run_analysis_pipeline(sid)
                except Exception:
                    logger.exception(f"[queue] pipeline error for session {sid}")
                finally:
                    self._processing.discard(sid)
                    self._semaphore.release()
                    logger.info(f"[queue] pipeline finished for {sid}")

            asyncio.create_task(_run(session_id))
            self._queue.task_done()


# Module-level singleton — imported everywhere
processing_queue = ProcessingQueue()
