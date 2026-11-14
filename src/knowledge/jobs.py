"""Serial ingestion worker for the single-process local application."""

import threading
from concurrent.futures import Future, ThreadPoolExecutor

from .store import KnowledgeStore

_lock = threading.RLock()
_executor: ThreadPoolExecutor | None = None
_pending: dict[tuple[str, str], Future] = {}


def submit_ingestion(store: KnowledgeStore, version_id: str) -> Future:
    global _executor
    key = (str(store.root), version_id)
    with _lock:
        previous = _pending.get(key)
        if previous is not None and not previous.done():
            return previous
        if _executor is None:
            _executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="knowledge-ingestion")
        future = _executor.submit(store.process, version_id)
        _pending[key] = future

        def completed(done):
            with _lock:
                if _pending.get(key) is done:
                    _pending.pop(key, None)

        future.add_done_callback(completed)
        return future


def shutdown_jobs(*, wait: bool = False) -> None:
    global _executor
    with _lock:
        executor, _executor = _executor, None
    if executor is not None:
        # Canceled queued jobs remain in SQLite and become explicitly retryable
        # errors on the next process start. In-flight parsers finish safely.
        executor.shutdown(wait=wait, cancel_futures=True)
