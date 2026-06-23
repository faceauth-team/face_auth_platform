"""In-memory enrollment progress tracker.

Lets the status endpoint report live progress while the background worker
processes the frame burst. Single-process only (same stand-in scope as the
BackgroundTasks queue); a real Celery/Kafka worker would publish progress to
Redis or the DB instead.
"""
from __future__ import annotations

import threading

_lock = threading.Lock()
_progress: dict[str, dict] = {}


def set_progress(enrollment_id: str, processed: int, total: int) -> None:
    with _lock:
        _progress[enrollment_id] = {"processed": processed, "total": total}


def get_progress(enrollment_id: str) -> dict | None:
    with _lock:
        return _progress.get(enrollment_id)


def clear_progress(enrollment_id: str) -> None:
    with _lock:
        _progress.pop(enrollment_id, None)
