"""Semaphore-based concurrency limiter for GPU/CPU inference endpoints."""
from __future__ import annotations

import asyncio
import logging

from fastapi import HTTPException, status

from app.core import config

logger = logging.getLogger(__name__)

_semaphore: asyncio.Semaphore | None = None


def _get_semaphore() -> asyncio.Semaphore:
    global _semaphore
    if _semaphore is None:
        _semaphore = asyncio.Semaphore(config.MAX_CONCURRENT_INFERENCES)
    return _semaphore


async def acquire_inference_slot():
    sem = _get_semaphore()
    if sem.locked():
        logger.warning("Inference concurrency limit reached (%d)", config.MAX_CONCURRENT_INFERENCES)
        raise HTTPException(
            status.HTTP_429_TOO_MANY_REQUESTS,
            f"server busy — max {config.MAX_CONCURRENT_INFERENCES} concurrent inference requests",
        )
    await sem.acquire()
    return sem


def release_inference_slot(sem: asyncio.Semaphore):
    sem.release()
