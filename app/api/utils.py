"""Shared helpers for the API layer."""
from __future__ import annotations

import cv2
import numpy as np
from fastapi import HTTPException, UploadFile, status

from app.core import config


async def decode_frames(files: list[UploadFile], max_bytes: int | None = None) -> list[np.ndarray]:
    limit = max_bytes if max_bytes is not None else config.MAX_UPLOAD_SIZE_BYTES
    frames = []
    total_bytes = 0
    for f in files:
        raw = await f.read()
        total_bytes += len(raw)
        if total_bytes > limit:
            raise HTTPException(
                status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                f"total upload size exceeds {limit // (1024*1024)}MB limit",
            )
        arr = np.frombuffer(raw, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            raise HTTPException(status.HTTP_400_BAD_REQUEST, f"could not decode image: {f.filename}")
        frames.append(img)
    return frames
