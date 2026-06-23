"""Request-ID middleware and global exception handler."""
from __future__ import annotations

import logging
import time
import uuid

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from app.core.logging import request_id_var

logger = logging.getLogger(__name__)


class RequestIDMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        rid = request.headers.get("X-Request-ID", str(uuid.uuid4()))
        request_id_var.set(rid)
        request.state.request_id = rid

        start = time.perf_counter()
        response = await call_next(request)
        elapsed_ms = (time.perf_counter() - start) * 1000

        response.headers["X-Request-ID"] = rid
        logger.info(
            "method=%s path=%s status=%d elapsed_ms=%.1f",
            request.method, request.url.path, response.status_code, elapsed_ms,
        )
        return response


def install_exception_handlers(app: FastAPI):
    @app.exception_handler(Exception)
    async def _unhandled(request: Request, exc: Exception):
        rid = getattr(request.state, "request_id", "")
        logger.exception("Unhandled exception request_id=%s", rid)
        return JSONResponse(
            status_code=500,
            content={
                "error": "internal_server_error",
                "detail": "An unexpected error occurred.",
                "request_id": rid,
            },
        )
