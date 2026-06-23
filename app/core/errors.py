"""Consistent error/denial response envelope across all endpoints."""
from __future__ import annotations

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.core.logging import request_id_var


def install_error_handlers(app: FastAPI):
    @app.exception_handler(StarletteHTTPException)
    async def _http_exc(request: Request, exc: StarletteHTTPException):
        rid = request_id_var.get("")
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "error": _status_to_code(exc.status_code),
                "detail": exc.detail,
                "request_id": rid,
            },
        )

    @app.exception_handler(RequestValidationError)
    async def _validation_exc(request: Request, exc: RequestValidationError):
        rid = request_id_var.get("")
        return JSONResponse(
            status_code=422,
            content={
                "error": "validation_error",
                "detail": exc.errors(),
                "request_id": rid,
            },
        )


def _status_to_code(status: int) -> str:
    return {
        400: "bad_request",
        401: "unauthorized",
        403: "forbidden",
        404: "not_found",
        409: "conflict",
        422: "validation_error",
        423: "locked",
        429: "too_many_requests",
        500: "internal_server_error",
        503: "service_unavailable",
    }.get(status, f"error_{status}")
