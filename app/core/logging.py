"""Structured logging configuration with request-ID propagation."""
from __future__ import annotations

import logging
import sys
import uuid
from contextvars import ContextVar

request_id_var: ContextVar[str] = ContextVar("request_id", default="")


def setup_logging():
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(_StructuredFormatter())
    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(logging.INFO)
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)


class _StructuredFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        rid = request_id_var.get("")
        ts = self.formatTime(record)
        level = record.levelname
        msg = record.getMessage()
        parts = [f"ts={ts}", f"level={level}", f"logger={record.name}"]
        if rid:
            parts.append(f"request_id={rid}")
        parts.append(f"msg={msg}")
        if record.exc_info and record.exc_info[1]:
            parts.append(f"exc={record.exc_info[1]!r}")
        return " ".join(parts)


def get_request_id() -> str:
    rid = request_id_var.get("")
    if not rid:
        rid = str(uuid.uuid4())
        request_id_var.set(rid)
    return rid
