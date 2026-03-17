import json
import logging
import sys
from datetime import UTC, datetime
from typing import Any


class JsonFormatter(logging.Formatter):
    _RESERVED: set[str] = {
        "name",
        "msg",
        "args",
        "levelname",
        "levelno",
        "pathname",
        "filename",
        "module",
        "exc_info",
        "exc_text",
        "stack_info",
        "lineno",
        "funcName",
        "created",
        "msecs",
        "relativeCreated",
        "thread",
        "threadName",
        "processName",
        "process",
        "message",
    }

    def _safe_value(self, value: Any) -> Any:
        if value is None:
            return None
        if isinstance(value, (str, int, float, bool)):
            return value
        if isinstance(value, (list, tuple)):
            return [self._safe_value(v) for v in value[:50]]
        if isinstance(value, dict):
            out: dict[str, Any] = {}
            for k, v in list(value.items())[:50]:
                out[str(k)] = self._safe_value(v)
            return out
        return str(value)

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": datetime.now(UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        # Common top-level keys we expect
        for key in ("event", "error", "request_id", "user_id", "chat_id", "alert_id", "symbol", "latency_ms"):
            if hasattr(record, key):
                payload[key] = self._safe_value(getattr(record, key))

        # Include any other `extra={...}` fields automatically
        for key, value in record.__dict__.items():
            if key in payload or key in self._RESERVED:
                continue
            payload[key] = self._safe_value(value)

        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=True)


def setup_logging(level: str = "INFO") -> None:
    root = logging.getLogger()
    root.handlers.clear()
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter())
    root.addHandler(handler)
    root.setLevel(level.upper())
