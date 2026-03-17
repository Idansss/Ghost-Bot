from __future__ import annotations

import logging
from typing import Any

try:
    from opentelemetry import trace
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
    from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
    from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor
except Exception:  # pragma: no cover
    trace = None  # type: ignore[assignment]
    OTLPSpanExporter = None  # type: ignore[assignment]
    FastAPIInstrumentor = None  # type: ignore[assignment]
    HTTPXClientInstrumentor = None  # type: ignore[assignment]
    Resource = None  # type: ignore[assignment]
    TracerProvider = None  # type: ignore[assignment]
    BatchSpanProcessor = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)


def _parse_headers(value: str) -> dict[str, str]:
    """Parse OTLP headers from 'k=v,k2=v2' form."""
    out: dict[str, str] = {}
    for chunk in (value or "").split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "=" not in chunk:
            continue
        k, v = chunk.split("=", 1)
        k = k.strip()
        v = v.strip()
        if k and v:
            out[k] = v
    return out


def configure_tracing(*, enabled: bool, service_name: str, otlp_endpoint: str, otlp_headers: str = "") -> None:
    """Best-effort OpenTelemetry setup.

    - If disabled, no-op.
    - If enabled but endpoint missing, keeps in-process tracing (no exporter).
    """
    if not enabled:
        return
    if trace is None or TracerProvider is None or Resource is None:
        logger.warning("otel_missing_dependencies", extra={"event": "otel_missing_dependencies"})
        return

    resource = Resource.create({"service.name": service_name or "ghost-bot"})
    provider = TracerProvider(resource=resource)
    trace.set_tracer_provider(provider)

    endpoint = (otlp_endpoint or "").strip()
    if endpoint and OTLPSpanExporter is not None and BatchSpanProcessor is not None:
        exporter = OTLPSpanExporter(endpoint=endpoint, headers=_parse_headers(otlp_headers))
        provider.add_span_processor(BatchSpanProcessor(exporter))
        logger.info("otel_configured", extra={"event": "otel_configured", "endpoint": endpoint})
    else:
        logger.info("otel_enabled_no_exporter", extra={"event": "otel_enabled_no_exporter"})


def instrument_app(app: Any) -> None:
    """Instrument FastAPI + httpx."""
    if FastAPIInstrumentor is None or HTTPXClientInstrumentor is None:
        logger.warning("otel_missing_dependencies", extra={"event": "otel_missing_dependencies"})
        return
    try:
        FastAPIInstrumentor.instrument_app(app)
    except Exception as exc:
        logger.warning("otel_fastapi_instrument_failed", extra={"event": "otel_fastapi_instrument_failed", "error": str(exc)})
    try:
        HTTPXClientInstrumentor().instrument()
    except Exception as exc:
        logger.warning("otel_httpx_instrument_failed", extra={"event": "otel_httpx_instrument_failed", "error": str(exc)})
