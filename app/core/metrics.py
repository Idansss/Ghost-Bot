from __future__ import annotations

import time
from collections.abc import Callable

from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest

HTTP_REQUESTS_TOTAL = Counter(
    "ghost_http_requests_total",
    "Total HTTP requests",
    ["method", "path", "status"],
)

HTTP_REQUEST_LATENCY_SECONDS = Histogram(
    "ghost_http_request_latency_seconds",
    "HTTP request latency (seconds)",
    ["method", "path"],
    buckets=(0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0),
)

FEATURE_RUNS_TOTAL = Counter(
    "ghost_feature_runs_total",
    "Feature executions (bot + tasks)",
    ["feature", "outcome"],
)

FEATURE_DURATION_SECONDS = Histogram(
    "ghost_feature_duration_seconds",
    "Feature execution duration (seconds)",
    ["feature"],
    buckets=(0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0),
)

LLM_REQUESTS_TOTAL = Counter(
    "ghost_llm_requests_total",
    "LLM requests by provider/model/endpoint/outcome",
    ["provider", "model", "endpoint", "outcome"],
)

LLM_LATENCY_SECONDS = Histogram(
    "ghost_llm_latency_seconds",
    "LLM request latency (seconds)",
    ["provider", "model", "endpoint"],
    buckets=(0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0),
)

ABUSE_ACTIONS_TOTAL = Counter(
    "ghost_abuse_actions_total",
    "Abuse protection actions taken",
    ["action"],
)

FEEDBACK_EVENTS_TOTAL = Counter(
    "ghost_feedback_events_total",
    "User feedback events captured from bot replies",
    ["sentiment", "source", "reason"],
)


def record_feature(feature: str, *, ok: bool) -> None:
    FEATURE_RUNS_TOTAL.labels(feature=feature, outcome="ok" if ok else "error").inc()


class FeatureTimer:
    def __init__(self, feature: str) -> None:
        self.feature = feature
        self._started = 0.0

    def __enter__(self):
        self._started = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc, tb):
        elapsed = max(time.perf_counter() - self._started, 0.0)
        FEATURE_DURATION_SECONDS.labels(feature=self.feature).observe(elapsed)
        return False


class LLMTimer:
    def __init__(self, *, provider: str, model: str, endpoint: str) -> None:
        self.provider = provider
        self.model = model
        self.endpoint = endpoint
        self._started = 0.0

    def __enter__(self):
        self._started = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc, tb):
        elapsed = max(time.perf_counter() - self._started, 0.0)
        LLM_LATENCY_SECONDS.labels(provider=self.provider, model=self.model, endpoint=self.endpoint).observe(elapsed)
        return False


def record_abuse(action: str) -> None:
    ABUSE_ACTIONS_TOTAL.labels(action=action).inc()


def record_feedback(*, sentiment: str, source: str, reason: str) -> None:
    FEEDBACK_EVENTS_TOTAL.labels(sentiment=sentiment, source=source, reason=reason).inc()


def metrics_response():
    return generate_latest(), 200, {"Content-Type": CONTENT_TYPE_LATEST}


def instrument_route(path: str) -> str:
    # Keep cardinality low: collapse dynamic path params where possible.
    return path or "/"


def metrics_middleware_factory(app_name: str = "ghost") -> Callable:
    async def middleware(request, call_next):
        started = time.perf_counter()
        path = instrument_route(getattr(request.url, "path", ""))
        method = str(getattr(request, "method", "")).upper() or "GET"
        status = "500"
        try:
            response = await call_next(request)
            status = str(getattr(response, "status_code", 200))
            return response
        finally:
            elapsed = max(time.perf_counter() - started, 0.0)
            HTTP_REQUESTS_TOTAL.labels(method=method, path=path, status=status).inc()
            HTTP_REQUEST_LATENCY_SECONDS.labels(method=method, path=path).observe(elapsed)

    return middleware
