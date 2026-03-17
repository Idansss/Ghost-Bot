## Observability (metrics, logs, traces)

### Metrics endpoints
- `/metrics` is **restricted** (same authorization as `/tasks/*`):
  - `Authorization: Bearer $CRON_SECRET` or `x-cron-secret: $CRON_SECRET`

### Key metrics (Prometheus)
#### HTTP
- `ghost_http_requests_total{method,path,status}`
- `ghost_http_request_latency_seconds_bucket{method,path,le=...}`

#### Feature outcomes
- `ghost_feature_runs_total{feature,outcome}` where outcome is `ok|error`

#### Abuse protection
- `ghost_abuse_actions_total{action}`

### Suggested alert rules (PromQL examples)
High error rate (HTTP 5xx):

```promql
sum(rate(ghost_http_requests_total{status=~"5.."}[5m]))
/
sum(rate(ghost_http_requests_total[5m]))
> 0.05
```

Latency p95 increase:

```promql
histogram_quantile(
  0.95,
  sum by (le) (rate(ghost_http_request_latency_seconds_bucket[5m]))
)
> 1.0
```

Feature failures (example: analysis):

```promql
sum(rate(ghost_feature_runs_total{feature="analysis",outcome="error"}[10m])) > 0
```

Abuse blocking spike:

```promql
sum(rate(ghost_abuse_actions_total{action=~"auto_block|admin_block"}[10m])) > 0
```

### Tracing (OpenTelemetry)
Enable:
- `OTEL_ENABLED=true`
- `OTEL_EXPORTER_OTLP_ENDPOINT=https://collector/v1/traces`
- Optional `OTEL_EXPORTER_OTLP_HEADERS=Authorization=Bearer xxx`

Once enabled, you get spans for:
- inbound FastAPI requests
- outbound httpx requests

