## Security posture (Ghost Alpha Bot)

This document describes the threat model, current mitigations, and operator responsibilities for running Ghost Alpha Bot safely.

### Assets we protect
- **Bot token** (`TELEGRAM_BOT_TOKEN`): full control of bot identity.
- **User data** (DB): Telegram chat IDs, alerts, journal/portfolio entries, giveaway state, wallet entries (if saved).
- **Infrastructure credentials**: Postgres/Redis URLs and secrets.
- **System availability**: preventing spam/flooding and expensive workloads from taking the bot down.

### Trust boundaries
- **Telegram → Webhook/Long-polling → Bot handlers**: untrusted input (messages, callback data).
- **HTTP API (FastAPI)**:
  - Public endpoints: `/health`, `/ready`, Telegram webhook path.
  - Restricted endpoints: `/metrics`, `/admin/stats`, `/tasks/*`, `/test/mock-price` (test mode only).
- **Outbound HTTP** to exchanges, RSS/news sources, and LLM providers: untrusted upstreams.
- **Redis**: shared cache + coordination; compromise impacts rate limits, locks, and dedupe.
- **Postgres**: source of truth for users, alerts, giveaways, journals, portfolios.

### Major threats and mitigations

#### 1) Webhook abuse / flood (DoS)
- **Mitigations**
  - Per-IP webhook rate limiting (`/telegram/webhook`).
  - Request-level rate limiting for chat interactions.
  - Per-user backpressure locks for heavy actions (analysis/scans/news/etc.).
  - Graceful degradation: serve cached results when upstream fetch fails.
  - Auto-block repeated rate-limit violators (Redis-backed abuse strikes + temporary blocklist).
- **Operator actions**
  - Run behind a reverse proxy/WAF if exposed publicly.
  - Prefer webhook secret token (`TELEGRAM_WEBHOOK_SECRET`) in production.

#### 2) Unauthorized task/metrics/admin endpoints
- **Mitigations**
  - `/tasks/*`, `/metrics`, `/admin/stats` require cron authorization (`CRON_SECRET`) or native Vercel cron header.
  - `/test/*` endpoints gated by `TEST_MODE`.
- **Operator actions**
  - Set a strong `CRON_SECRET` in staging/prod.
  - Disable `TEST_MODE` in staging/prod.

#### 3) Duplicate execution in multi-instance deployments
- **Mitigations**
  - Redis distributed locks around scheduled tasks.
  - Idempotency keys for alerts to prevent duplicates.
- **Operator actions**
  - Ensure all instances share the same Redis.
  - Monitor Redis availability; degraded Redis reduces safety controls.

#### 4) Secrets leakage
- **Mitigations**
  - Strict startup validation for required settings in staging/prod.
  - Structured logging avoids dumping large objects; do not log secrets.
- **Operator actions**
  - Never commit `.env`.
  - Use managed secrets (Vercel env vars / GitHub secrets / secret manager).
  - Rotate `TELEGRAM_BOT_TOKEN` and `CRON_SECRET` on suspected exposure.

#### 5) Injection / unsafe parsing
- **Mitigations**
  - DB access uses SQLAlchemy; avoid interpolating raw SQL.
  - Intent/entity parsing is mostly deterministic; LLM router is optional and guarded.
- **Operator actions**
  - Keep dependencies updated; run CI checks.

#### 6) Outbound dependency failures / malicious upstream responses
- **Mitigations**
  - Resilient outbound HTTP policy: retries + backoff + per-host circuit breaker.
  - Timeouts on outbound calls.
  - Cached fallbacks for some features.
- **Operator actions**
  - Prefer reliable upstream endpoints; consider running your own RPC where applicable.

### Recommended production settings
- `ENV=prod`
- `SERVERLESS_MODE=true` on Vercel (disables polling + APScheduler).
- `TELEGRAM_USE_WEBHOOK=true` with:
  - `TELEGRAM_WEBHOOK_URL`
  - `TELEGRAM_WEBHOOK_SECRET` (recommended)
- `CRON_SECRET` (required for `/tasks/*`, `/metrics`, `/admin/stats`)
- `TEST_MODE=false`

### Observability (security-relevant)
- Prometheus metrics (restricted endpoint): request rates, latency, feature outcomes.
- Abuse metrics: `ghost_abuse_actions_total{action=...}`.
- Optional distributed tracing (OpenTelemetry) via `OTEL_ENABLED=true`.

### Incident response (quick checklist)
- **Bot token exposed**: rotate `TELEGRAM_BOT_TOKEN`, revoke old token, redeploy.
- **Cron secret exposed**: rotate `CRON_SECRET`, invalidate old secret, redeploy.
- **Spam/flood**: increase rate limits conservatively, enable/raise auto-block, add WAF rules.
- **DB compromise suspected**: rotate DB credentials, snapshot DB, review access logs, notify users if required.

