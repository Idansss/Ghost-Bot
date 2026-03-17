## High availability / multi-instance strategy

This bot is safe to run with multiple instances **if and only if** all instances share the same:
- **Postgres**
- **Redis**

### What scales horizontally
- **FastAPI** `/health`, `/ready`, Telegram webhook endpoint: safe to scale behind a load balancer.
- Telegram handler processing: safe to scale, because:
  - duplicate Telegram updates are deduped (Redis key on message/callback IDs)
  - heavy features use per-user Redis locks (analysis/news/rsi/ema/etc.)

### What must be single-writer (or lock-protected)
- **Scheduled/cron tasks** (`/tasks/*` endpoints and APScheduler workers):
  - Use Redis distributed locks to ensure only one instance runs each job at a time.
  - If you run multiple workers, keep locks enabled (already implemented).

### Redis is critical for correctness
Redis is used for:
- distributed locks (prevent duplicated work)
- dedupe keys (prevent double-processing)
- rate limiting and abuse protection
- caching and graceful degradation

If Redis is down, safety controls degrade and multi-instance correctness is not guaranteed.

### Recommended deployment modes
- **Vercel serverless**
  - `SERVERLESS_MODE=true` disables polling and APScheduler.
  - Use Vercel cron or GitHub Actions scheduler to call `/tasks/*`.
  - Always set `CRON_SECRET`.

- **VPS / Docker**
  - Run one (or more) API instances behind a reverse proxy.
  - Run either:
    - one combined instance (API + APScheduler), or
    - multiple API instances + one dedicated worker instance
  - If running multiple workers, locks prevent duplicate processing but you still waste CPU; prefer one worker.

### Failure modes
- **Two workers without locks**: duplicate alerts/giveaway finalization/scanner refreshes.
- **Redis outage**: duplicate work, lost rate limits, stale cache behavior.
- **DB outage**: bot may respond degraded; alerts processing and persistence fail.

