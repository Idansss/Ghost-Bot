## Operations runbook (Ghost Alpha Bot)

This runbook is for operators running Ghost Alpha Bot on a VPS (Docker) or serverless (Vercel).

### Environments
- **dev**: local development, permissive defaults.
- **staging**: production-like, safe testing.
- **prod**: production.

In **staging/prod**, the app fails fast if critical settings are missing (see `app/core/config.py`).

### Required secrets / credentials
- `TELEGRAM_BOT_TOKEN` (bot identity)
- `DATABASE_URL` (Postgres)
- `REDIS_URL` (Redis)
- `CRON_SECRET` (protects `/tasks/*`, `/metrics`, `/admin/*`)

Recommended:
- `TELEGRAM_WEBHOOK_SECRET` (prevents webhook spoofing)
- `OTEL_*` if you export traces

### Deploy (Vercel)
1. Set env vars in Vercel Project Settings.
2. Run migrations against your hosted Postgres:

```bash
python -m alembic upgrade head
```

3. Deploy.
4. Verify:
   - `GET /health`
   - `GET /ready`
   - `GET /health/deep` (DB/Redis/exchange ping best-effort)

### Deploy (VPS / Docker Compose)
1. Pull latest code.
2. Ensure `.env` is present on the host (never commit it).
3. Start/restart:

```bash
docker compose up -d --build
```

4. Verify endpoints as above.

### Rollback (Vercel)
- Redeploy the previous successful build from the Vercel UI.
- If a DB migration was applied and is incompatible, you may need a forward-fix migration rather than downgrade.

### Rollback (VPS / Docker)
- Checkout the previous git SHA, rebuild, and redeploy:

```bash
git checkout <sha>
docker compose up -d --build
```

### Migrations
- Apply:

```bash
python -m alembic upgrade head
```

- Create (autogenerate):

```bash
python tools/dev.py makemigration -m "add x"
```

Notes:
- Prefer **forward-only migrations** in production systems.
- If you must downgrade, validate downtime and data impact first.

### Backups (Postgres)
Recommended baseline:
- Daily logical backup (and before deploying schema changes).
- Keep at least 7–30 days retention depending on your needs.

Example (pg_dump, adjust credentials/host):

```bash
pg_dump "$DATABASE_URL" --format=custom --file "backup_$(date +%F).dump"
```

Restore example:

```bash
pg_restore --clean --if-exists --dbname "$DATABASE_URL" "backup_YYYY-MM-DD.dump"
```

### Redis operational notes
Redis is used for:
- rate limiting / abuse controls
- dedupe keys
- caches
- distributed locks for tasks

If Redis is down:
- some safety controls degrade (rate limits/locks/dedupe)
- bot may still run but becomes riskier to operate under load

### Key rotation
- Rotate `TELEGRAM_BOT_TOKEN` if compromised (BotFather → revoke token).
- Rotate `CRON_SECRET` on any suspicion; update all callers (GitHub Actions, cron jobs).
- Rotate DB/Redis credentials per your provider’s guidelines.

### Incident response
#### Flood/spam
1. Check rate limit and abuse metrics (`/metrics`).
2. Consider raising auto-block aggressiveness:
   - `ABUSE_STRIKES_TO_BLOCK`
   - `ABUSE_STRIKE_WINDOW_SEC`
   - `ABUSE_BLOCK_TTL_SEC`
3. Add WAF rules (IP blocks) if needed.

#### Elevated errors / degraded upstreams
1. Check `/health/deep`.
2. Look for circuit breaker openings in logs.
3. Temporarily reduce expensive features or enable cached fallbacks (already present for key flows).

#### Unauthorized requests to admin/task endpoints
1. Rotate `CRON_SECRET`.
2. Ensure endpoints are not publicly exposed without auth headers.

