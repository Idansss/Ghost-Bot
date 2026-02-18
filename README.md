# Ghost Alpha Bot

Production-ready Telegram bot for crypto analysis, alerts, wallet scans, cycle checks, trade verification, watchlists, news briefs, and BTC-correlation checks.

## What it does

- Trade plans from natural language (`SOL long`, `ETH short`, `Long?`) using TA + market narrative
- Price alerts (`ping me when SOL hits 100`) with one-shot trigger + anti-spam dedupe/cooldown
- RSI scanner (`rsi top 10 1h oversold`, `top rsi 4h overbought`)
- Wallet scans for Solana/Tron (`scan solana <addr>`, `scan tron <addr>`)
- Cycle checks (`cycle check`, `are we near top?`)
- Trade verification (`check this trade from yesterday: ...`) with same-candle ambiguity modes
- Pair/price discovery (`find pair xion`, `coin around 0.155`)
- Giveaway admin flow (`/giveaway ...`, `/join`) with anti-back-to-back winner rule
- Watchlists (`Coins to watch 5`)
- Daily news briefs (`latest news today`) with source links
- Correlation (`is BIRB following BTC?`) with corr + beta context
- User settings (`/settings`): anon/formal/profanity/risk/timeframe/exchange

This bot is analysis only. It does **not** place trades.

## Stack

- Python 3.11
- aiogram 3 (Telegram)
- FastAPI (health, readiness, webhook, test-mode endpoints)
- Postgres + SQLAlchemy + Alembic
- Redis (cache + rate limiting + dedupe)
- APScheduler (background jobs)
- httpx (resilient adapters with retry/backoff + circuit breaker)
- pandas/numpy (TA + analytics)
- pytest (unit tests)

## Repository layout

- `app/main.py` - bootstrap, FastAPI app, webhook/polling runtime
- `app/bot/handlers.py` - Telegram handlers (commands + NLU routing + callbacks)
- `app/core/nlu.py` - deterministic regex/heuristic intent/entity parser
- `app/services/` - analysis/alerts/wallet/news/watchlist/cycle/correlation/trade-verify
- `app/adapters/` - Binance/CoinGecko/Derivatives/RSS/Solana/Tron adapters
- `app/db/models.py` - DB schema models
- `app/db/migrations/versions/0001_initial.py` - initial Alembic migration
- `app/workers/scheduler.py` - periodic alert monitor + news refresh
- `tests/` - parser + indicator + tradecheck tests

## Quick start (Docker Compose)

1. Copy env:

```bash
cp .env.example .env
```

2. Set at minimum:

- `TELEGRAM_BOT_TOKEN`

3. Start:

```bash
docker compose up --build
```

4. Health checks:

- `GET http://localhost:8000/health`
- `GET http://localhost:8000/ready`

The app runs migrations on start (`alembic upgrade head`) then launches the bot + API.

## Long polling vs webhook

Default is long polling:

- `TELEGRAM_USE_WEBHOOK=false`

Webhook mode:

- Set `TELEGRAM_USE_WEBHOOK=true`
- Set `TELEGRAM_WEBHOOK_URL=https://your-domain.com`
- Optional: set `TELEGRAM_WEBHOOK_SECRET`
- Ensure your reverse proxy routes `POST /telegram/webhook` to port `8000`

## Environment variables

Required:

- `TELEGRAM_BOT_TOKEN`

Primary runtime:

- `DATABASE_URL` (default: `postgresql+asyncpg://ghost:ghost@postgres:5432/ghost_bot`)
- `REDIS_URL` (default: `redis://redis:6379/0`)
- `TELEGRAM_USE_WEBHOOK`
- `TELEGRAM_WEBHOOK_URL`
- `TELEGRAM_WEBHOOK_PATH`
- `TELEGRAM_WEBHOOK_SECRET`
- `OPENAI_API_KEY` (optional, enables freeform Q&A fallback)
- `OPENAI_MODEL` (default `gpt-4.1-mini`)
- `OPENAI_MAX_OUTPUT_TOKENS` (default `350`)
- `OPENAI_TEMPERATURE` (default `0.7`)
- `OPENAI_ROUTER_MIN_CONFIDENCE` (default `0.6`, LLM intent router execution threshold)

Data source/adapters:

- `BINANCE_BASE_URL`
- `BINANCE_FUTURES_BASE_URL`
- `COINGECKO_BASE_URL`
- `NEWS_RSS_FEEDS`
- `CRYPTOPANIC_API_KEY` (optional)
- `SOLANA_RPC_URL`
- `TRON_API_URL`
- `TRONGRID_API_KEY` (optional)

Limits/reliability:

- `REQUEST_RATE_LIMIT_PER_MINUTE` (default 20)
- `WALLET_SCAN_LIMIT_PER_HOUR` (default 10)
- `ALERTS_CREATE_LIMIT_PER_DAY` (default 10)
- `ALERT_CHECK_INTERVAL_SEC` (default 30)
- `ALERT_COOLDOWN_MIN` (default 30)
- `ADMIN_CHAT_IDS` (comma-separated Telegram user IDs allowed to run giveaway admin commands)
- `GIVEAWAY_MIN_PARTICIPANTS` (default 2)
- `ANALYSIS_FAST_MODE` (default `true`)
- `ANALYSIS_DEFAULT_TIMEFRAMES` (default `1h`)
- `ANALYSIS_INCLUDE_DERIVATIVES_DEFAULT` (default `false`)
- `ANALYSIS_INCLUDE_NEWS_DEFAULT` (default `false`)
- `ANALYSIS_REQUEST_TIMEOUT_SEC` (default `8`)

Behavior/test mode:

- `DEFAULT_TIMEFRAME`
- `INCLUDE_BTC_ETH_WATCHLIST`
- `TEST_MODE` (default true in `.env.example`)
- `MOCK_PRICES` (e.g. `SOL:100,BTC:70000`)

## Reliability and safety

- Retries + exponential backoff for HTTP adapters
- Circuit breaker per upstream host
- Redis cache keys:
  - `price:<symbol>`
  - `ohlcv:<symbol>:<tf>:<limit>`
  - `news:today`
  - `funding:<symbol>`
- Alert anti-spam:
  - one-shot status transition to `triggered`
  - minute-bucket dedupe key
  - cooldown timestamp
- Per-user rate limits for requests/scans/alerts
- Minimal PII storage: Telegram chat ID and optional saved wallets
- Wallet scans are public-chain-only and include a no-attribution warning

## Telegram commands

- `/start`
- `/help`
- `/settings`
- `/watchlist [N]`
- `/alert add <symbol> <above|below|cross> <price>`
- `/alert list`
- `/alert delete <id>`
- `/alert clear`
- `/alert pause`
- `/alert resume`
- `/scan <chain> <address>`
- `/tradecheck` (interactive wizard)
- `/news`
- `/cycle`
- `/giveaway start <10m|1h|1d> prize "<text>"`
- `/giveaway end`
- `/giveaway reroll`
- `/giveaway status`
- `/join`

Natural language is fully supported, no strict command requirement.
When configured with `OPENAI_API_KEY`, an LLM JSON intent router handles free-form phrasing and maps to deterministic tools (analysis/alerts/news/scans/etc.). Low-confidence routes fall back to normal chat replies.
Analysis uses a fast default path (price + TA) and exposes on-demand detail buttons: `More detail`, `Derivatives`, `News`.

Advanced analysis syntax is supported:

- `SOL long 15m ema9 ema21 rsi14`
- `ETH short tf=1h,4h ema=20,50,200 rsi=14,21`
- `BTC long all timeframes all emas all rsis`
- `BTC limit long entry 66300 sl 64990 tp 69200 72000` (setup review mode)
- `watch btc` / `watch BTC` (quick analysis intent)
- `rsi top 10 1h oversold` / `top rsi 4h overbought`
- `find pair xion` / `find $PHB`
- `coin around 0.155`

Group behavior:

- Bot responds in groups only when mentioned (`@GhotalphaBot`), on `/commands`, or when replying directly to the bot.

## Manual QA (acceptance scenarios)

1. Send `SOL long`
- Expected: 1-2 line summary + Entry/TP1/TP2/SL block + signal/narrative bullets + risk line + inline buttons

2. Send `ETH short`
- Expected: short-bias trade plan in same structured format

3. Send `Coins to watch 5`
- Expected: day theme + 5-symbol watchlist with one-line catalysts

4. Send `what are the latest news for today`
- Expected: compact brief + headlines + links + vibe line

5. Send `ping me when SOL hits 100`
- Expected: alert created with ID and normalized condition

6. Trigger alert in test mode
- Set `TEST_MODE=true`
- Call:

```bash
curl -X POST http://localhost:8000/test/mock-price \
  -H "Content-Type: application/json" \
  -d '{"symbol":"SOL","price":100}'
```

- Expected: worker triggers alert once and marks it `triggered`

7. Send `scan solana <address>`
- Expected: native balance, token breakdown, tx context, warnings, `Save wallet` button

8. Send `check this trade from yesterday: ETH entry 2100 stop 2165 targets 2043 2027 1991 timeframe 1h`
- Expected: win/loss/ambiguous + first-hit time + MFE/MAE + R multiple

9. Send `is BIRB following BTC?`
- Expected: verdict + corr/beta + relative performance bullets

## Test mode (local-safe)

- Works without premium API keys (RSS + public endpoints + fallbacks)
- Mockable prices via:
  - `MOCK_PRICES` env at startup
  - `POST /test/mock-price` at runtime

## Local dev (without Docker)

```bash
python -m venv .venv
. .venv/Scripts/activate  # Windows
pip install -r requirements.txt
alembic upgrade head
python -m app.main
```

## Tests

```bash
python -m pytest -q
```

Includes:

- 30+ NLU parse cases
- TA deterministic checks
- Trade verification ambiguity mode checks

## VPS deployment notes

1. Provision VPS with Docker + Docker Compose
2. Clone repo and configure `.env`
3. Run `docker compose up -d --build`
4. Put Nginx/Caddy in front of `:8000` if using webhook
5. Monitor `/health` and `/ready`
6. Persist Docker volumes (`pgdata`, `redisdata`)

## Privacy note

- No exchange private keys stored
- No user API keys stored
- Wallet scans only process public-chain data
- Do not use outputs for doxxing, harassment, or attribution

## Assumptions

- Binance public endpoints are primary market/ohlcv source
- CoinGecko fallback handles symbols unavailable on Binance
- Derivatives sentiment is best-effort and omitted gracefully when unavailable
- Trade verification defaults to `ambiguous` same-candle mode
- Not financial advice
