## Privacy & data handling (Ghost Alpha Bot)

This document summarizes what data the bot stores, why it stores it, and how operators should handle retention and deletion.

### What data is stored
- **Telegram identifiers**
  - Chat ID (used to send replies/alerts)
  - Optional username/display name (if provided via settings)
- **Feature data**
  - Alerts (symbol, thresholds, status, created/triggered time)
  - Giveaways (participants and outcome)
  - Trade journal / portfolio entries (if the user uses those features)
  - Wallet addresses (only if user saves them)
- **Operational state**
  - Redis keys for rate limiting, dedupe, cache, distributed locks
  - Cached analysis/news outputs (short TTL)

### What is not stored
- **No exchange API keys**
- **No private wallet keys/seed phrases**
- **No payment details**

### Why we store it
- To deliver core functionality: alerts, history, preferences, and scheduled reports.
- To protect availability: rate limiting, dedupe, abuse controls.

### Retention
- **DB data**: retained until user requests deletion (or operator policy).
- **Redis cached data**: retained for short TTLs (minutes to days) depending on key type.

### User controls
- **Export**: users can export their stored data via bot export flows.
- **Deletion**: users can request deletion via the bot’s delete flow.

### Operator responsibilities
- Treat the database as sensitive (it contains identifiers + user-generated content).
- Limit access to DB/Redis credentials to operators only.
- Use encrypted connections and managed providers where possible.
- Define your own retention policy and communicate it to users if running publicly.

