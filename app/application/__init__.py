"""Application layer (use-cases).

The goal of this package is to keep transports (Telegram handlers, HTTP endpoints)
thin by moving orchestration (locks, caching, fallbacks) into explicit use-cases.
"""

