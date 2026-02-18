from __future__ import annotations

from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import NullPool

from app.core.config import get_settings

settings = get_settings()


def _normalize_database_url(raw_url: str) -> str:
    url = (raw_url or "").strip()
    if url.startswith("postgres://"):
        url = "postgresql+asyncpg://" + url[len("postgres://") :]
    elif url.startswith("postgresql://"):
        url = "postgresql+asyncpg://" + url[len("postgresql://") :]
    return url


def _normalize_asyncpg_query(url: str) -> tuple[str, dict]:
    parsed = urlsplit(url)
    pairs = parse_qsl(parsed.query, keep_blank_values=True)
    filtered: list[tuple[str, str]] = []
    connect_args: dict = {}

    sslmode = None
    for key, value in pairs:
        k = key.lower()
        if k == "sslmode":
            sslmode = (value or "").lower().strip()
            continue
        if k == "channel_binding":
            # libpq option not supported by asyncpg connect().
            continue
        filtered.append((key, value))

    if sslmode and sslmode not in {"disable", "allow"}:
        # Asyncpg expects `ssl`, not `sslmode`.
        connect_args["ssl"] = "require"

    rebuilt = parsed._replace(query=urlencode(filtered))
    return urlunsplit(rebuilt), connect_args


engine_kwargs = {"pool_pre_ping": True}
if settings.serverless_mode:
    # Serverless workers should avoid persistent pooled connections.
    engine_kwargs["poolclass"] = NullPool

normalized_url = _normalize_database_url(settings.database_url)
clean_url = normalized_url
connect_args: dict = {}
if normalized_url.startswith("postgresql+asyncpg://"):
    clean_url, connect_args = _normalize_asyncpg_query(normalized_url)

if connect_args:
    engine = create_async_engine(clean_url, connect_args=connect_args, **engine_kwargs)
else:
    engine = create_async_engine(clean_url, **engine_kwargs)
AsyncSessionLocal = async_sessionmaker(bind=engine, expire_on_commit=False, class_=AsyncSession)


async def get_db_session() -> AsyncSession:
    async with AsyncSessionLocal() as session:
        yield session
