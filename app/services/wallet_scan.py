from __future__ import annotations

from datetime import datetime

from sqlalchemy import select

from app.adapters.prices import PriceAdapter
from app.adapters.solana import SolanaAdapter
from app.adapters.tron import TronAdapter
from app.db.models import User, Wallet


class WalletScanService:
    def __init__(self, db_factory, solana: SolanaAdapter, tron: TronAdapter, price: PriceAdapter) -> None:
        self.db_factory = db_factory
        self.solana = solana
        self.tron = tron
        self.price = price

    def _flag_suspicious(self, tokens: list[dict]) -> list[str]:
        warns = []
        for token in tokens[:30]:
            symbol = token.get("symbol", "")
            amount = float(token.get("amount", 0) or 0)
            if amount < 0.000001:
                warns.append(f"Tiny dust token detected: {symbol}")
            if len(symbol) > 14:
                warns.append(f"Unknown/spam-like token label: {symbol}")
        return warns[:4]

    async def _ensure_user(self, chat_id: int):
        async with self.db_factory() as session:
            q = await session.execute(select(User).where(User.telegram_chat_id == chat_id))
            user = q.scalar_one_or_none()
            if user:
                return user
            user = User(telegram_chat_id=chat_id, settings_json={})
            session.add(user)
            await session.commit()
            await session.refresh(user)
            return user

    async def scan(self, chain: str, address: str, chat_id: int | None = None, save: bool = False, label: str | None = None) -> dict:
        chain = chain.lower()
        if chain == "solana":
            result = await self.solana.scan_wallet(address)
        elif chain == "tron":
            result = await self.tron.scan_wallet(address)
        else:
            raise RuntimeError("Unsupported chain")

        native_usd = None
        try:
            native_price = await self.price.get_price(result["native_symbol"])
            native_usd = result["native_balance"] * float(native_price["price"])
        except Exception:  # noqa: BLE001
            native_usd = None

        warnings = self._flag_suspicious(result.get("tokens", []))
        warnings.append("Public chain data only. No identity attribution.")

        if save and chat_id is not None:
            user = await self._ensure_user(chat_id)
            async with self.db_factory() as session:
                wallet = Wallet(
                    user_id=user.id,
                    chain=chain,
                    address=address,
                    label=label,
                    is_saved=True,
                    last_scanned_at=datetime.utcnow(),
                )
                session.add(wallet)
                await session.commit()

        result["native_usd"] = native_usd
        result["warnings"] = warnings
        return result
