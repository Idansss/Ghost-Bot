from __future__ import annotations

from app.core.http import ResilientHTTPClient


class SolanaAdapter:
    TOKEN_PROGRAM = "TokenkegQfeZyiNwAJbNbGKPFXCWuBvf9Ss623VQ5DA"

    def __init__(self, http: ResilientHTTPClient, rpc_url: str) -> None:
        self.http = http
        self.rpc_url = rpc_url

    async def scan_wallet(self, address: str, tx_limit: int = 10) -> dict:
        balance_resp = await self.http.post_json(
            self.rpc_url,
            {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "getBalance",
                "params": [address],
            },
        )
        lamports = balance_resp.get("result", {}).get("value", 0)

        token_resp = await self.http.post_json(
            self.rpc_url,
            {
                "jsonrpc": "2.0",
                "id": 2,
                "method": "getTokenAccountsByOwner",
                "params": [
                    address,
                    {"programId": self.TOKEN_PROGRAM},
                    {"encoding": "jsonParsed"},
                ],
            },
        )
        token_rows = token_resp.get("result", {}).get("value", [])

        tokens = []
        for row in token_rows:
            info = row.get("account", {}).get("data", {}).get("parsed", {}).get("info", {})
            token_amount = info.get("tokenAmount", {})
            amount = float(token_amount.get("uiAmount", 0) or 0)
            if amount <= 0:
                continue
            mint = info.get("mint", "")
            tokens.append(
                {
                    "symbol": mint[:6] + "...",
                    "mint": mint,
                    "amount": amount,
                }
            )

        tx_resp = await self.http.post_json(
            self.rpc_url,
            {
                "jsonrpc": "2.0",
                "id": 3,
                "method": "getSignaturesForAddress",
                "params": [address, {"limit": tx_limit}],
            },
        )

        txs = []
        for tx in tx_resp.get("result", [])[:tx_limit]:
            txs.append(
                {
                    "signature": tx.get("signature"),
                    "slot": tx.get("slot"),
                    "time": tx.get("blockTime"),
                    "err": tx.get("err"),
                }
            )

        return {
            "chain": "solana",
            "address": address,
            "native_balance": lamports / 1_000_000_000,
            "native_symbol": "SOL",
            "tokens": tokens,
            "resources": {},
            "recent_transactions": txs,
        }
