from __future__ import annotations

from app.core.http import ResilientHTTPClient


class TronAdapter:
    def __init__(self, http: ResilientHTTPClient, api_url: str, api_key: str = "") -> None:
        self.http = http
        self.api_url = api_url.rstrip("/")
        self.api_key = api_key

    def _headers(self) -> dict[str, str]:
        headers = {}
        if self.api_key:
            headers["TRON-PRO-API-KEY"] = self.api_key
        return headers

    async def scan_wallet(self, address: str, tx_limit: int = 10) -> dict:
        acct = await self.http.get_json(f"{self.api_url}/v1/accounts/{address}", headers=self._headers())
        data = acct.get("data", [])
        account = data[0] if data else {}

        trx_balance = float(account.get("balance", 0)) / 1_000_000

        trc20_rows = account.get("trc20", [])
        tokens = []
        for row in trc20_rows:
            for contract, amount in row.items():
                try:
                    parsed_amount = float(amount)
                except Exception:  # noqa: BLE001
                    continue
                if parsed_amount <= 0:
                    continue
                tokens.append(
                    {
                        "symbol": contract[:8] + "...",
                        "contract": contract,
                        "amount": parsed_amount,
                    }
                )

        resources = {}
        try:
            res = await self.http.post_json(
                f"{self.api_url}/wallet/getaccountresource",
                payload={"address": address, "visible": True},
                headers=self._headers(),
            )
            resources = {
                "free_net_limit": res.get("freeNetLimit"),
                "energy_limit": res.get("EnergyLimit"),
                "energy_used": res.get("EnergyUsed"),
            }
        except Exception:  # noqa: BLE001
            resources = {}

        tx_resp = await self.http.get_json(
            f"{self.api_url}/v1/accounts/{address}/transactions",
            params={"limit": tx_limit, "order_by": "block_timestamp,desc"},
            headers=self._headers(),
        )

        txs = []
        for tx in tx_resp.get("data", [])[:tx_limit]:
            txs.append(
                {
                    "txid": tx.get("txID"),
                    "type": tx.get("raw_data", {}).get("contract", [{}])[0].get("type"),
                    "time": tx.get("block_timestamp"),
                    "ret": tx.get("ret", [{}])[0].get("contractRet"),
                }
            )

        return {
            "chain": "tron",
            "address": address,
            "native_balance": trx_balance,
            "native_symbol": "TRX",
            "tokens": tokens,
            "resources": resources,
            "recent_transactions": txs,
        }
