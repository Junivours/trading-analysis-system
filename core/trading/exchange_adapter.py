import os
import time
import hmac
import hashlib
import json
from typing import Dict, Optional, Any
import requests

class ExchangeAdapter:
    """Thin REST adapter for Binance Spot/Futures. Supports public price, order, and account.
    Defaults to dry_run unless BINANCE_API_KEY/SECRET are provided and dry_run is False.
    """
    def __init__(self, base_url: Optional[str] = None, futures: bool = True, dry_run: bool = True):
        self.futures = futures
        self.base_url = base_url or ("https://fapi.binance.com" if futures else "https://api.binance.com")
        self.api_key = os.getenv("BINANCE_API_KEY")
        self.api_secret = os.getenv("BINANCE_API_SECRET")
        self.dry_run = dry_run or not (self.api_key and self.api_secret)

    def _sign(self, params: Dict[str, Any]) -> Dict[str, Any]:
        if not self.api_secret:
            return params
        query = "&".join([f"{k}={params[k]}" for k in sorted(params)])
        signature = hmac.new(self.api_secret.encode(), query.encode(), hashlib.sha256).hexdigest()
        params["signature"] = signature
        return params

    def _headers(self) -> Dict[str, str]:
        h = {"Content-Type": "application/json"}
        if self.api_key:
            h["X-MBX-APIKEY"] = self.api_key
        return h

    def get_price(self, symbol: str) -> float:
        try:
            url = f"{self.base_url}/fapi/v1/ticker/price" if self.futures else f"{self.base_url}/api/v3/ticker/price"
            r = requests.get(url, params={"symbol": symbol}, timeout=10)
            data = r.json()
            return float(data.get("price", 0))
        except Exception:
            return 0.0

    def place_order(self, symbol: str, side: str, qty: float, order_type: str = "MARKET", price: Optional[float] = None, reduce_only: bool = False) -> Dict[str, Any]:
        ts = int(time.time() * 1000)
        if self.dry_run:
            return {"dry_run": True, "symbol": symbol, "side": side, "qty": qty, "type": order_type, "price": price, "ts": ts, "reduceOnly": reduce_only}
        endpoint = "/fapi/v1/order" if self.futures else "/api/v3/order"
        url = f"{self.base_url}{endpoint}"
        params = {"symbol": symbol, "side": side.upper(), "type": order_type.upper(), "timestamp": ts}
        if order_type.upper() == "MARKET":
            params["quantity"] = qty
        else:
            params["quantity"] = qty
            if price is not None:
                params["price"] = price
                params["timeInForce"] = "GTC"
        if self.futures:
            params["reduceOnly"] = "true" if reduce_only else "false"
        params = self._sign(params)
        r = requests.post(url, headers=self._headers(), data=params, timeout=15)
        try:
            return r.json()
        except Exception:
            return {"error": r.text}

    def get_account(self) -> Dict[str, Any]:
        ts = int(time.time() * 1000)
        if self.dry_run:
            return {"dry_run": True}
        endpoint = "/fapi/v2/account" if self.futures else "/api/v3/account"
        url = f"{self.base_url}{endpoint}"
        params = self._sign({"timestamp": ts})
        r = requests.get(url, headers=self._headers(), params=params, timeout=15)
        return r.json()
