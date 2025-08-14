import os
import time
import hmac
import hashlib
import json
import math
from typing import Dict, Optional, Any
import requests

class MEXCExchangeAdapter:
    """MEXC Exchange adapter for Spot/Futures trading.
    Supports public price data, order placement, and account info.
    Defaults to dry_run unless MEXC_API_KEY/SECRET are provided and dry_run is False.
    """
    def __init__(self, base_url: Optional[str] = None, futures: bool = True, dry_run: bool = True):
        self.futures = futures
        # MEXC API URLs
        if futures:
            self.base_url = base_url or "https://contract.mexc.com"
        else:
            self.base_url = base_url or "https://api.mexc.com"
        
        self.api_key = os.getenv("MEXC_API_KEY")
        self.api_secret = os.getenv("MEXC_API_SECRET")
        self.dry_run = dry_run or not (self.api_key and self.api_secret)

    def _sign(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """MEXC signature method"""
        if not self.api_secret:
            return params
        
        # Sort parameters and create query string
        query = "&".join([f"{k}={params[k]}" for k in sorted(params)])
        
        # MEXC uses HMAC-SHA256 like Binance
        signature = hmac.new(
            self.api_secret.encode(), 
            query.encode(), 
            hashlib.sha256
        ).hexdigest()
        
        params["signature"] = signature
        return params

    def _headers(self) -> Dict[str, str]:
        """Headers for MEXC API"""
        h = {"Content-Type": "application/x-www-form-urlencoded"}
        if self.api_key:
            h["X-MEXC-APIKEY"] = self.api_key
        return h

    def _norm_symbol(self, symbol: str) -> str:
        """Normalize symbols for MEXC API differences."""
        s = (symbol or '').upper()
        # MEXC futures often uses underscore format like BTC_USDT
        if self.futures and '_' not in s and s.endswith('USDT'):
            s = s.replace('USDT', '_USDT')
        # Spot stays as BTCUSDT
        return s

    def get_price(self, symbol: str) -> float:
        """Get current price for symbol"""
        try:
            if self.futures:
                # MEXC Futures ticker endpoint
                url = f"{self.base_url}/api/v1/contract/ticker"
                params = {"symbol": self._norm_symbol(symbol)}
            else:
                # MEXC Spot ticker endpoint  
                url = f"{self.base_url}/api/v3/ticker/price"
                params = {"symbol": symbol}
            
            r = requests.get(url, params=params, timeout=10)
            data = r.json()
            
            if self.futures:
                # Futures response format
                return float(data.get("lastPrice", 0))
            else:
                # Spot response format
                return float(data.get("price", 0))
                
        except Exception as e:
            print(f"Error getting price for {symbol}: {e}")
            return 0.0

    def place_order(self, symbol: str, side: str, qty: float, order_type: str = "MARKET", price: Optional[float] = None, reduce_only: bool = False) -> Dict[str, Any]:
        """Place order on MEXC"""
        ts = int(time.time() * 1000)

        if self.dry_run:
            return {
                "dry_run": True,
                "symbol": symbol,
                "side": side,
                "qty": qty,
                "type": order_type,
                "price": price,
                "ts": ts,
                "reduceOnly": reduce_only,
                "exchange": "MEXC"
            }

        try:
            if self.futures:
                # MEXC Futures order endpoint
                endpoint = "/api/v1/private/order/submit"
                url = f"{self.base_url}{endpoint}"
                params = {
                    "symbol": self._norm_symbol(symbol),
                    "side": 1 if side.upper() == "BUY" else 2,  # 1=BUY, 2=SELL
                    "type": 1 if order_type.upper() == "MARKET" else 2,  # 1=MARKET, 2=LIMIT
                    "vol": qty,
                    "timestamp": ts
                }
                if order_type.upper() == "LIMIT" and price is not None:
                    params["price"] = price
                if reduce_only:
                    params["reduceOnly"] = True
            else:
                # MEXC Spot order endpoint
                endpoint = "/api/v3/order"
                url = f"{self.base_url}{endpoint}"
                params = {
                    "symbol": symbol.upper(),
                    "side": side.upper(),
                    "type": order_type.upper(),
                    "quantity": qty,
                    "timestamp": ts
                }
                if order_type.upper() == "LIMIT" and price is not None:
                    params["price"] = price
                    params["timeInForce"] = "GTC"

            # Sign and send
            params = self._sign(params)
            r = requests.post(url, headers=self._headers(), data=params, timeout=15)
            try:
                return r.json()
            except Exception:
                return {"error": r.text, "status_code": r.status_code}
        except Exception as e:
            return {"error": f"Order placement failed: {str(e)}"}

    def get_account(self) -> Dict[str, Any]:
        """Get account information"""
        ts = int(time.time() * 1000)
        
        if self.dry_run:
            return {"dry_run": True, "exchange": "MEXC"}

        try:
            if self.futures:
                # MEXC Futures account endpoint
                endpoint = "/api/v1/private/account/assets"
                url = f"{self.base_url}{endpoint}"
            else:
                # MEXC Spot account endpoint
                endpoint = "/api/v3/account"
                url = f"{self.base_url}{endpoint}"
            
            params = self._sign({"timestamp": ts})
            r = requests.get(url, headers=self._headers(), params=params, timeout=15)
            
            return r.json()
            
        except Exception as e:
            return {"error": f"Account info failed: {str(e)}"}

    def get_exchange_info(self, symbol: str) -> Dict[str, Any]:
        """Get symbol trading rules and filters"""
        try:
            if self.futures:
                url = f"{self.base_url}/api/v1/contract/detail"
                params = {"symbol": self._norm_symbol(symbol)}
            else:
                url = f"{self.base_url}/api/v3/exchangeInfo"
                params = {}
            
            r = requests.get(url, params=params, timeout=10)
            data = r.json()
            
            if self.futures:
                return data
            else:
                # Find symbol in spot exchange info
                symbols = data.get("symbols", [])
                for s in symbols:
                    if s.get("symbol") == symbol:
                        return s
                return {}
                
        except Exception as e:
            print(f"Error getting exchange info for {symbol}: {e}")
            return {}

    def format_quantity(self, symbol: str, quantity: float) -> float:
        """Format quantity according to symbol's step size"""
        try:
            info = self.get_exchange_info(symbol)
            
            if self.futures:
                # Futures step size handling
                step_size = info.get("priceScale", 8)  # Default to 8 decimals
                return round(quantity, step_size)
            else:
                # Spot step size handling
                filters = info.get("filters", [])
                for f in filters:
                    if f.get("filterType") == "LOT_SIZE":
                        step_size = float(f.get("stepSize", 0.00000001))
                        if step_size > 0:
                            decimals = max(0, int(round(-math.log10(step_size))))
                            return round(quantity, decimals)
            
            return round(quantity, 8)  # Default fallback
            
        except Exception:
            return round(quantity, 8)
