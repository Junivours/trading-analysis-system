import os
import time
from dataclasses import dataclass
from typing import Optional, Dict, Any

from core.scalper.engine import scan_scalp_signals, ScalpConfig
from core.trading.storage import TradeStorage

try:
    from core.trading.exchange_adapter import ExchangeAdapter
    from core.trading.mexc_adapter import MEXCExchangeAdapter
    EX_AVAILABLE = True
except Exception:
    EX_AVAILABLE = False


def trading_enabled() -> bool:
    # Never allow on Railway. Local only via ALLOW_TRADING=true
    try:
        for k in os.environ.keys():
            if str(k).startswith('RAILWAY_'):
                return False
    except Exception:
        pass
    val = (os.getenv('ALLOW_TRADING') or os.getenv('ENABLE_TRADING') or '').strip().lower()
    return val in ('1','true','yes','local')


@dataclass
class ScalpRunConfig:
    symbol: str = 'BTCUSDT'
    tf: str = '1m'
    exchange: str = 'binance'  # 'binance' | 'mexc'
    paper: bool = True
    equity: float = 1000.0
    risk_pct: float = 0.25
    min_delta_bps: float = 3.0   # require movement vs prev H/L in bps to avoid tiny flickers


class SimpleScalpExecutor:
    """Ultra-light executor: reads latest scalp signal and places a tiny paper trade.
    Completely isolated from long-term bot (separate panel and endpoints).
    """
    def __init__(self, storage: Optional[TradeStorage] = None):
        self.storage = storage or TradeStorage(base_path='data/scalper')

    def _adapter(self, cfg: ScalpRunConfig):
        if not EX_AVAILABLE:
            return None
        ex = (cfg.exchange or 'binance').lower()
        if ex == 'mexc':
            dry = cfg.paper or (os.getenv('MEXC_API_KEY') is None or os.getenv('MEXC_API_SECRET') is None)
            return MEXCExchangeAdapter(futures=False, dry_run=dry)
        dry = cfg.paper or (os.getenv('BINANCE_API_KEY') is None or os.getenv('BINANCE_API_SECRET') is None)
        return ExchangeAdapter(dry_run=dry)

    def _position_size(self, price: float, equity: float, risk_pct: float) -> float:
        risk_amount = equity * (risk_pct/100.0)
        # For a fast scalp, assume SL 0.2% away => notional ~ risk/0.2%
        denom = max(1e-8, price*0.002)
        qty = risk_amount / denom
        return max(0.0, qty)

    def run_once(self, cfg: ScalpRunConfig) -> Dict[str, Any]:
        if not trading_enabled():
            return {'success': False, 'error': 'trading_disabled', 'paper': True}
        out = scan_scalp_signals(cfg.symbol, ScalpConfig(tf=cfg.tf))
        if not out.get('success'):
            return {'success': False, 'error': out.get('error', 'scan_failed')}
        sigs = out.get('signals') or []
        if not sigs:
            return {'success': True, 'executed': [], 'note': 'no_signal'}
        s0 = sigs[0]
        side = 'BUY' if s0['type'] == 'SCALP_BUY' else 'SELL'
        price = float(out.get('last') or s0.get('price'))
        # tiny movement guard vs prev H/L using data from engine context is minimal here
        qty = self._position_size(price, cfg.equity, cfg.risk_pct)
        if qty <= 0:
            return {'success': True, 'executed': [], 'note': 'qty_zero'}

        ad = self._adapter(cfg)
        if ad is None:
            # record paper order
            rec = {'symbol': cfg.symbol.upper(), 'side': side, 'qty': qty, 'price': price, 'paper': True, 'ts': int(time.time()*1000), 'scalp': True}
            self.storage.append_order(rec)
            return {'success': True, 'executed': [rec], 'paper': True}
        # live/paper via adapter
        order = ad.place_order(cfg.symbol.upper(), side, qty, order_type='MARKET')
        fill = None
        try:
            if isinstance(order, dict):
                if 'fills' in order and order['fills']:
                    p = order['fills'][0].get('price')
                    if p is not None:
                        fill = float(p)
                if fill is None and order.get('avgPrice') is not None:
                    fill = float(order.get('avgPrice'))
        except Exception:
            pass
        if fill is None:
            try:
                fill = float(ad.get_price(cfg.symbol.upper()))
            except Exception:
                fill = price
        rec = {'symbol': cfg.symbol.upper(), 'side': side, 'qty': qty, 'price': fill, 'paper': getattr(ad, 'dry_run', True), 'ts': int(time.time()*1000), 'scalp': True}
        self.storage.append_order(rec)
        self.storage.update_position(cfg.symbol.upper(), delta_qty=(qty if side=='BUY' else -qty), entry_price=fill)
        return {'success': True, 'executed': [rec], 'paper': getattr(ad, 'dry_run', True)}
