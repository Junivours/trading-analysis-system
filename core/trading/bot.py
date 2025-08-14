import math
import time
from typing import Dict, Any, List, Optional

from core.orchestration.master_analyzer import MasterAnalyzer
from core.trading.exchange_adapter import ExchangeAdapter
from core.trading.mexc_adapter import MEXCExchangeAdapter
from core.trading.storage import TradeStorage
from core.binance_client import BinanceClient

class TradingBot:
    """Minimal automated trader. Pulls top setups from MasterAnalyzer and executes via ExchangeAdapter.
    Safety first: dry_run default, 1 trade per side per symbol, simple position sizing.
    Now supports both Binance and MEXC exchanges.
    """
    def __init__(self, analyzer: Optional[MasterAnalyzer] = None, adapter: Optional[Any] = None, storage: Optional[TradeStorage] = None, config: Optional[Dict[str, Any]] = None):
        self.analyzer = analyzer or MasterAnalyzer()
        
        # Auto-detect exchange or use provided adapter
        if adapter is None:
            exchange = config.get('exchange', 'binance').lower() if config else 'binance'
            if exchange == 'mexc':
                self.adapter = MEXCExchangeAdapter(dry_run=True)
            else:
                self.adapter = ExchangeAdapter(dry_run=True)
        else:
            self.adapter = adapter
            
        self.storage = storage or TradeStorage()
        self.cfg = config or {}

    def position_size(self, symbol: str, entry: float, stop: float) -> float:
        # risk-per-trade in quote currency
        equity = float(self.cfg.get('equity', 1000))
        risk_pct = float(self.cfg.get('risk_pct', 0.5))  # 0.5% default
        min_notional = float(self.cfg.get('min_notional', 10))
        risk_amount = equity * (risk_pct / 100.0)
        risk_per_unit = abs(entry - stop)
        if risk_per_unit <= 0: return 0.0
        qty = max(min_notional / entry, risk_amount / risk_per_unit)
        
        # round to lot size based on exchange type
        try:
            if hasattr(self.adapter, 'format_quantity'):
                # MEXC adapter has built-in quantity formatting
                qty = self.adapter.format_quantity(symbol, qty)
            else:
                # Binance adapter - use BinanceClient method
                f = BinanceClient.get_symbol_filters(symbol)
                step = f.get('stepSize') or 0.0
                if step:
                    decimals = max(0, int(round(-math.log10(step))))
                    qty = float(f"{qty:.{decimals}f}")
        except Exception:
            pass
        return max(0.0, qty)

    def select_setups(self, analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        setups = analysis.get('trade_setups') or []
        # rank already refined in analyzer; take best LONG and best SHORT
        best_long = None; best_short = None
        for s in setups:
            if s.get('direction') == 'LONG' and (best_long is None or (s.get('confidence',0) > best_long.get('confidence',0))):
                best_long = s
            if s.get('direction') == 'SHORT' and (best_short is None or (s.get('confidence',0) > best_short.get('confidence',0))):
                best_short = s
        out = []
        if best_long: out.append(best_long)
        if best_short: out.append(best_short)
        return out

    def should_trade(self, analysis: Dict[str, Any], setup: Dict[str, Any]) -> bool:
        # basic gates: enterprise_ready and probability
        fs = (analysis.get('final_score') or {}).get('validation') or {}
        if not fs.get('enterprise_ready', False):
            return False
        prob = setup.get('probability_estimate_pct') or 0
        if prob < float(self.cfg.get('min_probability', 52)):
            return False
        # sanity: RR > min
        rr = setup.get('risk_reward_ratio') or setup.get('primary_rr') or 0
        if rr < float(self.cfg.get('min_rr', 1.2)):
            return False
        return True

    def execute_setup(self, symbol: str, setup: Dict[str, Any]) -> Dict[str, Any]:
        direction = setup.get('direction')
        entry = setup.get('entry') or setup.get('entry_price')
        stop = setup.get('stop_loss')
        if not entry or not stop or not direction:
            return {"skipped": True, "reason": "missing_fields"}
        side = 'BUY' if direction == 'LONG' else 'SELL'
        qty = self.position_size(symbol, float(entry), float(stop))
        if qty <= 0:
            return {"skipped": True, "reason": "qty_zero"}
        order = self.adapter.place_order(symbol, side, qty, order_type="MARKET")
        # persist order
        rec = {"symbol": symbol, "side": side, "qty": qty, "entry": entry, "stop": stop, "direction": direction, "setup": setup, "ts": int(time.time()*1000), "order": order}
        self.storage.append_order(rec)
        # update paper position
        delta = qty if side == 'BUY' else -qty
        self.storage.update_position(symbol, delta_qty=delta, entry_price=float(entry))
        return rec

    def run_once(self, symbol: str, base_interval: str = '1h') -> Dict[str, Any]:
        analysis = self.analyzer.analyze_symbol(symbol.upper(), base_interval=base_interval)
        chosen = self.select_setups(analysis)
        executed: List[Dict[str, Any]] = []
        for s in chosen:
            if self.should_trade(analysis, s):
                executed.append(self.execute_setup(symbol, s))
            else:
                executed.append({"skipped": True, "reason": "gates", "setup": s})
        return {"symbol": symbol.upper(), "timeframe": base_interval, "executed": executed, "analysis_score": (analysis.get('final_score') or {}).get('score')}
