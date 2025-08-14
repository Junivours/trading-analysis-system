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
        # Defaults for safety/tuning
        self.cfg.setdefault('min_probability', 52)
        self.cfg.setdefault('min_rr', 1.2)
        self.cfg.setdefault('cooldown_minutes', 5)  # throttle per symbol
        self.cfg.setdefault('max_trades_per_day', 3)
        self.cfg.setdefault('require_trend_alignment', True)
        # soft cap for position notional as % of equity (post sizing)
        self.cfg.setdefault('max_notional_pct', 100.0)  # 100% of risk-based notional by default

    def position_size(self, symbol: str, entry: float, stop: float) -> float:
        # risk-per-trade in quote currency
        equity = float(self.cfg.get('equity', 1000))
        risk_pct = float(self.cfg.get('risk_pct', 0.5))  # 0.5% default
        min_notional = float(self.cfg.get('min_notional', 10))
        risk_amount = equity * (risk_pct / 100.0)
        risk_per_unit = abs(entry - stop)
        if risk_per_unit <= 0: return 0.0
        qty = max(min_notional / entry, risk_amount / risk_per_unit)

        # Apply notional soft cap (e.g., avoid oversized positions)
        try:
            max_notional_pct = float(self.cfg.get('max_notional_pct', 100.0))
            if max_notional_pct > 0:
                notional = qty * entry
                cap = (equity * max_notional_pct / 100.0)
                if cap > 0 and notional > cap:
                    qty = cap / entry
        except Exception:
            pass
        
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
        # take the single best setup overall by confidence (keeps "max 1 trade")
        if not setups:
            return []
        ranked = sorted(setups, key=lambda s: int(s.get('confidence', 0)), reverse=True)
        return [ranked[0]]

    def _get_recent_orders(self, symbol: str) -> List[Dict[str, Any]]:
        """Read recent orders from storage (JSONL tail) for gating purposes."""
        path = getattr(self.storage, 'orders_path', None)
        if not path:
            return []
        rows: List[Dict[str, Any]] = []
        try:
            with open(path, 'r', encoding='utf-8') as f:
                # read only last ~200 lines to limit IO
                tail = f.readlines()[-200:]
            for ln in tail:
                try:
                    rec = __import__('json').loads(ln)
                    if rec.get('symbol') == symbol.upper():
                        rows.append(rec)
                except Exception:
                    continue
        except Exception:
            return []
        return rows

    def _throttled(self, symbol: str) -> bool:
        mins = float(self.cfg.get('cooldown_minutes', 5))
        if mins <= 0:
            return False
        orders = self._get_recent_orders(symbol)
        if not orders:
            return False
        last_ts = max((o.get('ts') or 0) for o in orders)
        if not last_ts:
            return False
        return (time.time() * 1000 - float(last_ts)) < (mins * 60 * 1000)

    def _daily_limit_reached(self, symbol: str) -> bool:
        max_trades = int(self.cfg.get('max_trades_per_day', 0))
        if max_trades <= 0:
            return False
        orders = self._get_recent_orders(symbol)
        if not orders:
            return False
        try:
            from datetime import datetime, timezone
            now = datetime.now(timezone.utc)
            day_start = int(datetime(now.year, now.month, now.day, tzinfo=timezone.utc).timestamp() * 1000)
            todays = [o for o in orders if (o.get('ts') or 0) >= day_start]
            return len(todays) >= max_trades
        except Exception:
            return False

    def should_trade(self, analysis: Dict[str, Any], setup: Dict[str, Any]) -> tuple[bool, List[str]]:
        reasons: List[str] = []
        # basic gates: enterprise_ready and probability
        fs = (analysis.get('final_score') or {}).get('validation') or {}
        if not fs.get('enterprise_ready', False):
            reasons.append('enterprise_not_ready')
        prob = setup.get('probability_estimate_pct') or 0
        if prob < float(self.cfg.get('min_probability', 52)):
            reasons.append('probability_below_threshold')
        # sanity: RR > min
        rr = setup.get('risk_reward_ratio') or setup.get('primary_rr') or 0
        if rr < float(self.cfg.get('min_rr', 1.2)):
            reasons.append('rr_below_threshold')

        # cooldown / daily limits
        sym = setup.get('symbol') or analysis.get('symbol') or ''
        sym = (sym or '').upper() or (analysis.get('symbol') or '').upper()
        if self._throttled(sym):
            reasons.append('throttled_cooldown')
        if self._daily_limit_reached(sym):
            reasons.append('daily_limit_reached')

        # optional trend alignment with MTF consensus
        if bool(self.cfg.get('require_trend_alignment', True)):
            try:
                mtc = ((analysis.get('multi_timeframe') or {}).get('consensus') or {})
                mtf_dir = (mtc.get('primary') or mtc.get('final_decision') or '').upper()
                direction = (setup.get('direction') or '').upper()
                if direction == 'LONG' and mtf_dir == 'SHORT':
                    reasons.append('trend_conflict')
                if direction == 'SHORT' and mtf_dir == 'LONG':
                    reasons.append('trend_conflict')
            except Exception:
                pass

        return (len(reasons) == 0, reasons)

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
        details: Dict[str, Any] = {}
        # derive minimal decision context for transparency
        fs = (analysis.get('final_score') or {})
        decision = (fs.get('signal') or '').upper()
        mtf = ((analysis.get('multi_timeframe') or {}).get('consensus') or {}).get('primary')
        tech = analysis.get('technical_analysis') or {}
        rsi = ((tech.get('rsi') or {}).get('rsi'))
        macd = (tech.get('macd') or {}).get('curve_direction')
        details['decision'] = decision
        details['mtf'] = mtf
        details['rsi'] = rsi
        details['macd'] = macd
        details['ai_signal'] = (analysis.get('ai_analysis') or {}).get('signal')
        details['ai_confidence'] = (analysis.get('ai_analysis') or {}).get('confidence')

        for s in chosen:
            ok, gates = self.should_trade(analysis, s)
            if ok:
                rec = self.execute_setup(symbol, s)
                # Keep key rationale fields (direction, reason/rationale, rr, probability, timeframe)
                rec['rationale'] = s.get('rationale') or s.get('reason')
                rec['rr'] = s.get('risk_reward_ratio') or s.get('primary_rr')
                rec['probability_estimate_pct'] = s.get('probability_estimate_pct')
                rec['timeframe'] = s.get('timeframe')
                rec['strategy'] = s.get('strategy') or s.get('label')
                executed.append(rec)
            else:
                # include all gating reasons for transparency
                executed.append({"skipped": True, "reason": ",".join(gates) or "gates", "setup": s})

        return {
            "symbol": symbol.upper(),
            "timeframe": base_interval,
            "executed": executed,
            "analysis_score": (analysis.get('final_score') or {}).get('score'),
            "context": details
        }
