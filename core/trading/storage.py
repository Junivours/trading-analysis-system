import json
import os
import time
from typing import Dict, Any, List, Optional

class TradeStorage:
    """Simple JSONL storage for orders and positions (paper trading friendly)."""
    def __init__(self, base_path: str = "data/trading"):
        self.base_path = base_path
        os.makedirs(self.base_path, exist_ok=True)
        self.orders_path = os.path.join(self.base_path, "orders.jsonl")
        self.positions_path = os.path.join(self.base_path, "positions.json")
        if not os.path.exists(self.positions_path):
            with open(self.positions_path, 'w', encoding='utf-8') as f:
                json.dump({}, f)

    def append_order(self, order: Dict[str, Any]):
        order = dict(order)
        order.setdefault('ts', int(time.time()*1000))
        with open(self.orders_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(order) + "\n")

    def get_positions(self) -> Dict[str, Any]:
        try:
            with open(self.positions_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception:
            return {}

    def save_positions(self, pos: Dict[str, Any]):
        with open(self.positions_path, 'w', encoding='utf-8') as f:
            json.dump(pos, f, ensure_ascii=False, indent=2)

    def update_position(self, symbol: str, delta_qty: float, entry_price: Optional[float] = None):
        pos = self.get_positions()
        p = pos.get(symbol) or {"qty": 0.0, "avg_entry": 0.0}
        qty = p["qty"] + delta_qty
        if qty == 0:
            pos.pop(symbol, None)
        else:
            # update avg entry only when adding size in same direction
            if p["qty"] * delta_qty > 0 and entry_price is not None:
                p["avg_entry"] = (p["avg_entry"] * abs(p["qty"]) + entry_price * abs(delta_qty)) / abs(qty)
            elif p["qty"] == 0 and entry_price is not None:
                p["avg_entry"] = entry_price
            p["qty"] = qty
            pos[symbol] = p
        self.save_positions(pos)
