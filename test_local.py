import json, sys, time
from app import app

def check(label, resp):
    status = resp.status_code
    try:
        data = resp.get_json()
    except Exception:
        data = resp.data.decode()[:500]
    ok = 200 <= status < 300 and isinstance(data, dict) and data.get('success', True)
    print(f"[{label}] status={status} ok={ok}")
    if not ok:
        print(json.dumps(data, indent=2, ensure_ascii=False) if isinstance(data, dict) else data)
    return ok

with app.test_client() as c:
    symbol = 'BTCUSDT'
    print("--- SELF TEST START ---")
    r1 = c.get(f"/api/ai/status")
    check('ai_status', r1)
    r2 = c.get(f"/api/analyze/{symbol}?refresh=1")
    check('analyze', r2)
    r3 = c.get(f"/api/backtest/{symbol}?interval=1h&limit=500&refresh=1")
    check('backtest_1h', r3)
    r4 = c.get(f"/api/backtest/{symbol}?interval=15m&limit=500&refresh=1")
    check('backtest_15m', r4)
    r5 = c.get(f"/api/logs/recent?limit=15")
    check('recent_logs', r5)
    # Inspect RSI diff if analyze succeeded
    if r2.status_code==200:
        data = r2.get_json().get('data', {})
        rsi = data.get('technical_analysis', {}).get('rsi', {})
        print('RSI detail:', rsi)
    print("--- SELF TEST END ---")
