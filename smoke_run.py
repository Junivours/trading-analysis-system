"""Simple manual smoke script to exercise MasterAnalyzer without Flask.
Run: python smoke_run.py BTCUSDT
"""
import sys, json, time
from core.orchestration.master_analyzer import MasterAnalyzer

def main():
    symbol = (sys.argv[1] if len(sys.argv) > 1 else 'BTCUSDT').upper()
    ma = MasterAnalyzer()
    t0 = time.time()
    data = ma.analyze_symbol(symbol)
    dt = time.time() - t0
    if 'error' in data:
        print(f"ERROR: {data['error']}")
        return 1
    summary = {
        'symbol': data['symbol'],
        'price': data['current_price'],
        'score': data['final_score'].get('score'),
        'signal': data['final_score'].get('signal'),
        'trade_setups': len(data.get('trade_setups') or []),
        'ai_confidence': data.get('ai_analysis', {}).get('confidence'),
        'timing_ms': data.get('phase_timings_ms', {}),
    }
    print(json.dumps(summary, indent=2))
    # Print first trade setup (compact)
    setups = data.get('trade_setups') or []
    if setups:
        first = dict(setups[0])
        # Trim verbose fields
        for k in list(first.keys()):
            if k not in {'id','direction','strategy','entry','stop_loss','confidence','targets','primary_rr'}:
                first.pop(k, None)
        print("First setup:")
        print(json.dumps(first, indent=2))
    print(f"Done in {dt:.2f}s")
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
