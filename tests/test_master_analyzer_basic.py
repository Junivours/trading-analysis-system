import os
import pytest

# Ensure we can import project modules when tests run from repository root
import sys
ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from core.orchestration.master_analyzer import MasterAnalyzer

@pytest.fixture(scope="module")
def analyzer():
    return MasterAnalyzer()

@pytest.mark.timeout(25)
def test_analyze_symbol_structure(analyzer):
    """Smoke test: basic structure keys exist for a common symbol.
    NOTE: Relies on live Binance API; if offline, test is skipped.
    """
    symbol = os.environ.get("TEST_SYMBOL", "BTCUSDT")
    data = analyzer.analyze_symbol(symbol)
    if 'error' in data:
        pytest.skip(f"Live data fetch error: {data['error']}")
    required_top_level = [
        'symbol','current_price','market_data','technical_analysis',
        'pattern_analysis','multi_timeframe','position_analysis',
        'ai_analysis','final_score','trade_setups','adaptive_risk_targets'
    ]
    for k in required_top_level:
        assert k in data, f"Missing key: {k}"
    assert isinstance(data['trade_setups'], list)
    # At least one setup or gracefully empty list
    assert data['final_score'].get('score') is not None

@pytest.mark.timeout(25)
def test_run_backtest(analyzer):
    symbol = os.environ.get("TEST_SYMBOL", "BTCUSDT")
    bt = analyzer.run_backtest(symbol, interval='1h', limit=200)
    if 'error' in bt:
        pytest.skip(f"Backtest unavailable: {bt['error']}")
    assert 'metrics' in bt
    metrics = bt['metrics']
    for key in ['total_trades','win_rate_pct','total_return_pct']:
        assert key in metrics
