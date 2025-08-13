import pytest, sys, os
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
from core.orchestration.master_analyzer import MasterAnalyzer

@pytest.mark.timeout(30)
def test_analysis_includes_calibration_fields():
    ma = MasterAnalyzer()
    data = ma.analyze_symbol('BTCUSDT')
    fs = data.get('final_score', {})
    ai = data.get('ai_analysis', {})
    assert 'probability_bullish' in fs
    assert 'probability_bullish_source' in fs
    assert 'bull_probability_raw' in ai
    assert 'bull_probability_calibrated' in ai
    assert fs.get('probability_bullish_source') in ('score_logistic','ai_calibrated')
