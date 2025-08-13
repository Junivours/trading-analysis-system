import os
import sys
import pytest

# Ensure repo root on path
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from core.orchestration.master_analyzer import MasterAnalyzer


@pytest.mark.timeout(30)
def test_validation_block_present_and_sane():
    ma = MasterAnalyzer()
    symbol = os.environ.get('TEST_SYMBOL', 'BTCUSDT')
    data = ma.analyze_symbol(symbol)
    if 'error' in data:
        pytest.skip(f"Live data fetch error: {data['error']}")
    fs = data.get('final_score') or {}
    # Validation may be injected by app layer; if absent here, just check that we can compute indirectly via fields
    validation = fs.get('validation')
    # Ensure minimal keys exist in final score and technicals for validator inputs
    assert 'score' in fs
    assert 'technical_weight' in fs and 'pattern_weight' in fs and 'ai_weight' in fs
    tech = data.get('technical_analysis') or {}
    assert 'rsi' in tech and isinstance(tech['rsi'], dict)
    # If validation present, check schema
    if isinstance(validation, dict):
        for k in ['trading_action','risk_level','enterprise_ready','contradictions','warnings','confidence_factors']:
            assert k in validation
    # Accept both app-level and core-level variants
    allowed_actions = {'TRADE','WAIT','BUY','SELL','HOLD','STRONG_BUY','STRONG_SELL'}
    assert str(validation['trading_action']).upper() in allowed_actions
    allowed_risks = {'LOW','MEDIUM','HIGH','VERY_HIGH','UNKNOWN','low','medium','high','very_high'}
    assert str(validation['risk_level']) in allowed_risks
    assert isinstance(validation['enterprise_ready'], bool)
    assert isinstance(validation['contradictions'], list)
    assert isinstance(validation['warnings'], list)
    assert isinstance(validation['confidence_factors'], list)
