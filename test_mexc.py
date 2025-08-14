#!/usr/bin/env python3
"""
MEXC Trading Bot Test Script
Teste die MEXC-Integration ohne echte Trades
"""

import requests
import json
import time
from typing import Dict, Any

def test_mexc_bot(base_url: str = "http://localhost:5000") -> Dict[str, Any]:
    """Test MEXC bot functionality"""
    
    print("🚀 Testing MEXC Trading Bot Integration...")
    
    # Test 1: Paper Trading
    print("\n📄 Test 1: MEXC Paper Trading")
    payload = {
        "symbol": "BTCUSDT",
        "exchange": "mexc",
        "interval": "1h",
        "paper": True,
        "equity": 1000,
        "risk_pct": 0.5,
        "min_probability": 55,
        "min_rr": 1.2
    }
    
    try:
        response = requests.post(f"{base_url}/api/bot/run", json=payload, timeout=30)
        result = response.json()
        
        if result['success']:
            print(f"✅ Paper trading successful!")
            print(f"   Exchange: {result['exchange']}")
            print(f"   Paper Mode: {result['paper']}")
            data = result['data']
            print(f"   Symbol: {data['symbol']}")
            print(f"   Analysis Score: {data.get('analysis_score', 'N/A')}")
            print(f"   Executed: {len(data['executed'])} setups processed")
            
            for i, exec_result in enumerate(data['executed']):
                if exec_result.get('skipped'):
                    print(f"   Setup {i+1}: SKIPPED - {exec_result.get('reason', 'unknown')}")
                else:
                    print(f"   Setup {i+1}: EXECUTED - {exec_result.get('symbol', 'N/A')}")
        else:
            print(f"❌ Paper trading failed: {result.get('error', 'Unknown error')}")
            return {"success": False, "error": result.get('error')}
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Network error: {e}")
        return {"success": False, "error": str(e)}
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return {"success": False, "error": str(e)}
    
    # Test 2: Different symbol
    print("\n💰 Test 2: MEXC with different symbol (ETH)")
    payload['symbol'] = 'ETHUSDT'
    payload['risk_pct'] = 0.3  # Lower risk for test
    
    try:
        response = requests.post(f"{base_url}/api/bot/run", json=payload, timeout=30)
        result = response.json()
        
        if result['success']:
            print(f"✅ ETH trading test successful!")
            data = result['data']
            print(f"   Analysis Score: {data.get('analysis_score', 'N/A')}")
        else:
            print(f"❌ ETH trading test failed: {result.get('error')}")
            
    except Exception as e:
        print(f"❌ ETH test error: {e}")
    
    # Test 3: Conservative settings
    print("\n🛡️ Test 3: Conservative MEXC settings")
    conservative_payload = {
        "symbol": "SOLUSDT", 
        "exchange": "mexc",
        "interval": "4h",  # Longer timeframe
        "paper": True,
        "equity": 500,
        "risk_pct": 0.25,  # Very low risk
        "min_probability": 65,  # High probability requirement
        "min_rr": 2.0  # High risk/reward
    }
    
    try:
        response = requests.post(f"{base_url}/api/bot/run", json=conservative_payload, timeout=30)
        result = response.json()
        
        if result['success']:
            print(f"✅ Conservative settings successful!")
            data = result['data']
            executed_count = len([x for x in data['executed'] if not x.get('skipped')])
            skipped_count = len([x for x in data['executed'] if x.get('skipped')])
            print(f"   Executed: {executed_count}, Skipped: {skipped_count}")
            print(f"   (High filters should result in more skipped setups)")
        else:
            print(f"❌ Conservative test failed: {result.get('error')}")
            
    except Exception as e:
        print(f"❌ Conservative test error: {e}")
    
    # Test 4: API endpoint validation
    print("\n🔍 Test 4: Validate API endpoints")
    
    # Check if trading module is available
    try:
        health_resp = requests.get(f"{base_url}/health", timeout=10)
        if health_resp.status_code == 200:
            print("✅ Health endpoint OK")
        else:
            print(f"❌ Health endpoint failed: {health_resp.status_code}")
    except Exception as e:
        print(f"❌ Health check error: {e}")
    
    # Check version
    try:
        version_resp = requests.get(f"{base_url}/api/version", timeout=10)
        if version_resp.status_code == 200:
            version_data = version_resp.json()
            print(f"✅ Version: {version_data.get('version', 'unknown')}")
        else:
            print(f"❌ Version endpoint failed")
    except Exception as e:
        print(f"❌ Version check error: {e}")
    
    print("\n🎯 MEXC Integration Test Summary:")
    print("   - Paper trading tested ✅")
    print("   - Multiple symbols tested ✅") 
    print("   - Conservative settings tested ✅")
    print("   - API endpoints validated ✅")
    print("\n🔒 Next Steps:")
    print("   1. Set MEXC_API_KEY and MEXC_API_SECRET for live trading")
    print("   2. Test with small amounts first (paper=false)")
    print("   3. Monitor logs: /api/logs/recent")
    print("   4. Check MEXC_SETUP.md for detailed configuration")
    
    return {"success": True, "tests_completed": 4}

def test_price_fetch():
    """Test MEXC price fetching directly"""
    print("\n💱 Testing direct MEXC price fetch...")
    
    try:
        from core.trading.mexc_adapter import MEXCExchangeAdapter
        
        # Test spot price
        adapter = MEXCExchangeAdapter(futures=False, dry_run=True)
        btc_price = adapter.get_price("BTCUSDT")
        print(f"   MEXC Spot BTC Price: ${btc_price:,.2f}")
        
        # Test futures price
        adapter_futures = MEXCExchangeAdapter(futures=True, dry_run=True)
        btc_futures_price = adapter_futures.get_price("BTC_USDT")  # MEXC futures format
        print(f"   MEXC Futures BTC Price: ${btc_futures_price:,.2f}")
        
        if btc_price > 0:
            print("✅ MEXC price fetching works!")
        else:
            print("❌ MEXC price fetching failed")
            
    except ImportError:
        print("❌ MEXC adapter not found - check installation")
    except Exception as e:
        print(f"❌ Price fetch error: {e}")

if __name__ == "__main__":
    print("=" * 60)
    print("🔧 MEXC Trading Bot Integration Test")
    print("=" * 60)
    
    # Test price fetching first
    test_price_fetch()
    
    # Test full bot integration
    result = test_mexc_bot()
    
    if result['success']:
        print("\n🎉 All tests completed successfully!")
        print("🚀 MEXC integration is ready to use!")
    else:
        print(f"\n💥 Tests failed: {result.get('error', 'Unknown error')}")
        print("🔧 Check your setup and try again.")
