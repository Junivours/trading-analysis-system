#!/usr/bin/env python3
"""
Quick test ob MEXC API Keys richtig gesetzt sind
"""
import os

def test_mexc_keys():
    print("🔍 MEXC API Keys Test...")
    
    api_key = os.getenv('MEXC_API_KEY')
    api_secret = os.getenv('MEXC_API_SECRET')
    
    if not api_key:
        print("❌ MEXC_API_KEY nicht gefunden!")
        print("   Setze: export MEXC_API_KEY='dein_key'")
        return False
        
    if not api_secret:
        print("❌ MEXC_API_SECRET nicht gefunden!")  
        print("   Setze: export MEXC_API_SECRET='dein_secret'")
        return False
    
    print(f"✅ MEXC_API_KEY: {api_key[:8]}...")
    print(f"✅ MEXC_API_SECRET: {api_secret[:8]}...")
    
    # Test Connection
    try:
        from core.trading.mexc_adapter import MEXCExchangeAdapter
        adapter = MEXCExchangeAdapter(dry_run=False)
        
        # Test Account Info
        account = adapter.get_account()
        if 'error' in account:
            print(f"❌ MEXC Connection Error: {account['error']}")
            return False
        else:
            print("✅ MEXC Connection successful!")
            return True
            
    except Exception as e:
        print(f"❌ Test Error: {e}")
        return False

if __name__ == "__main__":
    if test_mexc_keys():
        print("\n🚀 MEXC Setup ist bereit!")
        print("   Du kannst jetzt mit paper=false traden")
    else:
        print("\n🔧 Bitte korrigiere die API Key Einstellungen")
