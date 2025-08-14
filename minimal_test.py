#!/usr/bin/env python3
"""
Minimal MEXC Test - Ohne Library Loading
"""
try:
    import requests
    import time
    
    print("🚀 Testing MEXC Bot Connection...")
    
    # Warte kurz bis Server bereit ist
    time.sleep(2)
    
    # Test Connection
    response = requests.get('http://localhost:5000/', timeout=10)
    print(f"Server Status: {response.status_code}")
    
    if response.status_code == 200:
        print("✅ Server is running!")
        
        # Test MEXC Bot
        bot_response = requests.post('http://localhost:5000/api/bot/run', 
            json={
                'symbol': 'BTCUSDT',
                'exchange': 'mexc',
                'paper': True,
                'risk_percentage': 0.5
            },
            timeout=30
        )
        
        print(f"Bot Status: {bot_response.status_code}")
        if bot_response.status_code == 200:
            data = bot_response.json()
            print("✅ MEXC Bot Test SUCCESS!")
            print(f"Exchange: {data.get('exchange', 'unknown')}")
            print(f"Symbol: {data.get('symbol', 'unknown')}")
            print(f"Mode: {data.get('mode', 'unknown')}")
        else:
            print(f"❌ Bot Error: {bot_response.text}")
    else:
        print("❌ Server not reachable")
        
except Exception as e:
    print(f"❌ Test Error: {e}")

print("📊 Test completed.")
