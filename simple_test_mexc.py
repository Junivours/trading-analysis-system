#!/usr/bin/env python3
"""
Simple MEXC Bot Test - Ohne TensorFlow Loading
"""
import requests
import json

print("🚀 Testing MEXC Bot (Simple)...")

try:
    # Test MEXC Bot via API
    response = requests.post('http://localhost:5000/api/bot/run', 
        json={
            'symbol': 'BTCUSDT',
            'exchange': 'mexc',
            'paper': True,
            'risk_percentage': 0.5
        },
        timeout=30
    )
    
    print(f"Status Code: {response.status_code}")
    print(f"Response: {response.text}")
    
    if response.status_code == 200:
        data = response.json()
        print("✅ MEXC Bot Test SUCCESS!")
        print(f"Exchange: {data.get('exchange', 'unknown')}")
        print(f"Symbol: {data.get('symbol', 'unknown')}")
        print(f"Mode: {data.get('mode', 'unknown')}")
        print(f"Action: {data.get('action', 'none')}")
    else:
        print("❌ MEXC Bot Test FAILED")
        
except Exception as e:
    print(f"❌ Error: {e}")

print("\n📊 Test completed.")
