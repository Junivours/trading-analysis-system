import requests
import json

# MEXC Bot Test
print("🚀 Testing MEXC Bot...")

response = requests.post('http://localhost:5000/api/bot/run', 
    json={
        'symbol': 'BTCUSDT',
        'exchange': 'mexc', 
        'paper': True,
        'equity': 1000,
        'risk_pct': 0.5
    }
)

result = response.json()

if result.get('success'):
    print('✅ MEXC Bot Test erfolgreich!')
    print(f'Exchange: {result.get("exchange")}')
    print(f'Paper Mode: {result.get("paper")}')
    print(f'Symbol: {result["data"]["symbol"]}')
    print(f'Setups processed: {len(result["data"]["executed"])}')
    
    for i, setup in enumerate(result["data"]["executed"]):
        if setup.get('skipped'):
            print(f'  Setup {i+1}: SKIPPED - {setup.get("reason")}')
        else:
            print(f'  Setup {i+1}: EXECUTED - {setup.get("direction")} {setup.get("qty")} at {setup.get("entry")}')
else:
    print(f'❌ Bot Test Fehler: {result.get("error")}')

print(f'\nFull Response: {json.dumps(result, indent=2)}')
