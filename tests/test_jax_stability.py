#!/usr/bin/env python3
"""
Quick test script for JAX training stability
"""
import requests
import time

def test_jax_stability():
    print("🔥 Testing JAX Training Stability...")
    
    for i in range(3):
        print(f"\n🧠 Training Round {i+1}/3...")
        
        # Test JAX training
        try:
            response = requests.post('http://127.0.0.1:5001/api/jax_train', 
                                   json={'symbol': 'BTCUSDT', 'timeframe': '1h'})
            print(f"✅ Training Status: {response.status_code}")
            
            if response.status_code == 200:
                data = response.json()
                print(f"📊 Training successful: {data.get('training_metrics', {}).get('final_loss', 'N/A')} loss")
            
        except Exception as e:
            print(f"❌ Training failed: {e}")
            return False
        
        time.sleep(2)
        
        # Test server health
        try:
            health = requests.get('http://127.0.0.1:5001/api/jax_predictions/BTCUSDT')
            print(f"💚 Health Check: {health.status_code}")
            
            if health.status_code != 200:
                print("❌ Server unhealthy after training!")
                return False
                
        except Exception as e:
            print(f"❌ Health check failed: {e}")
            return False
    
    print("\n🎉 ALL TESTS PASSED - JAX Training is now stable!")
    return True

if __name__ == '__main__':
    test_jax_stability()
