#!/usr/bin/env python3
"""
🔥 ULTIMATE TRADING V4 - System Monitor & Auto-Recovery
================================================================================
⚡ Monitors system health and automatically restarts if needed
🧠 Prevents JAX training from crashing the main system
🎯 Ensures live trading system stays online 24/7
================================================================================
"""

import time
import requests
import subprocess
import logging
import sys
from datetime import datetime

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - MONITOR - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class TradingSystemMonitor:
    def __init__(self, port=5001, check_interval=30):
        self.port = port
        self.check_interval = check_interval
        self.base_url = f"http://127.0.0.1:{port}"
        self.failure_count = 0
        self.max_failures = 3
        self.last_restart = datetime.now()
        
    def check_health(self):
        """Check if the trading system is responsive"""
        try:
            # Test main endpoint
            response = requests.get(f"{self.base_url}/", timeout=10)
            if response.status_code == 200:
                return True
                
            # Test API endpoint
            response = requests.get(f"{self.base_url}/api/jax_test", timeout=10)
            if response.status_code == 200:
                return True
                
            return False
            
        except Exception as e:
            logger.warning(f"Health check failed: {e}")
            return False
    
    def restart_system(self):
        """Restart the trading system"""
        try:
            time_since_restart = (datetime.now() - self.last_restart).total_seconds()
            if time_since_restart < 60:  # Prevent rapid restarts
                logger.warning("Restart prevented: Too soon since last restart")
                return False
                
            logger.error("🚨 RESTARTING TRADING SYSTEM")
            
            # Kill existing process (if any)
            try:
                subprocess.run(['taskkill', '/f', '/im', 'python.exe'], 
                             capture_output=True, timeout=10)
                time.sleep(3)
            except:
                pass
            
            # Start new process
            subprocess.Popen([
                sys.executable, 
                'app_turbo.py'
            ], cwd=r'c:\Users\faruk\Downloads\TRADING AKTUELL')
            
            self.last_restart = datetime.now()
            self.failure_count = 0
            
            # Wait for startup
            time.sleep(15)
            return True
            
        except Exception as e:
            logger.error(f"Failed to restart system: {e}")
            return False
    
    def monitor(self):
        """Main monitoring loop"""
        logger.info("🔥 Trading System Monitor Started")
        logger.info(f"⚡ Monitoring {self.base_url} every {self.check_interval}s")
        
        while True:
            try:
                if self.check_health():
                    if self.failure_count > 0:
                        logger.info("✅ System recovered")
                        self.failure_count = 0
                    
                    # Log healthy status every 10 minutes
                    if int(time.time()) % 600 == 0:
                        logger.info("💚 Trading system healthy")
                        
                else:
                    self.failure_count += 1
                    logger.warning(f"❌ Health check failed ({self.failure_count}/{self.max_failures})")
                    
                    if self.failure_count >= self.max_failures:
                        if self.restart_system():
                            logger.info("🔄 System restart initiated")
                        else:
                            logger.error("💀 Failed to restart system")
                            time.sleep(60)  # Wait longer on restart failure
                
                time.sleep(self.check_interval)
                
            except KeyboardInterrupt:
                logger.info("🛑 Monitor stopped by user")
                break
            except Exception as e:
                logger.error(f"Monitor error: {e}")
                time.sleep(60)

def main():
    monitor = TradingSystemMonitor()
    monitor.monitor()

if __name__ == '__main__':
    main()
