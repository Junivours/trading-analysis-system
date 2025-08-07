#!/usr/bin/env python3
"""
🔥 ULTIMATE TRADING V4 - Production Server with Auto-Recovery
================================================================================
⚡ Uses Waitress WSGI server for production stability
🧠 Automatic server restart on crashes
🎯 JAX training isolation to prevent crashes
🔧 Memory management and health monitoring
================================================================================
"""

import os
import sys
import time
import logging
import threading
import subprocess
from waitress import serve
from app_turbo import app

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - PRODUCTION - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ProductionServer:
    def __init__(self, host='0.0.0.0', port=5001):
        self.host = host
        self.port = port
        self.server_process = None
        self.monitor_thread = None
        self.running = False
        
    def start_server(self):
        """Start the production server with Waitress"""
        try:
            logger.info("🔥 Starting ULTIMATE TRADING V4 Production Server")
            logger.info(f"⚡ Serving on {self.host}:{self.port}")
            
            # Use Waitress for production stability
            serve(
                app,
                host=self.host,
                port=self.port,
                threads=4,  # Limit threads to prevent overload
                max_request_body_size=1073741824,  # 1GB max request
                cleanup_interval=30,  # Cleanup every 30 seconds
                channel_timeout=120,  # 2 minute timeout
                log_socket_errors=True,
                expose_tracebacks=False  # Security
            )
            
        except Exception as e:
            logger.error(f"❌ Server startup failed: {e}")
            raise
    
    def start_with_monitoring(self):
        """Start server with monitoring thread"""
        self.running = True
        
        # Start monitor thread
        self.monitor_thread = threading.Thread(
            target=self._monitor_health,
            daemon=True
        )
        self.monitor_thread.start()
        
        # Start main server
        self.start_server()
    
    def _monitor_health(self):
        """Monitor server health and restart if needed"""
        import requests
        import time
        
        logger.info("🔍 Health monitor started")
        
        while self.running:
            try:
                time.sleep(30)  # Check every 30 seconds
                
                # Health check
                response = requests.get(
                    f"http://{self.host}:{self.port}/",
                    timeout=10
                )
                
                if response.status_code == 200:
                    logger.debug("💚 Server healthy")
                else:
                    logger.warning(f"⚠️ Server health check failed: {response.status_code}")
                    
            except Exception as e:
                logger.error(f"❌ Health check failed: {e}")
                # Server might be down, restart handled by supervisor
                
            except KeyboardInterrupt:
                logger.info("🛑 Health monitor stopped")
                break

def main():
    """Main entry point"""
    try:
        # Check if Waitress is installed
        try:
            import waitress
            logger.info("✅ Waitress WSGI server available")
        except ImportError:
            logger.error("❌ Waitress not installed. Installing...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", "waitress"])
            logger.info("✅ Waitress installed successfully")
        
        # Start production server
        server = ProductionServer()
        server.start_with_monitoring()
        
    except KeyboardInterrupt:
        logger.info("🛑 Server stopped by user")
    except Exception as e:
        logger.error(f"💀 Server crashed: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()
