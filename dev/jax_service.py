#!/usr/bin/env python3
"""
🔥 JAX Training Isolation Service
================================================================================
⚡ Runs JAX training in isolated subprocess to prevent server crashes
🧠 Communicates via file-based queues
🎯 Ensures main trading server stays stable during AI training
================================================================================
"""

import os
import sys
import json
import time
import subprocess
import logging
from datetime import datetime
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - JAX_SERVICE - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class JAXTrainingService:
    def __init__(self, work_dir="c:/Users/faruk/Downloads/TRADING AKTUELL"):
        self.work_dir = Path(work_dir)
        self.queue_dir = self.work_dir / "jax_queue"
        self.queue_dir.mkdir(exist_ok=True)
        
    def submit_training_job(self, symbol="BTCUSDT", timeframe="1h"):
        """Submit a JAX training job to the isolated service"""
        try:
            job_id = f"train_{symbol}_{timeframe}_{int(time.time())}"
            
            job_data = {
                "job_id": job_id,
                "symbol": symbol,
                "timeframe": timeframe,
                "timestamp": datetime.now().isoformat(),
                "status": "pending"
            }
            
            # Write job to queue
            job_file = self.queue_dir / f"{job_id}.json"
            with open(job_file, 'w') as f:
                json.dump(job_data, f, indent=2)
            
            logger.info(f"🔥 JAX training job submitted: {job_id}")
            
            # Start isolated training process
            self._start_isolated_training(job_id)
            
            return {
                "job_id": job_id,
                "status": "submitted",
                "message": f"JAX training started for {symbol} on {timeframe}"
            }
            
        except Exception as e:
            logger.error(f"❌ Failed to submit training job: {e}")
            return {
                "status": "error",
                "message": str(e)
            }
    
    def _start_isolated_training(self, job_id):
        """Start JAX training in isolated subprocess"""
        try:
            # Create isolated training script
            training_script = f"""
#!/usr/bin/env python3
import sys
import json
import os
from pathlib import Path

# Add current directory to path
sys.path.insert(0, r'{self.work_dir}')

def run_training():
    try:
        # Import our JAX training code
        from app_turbo import JAXTradingAI
        
        # Load job data
        job_file = Path(r'{self.queue_dir}') / '{job_id}.json'
        with open(job_file, 'r') as f:
            job_data = json.load(f)
        
        symbol = job_data['symbol']
        timeframe = job_data['timeframe']
        
        print(f"🔥 Starting isolated JAX training: {{symbol}} on {{timeframe}}")
        
        # Initialize JAX AI
        jax_ai = JAXTradingAI()
        
        # Run training
        result = jax_ai.train_model(symbol, timeframe)
        
        # Update job status
        job_data['status'] = 'completed'
        job_data['result'] = result
        job_data['completed_at'] = '{datetime.now().isoformat()}'
        
        # Write result
        with open(job_file, 'w') as f:
            json.dump(job_data, f, indent=2)
        
        print(f"✅ JAX training completed: {{result.get('final_loss', 'N/A')}}")
        
    except Exception as e:
        print(f"❌ JAX training failed: {{e}}")
        
        # Update job status with error
        try:
            with open(job_file, 'r') as f:
                job_data = json.load(f)
            job_data['status'] = 'failed'
            job_data['error'] = str(e)
            job_data['failed_at'] = '{datetime.now().isoformat()}'
            with open(job_file, 'w') as f:
                json.dump(job_data, f, indent=2)
        except:
            pass

if __name__ == '__main__':
    run_training()
"""
            
            # Write training script
            script_file = self.queue_dir / f"train_{job_id}.py"
            with open(script_file, 'w') as f:
                f.write(training_script)
            
            # Start subprocess
            subprocess.Popen([
                sys.executable,
                str(script_file)
            ], cwd=str(self.work_dir))
            
            logger.info(f"🚀 Isolated training process started for {job_id}")
            
        except Exception as e:
            logger.error(f"❌ Failed to start isolated training: {e}")
    
    def get_job_status(self, job_id):
        """Get status of a training job"""
        try:
            job_file = self.queue_dir / f"{job_id}.json"
            
            if not job_file.exists():
                return {"status": "not_found", "message": "Job not found"}
            
            with open(job_file, 'r') as f:
                job_data = json.load(f)
            
            return job_data
            
        except Exception as e:
            logger.error(f"❌ Failed to get job status: {e}")
            return {"status": "error", "message": str(e)}
    
    def cleanup_old_jobs(self, max_age_hours=24):
        """Clean up old job files"""
        try:
            cutoff_time = time.time() - (max_age_hours * 3600)
            
            for job_file in self.queue_dir.glob("*.json"):
                if job_file.stat().st_mtime < cutoff_time:
                    job_file.unlink()
                    
            # Also clean up training scripts
            for script_file in self.queue_dir.glob("train_*.py"):
                if script_file.stat().st_mtime < cutoff_time:
                    script_file.unlink()
                    
            logger.info("🧹 Old training jobs cleaned up")
            
        except Exception as e:
            logger.error(f"❌ Failed to cleanup jobs: {e}")

# Global service instance
jax_service = JAXTrainingService()

def main():
    """Test the service"""
    print("🔥 Testing JAX Training Service")
    
    # Submit a test job
    result = jax_service.submit_training_job("BTCUSDT", "1h")
    print(f"Job submission result: {result}")
    
    if result.get("status") == "submitted":
        job_id = result["job_id"]
        
        # Monitor job progress
        for i in range(30):  # Wait up to 5 minutes
            status = jax_service.get_job_status(job_id)
            print(f"Job status: {status.get('status', 'unknown')}")
            
            if status.get("status") in ["completed", "failed"]:
                print(f"Final result: {status}")
                break
                
            time.sleep(10)

if __name__ == '__main__':
    main()
