#!/usr/bin/env python3
"""
Continuous parallel training runner
Runs multiple training processes simultaneously and restarts them when complete
"""

import subprocess
import time
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def run_training(worker_id: int) -> dict:
    """Run a single training iteration"""
    start_time = datetime.now()
    logger.info(f"[Worker {worker_id}] Starting training run")
    
    try:
        # Run the training script
        result = subprocess.run(
            ["python", "train_rl_policy.py"],
            capture_output=True,
            text=True,
            timeout=3600  # 1 hour timeout per training
        )
        
        duration = (datetime.now() - start_time).total_seconds()
        
        if result.returncode == 0:
            logger.info(f"[Worker {worker_id}] Training completed successfully in {duration:.1f}s")
            return {"worker_id": worker_id, "status": "success", "duration": duration}
        else:
            logger.error(f"[Worker {worker_id}] Training failed: {result.stderr[-500:]}")
            return {"worker_id": worker_id, "status": "failed", "error": result.stderr[-500:]}
    
    except subprocess.TimeoutExpired:
        logger.error(f"[Worker {worker_id}] Training timed out after 1 hour")
        return {"worker_id": worker_id, "status": "timeout"}
    
    except Exception as e:
        logger.error(f"[Worker {worker_id}] Unexpected error: {e}")
        return {"worker_id": worker_id, "status": "error", "error": str(e)}


def main():
    """Run continuous parallel training"""
    num_workers = 5
    logger.info(f"Starting continuous training with {num_workers} parallel workers")
    logger.info("Press Ctrl+C to stop")
    
    try:
        while True:
            logger.info(f"\n{'='*60}")
            logger.info(f"Starting new batch of {num_workers} training runs")
            logger.info(f"{'='*60}\n")
            
            # Run training in parallel
            with ProcessPoolExecutor(max_workers=num_workers) as executor:
                futures = {
                    executor.submit(run_training, i): i 
                    for i in range(1, num_workers + 1)
                }
                
                completed = 0
                for future in as_completed(futures):
                    result = future.result()
                    completed += 1
                    logger.info(f"Progress: {completed}/{num_workers} workers completed")
            
            # Short pause between batches
            logger.info("\nBatch complete. Starting next batch in 10 seconds...")
            time.sleep(10)
    
    except KeyboardInterrupt:
        logger.info("\n\nStopping continuous training (Ctrl+C pressed)")
        logger.info("Waiting for current trainings to finish...")
    except Exception as e:
        logger.error(f"Fatal error: {e}")


if __name__ == "__main__":
    main()
