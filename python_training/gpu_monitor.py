"""
GPU Monitoring and Load Balancing
Tracks GPU utilization, memory, temperature for optimal distributed training
"""
import subprocess
import json
import time
from typing import Dict, List, Optional
from dataclasses import dataclass
import threading


@dataclass
class GPUStats:
    """GPU statistics at a point in time"""
    gpu_id: int
    name: str
    utilization: float  # 0-100%
    memory_used: float  # GB
    memory_total: float  # GB
    memory_utilization: float  # 0-100%
    temperature: float  # Celsius
    power_draw: float  # Watts
    power_limit: float  # Watts


class GPUMonitor:
    """Real-time GPU monitoring"""
    
    def __init__(self, refresh_interval: float = 1.0):
        self.refresh_interval = refresh_interval
        self.is_monitoring = False
        self.monitor_thread = None
        self.gpu_stats_history: List[List[GPUStats]] = []
        self._telemetry_available = True
        self._telemetry_error_reported = False
        
    def start(self):
        """Start monitoring in background thread"""
        if self.is_monitoring:
            return
        
        self.is_monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        print("🔍 GPU monitoring started")
    
    def stop(self):
        """Stop monitoring"""
        self.is_monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=2.0)
        print("🛑 GPU monitoring stopped")
    
    def _monitor_loop(self):
        """Background monitoring loop"""
        while self.is_monitoring:
            try:
                stats = self.get_gpu_stats()
                self.gpu_stats_history.append(stats)
                
                # Keep last 1000 samples (adjust as needed)
                if len(self.gpu_stats_history) > 1000:
                    self.gpu_stats_history = self.gpu_stats_history[-1000:]
                
                time.sleep(self.refresh_interval)
            except Exception as e:
                print(f"⚠️ Monitoring error: {e}")
                time.sleep(self.refresh_interval)
    
    def get_gpu_stats(self) -> List[GPUStats]:
        """Get current GPU statistics using nvidia-smi"""
        if not self._telemetry_available:
            return []

        try:
            result = subprocess.run(
                [
                    'nvidia-smi',
                    '--query-gpu=index,name,utilization.gpu,memory.used,memory.total,memory.utilization,temperature.gpu,power.draw,power.limit',
                    '--format=csv,noheader,nounits'
                ],
                capture_output=True,
                text=True,
                check=True
            )
            
            stats = []
            for line in result.stdout.strip().split('\n'):
                if not line:
                    continue
                
                parts = [p.strip() for p in line.split(',')]
                if len(parts) < 9:
                    continue
                
                stats.append(GPUStats(
                    gpu_id=int(parts[0]),
                    name=parts[1],
                    utilization=float(parts[2]),
                    memory_used=float(parts[3]) / 1024,  # MB to GB
                    memory_total=float(parts[4]) / 1024,
                    memory_utilization=float(parts[5]),
                    temperature=float(parts[6]),
                    power_draw=float(parts[7]),
                    power_limit=float(parts[8])
                ))
            
            return stats
        
        except Exception as e:
            if not self._telemetry_error_reported:
                print(f"⚠️ GPU telemetry disabled: {e}")
                self._telemetry_error_reported = True
            self._telemetry_available = False
            return []
    
    def get_avg_stats(self, window: int = 10) -> List[GPUStats]:
        """Get average stats over last N samples"""
        if not self.gpu_stats_history:
            return self.get_gpu_stats()
        
        recent = self.gpu_stats_history[-window:]
        if not recent:
            return []
        
        # Average across time for each GPU
        num_gpus = len(recent[0])
        avg_stats = []
        
        for gpu_id in range(num_gpus):
            gpu_samples = [sample[gpu_id] for sample in recent if gpu_id < len(sample)]
            
            if not gpu_samples:
                continue
            
            avg_stats.append(GPUStats(
                gpu_id=gpu_id,
                name=gpu_samples[0].name,
                utilization=sum(s.utilization for s in gpu_samples) / len(gpu_samples),
                memory_used=sum(s.memory_used for s in gpu_samples) / len(gpu_samples),
                memory_total=gpu_samples[0].memory_total,
                memory_utilization=sum(s.memory_utilization for s in gpu_samples) / len(gpu_samples),
                temperature=sum(s.temperature for s in gpu_samples) / len(gpu_samples),
                power_draw=sum(s.power_draw for s in gpu_samples) / len(gpu_samples),
                power_limit=gpu_samples[0].power_limit
            ))
        
        return avg_stats
    
    def get_summary(self) -> Dict:
        """Get summary of GPU usage"""
        stats = self.get_avg_stats(window=30)
        
        if not stats:
            return {'available': False, 'error': 'No GPU data'}
        
        return {
            'available': True,
            'num_gpus': len(stats),
            'avg_utilization': sum(s.utilization for s in stats) / len(stats),
            'avg_memory_used': sum(s.memory_used for s in stats) / len(stats),
            'avg_temperature': sum(s.temperature for s in stats) / len(stats),
            'total_power': sum(s.power_draw for s in stats),
            'gpus': [
                {
                    'id': s.gpu_id,
                    'name': s.name,
                    'utilization': s.utilization,
                    'memory_used_gb': s.memory_used,
                    'memory_total_gb': s.memory_total,
                    'temperature_c': s.temperature
                }
                for s in stats
            ]
        }


class LoadBalancer:
    """Intelligent load balancing across GPUs"""
    
    def __init__(self, monitor: GPUMonitor):
        self.monitor = monitor
    
    def get_least_loaded_gpu(self) -> int:
        """Find GPU with lowest current load"""
        stats = self.monitor.get_avg_stats(window=5)
        
        if not stats:
            return 0
        
        # Score based on utilization and memory
        scores = []
        for s in stats:
            score = (s.utilization / 100) * 0.6 + (s.memory_utilization / 100) * 0.4
            scores.append((s.gpu_id, score))
        
        # Return GPU with lowest score
        return min(scores, key=lambda x: x[1])[0]
    
    def assign_workers_to_gpus(
        self,
        num_workers: int,
        gpu_ids: Optional[List[int]] = None
    ) -> List[int]:
        """Assign workers to GPUs for balanced load"""
        stats = self.monitor.get_gpu_stats()
        
        if not stats:
            # No GPU info, distribute evenly
            return [i % num_workers for i in range(num_workers)]
        
        if gpu_ids is None:
            gpu_ids = [s.gpu_id for s in stats]
        
        # Simple round-robin for now
        # Could be made smarter based on current load
        assignments = []
        for i in range(num_workers):
            assignments.append(gpu_ids[i % len(gpu_ids)])
        
        return assignments
    
    def check_thermal_throttling(self, threshold_temp: float = 80.0) -> Dict:
        """Check if any GPUs are thermal throttling"""
        stats = self.monitor.get_avg_stats(window=10)
        
        throttling = []
        for s in stats:
            if s.temperature > threshold_temp:
                throttling.append({
                    'gpu_id': s.gpu_id,
                    'temperature': s.temperature,
                    'status': 'throttling' if s.temperature > 85 else 'warning'
                })
        
        return {
            'is_throttling': len(throttling) > 0,
            'gpus': throttling
        }


def print_gpu_summary():
    """Print a nice GPU summary"""
    monitor = GPUMonitor()
    stats = monitor.get_gpu_stats()
    
    if not stats:
        print("❌ No GPUs detected or nvidia-smi not available")
        return
    
    print("\n" + "="*70)
    print("🎮 GPU Status Summary")
    print("="*70)
    
    for s in stats:
        print(f"\nGPU {s.gpu_id}: {s.name}")
        print(f"  Utilization:  {s.utilization:>5.1f}% {'▓' * int(s.utilization/10)}")
        print(f"  Memory:       {s.memory_used:>5.1f}/{s.memory_total:.1f} GB "
              f"({s.memory_utilization:.1f}%) {'▓' * int(s.memory_utilization/10)}")
        print(f"  Temperature:  {s.temperature:>5.1f}°C")
        print(f"  Power:        {s.power_draw:>5.1f}/{s.power_limit:.1f} W")
    
    print("\n" + "="*70 + "\n")


if __name__ == "__main__":
    print_gpu_summary()
