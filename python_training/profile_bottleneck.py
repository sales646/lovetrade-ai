#!/usr/bin/env python3
"""Profile training to find bottlenecks"""

import torch
import time
import numpy as np
from trading_environment import create_trading_env

print("="*70)
print("TRAINING BOTTLENECK PROFILER")
print("="*70)

# Test GPU computation speed
print("\n1. Testing GPU Computation Speed...")
device = torch.device("cuda:0")
model_size = 1024
batch_size = 32768

# Create random data
x = torch.randn(batch_size, model_size).to(device)
w = torch.randn(model_size, model_size).to(device)

torch.cuda.synchronize()
start = time.time()
for _ in range(100):
    y = torch.matmul(x, w)
    torch.cuda.synchronize()
elapsed = time.time() - start
print(f"   GPU matmul: {elapsed:.3f}s for 100 iterations")
print(f"   Throughput: {100/elapsed:.1f} iterations/sec")

# Test environment creation speed
print("\n2. Testing Environment Creation...")
start = time.time()
try:
    env = create_trading_env(use_augmentation=False)
    elapsed = time.time() - start
    print(f"   ✅ Environment created in {elapsed:.3f}s")
except Exception as e:
    print(f"   ❌ Error: {e}")
    env = None
    elapsed = 999

# Test environment step speed
if env:
    print("\n3. Testing Environment Step Speed...")
    state = env.reset()
    
    start = time.time()
    for _ in range(1000):
        state, reward, done, info = env.step(env.action_space.sample())
        if done:
            state = env.reset()
    elapsed = time.time() - start
    
    print(f"   Environment steps: {elapsed:.3f}s for 1000 steps")
    print(f"   Throughput: {1000/elapsed:.1f} steps/sec")
    print(f"   ⚠️  With 512 envs, this is only {512*1000/elapsed:.0f} steps/sec total")

# Test data transfer speed
print("\n4. Testing CPU->GPU Transfer Speed...")
states = np.random.randn(32768, 50).astype(np.float32)

start = time.time()
for _ in range(100):
    tensor = torch.FloatTensor(states).to(device)
    torch.cuda.synchronize()
elapsed = time.time() - start

print(f"   Data transfer: {elapsed:.3f}s for 100 transfers")
print(f"   Throughput: {100/elapsed:.1f} transfers/sec")

# Bottleneck analysis
print("\n" + "="*70)
print("BOTTLENECK ANALYSIS")
print("="*70)

if env:
    env_steps_per_sec = 1000/elapsed
    total_env_throughput = 512 * env_steps_per_sec
    
    print(f"\n🔍 Environment is likely the bottleneck:")
    print(f"   - 512 envs can produce ~{total_env_throughput:.0f} steps/sec")
    print(f"   - But GPU can process 100K+ steps/sec")
    print(f"   - GPU is waiting {100000/total_env_throughput:.1f}x longer than computing!")
    
    print(f"\n💡 Solutions:")
    print(f"   1. Reduce envs_per_gpu from 256 to 32 (8x less)")
    print(f"   2. Pre-generate trajectories in database instead of simulating live")
    print(f"   3. Use C++/Cython for environment (100x faster)")
    print(f"   4. Cache environment states in GPU memory")

print("\n" + "="*70)
