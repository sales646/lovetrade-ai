#!/usr/bin/env python3
"""Simple GPU Training - Single GPU, Clear Progress"""

import torch
import sys

print("="*60)
print("SIMPLE GPU TRAINING")
print("="*60)

# Step 1: Check GPU
print("\n[1/5] Checking GPU...")
if not torch.cuda.is_available():
    print("❌ CUDA not available! Install PyTorch with CUDA support.")
    sys.exit(1)

gpu_name = torch.cuda.get_device_name(0)
gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
print(f"✅ GPU detected: {gpu_name}")
print(f"   Memory: {gpu_memory:.1f} GB")

# Step 2: Load data
print("\n[2/5] Loading data from S3...")
from s3_data_loader import S3DataLoader

loader = S3DataLoader()
print("   Discovering symbols...")
symbols = loader.discover_all_symbols(max_symbols=100)  # Start with 100 symbols
print(f"✅ Found {len(symbols)} symbols")

print("   Loading market data...")
df = loader.load_multi_day_data(
    start_date="2024-01-01",
    end_date="2024-01-31",  # Just January for quick start
    symbols=symbols[:50]  # Use first 50 symbols
)
print(f"✅ Loaded {len(df):,} rows")

# Step 3: Create training environment
print("\n[3/5] Setting up training environment...")
from trading_environment import TradingEnvironment

env = TradingEnvironment(df, symbols=symbols[:50])
print(f"✅ Environment ready")

# Step 4: Initialize model
print("\n[4/5] Creating neural network...")
from transformer_policy import TransformerPolicy

policy = TransformerPolicy(
    state_dim=52,
    action_dim=3,
    d_model=512,  # Reasonable size
    nhead=8,
    num_layers=6,
    device='cuda'
)
print(f"✅ Model on GPU: {next(policy.parameters()).device}")

# Step 5: Train
print("\n[5/5] Starting training...")
print("   Training for 10 epochs (quick test)")
print("-"*60)

optimizer = torch.optim.Adam(policy.parameters(), lr=0.0003)

for epoch in range(10):
    state = env.reset()
    total_reward = 0
    steps = 0
    
    for step in range(100):  # 100 steps per epoch
        # Get action from policy
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).cuda()
            action_probs = policy(state_tensor)
            action = torch.argmax(action_probs, dim=-1).item()
        
        # Take step
        next_state, reward, done, _ = env.step(action)
        total_reward += reward
        steps += 1
        
        if done:
            state = env.reset()
        else:
            state = next_state
    
    print(f"Epoch {epoch+1}/10 | Steps: {steps} | Reward: {total_reward:.4f}")

print("\n" + "="*60)
print("✅ TRAINING COMPLETE!")
print("="*60)
print("\nNext steps:")
print("1. If this worked, increase epochs and data range")
print("2. Run: python train_rl_policy.py for full BC+PPO training")
print("3. Check results in dashboard")
