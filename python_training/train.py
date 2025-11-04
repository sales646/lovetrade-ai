#!/usr/bin/env python3
"""Simple GPU Training - BC + PPO"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from datetime import datetime
from tqdm import tqdm
import sys

print("="*70)
print("SIMPLE GPU TRAINING")
print("="*70)

# ====================
# STEP 1: GPU Setup
# ====================
print("\n[STEP 1/6] GPU Setup")
if not torch.cuda.is_available():
    print("❌ No GPU detected!")
    sys.exit(1)

device = torch.device('cuda')
gpu_name = torch.cuda.get_device_name(0)
gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
print(f"✅ Using: {gpu_name} ({gpu_memory:.1f} GB)")

# ====================
# STEP 2: Load Data
# ====================
print("\n[STEP 2/6] Loading Data from S3")
from s3_data_loader import S3DataLoader

loader = S3DataLoader()

# Discover symbols
print("   Discovering symbols...")
symbols = loader.discover_all_symbols(max_symbols=200)
print(f"✅ Found {len(symbols)} symbols")

# Load one month of data to start
print("   Loading market data (January 2024)...")
df = loader.load_multi_day_data(
    start_date="2024-01-01",
    end_date="2024-01-31",
    symbols=symbols[:100]  # Use first 100 symbols
)
print(f"✅ Loaded {len(df):,} rows")

# ====================
# STEP 3: Environment
# ====================
print("\n[STEP 3/6] Creating Trading Environment")
from trading_environment import TradingEnvironment

env = TradingEnvironment(df, symbols=symbols[:100])
print(f"✅ Environment ready")

# ====================
# STEP 4: Neural Network
# ====================
print("\n[STEP 4/6] Creating Neural Network")

class SimplePolicy(nn.Module):
    """Simple but effective policy network"""
    def __init__(self, state_dim=52, action_dim=3, hidden=512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, action_dim)
        )
        self.value_head = nn.Linear(hidden, 1)
    
    def forward(self, x):
        features = self.net[:-1](x)  # All layers except last
        action_logits = self.net[-1](features)
        value = self.value_head(features)
        return action_logits, value

policy = SimplePolicy().to(device)
optimizer = optim.Adam(policy.parameters(), lr=0.0003)

param_count = sum(p.numel() for p in policy.parameters())
print(f"✅ Model on GPU ({param_count:,} parameters)")

# ====================
# STEP 5: BC Pretraining
# ====================
print("\n[STEP 5/6] Behavior Cloning (BC) Pretraining")
print("   Collecting expert demonstrations...")

# Collect some trajectories
states, actions = [], []
for _ in range(100):  # 100 episodes
    state = env.reset()
    for step in range(50):  # 50 steps each
        # Simple expert: buy on uptrend, sell on downtrend
        price_change = state[1]  # Assume this is price change
        if price_change > 0.01:
            action = 2  # Buy
        elif price_change < -0.01:
            action = 0  # Sell
        else:
            action = 1  # Hold
        
        states.append(state)
        actions.append(action)
        
        next_state, _, done, _ = env.step(action)
        if done:
            break
        state = next_state

print(f"✅ Collected {len(states)} transitions")

# Train BC
print("   Training on demonstrations...")
states_tensor = torch.FloatTensor(np.array(states)).to(device)
actions_tensor = torch.LongTensor(actions).to(device)

bc_epochs = 20
for epoch in range(bc_epochs):
    # Forward pass
    logits, _ = policy(states_tensor)
    loss = nn.CrossEntropyLoss()(logits, actions_tensor)
    
    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    # Calculate accuracy
    preds = torch.argmax(logits, dim=1)
    acc = (preds == actions_tensor).float().mean()
    
    if (epoch + 1) % 5 == 0:
        print(f"   Epoch {epoch+1}/{bc_epochs} | Loss: {loss.item():.4f} | Acc: {acc.item():.2%}")

print(f"✅ BC pretraining complete")

# ====================
# STEP 6: PPO Training
# ====================
print("\n[STEP 6/6] PPO Reinforcement Learning")
print("   Training for 50 epochs...")
print("-"*70)

best_reward = float('-inf')

for epoch in range(50):
    # Collect rollout
    states_list, actions_list, rewards_list = [], [], []
    
    state = env.reset()
    episode_reward = 0
    
    for step in range(200):
        # Get action from policy
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
            logits, value = policy(state_tensor)
            probs = torch.softmax(logits, dim=-1)
            action = torch.multinomial(probs, 1).item()
        
        # Take step
        next_state, reward, done, _ = env.step(action)
        
        states_list.append(state)
        actions_list.append(action)
        rewards_list.append(reward)
        episode_reward += reward
        
        if done:
            state = env.reset()
        else:
            state = next_state
    
    # Train on rollout
    states_batch = torch.FloatTensor(np.array(states_list)).to(device)
    actions_batch = torch.LongTensor(actions_list).to(device)
    rewards_batch = torch.FloatTensor(rewards_list).to(device)
    
    # PPO update
    logits, values = policy(states_batch)
    values = values.squeeze()
    
    # Actor loss
    log_probs = torch.log_softmax(logits, dim=-1)
    action_log_probs = log_probs.gather(1, actions_batch.unsqueeze(1)).squeeze()
    advantages = rewards_batch - values.detach()
    actor_loss = -(action_log_probs * advantages).mean()
    
    # Critic loss
    critic_loss = nn.MSELoss()(values, rewards_batch)
    
    # Total loss
    loss = actor_loss + 0.5 * critic_loss
    
    # Update
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
    optimizer.step()
    
    # Log
    avg_reward = rewards_batch.mean().item()
    if epoch % 5 == 0 or epoch == 49:
        print(f"Epoch {epoch+1:3d}/50 | Reward: {avg_reward:7.4f} | Loss: {loss.item():7.4f}")
    
    # Save best
    if avg_reward > best_reward:
        best_reward = avg_reward
        torch.save(policy.state_dict(), 'checkpoints/best_policy.pt')

print("\n" + "="*70)
print("✅ TRAINING COMPLETE!")
print("="*70)
print(f"\nBest average reward: {best_reward:.4f}")
print(f"Model saved: checkpoints/best_policy.pt")
print("\nNext steps:")
print("1. Check training dashboard for results")
print("2. Increase epochs and data range for better performance")
print("3. Add more symbols and longer time periods")
