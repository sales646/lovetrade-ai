#!/usr/bin/env python3
"""Behavioral Cloning pretraining phase"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import List, Dict, Tuple
from pathlib import Path
import json

class ExpertTrajectoryDataset(Dataset):
    """Dataset from expert trajectories"""
    
    def __init__(self, symbols: List[str], data_env):
        self.states = []
        self.actions = []
        
        print("📚 Loading expert trajectories for BC pretraining...")
        
        # Collect trajectories by rolling out simple strategies
        for symbol in symbols:
            print(f"  Collecting trajectories for {symbol}...")
            self._collect_symbol_trajectories(symbol, data_env)
        
        print(f"✅ Loaded {len(self.states)} state-action pairs")
    
    def _collect_symbol_trajectories(self, symbol: str, env):
        """Collect expert trajectories using simple heuristics"""
        env.current_symbol = symbol
        
        for _ in range(10):  # 10 episodes per symbol
            state = env.reset()
            done = False
            
            while not done:
                # Simple expert policy based on indicators
                action = self._expert_action(state)
                
                self.states.append(state.copy())
                self.actions.append(action)
                
                next_state, reward, done, info = env.step(action)
                state = next_state
    
    def _expert_action(self, state: np.ndarray) -> int:
        """Simple expert policy: buy on dips, sell on peaks"""
        # Extract RSI and trend from state (assuming positions 20-21)
        rsi = state[20] if len(state) > 20 else 50.0
        
        # Denormalize RSI (was normalized to [-1, 1])
        rsi = (rsi + 1) * 50  # Back to 0-100
        
        # Simple strategy
        if rsi < 30:  # Oversold
            return 1  # Buy
        elif rsi > 70:  # Overbought
            return 2  # Sell
        else:
            return 0  # Hold
    
    def __len__(self):
        return len(self.states)
    
    def __getitem__(self, idx):
        return (
            torch.FloatTensor(self.states[idx]),
            torch.LongTensor([self.actions[idx]])
        )


def pretrain_bc(
    policy,
    symbols: List[str],
    env,
    device: str = "cuda",
    epochs: int = 5000,
    batch_size: int = 512,
    learning_rate: float = 3e-4,
    save_dir: str = "./checkpoints/bc"
) -> Dict:
    """Run BC pretraining phase"""
    
    print("\n" + "="*60)
    print("🎓 BEHAVIORAL CLONING PRETRAINING")
    print("="*60)
    
    # Create dataset
    dataset = ExpertTrajectoryDataset(symbols, env)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    # Setup training
    policy.train()
    policy.to(device)
    
    optimizer = optim.AdamW(policy.parameters(), lr=learning_rate, fused=True)
    criterion = nn.CrossEntropyLoss()
    
    best_loss = float('inf')
    patience = 10
    patience_counter = 0
    
    losses = []
    accuracies = []
    
    for epoch in range(epochs):
        epoch_loss = 0.0
        epoch_correct = 0
        epoch_total = 0
        
        for states, actions in dataloader:
            states = states.to(device, non_blocking=True)
            actions = actions.to(device, non_blocking=True).squeeze()
            
            # Forward pass
            action_probs, _ = policy(states)
            
            # BC loss: cross-entropy between predicted and expert actions
            loss = criterion(action_probs, actions)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            optimizer.step()
            
            # Track metrics
            epoch_loss += loss.item()
            
            pred_actions = torch.argmax(action_probs, dim=-1)
            epoch_correct += (pred_actions == actions).sum().item()
            epoch_total += actions.size(0)
        
        avg_loss = epoch_loss / len(dataloader)
        accuracy = epoch_correct / epoch_total
        
        losses.append(avg_loss)
        accuracies.append(accuracy)
        
        # Early stopping
        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0
            
            # Save best model
            Path(save_dir).mkdir(parents=True, exist_ok=True)
            torch.save({
                'epoch': epoch,
                'model_state_dict': policy.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_loss,
                'accuracy': accuracy,
            }, f"{save_dir}/bc_best.pt")
        else:
            patience_counter += 1
        
        if (epoch + 1) % 100 == 0:
            print(f"Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.4f} | Acc: {accuracy:.2%} | Best: {best_loss:.4f}")
        
        if patience_counter >= patience:
            print(f"✅ Early stopping at epoch {epoch+1}")
            break
    
    print("\n" + "="*60)
    print(f"✅ BC Pretraining Complete!")
    print(f"   Best Loss: {best_loss:.4f}")
    print(f"   Final Accuracy: {accuracies[-1]:.2%}")
    print("="*60 + "\n")
    
    return {
        'final_loss': avg_loss,
        'best_loss': best_loss,
        'final_accuracy': accuracies[-1],
        'epochs_trained': epoch + 1
    }


if __name__ == "__main__":
    from data_discovery import load_discovered_symbols
    from trading_environment import create_trading_env
    from transformer_policy import TransformerPolicy
    
    # Discover all symbols
    symbols_data = load_discovered_symbols()
    all_symbols = symbols_data['stocks'] + symbols_data['crypto']
    
    # Create environment
    env = create_trading_env(
        symbols=all_symbols,
        enable_multi_market=True,
        augment=True
    )
    
    # Create policy
    policy = TransformerPolicy(
        state_dim=52,
        action_dim=3,
        d_model=1024,
        nhead=16,
        num_layers=8,
        dim_feedforward=4096
    )
    
    # Run BC pretraining
    results = pretrain_bc(
        policy=policy,
        symbols=all_symbols,
        env=env,
        epochs=10000
    )
    
    print("BC pretraining completed!")
    print(json.dumps(results, indent=2))
