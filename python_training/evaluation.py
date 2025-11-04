#!/usr/bin/env python3
"""Automatic evaluation of trained models"""

import numpy as np
from typing import Dict, List, Tuple
import torch
from collections import defaultdict

def calculate_sharpe_ratio(returns: np.ndarray, risk_free_rate: float = 0.0) -> float:
    """Calculate Sharpe ratio"""
    if len(returns) < 2:
        return 0.0
    
    excess_returns = returns - risk_free_rate
    if np.std(excess_returns) == 0:
        return 0.0
    
    return np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252 * 390)  # Annualized


def calculate_sortino_ratio(returns: np.ndarray, risk_free_rate: float = 0.0) -> float:
    """Calculate Sortino ratio (only downside deviation)"""
    if len(returns) < 2:
        return 0.0
    
    excess_returns = returns - risk_free_rate
    downside_returns = excess_returns[excess_returns < 0]
    
    if len(downside_returns) == 0 or np.std(downside_returns) == 0:
        return 0.0
    
    return np.mean(excess_returns) / np.std(downside_returns) * np.sqrt(252 * 390)


def calculate_max_drawdown(equity_curve: np.ndarray) -> float:
    """Calculate maximum drawdown"""
    if len(equity_curve) < 2:
        return 0.0
    
    running_max = np.maximum.accumulate(equity_curve)
    drawdown = (equity_curve - running_max) / running_max
    return float(np.min(drawdown))


def calculate_win_rate(returns: np.ndarray) -> float:
    """Calculate win rate"""
    if len(returns) == 0:
        return 0.0
    
    winning_trades = np.sum(returns > 0)
    return winning_trades / len(returns)


def calculate_profit_factor(returns: np.ndarray) -> float:
    """Calculate profit factor"""
    gross_profit = np.sum(returns[returns > 0])
    gross_loss = abs(np.sum(returns[returns < 0]))
    
    if gross_loss == 0:
        return float('inf') if gross_profit > 0 else 0.0
    
    return gross_profit / gross_loss


def evaluate_model(env, policy, num_episodes: int = 100, device: str = "cuda") -> Dict:
    """Evaluate trained model on validation data"""
    print(f"\n📊 Evaluating model on {num_episodes} episodes...")
    
    policy.eval()
    
    episode_returns = []
    episode_sharpes = []
    equity_curves = []
    all_returns = []
    actions_taken = defaultdict(int)
    
    with torch.no_grad():
        for ep in range(num_episodes):
            state = env.reset()
            done = False
            episode_equity = [100000.0]  # Starting capital
            episode_pnl = []
            
            while not done:
                # Get action from policy
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
                action_probs, _ = policy(state_tensor)
                
                # Sample action (or take argmax for deterministic)
                action = torch.argmax(action_probs, dim=-1).item()
                actions_taken[action] += 1
                
                # Step environment
                next_state, reward, done, info = env.step(action)
                
                # Track equity
                if 'pnl' in info and info['pnl'] != 0:
                    episode_pnl.append(info['pnl'])
                    episode_equity.append(episode_equity[-1] + info['pnl'])
                
                state = next_state
            
            # Episode stats
            if len(episode_pnl) > 0:
                episode_return = (episode_equity[-1] - episode_equity[0]) / episode_equity[0]
                episode_returns.append(episode_return)
                all_returns.extend(episode_pnl)
                equity_curves.append(np.array(episode_equity))
                
                # Calculate episode Sharpe
                if len(episode_pnl) > 1:
                    ep_sharpe = calculate_sharpe_ratio(np.array(episode_pnl))
                    episode_sharpes.append(ep_sharpe)
            
            if (ep + 1) % 20 == 0:
                print(f"  Evaluated {ep + 1}/{num_episodes} episodes...")
    
    # Aggregate metrics
    episode_returns = np.array(episode_returns)
    all_returns = np.array(all_returns)
    
    # Combine all equity curves
    max_len = max(len(eq) for eq in equity_curves)
    padded_curves = []
    for eq in equity_curves:
        padded = np.pad(eq, (0, max_len - len(eq)), mode='edge')
        padded_curves.append(padded)
    
    avg_equity_curve = np.mean(padded_curves, axis=0)
    
    results = {
        "num_episodes": num_episodes,
        "mean_return": float(np.mean(episode_returns)),
        "std_return": float(np.std(episode_returns)),
        "sharpe_ratio": calculate_sharpe_ratio(all_returns),
        "sortino_ratio": calculate_sortino_ratio(all_returns),
        "max_drawdown": calculate_max_drawdown(avg_equity_curve),
        "win_rate": calculate_win_rate(all_returns),
        "profit_factor": calculate_profit_factor(all_returns),
        "total_trades": len(all_returns),
        "action_distribution": {
            "hold": actions_taken[0] / sum(actions_taken.values()) * 100,
            "buy": actions_taken[1] / sum(actions_taken.values()) * 100,
            "sell": actions_taken[2] / sum(actions_taken.values()) * 100,
        }
    }
    
    # Print results
    print("\n" + "="*60)
    print("📈 EVALUATION RESULTS")
    print("="*60)
    print(f"Mean Return:       {results['mean_return']:>8.2%}")
    print(f"Std Return:        {results['std_return']:>8.2%}")
    print(f"Sharpe Ratio:      {results['sharpe_ratio']:>8.2f}")
    print(f"Sortino Ratio:     {results['sortino_ratio']:>8.2f}")
    print(f"Max Drawdown:      {results['max_drawdown']:>8.2%}")
    print(f"Win Rate:          {results['win_rate']:>8.2%}")
    print(f"Profit Factor:     {results['profit_factor']:>8.2f}")
    print(f"Total Trades:      {results['total_trades']:>8}")
    print("\nAction Distribution:")
    print(f"  Hold: {results['action_distribution']['hold']:>6.2f}%")
    print(f"  Buy:  {results['action_distribution']['buy']:>6.2f}%")
    print(f"  Sell: {results['action_distribution']['sell']:>6.2f}%")
    print("="*60 + "\n")
    
    return results


def log_ppo_metrics(epoch: int, metrics: Dict):
    """Log PPO training metrics"""
    print(f"\n📊 Epoch {epoch} PPO Metrics:")
    print(f"  KL Divergence:  {metrics.get('kl_div', 0):.6f}")
    print(f"  Entropy:        {metrics.get('entropy', 0):.6f}")
    print(f"  Value Loss:     {metrics.get('value_loss', 0):.6f}")
    print(f"  Policy Loss:    {metrics.get('policy_loss', 0):.6f}")
    print(f"  Clip Fraction:  {metrics.get('clip_fraction', 0):.4f}")
    print(f"  Mean Reward:    {metrics.get('mean_reward', 0):.4f}")
