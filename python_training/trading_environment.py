"""
Real Trading Environment - Connects to Supabase Data
Uses expert trajectories and market data for realistic RL training
"""
import os
import numpy as np
from typing import Dict, List, Optional, Tuple
from supabase import create_client, Client
from datetime import datetime
import json


class TradingEnvironment:
    """
    Real trading environment that uses Supabase data
    
    Features:
    - Loads expert trajectories from Supabase
    - Uses real market features (price, indicators, etc.)
    - Calculates profit-based rewards
    - Supports multiple symbols and timeframes
    """
    
    def __init__(
        self,
        supabase_url: Optional[str] = None,
        supabase_key: Optional[str] = None,
        symbols: List[str] = None,
        timeframe: str = "5m",  # Match database format
        max_steps: int = 512,
        initial_balance: float = 100000.0
    ):
        # Connect to Supabase
        self.supabase_url = supabase_url or os.getenv("SUPABASE_URL")
        self.supabase_key = supabase_key or os.getenv("SUPABASE_SERVICE_ROLE_KEY")
        
        if not self.supabase_url or not self.supabase_key:
            raise ValueError("SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY must be set")
        
        self.supabase: Client = create_client(self.supabase_url, self.supabase_key)
        
        # Environment parameters
        self.symbols = symbols or ["AAPL", "MSFT", "GOOGL", "TSLA"]
        self.timeframe = timeframe if timeframe else "5m"  # Match database format
        self.max_steps = max_steps
        self.initial_balance = initial_balance
        
        # State tracking
        self.current_step = 0
        self.current_trajectory_idx = 0
        self.trajectories: List[Dict] = []
        self.current_position = 0.0  # -1 (short), 0 (flat), 1 (long)
        self.entry_price = 0.0
        self.balance = initial_balance
        self.equity_history = [initial_balance]
        
        # Load trajectories on initialization
        self._load_trajectories()
        
        # Define observation space dimensions
        self.state_dim = 50  # Match config
        self.action_space_dim = 3  # position_size (-1 to 1), stop_loss, take_profit
        
    def _load_trajectories(self):
        """Load expert trajectories from Supabase"""
        print(f"📊 Loading trajectories from Supabase...")
        
        try:
            # Query trajectories ordered by timestamp
            response = self.supabase.table("expert_trajectories") \
                .select("*") \
                .in_("symbol", self.symbols) \
                .eq("timeframe", self.timeframe) \
                .order("timestamp") \
                .limit(10000) \
                .execute()
            
            if response.data:
                self.trajectories = response.data
                print(f"✅ Loaded {len(self.trajectories)} trajectories")
                print(f"   Symbols: {set(t['symbol'] for t in self.trajectories)}")
                print(f"   Timeframe: {self.timeframe}")
            else:
                print("⚠️  No trajectories found - using synthetic data")
                self._generate_synthetic_trajectories()
                
        except Exception as e:
            print(f"⚠️  Error loading trajectories: {e}")
            print("   Falling back to synthetic data")
            self._generate_synthetic_trajectories()
    
    def _generate_synthetic_trajectories(self):
        """Generate synthetic trajectories if database is empty"""
        print("🔧 Generating synthetic training data...")
        
        num_samples = 1000
        for i in range(num_samples):
            # Create realistic-looking market features
            price = 100 + np.random.randn() * 10
            features = {
                "close": price,
                "open": price + np.random.randn() * 0.5,
                "high": price + abs(np.random.randn()) * 1.5,
                "low": price - abs(np.random.randn()) * 1.5,
                "volume": int(1000000 + np.random.randn() * 500000),
                "rsi_14": 30 + np.random.rand() * 40,  # 30-70 range
                "atr_14": abs(np.random.randn()) * 2,
                "ema_20": price + np.random.randn() * 2,
                "ema_50": price + np.random.randn() * 5,
                "vwap": price + np.random.randn() * 1,
            }
            
            # Action: -1 (sell), 0 (hold), 1 (buy)
            action = np.random.choice([-1, 0, 1])
            
            # Calculate reward based on next price movement
            next_price_change = np.random.randn() * 0.02  # ±2% typical
            if action == 1:  # Buy
                reward = next_price_change * 100  # Profit on long
            elif action == -1:  # Sell
                reward = -next_price_change * 100  # Profit on short
            else:  # Hold
                reward = -0.1  # Small holding cost
            
            self.trajectories.append({
                "symbol": np.random.choice(self.symbols),
                "timeframe": self.timeframe,
                "timestamp": datetime.now().isoformat(),
                "obs_features": features,
                "action": action,
                "reward": reward,
                "delta_equity": reward,
                "fees": -0.5,  # Typical commission
                "slippage": -0.1
            })
        
        print(f"✅ Generated {num_samples} synthetic trajectories")
    
    def reset(self) -> np.ndarray:
        """Reset environment to initial state"""
        self.current_step = 0
        self.current_position = 0.0
        self.entry_price = 0.0
        self.balance = self.initial_balance
        self.equity_history = [self.initial_balance]
        
        # Randomly sample a starting trajectory
        if len(self.trajectories) > 0:
            self.current_trajectory_idx = np.random.randint(0, len(self.trajectories))
        
        return self._get_observation()
    
    def _get_observation(self) -> np.ndarray:
        """Get current observation/state"""
        if not self.trajectories or self.current_trajectory_idx >= len(self.trajectories):
            # Return random state if no data
            return np.random.randn(self.state_dim).astype(np.float32)
        
        # Get current trajectory
        traj = self.trajectories[self.current_trajectory_idx]
        features = traj.get("obs_features", {})
        
        # Build state vector from features
        state = np.zeros(self.state_dim, dtype=np.float32)
        
        # Price features (normalized)
        state[0] = features.get("close", 100) / 100.0
        state[1] = features.get("open", 100) / 100.0
        state[2] = features.get("high", 100) / 100.0
        state[3] = features.get("low", 100) / 100.0
        state[4] = features.get("volume", 1000000) / 1000000.0
        
        # Technical indicators
        state[5] = features.get("rsi_14", 50) / 100.0
        state[6] = features.get("atr_14", 1) / 10.0
        state[7] = features.get("ema_20", 100) / 100.0
        state[8] = features.get("ema_50", 100) / 100.0
        state[9] = features.get("vwap", 100) / 100.0
        
        # Position information
        state[10] = self.current_position
        state[11] = (self.balance - self.initial_balance) / self.initial_balance
        state[12] = self.current_step / self.max_steps
        
        # Fill remaining with expert features or zeros
        # This allows the model to learn from expert demonstrations
        if len(state) > 13:
            state[13:] = np.random.randn(len(state) - 13) * 0.1
        
        return state.astype(np.float32)
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        Execute one step in the environment
        
        Args:
            action: [position_size, stop_loss, take_profit]
                    position_size: -1 (full short) to 1 (full long)
                    stop_loss: 0 to 1 (percentage)
                    take_profit: 0 to 1 (percentage)
        
        Returns:
            next_state, reward, done, info
        """
        # Parse action
        position_size = np.clip(action[0] if len(action) > 0 else 0, -1, 1)
        
        # Get current trajectory data
        if self.current_trajectory_idx >= len(self.trajectories):
            # Episode ended
            return self._get_observation(), 0.0, True, {}
        
        traj = self.trajectories[self.current_trajectory_idx]
        
        # Calculate reward based on position and price movement
        if "reward" in traj:
            # Use expert trajectory reward
            base_reward = float(traj["reward"])
        else:
            # Calculate reward from price change
            price_change_pct = np.random.randn() * 0.02  # Simulated ±2%
            base_reward = position_size * price_change_pct * 100
        
        # Add penalties
        commission = -0.5 if abs(position_size) > 0.1 else 0
        slippage = -abs(position_size) * 0.1
        
        reward = base_reward + commission + slippage
        
        # Update balance
        self.balance += reward
        self.equity_history.append(self.balance)
        
        # Update state
        self.current_position = position_size
        self.current_step += 1
        self.current_trajectory_idx += 1
        
        # Check if episode is done
        done = (
            self.current_step >= self.max_steps or
            self.current_trajectory_idx >= len(self.trajectories) or
            self.balance <= self.initial_balance * 0.5  # Stop if 50% drawdown
        )
        
        # Get next observation
        next_state = self._get_observation()
        
        # Info dict
        info = {
            "balance": self.balance,
            "position": self.current_position,
            "step": self.current_step,
            "pnl": self.balance - self.initial_balance,
            "return_pct": (self.balance - self.initial_balance) / self.initial_balance * 100
        }
        
        return next_state, reward, done, info
    
    def get_episode_stats(self) -> Dict:
        """Get statistics for completed episode"""
        equity_array = np.array(self.equity_history)
        returns = np.diff(equity_array) / equity_array[:-1]
        
        total_return = (self.balance - self.initial_balance) / self.initial_balance * 100
        
        # Calculate Sharpe ratio (simplified)
        if len(returns) > 1 and np.std(returns) > 0:
            sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252)
        else:
            sharpe = 0
        
        # Calculate max drawdown
        cummax = np.maximum.accumulate(equity_array)
        drawdown = (equity_array - cummax) / cummax * 100
        max_drawdown = np.min(drawdown)
        
        return {
            "total_return_pct": total_return,
            "final_balance": self.balance,
            "sharpe_ratio": sharpe,
            "max_drawdown_pct": max_drawdown,
            "num_steps": self.current_step,
            "trajectories_used": self.current_trajectory_idx
        }


def create_trading_env() -> TradingEnvironment:
    """Factory function to create trading environment"""
    return TradingEnvironment()


if __name__ == "__main__":
    # Test the environment
    print("Testing Trading Environment...")
    env = create_trading_env()
    
    state = env.reset()
    print(f"\nInitial state shape: {state.shape}")
    print(f"State dim: {env.state_dim}")
    print(f"Action dim: {env.action_space_dim}")
    print(f"Trajectories loaded: {len(env.trajectories)}")
    
    # Run a few steps
    total_reward = 0
    for i in range(10):
        action = np.random.randn(3)  # Random action
        next_state, reward, done, info = env.step(action)
        total_reward += reward
        print(f"Step {i+1}: reward={reward:.2f}, balance=${info['balance']:.2f}, done={done}")
        
        if done:
            break
    
    stats = env.get_episode_stats()
    print(f"\nEpisode Stats:")
    print(f"  Total Return: {stats['total_return_pct']:.2f}%")
    print(f"  Sharpe Ratio: {stats['sharpe_ratio']:.2f}")
    print(f"  Max Drawdown: {stats['max_drawdown_pct']:.2f}%")
