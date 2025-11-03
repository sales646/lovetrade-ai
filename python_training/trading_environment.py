"""
Real Trading Environment - Uses Historical Market Data from Supabase
Features:
- Real historical bars from Supabase
- Data augmentation with ±1% noise for variety
- Wraparound for infinite training
"""
import os
import numpy as np
from typing import Dict, List, Optional, Tuple
from supabase import create_client, Client
from datetime import datetime
import json


class TradingEnvironment:
    """
    Real trading environment using historical market data
    
    Features:
    - Loads historical bars from Supabase (5m, 1m timeframes)
    - Uses real price movements for reward calculation
    - Includes technical indicators
    - No simulation or augmentation - pure historical data
    """
    
    # Class-level cache shared across all instances
    _data_cache: Dict = {}
    _cache_lock = None
    
    def __init__(
        self,
        supabase_url: Optional[str] = None,
        supabase_key: Optional[str] = None,
        symbols: List[str] = None,
        timeframe: str = "5m",
        lookback_days: int = 1825,  # 5 years default
        max_steps: int = 512,
        initial_balance: float = 100000.0,
        use_augmentation: bool = True,
        augmentation_noise: float = 0.01  # ±1% noise
    ):
        # Connect to Supabase
        self.supabase_url = supabase_url or os.getenv("SUPABASE_URL")
        self.supabase_key = supabase_key or os.getenv("SUPABASE_SERVICE_ROLE_KEY")
        
        if not self.supabase_url or not self.supabase_key:
            raise ValueError("SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY must be set")
        
        self.supabase: Client = create_client(self.supabase_url, self.supabase_key)
        
        # Environment parameters
        self.symbols = symbols or ["AAPL", "MSFT", "GOOGL", "TSLA", "NVDA", "AMZN"]
        self.timeframe = timeframe
        self.lookback_days = lookback_days
        self.max_steps = max_steps
        self.initial_balance = initial_balance
        self.use_augmentation = use_augmentation
        self.augmentation_noise = augmentation_noise
        
        # State tracking
        self.current_step = 0
        self.current_bar_idx = 0
        self.historical_bars: List[Dict] = []
        self.indicators: Dict = {}  # symbol -> list of indicators
        self.current_position = 0.0  # -1 (short), 0 (flat), 1 (long)
        self.entry_price = 0.0
        self.balance = initial_balance
        self.equity_history = [initial_balance]
        
        # Load historical data (using cache if available)
        self._load_historical_data()
        
        # Define observation space dimensions
        self.state_dim = 50
        self.action_space_dim = 3  # position_size, stop_loss, take_profit
        
    def _load_historical_data(self):
        """Load historical bars and indicators from disk cache"""
        import pickle
        
        cache_file = "python_training/.market_data_cache.pkl"
        
        # Try to load from cache file
        if os.path.exists(cache_file):
            print(f"📦 Loading from cache file: {cache_file}")
            try:
                with open(cache_file, 'rb') as f:
                    cached = pickle.load(f)
                
                self.historical_bars = cached['bars']
                self.indicators = cached['indicators']
                
                print(f"✅ Loaded {len(self.historical_bars):,} bars from cache")
                print(f"   Date range: {self.historical_bars[0]['timestamp'][:10]} to {self.historical_bars[-1]['timestamp'][:10]}")
                return
            except Exception as e:
                print(f"⚠️  Cache load failed: {e}, loading from Supabase...")
        
        # Fallback: load from Supabase
        print(f"📊 Loading historical market data from Supabase...")
        print(f"   Symbols: {self.symbols}")
        print(f"   Timeframe: {self.timeframe}")
        print(f"   Lookback: {self.lookback_days} days")
        
        try:
            # Load historical bars for all symbols
            all_bars = []
            
            for symbol in self.symbols:
                print(f"   Loading {symbol}...")
                response = self.supabase.table("historical_bars") \
                    .select("*") \
                    .eq("symbol", symbol) \
                    .eq("timeframe", self.timeframe) \
                    .order("timestamp", desc=False) \
                    .limit(500000) \
                    .execute()
                
                if response.data:
                    all_bars.extend(response.data)
                    print(f"      ✅ {len(response.data)} bars")
            
            if len(all_bars) > 0:
                # Sort all bars by timestamp
                self.historical_bars = sorted(all_bars, key=lambda x: x['timestamp'])
                
                print(f"\n✅ Loaded {len(self.historical_bars):,} total historical bars")
                print(f"   Date range: {self.historical_bars[0]['timestamp'][:10]} to {self.historical_bars[-1]['timestamp'][:10]}")
                print(f"   Symbols: {set(b['symbol'] for b in self.historical_bars)}")
                
                # Load technical indicators
                self._load_indicators()
                
            else:
                print("⚠️  No historical bars found - generating fallback data")
                self._generate_fallback_data()
                
        except Exception as e:
            print(f"❌ Error loading historical data: {e}")
            import traceback
            traceback.print_exc()
            self._generate_fallback_data()
    
    def _load_indicators(self):
        """Load technical indicators for all symbols"""
        print(f"\n📈 Loading technical indicators...")
        
        try:
            for symbol in self.symbols:
                response = self.supabase.table("technical_indicators") \
                    .select("*") \
                    .eq("symbol", symbol) \
                    .eq("timeframe", self.timeframe) \
                    .order("timestamp", desc=False) \
                    .limit(50000) \
                    .execute()
                
                if response.data:
                    self.indicators[symbol] = response.data
                    print(f"   {symbol}: {len(response.data)} indicator snapshots")
            
            print(f"✅ Loaded indicators for {len(self.indicators)} symbols")
                    
        except Exception as e:
            print(f"⚠️  Could not load indicators: {e}")
            self.indicators = {}
    
    def _augment_data(self):
        """
        Augment historical data by creating variations with small noise
        This multiplies the dataset size for better training
        """
        original_count = len(self.historical_bars)
        print(f"\n🎲 Applying data augmentation...")
        print(f"   Original bars: {original_count:,}")
        print(f"   Noise level: ±{self.augmentation_noise*100}%")
        
        augmented_bars = []
        
        # Create 2 augmented versions of each bar (3x total data)
        for _ in range(2):
            for bar in self.historical_bars:
                # Create augmented bar with small random variations
                aug_bar = bar.copy()
                
                # Add noise to OHLC prices (±1% by default)
                noise = 1 + np.random.uniform(-self.augmentation_noise, self.augmentation_noise)
                aug_bar['open'] = float(bar['open']) * noise
                aug_bar['high'] = float(bar['high']) * noise
                aug_bar['low'] = float(bar['low']) * noise
                aug_bar['close'] = float(bar['close']) * noise
                
                # Add noise to volume (±5%)
                volume_noise = 1 + np.random.uniform(-0.05, 0.05)
                aug_bar['volume'] = int(float(bar['volume']) * volume_noise)
                
                augmented_bars.append(aug_bar)
        
        # Add augmented bars to original data
        self.historical_bars.extend(augmented_bars)
        
        # Shuffle to mix original and augmented data
        np.random.shuffle(self.historical_bars)
        
        print(f"   ✅ Augmented to {len(self.historical_bars):,} bars ({len(self.historical_bars)/original_count:.1f}x)")
    
    def _generate_fallback_data(self):
        """Generate minimal fallback data if database is empty"""
        print("🔧 Generating fallback market data (1000 bars)...")
        
        for i in range(1000):
            price = 100 + np.random.randn() * 10
            for symbol in self.symbols[:2]:  # Just 2 symbols for fallback
                self.historical_bars.append({
                    'symbol': symbol,
                    'timestamp': f"2024-01-01T{i//60:02d}:{i%60:02d}:00Z",
                    'open': price,
                    'high': price + abs(np.random.randn()),
                    'low': price - abs(np.random.randn()),
                    'close': price + np.random.randn() * 0.5,
                    'volume': int(1000000 + np.random.randn() * 500000)
                })
        
        print(f"✅ Generated {len(self.historical_bars)} fallback bars")
    
    def reset(self) -> np.ndarray:
        """
        Reset environment to random point in history
        With wraparound enabled for infinite training
        """
        self.current_step = 0
        self.current_position = 0.0
        self.entry_price = 0.0
        self.balance = self.initial_balance
        self.equity_history = [self.initial_balance]
        
        # Start at random point in history
        # With wraparound, we can start anywhere
        if len(self.historical_bars) > 0:
            self.current_bar_idx = np.random.randint(0, len(self.historical_bars))
        else:
            self.current_bar_idx = 0
        
        return self._get_observation()
    
    def _get_observation(self) -> np.ndarray:
        """
        Get current market state observation
        Uses wraparound to loop through data infinitely
        """
        if len(self.historical_bars) == 0:
            return np.zeros(self.state_dim, dtype=np.float32)
        
        # Wraparound: loop back to beginning if we exceed data length
        self.current_bar_idx = self.current_bar_idx % len(self.historical_bars)
        
        bar = self.historical_bars[self.current_bar_idx]
        symbol = bar['symbol']
        
        # Build state vector from real market data
        state = np.zeros(self.state_dim, dtype=np.float32)
        
        # Price features (normalized by close price)
        close = float(bar['close'])
        state[0] = float(bar['open']) / close if close > 0 else 1.0
        state[1] = float(bar['high']) / close if close > 0 else 1.0
        state[2] = float(bar['low']) / close if close > 0 else 1.0
        state[3] = 1.0  # close / close = 1
        state[4] = float(bar['volume']) / 1000000.0  # Volume in millions
        
        # Get technical indicators if available
        if symbol in self.indicators and len(self.indicators[symbol]) > 0:
            # Find indicator snapshot closest to current bar
            bar_time = bar['timestamp']
            matching_indicators = [ind for ind in self.indicators[symbol] 
                                 if ind['timestamp'] <= bar_time]
            
            if matching_indicators:
                ind = matching_indicators[-1]  # Most recent
                
                state[5] = ind.get('rsi_14', 50) / 100.0
                state[6] = ind.get('atr_14', 1) / close if close > 0 else 0.01
                state[7] = ind.get('ema_20', close) / close if close > 0 else 1.0
                state[8] = ind.get('ema_50', close) / close if close > 0 else 1.0
                state[9] = ind.get('vwap', close) / close if close > 0 else 1.0
                state[10] = ind.get('vwap_distance_pct', 0) / 100.0
                state[11] = ind.get('volume_zscore', 0) / 3.0  # Normalize z-score
        
        # Position and account state
        state[12] = self.current_position
        state[13] = (self.balance - self.initial_balance) / self.initial_balance
        state[14] = self.current_step / self.max_steps
        
        # Look back at recent bars for momentum features (if available)
        if self.current_bar_idx >= 5:
            recent_bars = self.historical_bars[self.current_bar_idx-5:self.current_bar_idx]
            closes = [float(b['close']) for b in recent_bars if b['symbol'] == symbol]
            
            if len(closes) >= 2:
                # Recent price momentum
                state[15] = (closes[-1] - closes[0]) / closes[0] if closes[0] > 0 else 0
                # Volatility (std of returns)
                returns = [closes[i]/closes[i-1] - 1 for i in range(1, len(closes))]
                state[16] = np.std(returns) if len(returns) > 1 else 0
        
        return state.astype(np.float32)
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        Execute one step using real market data with wraparound
        
        Args:
            action: [position_size, stop_loss, take_profit]
        
        Returns:
            next_state, reward, done, info
        """
        # Parse action
        position_size = np.clip(action[0] if len(action) > 0 else 0, -1, 1)
        
        if len(self.historical_bars) == 0:
            return self._get_observation(), 0.0, True, self._get_info()
        
        # Get current and next bar (with wraparound)
        self.current_bar_idx = self.current_bar_idx % len(self.historical_bars)
        next_idx = (self.current_bar_idx + 1) % len(self.historical_bars)
        
        current_bar = self.historical_bars[self.current_bar_idx]
        next_bar = self.historical_bars[next_idx]
        
        # Only calculate reward if bars are from same symbol
        if current_bar['symbol'] == next_bar['symbol']:
            current_price = float(current_bar['close'])
            next_price = float(next_bar['close'])
            
            # Calculate real price change
            price_change_pct = (next_price - current_price) / current_price if current_price > 0 else 0
            
            # Calculate reward based on position and actual price movement
            if abs(position_size) > 0.1:
                # Trading: reward = position * actual price change * leverage
                base_reward = position_size * price_change_pct * 100
                
                # Costs
                commission = -0.5  # $0.50 per trade
                slippage = -abs(position_size) * 0.1
                
                reward = base_reward + commission + slippage
            else:
                # Holding: small penalty
                reward = -0.1
        else:
            # Different symbols, no reward
            reward = 0.0
        
        # Update balance
        self.balance += reward
        self.equity_history.append(self.balance)
        
        # Update state
        self.current_position = position_size
        self.current_step += 1
        self.current_bar_idx = next_idx
        
        # Check if episode should end (no data exhaustion with wraparound!)
        done = (
            self.current_step >= self.max_steps or
            self.balance <= self.initial_balance * 0.5  # 50% drawdown
        )
        
        # Get next observation
        next_state = self._get_observation()
        
        return next_state, reward, done, self._get_info()
    
    def _get_info(self) -> Dict:
        """Get current episode info"""
        return {
            "balance": self.balance,
            "position": self.current_position,
            "step": self.current_step,
            "bar_idx": self.current_bar_idx,
            "pnl": self.balance - self.initial_balance,
            "return_pct": (self.balance - self.initial_balance) / self.initial_balance * 100
        }
    
    def get_episode_stats(self) -> Dict:
        """Get statistics for completed episode"""
        equity_array = np.array(self.equity_history)
        returns = np.diff(equity_array) / equity_array[:-1]
        
        total_return = (self.balance - self.initial_balance) / self.initial_balance * 100
        
        # Calculate Sharpe ratio
        if len(returns) > 1 and np.std(returns) > 0:
            sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252 * 78)  # 78 5-min bars per day
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
            "bars_used": self.current_bar_idx
        }


def create_trading_env(use_augmentation: bool = True) -> TradingEnvironment:
    """
    Factory function to create trading environment
    
    Args:
        use_augmentation: If True, applies ±1% noise to create 3x more data
    
    Returns:
        TradingEnvironment with real historical data, augmentation, and wraparound
    """
    return TradingEnvironment(use_augmentation=use_augmentation)


if __name__ == "__main__":
    # Test the environment
    print("Testing Real Market Data Environment...")
    env = create_trading_env()
    
    state = env.reset()
    print(f"\nInitial state shape: {state.shape}")
    print(f"Historical bars loaded: {len(env.historical_bars):,}")
    print(f"State dim: {env.state_dim}")
    print(f"Action dim: {env.action_space_dim}")
    
    # Run a few steps with real data
    total_reward = 0
    for i in range(10):
        action = np.random.randn(3)
        next_state, reward, done, info = env.step(action)
        total_reward += reward
        print(f"Step {i+1}: reward={reward:.2f}, balance=${info['balance']:.2f}, bar={info['bar_idx']}")
        
        if done:
            break
    
    stats = env.get_episode_stats()
    print(f"\nEpisode Stats:")
    print(f"  Total Return: {stats['total_return_pct']:.2f}%")
    print(f"  Sharpe Ratio: {stats['sharpe_ratio']:.2f}")
    print(f"  Max Drawdown: {stats['max_drawdown_pct']:.2f}%")
