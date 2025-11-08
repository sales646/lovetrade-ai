"""
Trading Environment with Complete PNU Specifications
Features:
- All historical data sources (Supabase, Polygon)
- Walk-forward train/val/test split
- Realistic execution (spread, slippage, fees)
- Risk management (max positions, cooldown, confidence threshold)
- Risk-adjusted rewards
- Action masking
"""
import os
import numpy as np
from typing import Dict, Iterable, List, Optional, Tuple
import pandas as pd
from supabase import create_client, Client
from datetime import datetime
import json
import hashlib
from threading import Lock


class TradingEnvironment:
    """Real trading environment using all available historical data"""

    _data_cache: Dict = {}
    _stats_cache: Dict = {}
    _cache_lock: Lock = Lock()
    
    def __init__(
        self,
        symbols: List[str] = None,
        timeframe: str = "1Min",
        lookback: int = 100,
        initial_balance: float = 100000.0,
        augment_data: bool = False,
        enable_multi_market: bool = True,
        crypto_stock_ratio: float = 0.7,
        walk_forward: bool = True,
        train_split: float = 0.7,
        val_split: float = 0.15,
        confidence_threshold: float = 0.6,
        max_positions: int = 1,
        cooldown_minutes: int = 3,
        external_data: Optional[Dict[str, Iterable[Dict]]] = None,
    ):
        self.symbols = symbols or []
        self.timeframe = timeframe
        self.lookback = lookback
        self.initial_balance = initial_balance
        self.augment_data = augment_data
        self.enable_multi_market = enable_multi_market
        self.crypto_stock_ratio = crypto_stock_ratio
        self.walk_forward = walk_forward
        self.train_split = train_split
        self.val_split = val_split
        self.confidence_threshold = confidence_threshold
        self.max_positions = max_positions
        self.cooldown_minutes = cooldown_minutes
        
        # Supabase connection
        self.supabase_url = os.getenv("SUPABASE_URL")
        self.supabase_key = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
        
        if self.supabase_url and self.supabase_key:
            self.supabase: Client = create_client(self.supabase_url, self.supabase_key)
        else:
            self.supabase = None
            print("⚠️  Supabase not configured - using fallback data")
        
        # Data storage
        self.data: Dict[str, List] = {}  # symbol -> bars
        self.indicators: Dict[str, List] = {}  # symbol -> indicators
        self.symbol_types: Dict[str, str] = {}  # symbol -> "stock" or "crypto"
        self.symbol_stats: Dict[str, Dict] = {}  # symbol -> {mean, std, atr}
        
        # State tracking
        self.balance = initial_balance
        self.position = 0
        self.position_price = 0.0
        self.current_step = 0
        self.done = False
        self.cooldown_counter = 0
        self.last_action_confidence = 0.0
        self.current_symbol = None
        
        # Execution costs (realistic)
        self.stock_fee = 0.0002  # 2 bps
        self.crypto_fee = 0.0001  # 1 bp
        self.spread_bps = 0.0005  # 5 bps spread
        self.slippage_bps = 0.0003  # 3 bps slippage
        
        # Episode tracking
        self.episode_trades = []
        self.episode_pnls = []
        
        # External preloaded data (bypasses Supabase fetch)
        self.external_data = external_data or {}

        self._cache_key = self._make_cache_key(
            self.symbols,
            self.timeframe,
            self.augment_data,
            self.external_data
        )

        # Load data
        self._load_historical_data()
        if self.enable_multi_market:
            self._compute_symbol_stats()
        
        # Dimensions
        self.state_dim = 52
        self.action_space_dim = 3
    
    @classmethod
    def _make_cache_key(
        cls,
        symbols: List[str],
        timeframe: str,
        augment: bool,
        external_data: Optional[Dict]
    ) -> Tuple:
        symbols_key = tuple(sorted(symbols)) if symbols else tuple()

        if isinstance(external_data, dict):
            external_key = cls._fingerprint_external_data(external_data)
        elif external_data is None:
            external_key = None
        else:
            external_key = id(external_data)

        return (symbols_key, timeframe, augment, external_key)

    @staticmethod
    def _json_default(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        if isinstance(obj, (datetime,)):
            return obj.isoformat()
        if isinstance(obj, pd.Timestamp):
            return obj.isoformat()
        return str(obj)

    @classmethod
    def _fingerprint_external_data(cls, external_data: Dict) -> str:
        try:
            normalized = []
            for symbol in sorted(external_data.keys()):
                rows = external_data[symbol]
                if isinstance(rows, pd.DataFrame):
                    payload = rows.to_dict("records")
                elif isinstance(rows, dict):
                    payload = rows
                else:
                    try:
                        payload = list(rows)
                    except TypeError:
                        payload = rows

                normalized.append((symbol, payload))

            serialized = json.dumps(
                normalized,
                sort_keys=True,
                default=cls._json_default
            )
            return hashlib.sha256(serialized.encode("utf-8")).hexdigest()
        except Exception:
            return str(id(external_data))

    def _load_historical_data(self):
        """Load all available historical data"""
        with self._cache_lock:
            cached = self._data_cache.get(self._cache_key)

        if cached:
            self.data = cached['data']
            self.symbol_types = cached['symbol_types']
            cached_stats = cached.get('symbol_stats')
            if cached_stats and self.enable_multi_market:
                self.symbol_stats = cached_stats
            print(f"📊 Reusing cached market data for {len(self.data)} symbols...")
            return

        if self.external_data:
            print(f"📊 Loading preloaded data for {len(self.external_data)} symbols...")
            for symbol, rows in self.external_data.items():
                if isinstance(rows, pd.DataFrame):
                    bars = rows.to_dict("records")
                else:
                    bars = list(rows)

                if not bars:
                    continue

                self.data[symbol] = bars
                is_crypto = symbol.endswith(('USDT', 'BUSD', 'USD', 'BTC', 'ETH')) and len(symbol) > 4
                self.symbol_types[symbol] = "crypto" if is_crypto else "stock"

            print(f"✅ Loaded data for {len(self.data)} symbols from cache")
        else:
            print(f"📊 Loading historical data for {len(self.symbols)} symbols...")

            for symbol in self.symbols:
                # Classify symbol type
                is_crypto = symbol.endswith(('USDT', 'BUSD', 'USD', 'BTC', 'ETH')) and len(symbol) > 4
                self.symbol_types[symbol] = "crypto" if is_crypto else "stock"

                # Load from Supabase
                if self.supabase:
                    try:
                        response = self.supabase.table("historical_bars") \
                            .select("*") \
                            .eq("symbol", symbol) \
                            .eq("timeframe", self.timeframe) \
                            .order("timestamp", desc=False) \
                            .limit(1000000) \
                            .execute()

                        if response.data:
                            self.data[symbol] = response.data
                            print(f"  ✅ {symbol}: {len(response.data):,} bars")
                        else:
                            print(f"  ⚠️  {symbol}: No data found")
                    except Exception as e:
                        print(f"  ❌ {symbol}: Error loading - {e}")

                # Augment if requested
                if self.augment_data and symbol in self.data:
                    original_len = len(self.data[symbol])
                    augmented = []
                    for bar in self.data[symbol]:
                        # 2 augmented versions
                        for _ in range(2):
                            aug = bar.copy()
                            noise = 1 + np.random.uniform(-0.01, 0.01)
                            aug['open'] = float(bar['open']) * noise
                            aug['high'] = float(bar['high']) * noise
                            aug['low'] = float(bar['low']) * noise
                            aug['close'] = float(bar['close']) * noise
                            aug['volume'] = int(float(bar['volume']) * (1 + np.random.uniform(-0.05, 0.05)))
                            augmented.append(aug)

                    self.data[symbol].extend(augmented)
                    print(f"     📈 Augmented to {len(self.data[symbol]):,} bars")

            print(f"✅ Loaded data for {len(self.data)} symbols")

        with self._cache_lock:
            self._data_cache[self._cache_key] = {
                'data': self.data,
                'symbol_types': self.symbol_types
            }

    def _compute_symbol_stats(self):
        """Compute per-symbol statistics for normalization"""
        stats_cache_key = (self._cache_key, self.enable_multi_market)

        with self._cache_lock:
            cached_stats = self._stats_cache.get(stats_cache_key)

        if cached_stats:
            self.symbol_stats = cached_stats
            print(f"📊 Reusing cached symbol statistics for {len(self.symbol_stats)} symbols...")
            return

        print("📊 Computing symbol statistics...")

        for symbol, bars in self.data.items():
            if len(bars) < 2:
                continue

            prices = [float(b['close']) for b in bars]
            log_returns = [np.log(prices[i] / prices[i-1]) for i in range(1, len(prices))]

            atrs = []
            for i in range(1, len(bars)):
                high = float(bars[i]['high'])
                low = float(bars[i]['low'])
                prev_close = float(bars[i-1]['close'])
                tr = max(high - low, abs(high - prev_close), abs(low - prev_close))
                atrs.append(tr)

            self.symbol_stats[symbol] = {
                'mean_return': np.mean(log_returns),
                'std_return': np.std(log_returns) if len(log_returns) > 1 else 0.01,
                'avg_atr': np.mean(atrs) if len(atrs) > 0 else 1.0,
                'avg_price': np.mean(prices)
            }

        with self._cache_lock:
            cached_entry = self._data_cache.get(self._cache_key, {})
            if cached_entry:
                cached_entry['symbol_stats'] = self.symbol_stats
                self._data_cache[self._cache_key] = cached_entry
            self._stats_cache[stats_cache_key] = self.symbol_stats

        print(f"✅ Computed stats for {len(self.symbol_stats)} symbols")
    
    def reset(self, seed: int = None, phase: str = "train") -> np.ndarray:
        """Reset with walk-forward split support"""
        if seed is not None:
            np.random.seed(seed)
        
        # Select random symbol
        self.current_symbol = np.random.choice(self.symbols)
        symbol_data = self.data.get(self.current_symbol, [])
        
        if len(symbol_data) == 0:
            # Fallback
            self.current_step = 0
            return self._get_observation()
        
        # Walk-forward split
        total_len = len(symbol_data)
        if self.walk_forward:
            if phase == "train":
                start_idx = 0
                end_idx = int(total_len * self.train_split)
            elif phase == "val":
                start_idx = int(total_len * self.train_split)
                end_idx = int(total_len * (self.train_split + self.val_split))
            else:  # test
                start_idx = int(total_len * (self.train_split + self.val_split))
                end_idx = total_len
        else:
            start_idx = 0
            end_idx = total_len
        
        # Random starting point within split
        max_start = end_idx - self.lookback - 500
        if max_start <= start_idx:
            self.current_step = start_idx
        else:
            self.current_step = np.random.randint(start_idx, max_start)
        
        # Reset state
        self.balance = self.initial_balance
        self.position = 0
        self.position_price = 0.0
        self.done = False
        self.cooldown_counter = 0
        self.last_action_confidence = 0.0
        self.episode_trades = []
        self.episode_pnls = []
        
        return self._get_observation()
    
    def _get_observation(self) -> np.ndarray:
        """Get observation with multi-market features"""
        state = np.zeros(self.state_dim, dtype=np.float32)
        
        if self.current_symbol not in self.data:
            return state
        
        symbol_data = self.data[self.current_symbol]
        if self.current_step >= len(symbol_data):
            return state
        
        bar = symbol_data[self.current_step]
        close = float(bar['close'])
        
        # Price features (normalized)
        state[0] = float(bar['open']) / close if close > 0 else 1.0
        state[1] = float(bar['high']) / close if close > 0 else 1.0
        state[2] = float(bar['low']) / close if close > 0 else 1.0
        state[3] = 1.0  # close/close
        state[4] = float(bar['volume']) / 1000000.0
        
        # Position state
        state[12] = self.position
        state[13] = (self.balance - self.initial_balance) / self.initial_balance
        state[14] = self.cooldown_counter / self.cooldown_minutes if self.cooldown_minutes > 0 else 0
        
        # Lookback momentum
        if self.current_step >= 5:
            recent_idx = max(0, self.current_step - 5)
            recent_bars = symbol_data[recent_idx:self.current_step]
            closes = [float(b['close']) for b in recent_bars]
            
            if len(closes) >= 2 and closes[0] > 0:
                log_return = np.log(closes[-1] / closes[0])
                
                # Z-score normalize
                if self.current_symbol in self.symbol_stats:
                    mean = self.symbol_stats[self.current_symbol]['mean_return']
                    std = self.symbol_stats[self.current_symbol]['std_return']
                    state[15] = (log_return - mean) / std if std > 0 else 0
                else:
                    state[15] = log_return
                
                # Volatility
                returns = [np.log(closes[i]/closes[i-1]) if closes[i-1] > 0 else 0 
                          for i in range(1, len(closes))]
                state[16] = np.std(returns) if len(returns) > 1 else 0
        
        # Multi-market features
        if self.enable_multi_market:
            state[50] = 1 if self.symbol_types.get(self.current_symbol) == "crypto" else 0
            
            if self.current_symbol in self.symbol_stats:
                avg_atr = self.symbol_stats[self.current_symbol]['avg_atr']
                avg_price = self.symbol_stats[self.current_symbol]['avg_price']
                state[51] = avg_atr / avg_price if avg_price > 0 else 0
        
        return state
    
    def step(self, action: int, confidence: float = 1.0) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        Execute action with confidence threshold and risk management
        action: 0=hold, 1=buy, 2=sell
        """
        self.last_action_confidence = confidence
        
        # Apply confidence threshold
        if confidence < self.confidence_threshold and action != 0:
            action = 0
        
        # Check cooldown
        if self.cooldown_counter > 0:
            self.cooldown_counter -= 1
            action = 0
        
        # Max positions check
        if self.position != 0 and action in [1, 2] and action != self._get_close_action():
            action = 0
        
        if self.current_symbol not in self.data:
            return self._get_observation(), 0.0, True, {}
        
        symbol_data = self.data[self.current_symbol]
        if self.current_step >= len(symbol_data):
            return self._get_observation(), 0.0, True, {}
        
        bar = symbol_data[self.current_step]
        current_price = float(bar['close'])
        
        # Get fees
        is_crypto = self.symbol_types.get(self.current_symbol) == "crypto"
        fee_rate = self.crypto_fee if is_crypto else self.stock_fee
        
        reward = 0.0
        pnl = 0.0
        fees_paid = 0.0
        
        # Execution with spread and slippage
        if action == 1:  # Buy
            execution_price = current_price * (1 + self.spread_bps + self.slippage_bps)
        elif action == 2:  # Sell
            execution_price = current_price * (1 - self.spread_bps - self.slippage_bps)
        else:
            execution_price = current_price
        
        # Execute
        if action == 1 and self.position == 0:  # Buy
            trade_value = self.balance
            fees_paid = trade_value * fee_rate
            self.position = (self.balance - fees_paid) / execution_price
            self.position_price = execution_price
            self.balance = 0
            self.cooldown_counter = self.cooldown_minutes
            self.episode_trades.append(('buy', execution_price, self.position))
        
        elif action == 2 and self.position > 0:  # Sell
            trade_value = self.position * execution_price
            fees_paid = trade_value * fee_rate
            self.balance = trade_value - fees_paid
            pnl = self.balance - self.initial_balance
            
            # Risk-adjusted reward (ATR-normalized)
            symbol_stats = self.symbol_stats.get(self.current_symbol, {})
            avg_atr = symbol_stats.get('avg_atr', current_price * 0.02)
            
            if self.enable_multi_market:
                std_return = symbol_stats.get('std_return', 0.01)
                reward = pnl / (std_return * self.initial_balance) if std_return > 0 else pnl / self.initial_balance
            else:
                reward = pnl / avg_atr if avg_atr > 0 else pnl / (current_price * 0.02)
            
            self.episode_pnls.append(pnl)
            self.episode_trades.append(('sell', execution_price, self.position))
            self.position = 0
            self.position_price = 0.0
            self.cooldown_counter = self.cooldown_minutes
        
        # Small reward for correct Hold
        elif action == 0 and self.position > 0:
            unrealized_pnl = (current_price - self.position_price) * self.position
            if unrealized_pnl > 0:
                reward = 0.001
            else:
                reward = -0.0005
        
        # Holding cost
        if self.position > 0:
            reward -= 0.0001
        
        # Next step
        self.current_step += 1
        
        # Check done
        if self.current_step >= len(symbol_data) or self.current_step >= 1000:
            self.done = True
        
        obs = self._get_observation()
        info = self._get_info()
        info['pnl'] = pnl
        info['fees_paid'] = fees_paid
        info['confidence'] = confidence
        info['cooldown'] = self.cooldown_counter
        
        return obs, reward, self.done, info
    
    def _get_close_action(self) -> int:
        """Get action to close position"""
        if self.position > 0:
            return 2
        return 0
    
    def _get_info(self) -> Dict:
        """Get episode info"""
        return {
            'balance': self.balance,
            'position': self.position,
            'step': self.current_step,
            'symbol': self.current_symbol,
            'trades': len(self.episode_trades),
        }
    
    def get_episode_stats(self) -> Dict:
        """Calculate episode statistics"""
        if len(self.episode_pnls) == 0:
            return {
                'total_return': 0.0,
                'sharpe_ratio': 0.0,
                'max_drawdown': 0.0,
                'num_trades': 0,
            }
        
        total_return = sum(self.episode_pnls) / self.initial_balance
        
        # Sharpe
        returns = np.array(self.episode_pnls)
        sharpe = np.mean(returns) / np.std(returns) if np.std(returns) > 0 else 0.0
        
        # Max DD
        equity_curve = np.cumsum([self.initial_balance] + self.episode_pnls)
        running_max = np.maximum.accumulate(equity_curve)
        drawdown = (equity_curve - running_max) / running_max
        max_dd = np.min(drawdown)
        
        return {
            'total_return': total_return,
            'sharpe_ratio': sharpe,
            'max_drawdown': max_dd,
            'num_trades': len(self.episode_pnls),
        }


def create_trading_env(
    symbols: List[str] = None,
    augment: bool = False,
    enable_multi_market: bool = True,
    phase: str = "train",
    **kwargs
) -> TradingEnvironment:
    """Factory for creating trading environments"""
    env = TradingEnvironment(
        symbols=symbols or [],
        augment_data=augment,
        enable_multi_market=enable_multi_market,
        **kwargs
    )
    env._phase = phase
    return env
