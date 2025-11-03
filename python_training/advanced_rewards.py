"""
Advanced Reward Shaping for Maximum Profit
Optimized for: High Sharpe ratio, leverage efficiency, drawdown control
"""
import numpy as np
from typing import Dict, Optional


class ProfitOptimizedRewardShaper:
    """
    Advanced reward shaping focused on maximizing profit with controlled risk
    
    Key features:
    - Risk-adjusted returns (Sharpe-based rewards)
    - Leverage utilization rewards
    - Drawdown penalties
    - Win streak bonuses
    - Market regime adaptation
    """
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        
        # Reward weights
        self.w_profit = self.config.get("w_profit", 1.0)
        self.w_sharpe = self.config.get("w_sharpe", 2.0)
        self.w_leverage = self.config.get("w_leverage", 0.5)
        self.w_drawdown = self.config.get("w_drawdown", -3.0)
        self.w_streak = self.config.get("w_streak", 0.3)
        self.w_timing = self.config.get("w_timing", 0.2)
        
        # State tracking
        self.trade_history = []
        self.equity_curve = []
        self.drawdown_history = []
        self.win_streak = 0
        self.loss_streak = 0
        
        # Risk-free rate for Sharpe calculation
        self.risk_free_rate = self.config.get("risk_free_rate", 0.02 / 252)  # Daily
        
    def compute_reward(
        self,
        action: str,
        profit: float,
        equity_before: float,
        equity_after: float,
        position_size: float,
        max_position_size: float,
        market_state: Dict,
        trade_info: Optional[Dict] = None
    ) -> float:
        """
        Compute comprehensive reward for a trading action
        
        Args:
            action: "buy", "sell", or "hold"
            profit: Realized profit from trade
            equity_before: Account equity before trade
            equity_after: Account equity after trade
            position_size: Actual position size used
            max_position_size: Maximum allowed position size
            market_state: Current market conditions
            trade_info: Additional trade information
        """
        reward = 0.0
        
        # 1. Base profit reward (risk-adjusted)
        if profit != 0:
            pct_return = profit / equity_before
            reward += self.w_profit * pct_return
            
            # Track trade
            self.trade_history.append({
                'profit': profit,
                'return': pct_return,
                'action': action
            })
            
            # Update win/loss streaks
            if profit > 0:
                self.win_streak += 1
                self.loss_streak = 0
            else:
                self.loss_streak += 1
                self.win_streak = 0
        
        # 2. Sharpe ratio component
        self.equity_curve.append(equity_after)
        if len(self.equity_curve) > 30:  # Need history for Sharpe
            sharpe_reward = self._compute_sharpe_reward()
            reward += self.w_sharpe * sharpe_reward
        
        # 3. Leverage utilization reward
        if action in ["buy", "sell"] and max_position_size > 0:
            leverage_ratio = position_size / max_position_size
            # Reward using 70-90% of available leverage (not too conservative, not max risk)
            if 0.7 <= leverage_ratio <= 0.9:
                reward += self.w_leverage * 0.5
            elif leverage_ratio < 0.5:
                reward += self.w_leverage * (-0.3)  # Penalty for underutilization
        
        # 4. Drawdown penalty
        drawdown = self._compute_drawdown(equity_after)
        self.drawdown_history.append(drawdown)
        
        if drawdown > 0.15:  # >15% drawdown
            reward += self.w_drawdown * (drawdown - 0.15)
        
        # 5. Win streak bonus
        if self.win_streak >= 3:
            reward += self.w_streak * min(self.win_streak / 10, 0.5)
        
        # Loss streak penalty (risk of overtrading)
        if self.loss_streak >= 3:
            reward += self.w_streak * (-0.3) * (self.loss_streak / 5)
        
        # 6. Timing reward (based on market conditions)
        timing_reward = self._compute_timing_reward(action, market_state)
        reward += self.w_timing * timing_reward
        
        # 7. Risk management bonus
        if trade_info:
            risk_reward = self._compute_risk_management_reward(trade_info)
            reward += risk_reward
        
        return reward
    
    def _compute_sharpe_reward(self, window: int = 30) -> float:
        """Compute rolling Sharpe ratio as reward"""
        if len(self.equity_curve) < window:
            return 0.0
        
        recent_equity = np.array(self.equity_curve[-window:])
        returns = np.diff(recent_equity) / recent_equity[:-1]
        
        if len(returns) < 2:
            return 0.0
        
        excess_returns = returns - self.risk_free_rate
        
        if np.std(returns) == 0:
            return 0.0
        
        sharpe = np.mean(excess_returns) / (np.std(returns) + 1e-8)
        
        # Normalize Sharpe to [-1, 1] range for reward
        # Sharpe > 2 is excellent, Sharpe < 0 is bad
        normalized_sharpe = np.tanh(sharpe / 2)
        
        return normalized_sharpe
    
    def _compute_drawdown(self, current_equity: float) -> float:
        """Compute current drawdown from peak equity"""
        if not self.equity_curve:
            return 0.0
        
        peak_equity = max(self.equity_curve)
        if peak_equity == 0:
            return 0.0
        
        drawdown = (peak_equity - current_equity) / peak_equity
        return max(0, drawdown)
    
    def _compute_timing_reward(self, action: str, market_state: Dict) -> float:
        """Reward good timing based on market conditions"""
        reward = 0.0
        
        rsi = market_state.get("rsi", 50)
        volatility = market_state.get("volatility", 0)
        volume_zscore = market_state.get("volume_zscore", 0)
        
        if action == "buy":
            # Reward buying in oversold conditions
            if rsi < 30:
                reward += 0.3
            # Reward buying with volume confirmation
            if volume_zscore > 1.5:
                reward += 0.2
        
        elif action == "sell":
            # Reward selling in overbought conditions
            if rsi > 70:
                reward += 0.3
            # Reward selling with volume confirmation
            if volume_zscore > 1.5:
                reward += 0.2
        
        # Penalize trading in high volatility (unless it's part of strategy)
        if action in ["buy", "sell"] and volatility > 0.03:
            reward -= 0.2
        
        return reward
    
    def _compute_risk_management_reward(self, trade_info: Dict) -> float:
        """Reward good risk management practices"""
        reward = 0.0
        
        # Stop loss placement
        if trade_info.get("has_stop_loss"):
            reward += 0.1
        
        # Take profit placement
        if trade_info.get("has_take_profit"):
            reward += 0.1
        
        # Risk-reward ratio
        rr_ratio = trade_info.get("risk_reward_ratio", 0)
        if rr_ratio >= 2.0:
            reward += 0.2
        elif rr_ratio < 1.0:
            reward -= 0.3
        
        # Position sizing appropriateness
        risk_pct = trade_info.get("risk_percentage", 0)
        if 0.01 <= risk_pct <= 0.02:  # 1-2% risk per trade
            reward += 0.15
        elif risk_pct > 0.05:  # >5% risk is too aggressive
            reward -= 0.4
        
        return reward
    
    def get_stats(self) -> Dict:
        """Get reward shaping statistics"""
        if not self.trade_history:
            return {}
        
        profits = [t['profit'] for t in self.trade_history]
        returns = [t['return'] for t in self.trade_history]
        
        winning_trades = [p for p in profits if p > 0]
        losing_trades = [p for p in profits if p < 0]
        
        return {
            'total_trades': len(self.trade_history),
            'winning_trades': len(winning_trades),
            'losing_trades': len(losing_trades),
            'win_rate': len(winning_trades) / len(self.trade_history) if self.trade_history else 0,
            'avg_win': np.mean(winning_trades) if winning_trades else 0,
            'avg_loss': np.mean(losing_trades) if losing_trades else 0,
            'profit_factor': abs(sum(winning_trades) / sum(losing_trades)) if losing_trades and sum(losing_trades) != 0 else 0,
            'total_profit': sum(profits),
            'sharpe_ratio': self._compute_sharpe_reward(),
            'max_drawdown': max(self.drawdown_history) if self.drawdown_history else 0,
            'current_win_streak': self.win_streak,
            'current_loss_streak': self.loss_streak
        }
    
    def reset(self):
        """Reset tracking for new episode"""
        self.trade_history = []
        self.equity_curve = []
        self.drawdown_history = []
        self.win_streak = 0
        self.loss_streak = 0


class AdaptiveRewardShaper(ProfitOptimizedRewardShaper):
    """
    Adaptive reward shaper that adjusts weights based on current performance
    """
    
    def __init__(self, config: Optional[Dict] = None):
        super().__init__(config)
        self.performance_window = []
    
    def adapt_weights(self):
        """Dynamically adjust reward weights based on recent performance"""
        if len(self.trade_history) < 20:
            return
        
        recent_trades = self.trade_history[-20:]
        win_rate = len([t for t in recent_trades if t['profit'] > 0]) / len(recent_trades)
        
        # If win rate is low, focus more on risk management
        if win_rate < 0.45:
            self.w_drawdown = -4.0  # Increase drawdown penalty
            self.w_sharpe = 2.5     # Focus on risk-adjusted returns
            self.w_leverage = 0.3   # Reduce leverage
        
        # If win rate is high, allow more aggressive trading
        elif win_rate > 0.55:
            self.w_profit = 1.5     # Increase profit weight
            self.w_leverage = 0.7   # Allow more leverage
            self.w_drawdown = -2.0  # Reduce drawdown penalty slightly
        
        print(f"🔧 Adapted reward weights for win_rate={win_rate:.2%}")
