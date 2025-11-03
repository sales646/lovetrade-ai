# Enterprise-Grade RL Trading Training Pipeline

## 🚀 New Capabilities for 8x H100 Training

This upgraded training pipeline is designed to leverage massive GPU compute (8x H100 or similar) for production-quality trading strategies.

### ✨ What's New

#### 1. **Massively Expanded Data** (Tier 1)
- ✅ **50+ liquid stocks** across all sectors (was: 20)
- ✅ **365 days** of historical data (was: 180)
- ✅ **5x data augmentation** - temporal jitter, price scaling, noise injection
- ✅ **Regime-based sampling** for balanced training

#### 2. **Advanced Features** (Tier 2)
- ✅ Order flow imbalance
- ✅ Bid-ask spread dynamics
- ✅ Intraday volume profile
- ✅ Options market data (IV rank, put/call ratio, skew)
- ✅ Sector rotation metrics

#### 3. **Ensemble Architecture** (Tier 2)
- ✅ 5 specialized policies:
  - Trend Follower
  - Mean Reversion
  - Breakout Specialist
  - Momentum Trader
  - Volatility Trader
- ✅ Intelligent regime detection
- ✅ Adaptive policy weighting

#### 4. **Hyperparameter Optimization** (Tier 2)
- ✅ Optuna-based systematic search
- ✅ 100+ trials for BC and PPO
- ✅ Parallel optimization support
- ✅ Top-K ensemble selection

#### 5. **Robustness Testing** (Tier 3)
- ✅ 9 stress test scenarios:
  - Market crash (-30%)
  - Flash crash + recovery
  - Volatility spike (3x)
  - Low liquidity
  - Bull/bear trends
  - Choppy sideways
  - Gap up/down
- ✅ Pass/fail criteria
- ✅ Comprehensive reporting

#### 6. **Model Versioning** (Tier 3)
- ✅ Model registry with metadata
- ✅ Performance tracking
- ✅ Model comparison
- ✅ Deployment management
- ✅ Model lineage tracking

---

## 📊 Training Configurations

### Quick Start Presets

```python
from hyperparameter_search import get_preset_config

# Fast iteration (1-2 hours on 8x H100)
config = get_preset_config('fast')

# Balanced (4-6 hours)
config = get_preset_config('balanced')

# Aggressive (12-24 hours)
config = get_preset_config('aggressive')
```

### Scale Comparison

| Config | BC Epochs | PPO Timesteps | Batch Size | Frame Stack | Training Time (8x H100) |
|--------|-----------|---------------|------------|-------------|------------------------|
| **Fast** | 1,000 | 100K | 2048 | 30 | ~1-2 hours |
| **Balanced** | 3,000 | 500K | 4096 | 60 | ~4-6 hours |
| **Aggressive** | 10,000 | 50M | 32768 | 120 | ~12-24 hours |

---

## 🎯 Usage Examples

### 1. Basic Training with All Features

```python
from train_rl_policy import TrainingConfig, main

# Create config with all new features enabled
config = TrainingConfig(
    # Data
    symbols=[...],  # 50 stocks
    days_lookback=365,
    
    # Training scale
    bc_batch_size=16384,      # 32x larger
    ppo_batch_size=32768,     # 8x larger
    ppo_total_timesteps=50_000_000,  # 100x more
    
    # Advanced features
    use_data_augmentation=True,
    augmentation_factor=5,
    use_ensemble=True,
    ensemble_mode='weighted_vote'
)

# Run training
main()
```

### 2. Hyperparameter Search

```python
from hyperparameter_search import HyperparameterSearch

# Initialize search
hpo = HyperparameterSearch(
    study_name="rl_trading_hpo",
    n_trials=100,
    n_jobs=8  # Parallel search on 8 GPUs
)

# Define training function
def train_fn(hyperparameters, trial):
    # ... run training with hyperparameters
    return metrics

# Optimize
study = hpo.optimize(
    train_fn=train_fn,
    metric_name='val_sharpe',
    mode='max'
)

# Get best config
best_params = hpo.get_best_hyperparameters(study)
```

### 3. Ensemble Training

```python
from ensemble_policies import PolicyEnsemble

# Create ensemble
ensemble = PolicyEnsemble(
    input_dim=4200,  # 120 frames * 35 features
    hidden_dims=[1024, 512, 256],
    device='cuda'
)

# Predict with ensemble
action, confidence, details = ensemble.predict(
    state=current_state,
    mode='weighted_vote'  # or 'regime_based'
)

# Get performance stats
stats = ensemble.get_policy_stats()
```

### 4. Robustness Testing

```python
from robustness_testing import RobustnessTester, StressScenario

# Initialize tester
tester = RobustnessTester(
    min_sharpe=0.5,
    max_drawdown=0.15,
    min_win_rate=0.45
)

# Run full test suite
results = tester.run_full_suite(
    policy=trained_policy,
    base_data=historical_data
)

# Check which scenarios passed
for scenario, result in results.items():
    if not result.passed:
        print(f"Failed {scenario}: {result.failure_reason}")
```

### 5. Model Versioning

```python
from model_versioning import ModelRegistry

# Initialize registry
registry = ModelRegistry(registry_dir="models/registry")

# Register new model
version = registry.register_model(
    model_path="checkpoints/bc_policy_best.pt",
    model_type="bc",
    hyperparameters=config.__dict__,
    performance_metrics={
        'val_sharpe': 1.2,
        'win_rate': 0.52,
        'max_drawdown': 0.08
    },
    training_data={
        'symbols': config.symbols,
        'timeframe': config.timeframe,
        'date_range': '2024-01-01 to 2024-12-31'
    },
    notes="First production model with 8x H100 training"
)

# Compare models
comparison = registry.compare_models(
    version1="bc-20250103_143022",
    version2="bc-20250103_165544"
)

# Deploy best model
best_model = registry.get_best_model(metric='sharpe_ratio')
registry.deploy_model(best_model.version, "production/model.pt")
```

### 6. Data Augmentation

```python
from data_augmentation import TimeSeriesAugmenter, get_feature_indices

# Initialize augmenter
augmenter = TimeSeriesAugmenter(
    temporal_jitter_prob=0.3,
    price_scaling_prob=0.3,
    noise_injection_prob=0.2,
    regime_sampling_prob=0.4
)

# Augment trajectories
feature_indices = get_feature_indices()
augmented_trajectories = augmenter.augment_batch(
    trajectories=original_trajectories,
    feature_indices=feature_indices,
    augmentation_factor=5  # 5x more data
)
```

---

## 🏗️ Architecture Overview

```
┌─────────────────────────────────────────────────────────┐
│                   DATA PIPELINE                         │
├─────────────────────────────────────────────────────────┤
│  50 Symbols × 365 Days × Advanced Features             │
│  → Data Augmentation (5x)                              │
│  → Regime-Based Sampling                                │
└─────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────┐
│              HYPERPARAMETER SEARCH (Optional)           │
├─────────────────────────────────────────────────────────┤
│  Optuna: 100 Trials × 8 Parallel Workers               │
│  → Best Config Selection                                │
└─────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────┐
│                 TRAINING PIPELINE                       │
├─────────────────────────────────────────────────────────┤
│  1. Behavior Cloning (10K epochs, batch=16K)           │
│  2. PPO Finetuning (50M steps, batch=32K)              │
│  3. Early Stopping + Top-K Checkpoints                 │
└─────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────┐
│              ENSEMBLE ARCHITECTURE                      │
├─────────────────────────────────────────────────────────┤
│  ┌──────────┐  ┌──────────┐  ┌──────────┐            │
│  │  Trend   │  │   Mean   │  │ Breakout │            │
│  │ Follower │  │ Reversion│  │Specialist│ ...        │
│  └──────────┘  └──────────┘  └──────────┘            │
│          │          │            │                     │
│          └──────────┴────────────┘                     │
│                     │                                   │
│            Intelligent Router                          │
│         (Regime Detection + Weighting)                 │
└─────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────┐
│              ROBUSTNESS TESTING                         │
├─────────────────────────────────────────────────────────┤
│  9 Stress Scenarios × Multiple Episodes                │
│  → Pass/Fail Validation                                │
└─────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────┐
│              MODEL VERSIONING                           │
├─────────────────────────────────────────────────────────┤
│  Registry → Comparison → Deployment                     │
└─────────────────────────────────────────────────────────┘
```

---

## 📈 Expected Performance

### Single GPU (RTX 4090)
- **Training Time**: ~12-20 hours for aggressive config
- **Throughput**: ~5K samples/sec

### 8x H100 GPUs
- **Training Time**: ~2-4 hours for aggressive config
- **Throughput**: ~80K+ samples/sec
- **Speedup**: ~8-10x vs single GPU

### Target Metrics (After Full Training)
- **Sharpe Ratio**: > 1.5
- **Win Rate**: > 52%
- **Max Drawdown**: < 10%
- **Robustness**: Pass 8/9 stress tests

---

## 🔧 Setup for Cloud GPU

### Option 1: RunPod (8x H100)

```bash
# 1. Launch 8x H100 instance on RunPod
# 2. SSH into instance
# 3. Clone repo and install
pip install -r requirements.txt

# 4. Set environment variables
export SUPABASE_URL="your-url"
export SUPABASE_SERVICE_ROLE_KEY="your-key"

# 5. Run distributed training
python train_rl_policy.py
```

### Option 2: Lambda Labs (8x H100)

```bash
# Similar setup as RunPod
# Lambda Labs pricing: ~$16-24/hour for 8x H100
```

---

## 📚 Module Reference

| Module | Purpose | Key Classes/Functions |
|--------|---------|----------------------|
| `train_rl_policy.py` | Main training pipeline | `TrainingConfig`, `main()` |
| `data_augmentation.py` | Data augmentation | `TimeSeriesAugmenter` |
| `hyperparameter_search.py` | HPO with Optuna | `HyperparameterSearch` |
| `ensemble_policies.py` | Ensemble of specialists | `PolicyEnsemble`, `RegimeDetector` |
| `robustness_testing.py` | Stress testing | `RobustnessTester` |
| `model_versioning.py` | Model management | `ModelRegistry` |
| `advanced_features.py` | Feature extraction | `AdvancedFeatureExtractor` |

---

## 🎓 Best Practices

1. **Start Small, Scale Up**
   - Test with `fast` preset first
   - Verify data quality
   - Then scale to `aggressive`

2. **Monitor Training**
   - Check Supabase `training_metrics` table
   - Watch for overfitting (train-val gap)
   - Use early stopping

3. **Hyperparameter Search**
   - Run HPO on smaller subset first
   - Use top-K ensemble of best configs
   - Re-train winners on full data

4. **Robustness is Key**
   - Always run stress tests
   - A model that fails crash scenarios is NOT production-ready
   - Target: Pass 8/9 stress tests minimum

5. **Version Everything**
   - Use model registry
   - Track all hyperparameters
   - Document what worked (and what didn't)

---

## 🐛 Troubleshooting

### OOM (Out of Memory) Errors
```python
# Reduce batch sizes
config.bc_batch_size = 8192  # Half size
config.ppo_batch_size = 16384
```

### Slow Training
```python
# Check GPU utilization
nvidia-smi

# Enable mixed precision (if supported)
# Add to training loop
from torch.cuda.amp import autocast, GradScaler
```

### Poor Convergence
```python
# Try different learning rates
config.bc_lr = 1e-4  # Lower
config.ppo_learning_rate = 1e-4

# Or increase data augmentation
config.augmentation_factor = 10
```

---

## 📞 Support

For issues or questions, check:
1. This README
2. Code comments
3. Supabase logs: `training_metrics`, `training_runs`

---

**Built for production. Tested at scale. Ready for 8x H100.**
