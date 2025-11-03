# Distributed RL Training System - Complete Guide

## Overview

This system trains reinforcement learning agents for trading using:
- **8 GPUs** with distributed data parallel (DDP)
- **Transformer policy networks** for temporal pattern recognition
- **Population-Based Training (PBT)** for hyperparameter optimization
- **Real market data** from Supabase with fallback synthetic data
- **BF16 mixed precision** for faster training

## Quick Start

### 1. Verify System (IMPORTANT - Run this first!)

```bash
VERIFY_BEFORE_TRAINING.bat
```

This will check:
- Python environment and dependencies
- GPU availability and CUDA
- All imports and components
- Model creation and forward pass
- Environment data loading
- PBT scheduler
- Supabase connection (uses fallback if not available)

### 2. Start Training

```bash
START_DISTRIBUTED_TRAINING.bat
```

The system will:
- Start with **10 environments per GPU** (80 total) for stability testing
- Run distributed training across all GPUs
- Apply PBT for hyperparameter optimization
- Save checkpoints every 50 epochs
- Monitor GPU utilization and temperature

## System Architecture

### Components

1. **distributed_training.py**
   - Manages DDP across GPUs
   - Implements PPO algorithm
   - Handles rollout collection and training

2. **transformer_policy.py**
   - Transformer-based policy network
   - Separate actor (policy) and critic (value) heads
   - Positional encoding for time series

3. **trading_environment.py**
   - Loads real market data from Supabase
   - Provides fallback synthetic data if DB is empty
   - Implements realistic trading simulation

4. **pbt_scheduler.py**
   - Population-Based Training scheduler
   - Exploit & explore for hyperparameter optimization
   - Adaptive learning rate, gamma, clip_param

5. **distributed_orchestrator.py**
   - Master coordinator for all components
   - Manages training loop and checkpointing
   - Integrates GPU monitoring

6. **gpu_monitor.py**
   - Real-time GPU utilization tracking
   - Temperature and memory monitoring
   - Load balancing recommendations

## Configuration

Edit `distributed_orchestrator.py` to change:

```python
{
    'world_size': 8,        # Number of GPUs
    'envs_per_gpu': 10,     # Environments per GPU (start small)
    'use_bf16': True,       # BF16 mixed precision
    
    'epochs': 100,          # Training epochs
    'batch_size': 256,      # PPO batch size
    'learning_rate': 3e-4,  # Initial learning rate
    
    'model_type': 'transformer',  # Policy type
    'd_model': 256,         # Transformer dimension
    'num_layers': 4,        # Transformer layers
}
```

## Training Flow

1. **Setup Phase**
   - Check GPU availability
   - Initialize distributed trainer
   - Load historical market data
   - Create PBT population

2. **Training Loop** (per epoch)
   - Launch 8 workers (one per GPU)
   - Each worker:
     - Creates 10 parallel environments
     - Collects 512-step rollouts
     - Trains with PPO for 4 epochs
   - Synchronize gradients across GPUs
   - PBT exploit & explore (every 5 epochs)

3. **Checkpointing**
   - Save every 50 epochs
   - Save on interruption
   - Includes model, config, PBT state

## Data Sources

### Real Market Data (Preferred)
- Loaded from Supabase `historical_bars` table
- 5-minute timeframe by default
- Multiple symbols: AAPL, MSFT, GOOGL, TSLA, NVDA, AMZN
- Technical indicators from `technical_indicators` table

### Fallback Synthetic Data
- Used if Supabase is unavailable or empty
- 1000 bars of random walk data
- Ensures training can always proceed

## Monitoring

### During Training

Watch for:
- `[GPU X] Setting up...` - Worker initialization
- `[GPU X] Epoch Y: Collected Z steps` - Rollout collection
- `Epoch X: Loss=Y, Reward=Z` - Training progress
- `🧬 PBT: Best performer: ...` - Hyperparameter evolution

### GPU Stats (every 10 epochs)
- Average utilization %
- Memory usage
- Temperature
- Thermal throttling warnings

### Checkpoints
- `checkpoints/dist_run_YYYYMMDD_HHMMSS/`
  - `config.json` - Training configuration
  - `checkpoint_epoch_X.pt` - Model checkpoints
  - `metrics.jsonl` - Training metrics log

## Troubleshooting

### Workers hang on setup
**Cause**: Too many environments per GPU blocking initialization
**Fix**: Already set to 10 per GPU. If issues persist, reduce to 5.

### CUDA out of memory
**Cause**: Model too large or batch size too big
**Fix**: Reduce `batch_size` or use `model_type: 'lightweight'`

### No historical data loaded
**Cause**: Supabase connection issue
**Fix**: System will automatically use fallback data. Check logs for details.

### nvidia-smi errors
**Not a problem**: These are monitoring warnings. Training uses PyTorch directly.

### Training very slow
**Check**: 
- GPU utilization (should be >80%)
- BF16 enabled (2x speedup)
- Thermal throttling (reduce temp if >85°C)

## Scaling Up

After successful test run with 10 envs/GPU:

1. Stop training (Ctrl+C)
2. Edit `distributed_orchestrator.py`:
   ```python
   'envs_per_gpu': 100,  # Scale to 100
   ```
3. Restart training

You can scale up to 1000 envs/GPU for maximum throughput (8000 total environments).

## Performance Expectations

### With 8 GPUs, 10 envs/GPU (80 total)
- ~500-1000 steps/second
- ~5-10 epochs/hour
- Good for testing stability

### With 8 GPUs, 1000 envs/GPU (8000 total)
- ~50,000-100,000 steps/second  
- ~100-200 epochs/hour
- Production throughput

## Files You Can Safely Modify

- `distributed_orchestrator.py` - Training config
- `trading_environment.py` - Environment logic
- `advanced_rewards.py` - Reward shaping
- `transformer_policy.py` - Model architecture

## Files You Should NOT Modify

- `distributed_training.py` - Core DDP logic (unless you know what you're doing)
- `gpu_monitor.py` - Monitoring infrastructure

## Next Steps After Training

1. **Evaluate Models**
   - Load checkpoint from `checkpoints/`
   - Test on held-out data
   - Compare different PBT populations

2. **Deploy Best Policy**
   - Export to ONNX for production
   - Integrate with trading system
   - Backtest on historical data

3. **Continue Training**
   - Resume from checkpoint
   - Adjust hyperparameters based on results
   - Scale up environments for more data

## Support

If issues persist after verification:
1. Check `verify_system.py` output
2. Review error messages in training logs
3. Ensure CUDA drivers are up to date
4. Verify PyTorch CUDA compatibility

## Environment Variables

Required (already configured in `.env`):
```
SUPABASE_URL=https://rgpgssvakgutmgejazjq.supabase.co
SUPABASE_SERVICE_ROLE_KEY=<your_key>
```

Optional:
```
TRAINING_MODE=distributed
GPU_COUNT=8
ENVS_PER_GPU=10
```
