# RL Trading Policy Training

This directory contains the Python training pipeline for Behavior Cloning (BC) and PPO finetuning.

## Setup

1. **Install dependencies**:
```bash
pip install -r requirements.txt
```

2. **Set environment variables**:
```bash
export SUPABASE_URL="your_supabase_url"
export SUPABASE_SERVICE_ROLE_KEY="your_service_role_key"
```

You can find these in your Lovable project's backend settings.

3. **Prepare data**:
Before running training, make sure you have generated expert trajectories:
- Go to Training page in the UI
- Select symbols (e.g., SPY, QQQ, AAPL, TSLA)
- Run the data pipeline (Fetch Data → Compute Indicators → Generate Trajectories)

## Training

Run the full pipeline:
```bash
python train_rl_policy.py
```

This will:
1. Load expert trajectories from Supabase
2. Train a Behavior Cloning (BC) policy to warm-start
3. Finetune with PPO using realistic rewards
4. Log all metrics back to Supabase
5. Save checkpoints in `checkpoints/` directory

## Configuration

Edit `TrainingConfig` in `train_rl_policy.py`:

- **Data**: symbols, timeframe, train/val/test splits
- **BC**: epochs, batch size, learning rate
- **PPO**: timesteps, n_steps, batch size, learning rate
- **Reward**: lambda_risk (risk penalty coefficient)

## Outputs

- `checkpoints/policy_bc_{run_id}.pt` - BC warm-start weights
- `checkpoints/policy_ppo_{run_id}.zip` - Final PPO model
- Metrics logged to `training_metrics` table in Supabase
- Training runs tracked in `training_runs` table

## Walk-Forward Validation

To implement walk-forward validation:
1. Modify `main()` to loop over rolling windows
2. For each window:
   - Train on past data
   - Validate on future data
   - Store metrics per window
3. Report median performance across windows

## GPU Support

This script supports GPU training. PyTorch will automatically use CUDA if available.

To check:
```python
import torch
print(torch.cuda.is_available())
```

## Next Steps

After training:
1. Evaluate on test set
2. Run ablation studies (with/without news features)
3. Analyze action distribution (%BUY/%SELL/%HOLD)
4. Calculate profit factor, Sharpe, win rate, max DD
5. Export metrics for reporting
