# GPU Training Setup Instructions

Your local GPU training is now connected to the cloud! All training metrics automatically sync to your dashboard.

## Installation (Required First!)

**Before running training, install dependencies:**

```bash
# Navigate to python_training folder
cd python_training

# Install all required packages
pip install -r requirements.txt

# For GPU support (CUDA 12.1)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Verify GPU is detected
python check_gpu.py
```

You should see "CUDA Available: True" and your GPU name. If not, see troubleshooting below.

## Quick Start

### Option 1: Single Training Run (Recommended for Testing)
```bash
# Windows
START_GPU_TRAINING.bat

# Linux/Mac
python train_rl_policy.py
```

### Option 2: Continuous Parallel Training (5 Workers)
```bash
# Windows
START_PARALLEL_GPU_TRAINING.bat

# Linux/Mac
python run_continuous_training.py
```

## What Happens During Training

### Data Flow
1. **Loads Training Data**: Fetches expert trajectories from Supabase cloud database
2. **BC Training**: Trains behavior cloning policy to mimic expert strategies (15 epochs)
3. **PPO Training**: Finetunes with reinforcement learning (50 epochs, walk-forward validation)
4. **Uploads Results**: All metrics automatically sync to cloud dashboard

### Training Metrics Logged
- Mean reward, win rate, profit factor, Sharpe ratio
- Policy loss, value loss, entropy
- Action distribution (buy/sell/hold percentages)
- Max drawdown, average R:R ratio
- Validation splits (train/val/test)

### Performance Expectations
- **GPU Training**: ~2-5 minutes per full BC+PPO cycle
- **CPU Training**: ~15-30 minutes per cycle
- **Parallel (5 workers)**: 5x throughput

## Viewing Results

Open your training dashboard:
https://7b040b8f-dffe-48c1-aedd-fecc6dccf027.lovableproject.com/training

Local GPU training metrics will appear alongside cloud training in real-time!

## Architecture

```
┌─────────────────┐
│  Your GPU PC    │
│                 │
│  PyTorch +      │──┐
│  CUDA           │  │
│                 │  │
│  BC + PPO       │  │
│  Training       │  │
└─────────────────┘  │
                     │  Metrics, Results
                     ↓
              ┌──────────────┐
              │   Supabase   │
              │   Database   │
              └──────────────┘
                     ↑
                     │  Dashboard Queries
┌─────────────────┐  │
│  Web Dashboard  │──┘
│                 │
│  Training Tab   │
│  Live Metrics   │
└─────────────────┘
```

## Training Configuration

Edit `train_rl_policy.py` to customize:
- `bc_epochs`: Behavior cloning epochs (default: 15)
- `ppo_epochs`: PPO training epochs (default: 50)
- `bc_lr`: BC learning rate (default: 3e-4)
- `ppo_lr`: PPO learning rate (default: 3e-4)
- `n_steps`: PPO rollout steps (default: 512)
- `batch_size`: Training batch size (default: 64)

## Troubleshooting

### "CUDA Available: False"
- Install CUDA-enabled PyTorch:
  ```bash
  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
  ```
- Check GPU: `nvidia-smi`

### "No trajectories found"
- Generate training data first from the web dashboard
- Click "Generate Data" or start autonomous generation

### Import Errors
- Install dependencies:
  ```bash
  pip install -r requirements.txt
  ```

## Next Steps

1. **Generate Training Data**: Use the web dashboard to generate realistic market scenarios
2. **Start Single Training**: Run `START_GPU_TRAINING.bat` to test GPU setup
3. **Monitor Dashboard**: Watch metrics appear in real-time
4. **Scale Up**: Start parallel training with 5 workers for 5x throughput
5. **Iterate**: Cloud auto-generation + local GPU training = continuous learning
