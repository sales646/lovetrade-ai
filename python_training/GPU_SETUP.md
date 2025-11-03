# GPU Server Setup Guide

## Prerequisites
- GPU server with 8 NVIDIA GPUs (RTX 4090 or similar)
- Ubuntu 20.04+ or similar Linux distro
- CUDA 11.8+ and cuDNN installed
- SSH access to the server

## Step 1: Clone Repository

```bash
# SSH into your GPU server
ssh your-username@your-gpu-server

# Clone the repository
git clone https://github.com/YOUR_USERNAME/tradepilot.git
cd tradepilot/python_training
```

## Step 2: Install Dependencies

```bash
# Install Python 3.10+ if not available
sudo apt update
sudo apt install python3.10 python3-pip

# Create virtual environment
python3.10 -m venv venv
source venv/bin/activate

# Install PyTorch with CUDA support (adjust CUDA version as needed)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Install distributed training requirements
pip install -r requirements_distributed.txt

# Verify GPU setup
python check_gpu.py
```

Expected output:
```
✅ CUDA available: True
✅ GPU count: 8
✅ GPU 0: NVIDIA GeForce RTX 4090 (24GB)
✅ GPU 1: NVIDIA GeForce RTX 4090 (24GB)
...
```

## Step 3: Configure Environment Variables

```bash
# Copy example env file
cp .env.example .env

# Edit with your credentials
nano .env
```

**Required variables:**
- `SUPABASE_URL`: Already set to your project URL
- `SUPABASE_SERVICE_ROLE_KEY`: Get from Lovable Cloud dashboard

**To get your Service Role Key:**
1. Go to your Lovable project
2. Open Backend settings (in chat: click "View Backend" button)
3. Navigate to Settings > API
4. Copy the `service_role` key (keep this SECRET!)

Paste it in `.env`:
```bash
SUPABASE_SERVICE_ROLE_KEY=eyJhbGc... (your actual key)
```

## Step 4: Test Supabase Connection

```bash
python -c "
import os
from supabase import create_client
from dotenv import load_dotenv

load_dotenv()
supabase = create_client(
    os.getenv('SUPABASE_URL'),
    os.getenv('SUPABASE_SERVICE_ROLE_KEY')
)
result = supabase.table('training_runs').select('*').limit(1).execute()
print('✅ Supabase connection successful!')
"
```

## Step 5: Start Distributed Training

### Option A: Start with defaults (Recommended first time)

```bash
# Windows (if you copied to Windows GPU server)
START_DISTRIBUTED_TRAINING.bat

# Linux/Mac
python distributed_orchestrator.py
```

### Option B: Custom configuration

```bash
python distributed_orchestrator.py --config custom_config.json
```

Example `custom_config.json`:
```json
{
  "world_size": 8,
  "envs_per_gpu": 1000,
  "epochs": 100,
  "pbt_enabled": true,
  "model_type": "transformer",
  "learning_rate": 3e-4
}
```

## Step 6: Monitor Training

### View in Web App
- Open your Lovable app at: https://YOUR_APP_URL/strategies
- You'll see live metrics: GPUs, environments, rewards, PBT status

### View in Terminal
Training outputs real-time stats:
```
🚀 Launching distributed training on 8 GPUs
📊 Total environments: 8000

📅 Epoch 1/100
──────────────────────────────────────
🚀 Launching 8 workers...
📊 Epoch 1 Results:
   Avg Loss: 0.3421
   Avg Reward: 156.34
🎮 GPU Stats:
   Avg Utilization: 87.3%
   Avg Memory: 18.2 GB
   Avg Temp: 72.5°C
```

### Check GPU Usage
```bash
# In another terminal
watch -n 1 nvidia-smi
```

## Troubleshooting

### GPU Out of Memory
Reduce batch size or envs per GPU:
```python
# In distributed_orchestrator.py or via config
config['envs_per_gpu'] = 500  # Instead of 1000
config['batch_size'] = 128     # Instead of 256
```

### CUDA Errors
Verify CUDA installation:
```bash
nvcc --version
nvidia-smi
```

### Supabase Connection Failed
- Check `.env` file has correct credentials
- Verify internet connection
- Test with curl: `curl https://rgpgssvakgutmgejazjq.supabase.co`

### Training Too Slow
Enable BF16 precision (faster on modern GPUs):
```python
config['use_bf16'] = True  # Already default
```

## Performance Optimization

### For 8x RTX 4090:
- **Expected speed**: ~2000 steps/sec total (250/sec per GPU)
- **Memory per GPU**: 16-20 GB
- **Training time**: 50M timesteps in ~7-8 hours

### Adjust for your hardware:
```python
# Smaller GPUs (16GB VRAM)
config['envs_per_gpu'] = 500
config['batch_size'] = 128

# Larger GPUs (48GB VRAM like A6000)
config['envs_per_gpu'] = 2000
config['batch_size'] = 512
```

## Checkpoints & Results

Training saves checkpoints to:
```
checkpoints/dist_run_YYYYMMDD_HHMMSS/
├── config.json
├── metrics.jsonl
├── checkpoint_epoch_0.pt
├── checkpoint_epoch_50.pt
└── checkpoint_epoch_100.pt
```

Best checkpoint is automatically uploaded to Supabase.

## Next Steps

1. **Let it train**: First run should be 24-48 hours for good results
2. **Monitor PBT**: Check Strategies page for hyperparameter evolution
3. **Evaluate**: Compare Sharpe ratios across PBT generations
4. **Deploy**: Best policy is auto-deployed to `autonomous-trader` function

## Support

If issues persist:
1. Check logs in `checkpoints/*/training.log`
2. Share error messages in GitHub issues
3. Contact support with your setup details

Happy training! 🚀
