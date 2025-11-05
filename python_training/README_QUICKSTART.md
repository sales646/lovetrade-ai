# 🚀 Quickstart: Drop-in Distributed Training

## TL;DR

```bash
git clone <your-repo>
cd python_training
./START_DISTRIBUTED.sh  # Linux/Mac
# or
START_DISTRIBUTED.bat   # Windows
```

**That's it!** The system will:
1. ✅ Install dependencies
2. ✅ Fetch real market data (Polygon S3 + Binance + Yahoo)
3. ✅ Store to Supabase
4. ✅ Start distributed GPU training
5. ✅ Log metrics visible in Strategies dashboard

---

## What You Get

### Data Sources (Real Market Data Only)
- **Polygon S3 (Massive)**: US Stocks minute data (2023-2024)
- **Binance**: Crypto minute data (2023-2024)
- **Yahoo Finance**: Supplementary data + news sentiment

### Training System
- **2x H100 GPUs** (or whatever you have)
- **512 parallel environments** (256 per GPU)
- **Transformer policy** (1024 hidden dim, 8 layers)
- **PPO algorithm** with GAE
- **Production-grade risk management**

### Results
- Training metrics logged to Supabase
- Real-time dashboard on `/strategies` page
- Checkpoints saved every 10 epochs

---

## Requirements

- NVIDIA GPU(s) with CUDA
- Python 3.8+
- Git

---

## API Keys (Already in .env)

All keys are **already configured** in `python_training/.env`:

```env
# Supabase (auto-configured)
SUPABASE_URL=https://rgpgssvakgutmgejazjq.supabase.co
SUPABASE_SERVICE_ROLE_KEY=<already set>

# Polygon S3 (Massive) - OPEN in repo
POLYGON_S3_ACCESS_KEY=TDADiVQVkn38UZMEF4jP
POLYGON_S3_SECRET_KEY=SsjXmFJYLGGPWYGLqOUxMFpC5FWVwbPUqjWEuI4g
POLYGON_S3_ENDPOINT=https://files.polygon.io
POLYGON_S3_BUCKET=flatfiles

# Binance (public data, no auth needed)
BINANCE_API_KEY=not_required_for_market_data
BINANCE_SECRET_KEY=not_required_for_market_data
```

**No setup needed!** Just clone and run.

---

## Manual Steps (Optional)

If you want more control:

### 1. Install Dependencies
```bash
pip install -r requirements_production.txt
pip install -r requirements_distributed.txt
```

### 2. Check GPU
```bash
python -c "import torch; print(f'GPUs: {torch.cuda.device_count()}')"
```

### 3. Run Training
```bash
python distributed_orchestrator.py
```

---

## Monitoring

### Dashboard
Navigate to `/strategies` in your app to see:
- Active training status
- GPU utilization
- Performance metrics (reward, loss)
- Recent runs

### Logs
Training outputs to console:
```
🚀 PNU Training System Setup
====================================================================
📥 Fetching REAL market data (Polygon S3 + Binance + Yahoo)...
📊 US Stocks: 250 symbols
📊 Crypto: 350 symbols
💾 Storing 600 symbols to Supabase...
✅ Ready with 600 symbols
🎯 Starting BC→PPO Training Pipeline
```

---

## Configuration

Edit `distributed_orchestrator.py` if you want to change:

```python
'world_size': 2,              # Number of GPUs
'envs_per_gpu': 256,          # Parallel envs per GPU
'd_model': 1024,              # Model size
'epochs': 100,                # Training epochs
'total_timesteps': 50_000_000 # Total steps
```

---

## Troubleshooting

### Out of Memory
Reduce batch size:
```python
'batch_size': 32768 → 16384
'envs_per_gpu': 256 → 128
```

### No GPUs Detected
```bash
# Check CUDA
nvidia-smi

# Install PyTorch with CUDA
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Data Fetch Fails
- Check internet connection
- Polygon S3 keys are valid (already in .env)
- Supabase URL is reachable

---

## Next Steps

1. **Monitor training** on `/strategies` page
2. **Adjust hyperparameters** in `distributed_orchestrator.py`
3. **Extend to more symbols** by changing data fetch range
4. **Deploy best model** via `autonomous-trader` edge function

---

## Support

Issues? Check:
- Console logs for errors
- Supabase tables: `training_metrics`, `historical_bars`
- GPU memory: `nvidia-smi`

Happy training! 🎯
