# 🎯 How to Run Training

## Quick Commands

### Test System First (Recommended)
```bash
python test_complete_system.py
```
Verifies all connections and data sources are working.

---

### Distributed Training (GPU-intensive)
```bash
python start_distributed.py
```

**What it does:**
- Fetches real data from Polygon S3 + Binance + Yahoo Finance
- Stores to Supabase
- Starts distributed PPO training across all GPUs
- Logs metrics to Supabase (visible on `/strategies` dashboard)

**Requirements:**
- 2+ NVIDIA GPUs recommended
- ~60GB GPU memory per H100
- CUDA 11.8+

---

### Production Training (Single GPU)
```bash
python start_production.py
```

**What it does:**
- Fetches 60 days of recent real market data
- Trains Transformer policy with PPO
- Logs to Supabase
- Saves checkpoints every 20 epochs

**Requirements:**
- 1 NVIDIA GPU (12GB+ VRAM)
- Less intensive than distributed

---

## What Happens During Training

### Phase 1: Data Fetch
```
📥 Fetching REAL market data...
📊 Polygon S3: US Stocks (2023-2024)
📊 Binance: Crypto (2023-2024)  
📊 Yahoo Finance: Supplementary data + news
💾 Storing to Supabase historical_bars...
```

### Phase 2: Training
```
🎯 Starting BC→PPO Training Pipeline
🧠 Transformer Policy (1024 hidden, 8 layers)
⚡ 512 parallel environments (256 per GPU)
📈 Epoch 1/100...
```

### Phase 3: Logging
```
💾 Logging to Supabase training_metrics...
✅ Epoch 10 complete
   Mean Reward: 2.34
   Policy Loss: 0.0023
   GPU Utilization: 87%
```

---

## Monitoring Training

### Real-time Dashboard
Navigate to `/strategies` in your app:
- **Overview Tab**: Current status, GPU count, environments
- **Training Progress Tab**: Live charts (reward, loss, GPU utilization)
- **Recent Runs**: All training sessions with config details

### Supabase Tables
Direct access to raw data:
- `training_metrics`: All epoch metrics
- `historical_bars`: Market data
- `pbt_populations`: PBT evolution (if enabled)

### Console Output
Real-time logs show:
- Data fetch progress
- Training epochs
- Metrics per epoch
- Checkpoint saves

---

## Configuration

Edit `distributed_orchestrator.py` to customize:

```python
config = {
    # GPUs
    'world_size': 2,              # Number of GPUs
    'envs_per_gpu': 256,          # Parallel environments per GPU
    
    # Model
    'd_model': 1024,              # Transformer hidden size
    'num_layers': 8,              # Depth
    
    # Training
    'epochs': 100,                # Total epochs
    'batch_size': 32768,          # PPO batch size
    'learning_rate': 3e-4,
    
    # Data
    'data_days': 90,              # Days of historical data
}
```

---

## Expected Performance

### 2x H100 GPUs
- **Throughput**: 500K+ environment steps/second
- **GPU Utilization**: 80-95%
- **Memory Usage**: ~60GB per GPU
- **Training Time**: ~6-8 hours for 100 epochs

### 1x RTX 3090
- **Throughput**: ~50K steps/second
- **GPU Utilization**: 70-85%
- **Memory Usage**: ~20GB
- **Training Time**: ~24 hours for 100 epochs

---

## Stopping Training

Press `Ctrl+C` to gracefully stop:
```
⚠️  Training interrupted by user
💾 Saving checkpoint...
✅ Checkpoint saved: checkpoints/pnu_20250115_143022/epoch_45.pt
```

Resume later by loading checkpoint in `distributed_orchestrator.py`.

---

## Next Steps After Training

1. **Check Metrics**: View `/strategies` dashboard
2. **Download Checkpoints**: Find in `checkpoints/<run_id>/`
3. **Deploy Best Model**: Use `autonomous-trader` edge function
4. **Analyze Results**: Query `training_metrics` table

---

## Troubleshooting

### "CUDA out of memory"
Reduce batch size and envs:
```python
'batch_size': 32768 → 16384
'envs_per_gpu': 256 → 128
```

### "No module named 'torch'"
```bash
pip install -r requirements_distributed.txt
```

### "Connection refused" to Supabase
Check `.env` file has correct `SUPABASE_URL` and `SUPABASE_SERVICE_ROLE_KEY`.

### No data fetched
Run test first:
```bash
python test_complete_system.py
```

---

Happy Training! 🚀
