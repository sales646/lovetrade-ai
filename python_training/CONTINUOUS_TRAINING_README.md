# Continuous Training - Maximum A100 Power

## What This Does

**Automatically repeating training** that uses **ALL** your GPU power and pulls data from **three sources**:
- 📊 **Polygon S3** - High-quality stock & crypto minute data
- 🪙 **Binance** - Real-time crypto data (20+ pairs)
- 📈 **Yahoo Finance** - Major stock data

## Quick Start

```bash
# 1. Install dependencies (if not already done)
pip install -r requirements_simple.txt

# 2. Make sure your .env has Polygon credentials
# POLYGON_S3_ACCESS_KEY=your_key
# POLYGON_S3_SECRET_KEY=your_secret

# 3. Start continuous training
bash START_CONTINUOUS.sh
# or on Windows:
START_CONTINUOUS.bat
```

## Configuration (Optimized for 4x A100)

```
🔥 Maximum GPU Utilization:
- 4 GPUs × 512 parallel environments = 2,048 total environments
- Batch size: 65,536 (massive batches)
- Model: Large Transformer (2048-dim, 12 layers, 32 heads)
- BF16 + TF32 enabled for A100 optimization
- Gradient accumulation for even larger effective batches

📊 Data Loading:
- Pulls 2 years of data (2022-2024)
- Polygon S3: US stocks + global crypto
- Binance: 20+ crypto pairs (BTC, ETH, SOL, etc.)
- Yahoo Finance: 22 major stocks (AAPL, MSFT, NVDA, etc.)
- Minimum 2000 bars per symbol for quality

🔄 Continuous Mode:
- Completes 200 epochs per run
- Automatically starts next run after completion
- Refreshes data from all sources each run
- Saves checkpoints every 25 epochs
- Runs forever until you stop it (Ctrl+C)
```

## What Gets Created

```
checkpoints/
├── run1_epoch25.pt
├── run1_epoch50.pt
├── run1_epoch75.pt
├── ...
├── run2_epoch25.pt
└── ...
```

Each checkpoint contains:
- Model weights
- Training configuration
- List of symbols used
- Run and epoch numbers

## Monitoring

The training will print:
- Data loading progress from each source
- GPU utilization stats
- Training metrics (loss, reward, etc.)
- Checkpoint saves
- Run completion summaries

## Stopping

Press `Ctrl+C` to stop. It will:
- Finish the current epoch gracefully
- Print summary statistics
- Save the final checkpoint

## Memory Optimization

The script automatically sets:
```python
PYTORCH_CUDA_ALLOC_CONF='max_split_size_mb:512'
```

This prevents memory fragmentation on A100s with massive batch sizes.

## Customization

Edit `continuous_training.py` to change:
- `envs_per_gpu`: More parallel environments (default: 512)
- `batch_size`: Larger batches for bigger GPUs (default: 65,536)
- `d_model`, `num_layers`: Model architecture
- `epochs`: How many epochs per run (default: 200)
- Date ranges for data fetching

## Data Sources Details

### Polygon S3
- Minute-level aggregated bars
- US stocks (SIP feed)
- Global crypto
- 2+ years of history

### Binance
- 1-minute klines
- Major pairs: BTC, ETH, BNB, SOL, ADA, DOT, AVAX
- DeFi: UNI, AAVE, LINK
- Layer-2: MATIC, ARB
- Memecoins: DOGE, SHIB

### Yahoo Finance
- 1-minute intraday data
- Mega caps: AAPL, MSFT, GOOGL, AMZN, NVDA, META, TSLA
- Finance: JPM, BAC, V, MA
- Tech: ORCL, INTC, CSCO, ADBE, CRM
- Retail: WMT
- Healthcare: UNH
- Consumer: PG, DIS

## Performance Expectations

On 4x A100 (40GB each):
- **~2048 parallel environments**
- **~50-100 episodes per second** (depends on episode length)
- **~1-2 hours per 200-epoch run** (depends on data size)
- **Continuous operation** (no human intervention needed)

## Troubleshooting

**"Unable to locate credentials"**
→ Add Polygon S3 credentials to `python_training/.env`

**"Out of memory"**
→ Reduce `envs_per_gpu` or `batch_size` in `continuous_training.py`

**"No data available"**
→ Check internet connection and API credentials

**Training too slow**
→ Increase `envs_per_gpu` or `batch_size` for better GPU utilization
