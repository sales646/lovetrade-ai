# Troubleshooting Guide - Distributed Training

## Quick Diagnosis

Run this first to identify issues:
```bash
python verify_system.py
```

## Common Issues & Solutions

### 1. Workers Hang After "Starting worker on GPU X/8"

**Symptoms:**
- See "🚀 Starting worker on GPU X/8" but nothing after
- No progress for >1 minute
- May see TCPStore connection errors

**Root Cause:**
- Too many environments per GPU blocking distributed setup
- Workers trying to initialize before process group is ready

**Solution:**
Already fixed! Config set to 10 envs/GPU. If still hanging:

```python
# In distributed_orchestrator.py, line 57:
'envs_per_gpu': 5,  # Reduce to 5 or even 1 for testing
```

### 2. CUDA Out of Memory

**Symptoms:**
```
RuntimeError: CUDA out of memory
```

**Solutions:**

Option A: Reduce batch size
```python
# In distributed_orchestrator.py:
'batch_size': 128,  # Reduce from 256
```

Option B: Use lightweight model
```python
'model_type': 'lightweight',  # Instead of 'transformer'
```

Option C: Reduce environments
```python
'envs_per_gpu': 5,  # Reduce from 10
```

### 3. nvidia-smi Errors (Not Actual Errors!)

**Symptoms:**
```
⚠️ nvidia-smi error: Command ... returned non-zero exit status 2.
```

**What's happening:**
- These are from the monitoring system
- Training uses PyTorch CUDA directly, not nvidia-smi
- Safe to ignore - training continues normally

**Why they occur:**
- nvidia-smi might be busy or locked
- WSL2/virtualization environment
- Driver compatibility issues

**Impact:** NONE - Training works fine

### 4. No Historical Data / Empty Database

**Symptoms:**
```
⚠️ No historical bars found - generating fallback data
```

**What happens:**
- System automatically generates synthetic data
- Training proceeds normally with 1000 fallback bars
- You'll see: "✅ Generated 1000 fallback bars"

**To use real data:**
1. Ensure `.env` has correct Supabase credentials
2. Populate database with:
   ```bash
   python populate_data.py  # If available
   ```
3. Or use the web app to fetch historical data

### 5. Import Errors

**Symptoms:**
```
ModuleNotFoundError: No module named 'X'
```

**Solution:**
```bash
# Activate virtual environment
.venv\Scripts\activate

# Reinstall requirements
pip install -r requirements.txt

# For distributed training:
pip install -r requirements_distributed.txt
```

### 6. Process Group Timeout

**Symptoms:**
```
RuntimeError: [c10d] TCP client failed to connect/validate to host
```

**Causes:**
- Port already in use
- Firewall blocking communication
- Workers initializing too slowly

**Solutions:**

Option A: Let system find free port (already implemented)
- The code automatically finds free ports

Option B: Reduce environments to speed up init
```python
'envs_per_gpu': 1,  # Fastest initialization
```

Option C: Increase timeout (if needed)
```python
# In distributed_training.py, setup() method:
dist.init_process_group(
    self.backend, 
    rank=rank, 
    world_size=world_size,
    timeout=datetime.timedelta(minutes=10)  # Add this
)
```

### 7. BF16 Not Supported

**Symptoms:**
```
RuntimeError: torch.cuda.is_bf16_supported() returned False
```

**Solution:**
```python
# In distributed_orchestrator.py:
'use_bf16': False,  # Disable BF16
```

**Performance impact:** ~2x slower training

### 8. Training Extremely Slow

**Check these:**

1. **GPU Utilization**
   - Should be >80% during training
   - If <50%, increase batch_size or envs_per_gpu

2. **Thermal Throttling**
   ```
   🔥 Warning: Thermal throttling detected!
   ```
   - Improve cooling
   - Reduce clock speeds
   - Take breaks between training runs

3. **Data Loading**
   - If fallback data is used, it's slower
   - Populate real historical data for better performance

4. **Too Few Environments**
   - Scale up from 10 to 100 or 1000 per GPU
   - More environments = better GPU utilization

### 9. Training Loss is NaN

**Symptoms:**
```
Epoch X: Loss=nan, Reward=nan
```

**Causes & Solutions:**

Exploding gradients:
```python
# Already implemented in code:
torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
```

Bad learning rate:
```python
'learning_rate': 1e-4,  # Reduce from 3e-4
```

Bad reward scale:
```python
# In advanced_rewards.py, adjust reward weights
```

### 10. Checkpoint Loading Fails

**Symptoms:**
```
RuntimeError: Error loading checkpoint
```

**Solutions:**

Check file exists:
```bash
dir checkpoints\dist_run_*\checkpoint_*.pt
```

Load with correct config:
```python
checkpoint = torch.load('checkpoint.pt')
config = checkpoint['config']
model = TransformerPolicy(config=config)
```

## Performance Tuning

### For Maximum Speed

```python
{
    'world_size': 8,
    'envs_per_gpu': 1000,      # Maximum
    'use_bf16': True,          # 2x speedup
    'batch_size': 512,         # Large batches
    'model_type': 'lightweight', # Faster model
    'num_layers': 2,           # Fewer layers
}
```

### For Maximum Quality

```python
{
    'world_size': 8,
    'envs_per_gpu': 100,       # Moderate
    'use_bf16': True,
    'batch_size': 256,
    'model_type': 'transformer', # Full model
    'num_layers': 6,           # More layers
    'd_model': 512,            # Larger model
}
```

### For Stability (Testing)

```python
{
    'world_size': 1,           # Single GPU
    'envs_per_gpu': 10,        # Few envs
    'use_bf16': False,         # FP32
    'batch_size': 64,          # Small batches
    'learning_rate': 1e-4,     # Conservative
}
```

## Getting Help

1. **Run verification first:**
   ```bash
   python verify_system.py
   ```

2. **Check logs carefully:**
   - Look for the FIRST error, not subsequent errors
   - Errors cascade, fix the root cause

3. **Test components individually:**
   ```bash
   # Test environment only
   python trading_environment.py
   
   # Test model only
   python transformer_policy.py
   ```

4. **Check GPU status:**
   ```bash
   nvidia-smi
   ```

5. **Reduce scale for testing:**
   - Start with 1 GPU, 1 environment
   - Gradually scale up

## Debug Mode

Add more logging:

```python
# In distributed_training.py, add at top:
import logging
logging.basicConfig(level=logging.DEBUG)

# In train_worker method, add prints:
print(f"[GPU {rank}] DEBUG: Step {step}, State shape: {state.shape}")
```

## Emergency Recovery

If training is completely broken:

1. **Reset to defaults:**
   ```bash
   copy distributed_orchestrator.py distributed_orchestrator.py.backup
   # Edit distributed_orchestrator.py and set:
   'world_size': 1
   'envs_per_gpu': 1
   'use_bf16': False
   ```

2. **Test minimal setup:**
   ```bash
   python verify_system.py
   ```

3. **Gradually increase complexity:**
   - 1 GPU, 1 env → Works? 
   - 1 GPU, 10 envs → Works?
   - 2 GPUs, 10 envs → Works?
   - 8 GPUs, 10 envs → Works?
   - 8 GPUs, 100 envs → Works?

## Known Limitations

1. **Windows WSL2:** nvidia-smi may not work, but CUDA training works fine
2. **Virtual GPUs:** Some VM GPU passthrough has issues with DDP
3. **Mixed GPU types:** All GPUs should be same model for best performance
4. **Memory:** Each GPU needs ~4-8GB VRAM minimum

## Still Having Issues?

Create a minimal reproduction:

```python
# test_minimal.py
import torch
from transformer_policy import TransformerPolicy
from trading_environment import create_trading_env

# Test 1: GPU
print(f"CUDA: {torch.cuda.is_available()}")
print(f"GPUs: {torch.cuda.device_count()}")

# Test 2: Model
model = TransformerPolicy(config={'state_dim': 50, 'action_dim': 3})
state = torch.randn(1, 50)
action, value, log_prob = model(state)
print(f"Model works: {action.shape}")

# Test 3: Environment
env = create_trading_env(use_augmentation=False)
state = env.reset()
action = [0.5, 0.1, 0.1]
next_state, reward, done, info = env.step(action)
print(f"Environment works: reward={reward}")

print("All tests passed!")
```

Run:
```bash
python test_minimal.py
```
