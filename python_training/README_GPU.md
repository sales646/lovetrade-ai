# GPU-Accelerated Training Setup

## Prerequisites

### 1. Install NVIDIA Drivers
Download and install the latest NVIDIA drivers for your GPU:
- **Windows/Linux**: https://www.nvidia.com/drivers
- Verify installation: `nvidia-smi`

### 2. Install CUDA Toolkit
Download CUDA Toolkit 12.1 (or 11.8):
- https://developer.nvidia.com/cuda-downloads
- Follow the installation wizard for your OS

### 3. Verify CUDA Installation
```bash
nvcc --version
nvidia-smi
```

## Installation

### 1. Create Virtual Environment
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate
```

### 2. Install PyTorch with CUDA Support

**For CUDA 12.1:**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

**For CUDA 11.8:**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### 3. Install Other Dependencies
```bash
pip install -r requirements.txt
```

## Verify GPU Setup

Run the GPU check script:
```bash
python check_gpu.py
```

Expected output:
```
============================================================
GPU Configuration Check
============================================================

✓ PyTorch version: 2.x.x+cu121
✓ CUDA available: True
✓ Number of GPUs: 1

  GPU 0: NVIDIA GeForce RTX 4090
    Memory: 24.00 GB

✓ Current CUDA device: 0
✓ GPU tensor operations working!
  Test tensor device: cuda:0

============================================================
✓ GPU is ready for training!
============================================================
```

## Training with GPU

The training scripts automatically use GPU when available:

```bash
# Single training run (uses GPU automatically)
python train_rl_policy.py

# Continuous parallel training
python run_continuous_training.py
```

## Performance Tips

### 1. Increase Batch Sizes
With GPU, you can use larger batches:
- BC batch size: 64-128 (default: 64)
- PPO batch size: 128-256 (default: 128)

### 2. Monitor GPU Usage
```bash
# Real-time monitoring
watch -n 1 nvidia-smi

# Or use gpustat
gpustat -i 1
```

### 3. Memory Management
If you get out-of-memory errors:
- Reduce batch sizes
- Reduce hidden layer dimensions
- Enable gradient accumulation

### 4. Multi-GPU Training
For multiple GPUs, modify `TrainingConfig`:
```python
device: str = "cuda:0"  # Use specific GPU
```

## Troubleshooting

### CUDA Out of Memory
```python
# Add to train_rl_policy.py
torch.cuda.empty_cache()
```

### CUDA Not Available
1. Check NVIDIA drivers: `nvidia-smi`
2. Check PyTorch installation: `python check_gpu.py`
3. Reinstall PyTorch with correct CUDA version

### Slow Training
1. Verify GPU is being used: `nvidia-smi` during training
2. Check GPU utilization (should be >80%)
3. Increase batch sizes if GPU memory allows

## Expected Performance Improvements

With a modern GPU (e.g., RTX 3080/4090):
- **BC Training**: 5-10x faster
- **PPO Training**: 3-5x faster
- **Overall**: Training completes in minutes instead of hours

## Environment Variables

Optional: Force CPU or specific GPU
```bash
# Force CPU
export CUDA_VISIBLE_DEVICES=-1

# Use specific GPU
export CUDA_VISIBLE_DEVICES=0

# Use multiple GPUs
export CUDA_VISIBLE_DEVICES=0,1
```
