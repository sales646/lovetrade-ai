#!/usr/bin/env python3
"""
GPU verification script
Run this to check if PyTorch can access your GPU
"""

import torch
import sys

def check_gpu():
    print("=" * 60)
    print("GPU Configuration Check")
    print("=" * 60)
    
    # Check PyTorch version
    print(f"\n✓ PyTorch version: {torch.__version__}")
    
    # Check CUDA availability
    cuda_available = torch.cuda.is_available()
    print(f"\n{'✓' if cuda_available else '✗'} CUDA available: {cuda_available}")
    
    if not cuda_available:
        print("\n⚠️  No CUDA support detected!")
        print("\nTo enable GPU training:")
        print("1. Install NVIDIA drivers: https://www.nvidia.com/drivers")
        print("2. Install CUDA Toolkit: https://developer.nvidia.com/cuda-downloads")
        print("3. Reinstall PyTorch with CUDA:")
        print("   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121")
        return False
    
    # GPU details
    gpu_count = torch.cuda.device_count()
    print(f"✓ Number of GPUs: {gpu_count}")
    
    for i in range(gpu_count):
        print(f"\n  GPU {i}: {torch.cuda.get_device_name(i)}")
        print(f"    Memory: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.2f} GB")
    
    # Current device
    current_device = torch.cuda.current_device()
    print(f"\n✓ Current CUDA device: {current_device}")
    
    # Test tensor operation
    try:
        x = torch.randn(1000, 1000).cuda()
        y = torch.randn(1000, 1000).cuda()
        z = torch.matmul(x, y)
        print("\n✓ GPU tensor operations working!")
        print(f"  Test tensor device: {z.device}")
    except Exception as e:
        print(f"\n✗ GPU test failed: {e}")
        return False
    
    print("\n" + "=" * 60)
    print("✓ GPU is ready for training!")
    print("=" * 60)
    return True

if __name__ == "__main__":
    success = check_gpu()
    sys.exit(0 if success else 1)
