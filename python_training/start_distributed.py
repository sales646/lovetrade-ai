#!/usr/bin/env python3
"""
Distributed RL Training Launcher
Cross-platform Python script to start distributed GPU training
"""

import os
import sys
import subprocess
import platform

def print_header():
    print("=" * 70)
    print("  DISTRIBUTED RL TRAINING - Drop-in & Go")
    print("=" * 70)
    print()
    print("This script will:")
    print("  1. Install all dependencies")
    print("  2. Fetch real market data (Polygon S3 + Binance + Yahoo)")
    print("  3. Store data to Supabase")
    print("  4. Start distributed GPU training")
    print()
    print(f"Platform: {platform.system()} {platform.release()}")
    print(f"Python: {sys.version.split()[0]}")
    print()

def install_dependencies():
    print("[1/3] Installing Python dependencies...")
    print("-" * 70)
    
    requirements_files = [
        "requirements_production.txt",
        "requirements_distributed.txt"
    ]
    
    for req_file in requirements_files:
        if os.path.exists(req_file):
            print(f"Installing from {req_file}...")
            try:
                subprocess.run(
                    [sys.executable, "-m", "pip", "install", "-r", req_file],
                    check=True
                )
                print(f"✅ {req_file} installed")
            except subprocess.CalledProcessError as e:
                print(f"❌ Failed to install {req_file}: {e}")
                return False
        else:
            print(f"⚠️  {req_file} not found")
    
    print()
    return True

def check_gpu():
    print("[2/3] Checking GPU availability...")
    print("-" * 70)
    
    try:
        import torch
        
        cuda_available = torch.cuda.is_available()
        if cuda_available:
            gpu_count = torch.cuda.device_count()
            print(f"✅ CUDA available: {cuda_available}")
            print(f"✅ GPU count: {gpu_count}")
            
            for i in range(gpu_count):
                props = torch.cuda.get_device_properties(i)
                print(f"   GPU {i}: {props.name}")
                print(f"          Memory: {props.total_memory / 1e9:.1f} GB")
        else:
            print("⚠️  No GPU detected - training will use CPU (slow)")
            print("   Install CUDA: https://developer.nvidia.com/cuda-downloads")
    except ImportError:
        print("⚠️  PyTorch not installed yet, skipping GPU check")
    
    print()
    return True

def start_training():
    print("[3/3] Starting distributed training...")
    print("-" * 70)
    print()
    
    try:
        # Run distributed_orchestrator.py
        subprocess.run(
            [sys.executable, "distributed_orchestrator.py"],
            check=True
        )
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Training failed: {e}")
        return False
    except KeyboardInterrupt:
        print("\n\n⚠️  Training interrupted by user")
        return False
    
    return True

def main():
    print_header()
    
    # Ask for confirmation
    try:
        input("Press Enter to start...")
    except KeyboardInterrupt:
        print("\n\nCancelled by user")
        sys.exit(0)
    
    print()
    
    # Step 1: Install dependencies
    if not install_dependencies():
        print("\n❌ Dependency installation failed")
        sys.exit(1)
    
    # Step 2: Check GPU
    check_gpu()
    
    # Step 3: Start training
    if start_training():
        print()
        print("=" * 70)
        print("✅ Training complete! Check Supabase for metrics and /strategies page")
        print("=" * 70)
    else:
        print()
        print("=" * 70)
        print("❌ Training failed or was interrupted")
        print("=" * 70)
        sys.exit(1)

if __name__ == "__main__":
    main()
