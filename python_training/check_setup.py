#!/usr/bin/env python3
"""Verify everything is ready for training"""

import sys

print("="*60)
print("SETUP VERIFICATION")
print("="*60)

# Check 1: PyTorch
print("\n[1/4] Checking PyTorch...")
try:
    import torch
    print(f"✅ PyTorch {torch.__version__}")
except ImportError:
    print("❌ PyTorch not installed")
    print("   Install: pip install torch>=2.0.0")
    sys.exit(1)

# Check 2: CUDA/GPU
print("\n[2/4] Checking GPU...")
if not torch.cuda.is_available():
    print("❌ CUDA not available")
    print("   Your PyTorch installation doesn't have GPU support")
    print("   Reinstall: pip install torch --index-url https://download.pytorch.org/whl/cu121")
    sys.exit(1)

gpu_count = torch.cuda.device_count()
print(f"✅ {gpu_count} GPU(s) detected")
for i in range(gpu_count):
    name = torch.cuda.get_device_name(i)
    memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
    print(f"   GPU {i}: {name} ({memory:.1f} GB)")

# Check 3: Dependencies
print("\n[3/4] Checking dependencies...")
missing = []
for pkg in ['numpy', 'pandas', 'boto3', 'supabase', 'gymnasium', 'tqdm', 'dotenv']:
    try:
        __import__(pkg)
        print(f"✅ {pkg}")
    except ImportError:
        print(f"❌ {pkg}")
        missing.append(pkg)

if missing:
    print(f"\n❌ Missing packages: {', '.join(missing)}")
    print("   Install: pip install -r requirements_simple.txt")
    sys.exit(1)

# Check 4: Data access
print("\n[4/4] Checking data access...")
try:
    from dotenv import load_dotenv
    import os
    load_dotenv()
    
    supabase_url = os.getenv('SUPABASE_URL')
    if not supabase_url:
        print("❌ SUPABASE_URL not found in .env")
        sys.exit(1)
    
    print(f"✅ Environment configured")
    print(f"   Supabase: {supabase_url}")
    
except Exception as e:
    print(f"❌ Error: {e}")
    sys.exit(1)

print("\n" + "="*60)
print("✅ ALL CHECKS PASSED - READY TO TRAIN")
print("="*60)
print("\nRun: python train.py")
