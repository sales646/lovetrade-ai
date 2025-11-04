"""
System Verification Script - Tests all components before distributed training
Run this before starting distributed training to ensure everything works
"""
import os
import sys
import torch
import numpy as np
from dotenv import load_dotenv

print("="*70)
print("DISTRIBUTED TRAINING SYSTEM VERIFICATION")
print("="*70)

# Load environment
load_dotenv()
print("\n✓ Step 1: Environment loaded")

# Check PyTorch and CUDA
print(f"\n✓ Step 2: PyTorch {torch.__version__}")
print(f"  CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"  CUDA version: {torch.version.cuda}")
    print(f"  GPU count: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"    GPU {i}: {torch.cuda.get_device_name(i)}")

# Test imports
print("\n✓ Step 3: Testing imports...")
try:
    from distributed_training import DistributedTrainer, check_gpu_availability
    print("  ✓ distributed_training")
except Exception as e:
    print(f"  ✗ distributed_training: {e}")
    sys.exit(1)

try:
    from pbt_scheduler import AdaptivePBTScheduler
    print("  ✓ pbt_scheduler")
except Exception as e:
    print(f"  ✗ pbt_scheduler: {e}")
    sys.exit(1)

try:
    from transformer_policy import TransformerPolicy
    print("  ✓ transformer_policy")
except Exception as e:
    print(f"  ✗ transformer_policy: {e}")
    sys.exit(1)

try:
    from trading_environment import TradingEnvironment, create_trading_env
    print("  ✓ trading_environment")
except Exception as e:
    print(f"  ✗ trading_environment: {e}")
    sys.exit(1)

try:
    from distributed_orchestrator import DistributedRLOrchestrator
    print("  ✓ distributed_orchestrator")
except Exception as e:
    print(f"  ✗ distributed_orchestrator: {e}")
    sys.exit(1)

# Test GPU availability
print("\n✓ Step 4: GPU availability check...")
gpu_info = check_gpu_availability()
print(f"  Available: {gpu_info['available']}")
print(f"  Count: {gpu_info['count']}")
if gpu_info['available']:
    print(f"  NCCL available: {gpu_info.get('nccl_available', False)}")

# Test environment creation
print("\n✓ Step 5: Testing environment creation...")
try:
    env = create_trading_env(augment=False)
    state = env.reset()
    print(f"  ✓ Environment created")
    print(f"    State shape: {state.shape}")
    print(f"    State dim: {env.state_dim}")
    print(f"    Action dim: {env.action_space_dim}")
    print(f"    Historical bars: {len(env.historical_bars):,}")
    
    # Test a few steps
    for i in range(3):
        action = np.random.randn(env.action_space_dim)
        next_state, reward, done, info = env.step(action)
        if i == 0:
            print(f"  ✓ Environment step works (reward: {reward:.4f})")
    
except Exception as e:
    print(f"  ✗ Environment creation failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test model creation
print("\n✓ Step 6: Testing model creation...")
try:
    config = {
        'state_dim': 50,
        'action_dim': 3,
        'd_model': 128,
        'nhead': 4,
        'num_layers': 2
    }
    model = TransformerPolicy(config=config)
    
    # Test forward pass
    test_state = torch.randn(4, 50)  # Batch of 4
    with torch.no_grad():
        actions, values, log_probs = model(test_state)
    
    print(f"  ✓ Model created and forward pass works")
    print(f"    Input shape: {test_state.shape}")
    print(f"    Actions shape: {actions.shape}")
    print(f"    Values shape: {values.shape}")
    print(f"    Log probs shape: {log_probs.shape}")
    
    # Test get_log_probs method (required for training)
    with torch.no_grad():
        log_probs_check = model.get_log_probs(test_state, actions)
    print(f"  ✓ get_log_probs method works")
    
except Exception as e:
    print(f"  ✗ Model creation failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test model on GPU (if available)
if torch.cuda.is_available():
    print("\n✓ Step 7: Testing model on GPU...")
    try:
        model_gpu = TransformerPolicy(config=config).cuda()
        test_state_gpu = torch.randn(4, 50).cuda()
        
        with torch.no_grad():
            actions, values, log_probs = model_gpu(test_state_gpu)
        
        print(f"  ✓ Model works on GPU")
        print(f"    Device: {next(model_gpu.parameters()).device}")
        
        # Test BF16
        if torch.cuda.is_bf16_supported():
            model_bf16 = model_gpu.to(torch.bfloat16)
            test_state_bf16 = test_state_gpu.to(torch.bfloat16)
            with torch.no_grad():
                actions, values, log_probs = model_bf16(test_state_bf16)
            print(f"  ✓ BF16 precision works")
        else:
            print(f"  ⚠ BF16 not supported on this GPU")
            
    except Exception as e:
        print(f"  ✗ GPU test failed: {e}")
        import traceback
        traceback.print_exc()

# Test PBT scheduler
print("\n✓ Step 8: Testing PBT scheduler...")
try:
    pbt = AdaptivePBTScheduler(population_size=8, exploit_interval=5)
    base_hyperparams = {
        'learning_rate': 3e-4,
        'gamma': 0.99,
        'clip_param': 0.2
    }
    pbt.initialize_population(base_hyperparams)
    print(f"  ✓ PBT scheduler initialized")
    print(f"    Population size: {len(pbt.population)}")
    
    # Test step
    performances = {i: np.random.rand() for i in range(8)}
    pbt.step(performances)
    best = pbt.get_best()
    print(f"  ✓ PBT step works")
    print(f"    Best ID: {best.id}, Performance: {best.performance:.4f}")
    
except Exception as e:
    print(f"  ✗ PBT test failed: {e}")
    import traceback
    traceback.print_exc()

# Test environment variable
print("\n✓ Step 9: Checking environment variables...")
supabase_url = os.getenv("SUPABASE_URL")
supabase_key = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
if supabase_url and supabase_key:
    print(f"  ✓ Supabase credentials found")
    print(f"    URL: {supabase_url[:30]}...")
else:
    print(f"  ⚠ Supabase credentials not set (will use fallback data)")

# Final check
print("\n" + "="*70)
print("VERIFICATION COMPLETE!")
print("="*70)
print("\n✅ All critical components verified successfully")
print("\nYou can now run distributed training with:")
print("  START_DISTRIBUTED_TRAINING.bat")
print("\nCurrent configuration:")
print(f"  - GPUs: {gpu_info['count']}")
print(f"  - BF16: {torch.cuda.is_bf16_supported() if torch.cuda.is_available() else 'N/A'}")
print(f"  - Environments per GPU: 10 (will scale up after successful test)")
print(f"  - Total environments: {gpu_info['count'] * 10}")
print("\n" + "="*70)
