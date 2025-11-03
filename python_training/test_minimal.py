"""
Minimal Test - Quick sanity check before distributed training
"""
import torch
import numpy as np
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

print("="*70)
print("MINIMAL SYSTEM TEST")
print("="*70)

# Test 1: PyTorch & CUDA
print("\n1. Testing PyTorch & CUDA...")
print(f"   PyTorch: {torch.__version__}")
print(f"   CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"   GPU count: {torch.cuda.device_count()}")
    print(f"   GPU 0: {torch.cuda.get_device_name(0)}")
else:
    print("   ⚠ No CUDA - training will be VERY slow")

# Test 2: Model
print("\n2. Testing model creation...")
try:
    from transformer_policy import TransformerPolicy
    
    config = {
        'state_dim': 50,
        'action_dim': 3,
        'd_model': 128,
        'nhead': 4,
        'num_layers': 2
    }
    
    model = TransformerPolicy(config=config)
    state = torch.randn(2, 50)  # Batch of 2
    
    with torch.no_grad():
        action, value, log_prob = model(state)
    
    print(f"   ✓ Model works")
    print(f"     Input: {state.shape}")
    print(f"     Action: {action.shape}")
    print(f"     Value: {value.shape}")
    
    # Test get_log_probs (critical for training)
    with torch.no_grad():
        lp = model.get_log_probs(state, action)
    print(f"   ✓ get_log_probs works: {lp.shape}")
    
except Exception as e:
    print(f"   ✗ Model test FAILED: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# Test 3: Environment
print("\n3. Testing trading environment...")
try:
    from trading_environment import create_trading_env
    
    env = create_trading_env(use_augmentation=False)
    state = env.reset()
    
    print(f"   ✓ Environment created")
    print(f"     State shape: {state.shape}")
    print(f"     Historical bars: {len(env.historical_bars):,}")
    
    # Test a step
    action = np.array([0.5, 0.1, 0.1])
    next_state, reward, done, info = env.step(action)
    
    print(f"   ✓ Environment step works")
    print(f"     Reward: {reward:.4f}")
    print(f"     Balance: ${info['balance']:.2f}")
    
except Exception as e:
    print(f"   ✗ Environment test FAILED: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# Test 4: GPU Training (if available)
if torch.cuda.is_available():
    print("\n4. Testing GPU training...")
    try:
        model_gpu = TransformerPolicy(config=config).cuda()
        state_gpu = torch.randn(2, 50).cuda()
        
        with torch.no_grad():
            action, value, log_prob = model_gpu(state_gpu)
        
        print(f"   ✓ GPU forward pass works")
        
        # Test BF16
        if torch.cuda.is_bf16_supported():
            model_bf16 = model_gpu.to(torch.bfloat16)
            state_bf16 = state_gpu.to(torch.bfloat16)
            with torch.no_grad():
                action, value, log_prob = model_bf16(state_bf16)
            print(f"   ✓ BF16 precision works")
        else:
            print(f"   ⚠ BF16 not supported (will be slower)")
            
    except Exception as e:
        print(f"   ✗ GPU test FAILED: {e}")
        import traceback
        traceback.print_exc()
        exit(1)

# Test 5: Distributed components
print("\n5. Testing distributed training components...")
try:
    from distributed_training import DistributedTrainer, check_gpu_availability
    from pbt_scheduler import AdaptivePBTScheduler
    
    gpu_info = check_gpu_availability()
    print(f"   ✓ GPU check: {gpu_info['count']} GPUs available")
    
    pbt = AdaptivePBTScheduler(population_size=4, exploit_interval=5)
    pbt.initialize_population({'learning_rate': 3e-4, 'gamma': 0.99})
    print(f"   ✓ PBT scheduler initialized")
    
except Exception as e:
    print(f"   ✗ Distributed components test FAILED: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

print("\n" + "="*70)
print("✅ ALL TESTS PASSED!")
print("="*70)
print("\nYou can now run distributed training:")
print("  START_DISTRIBUTED_TRAINING.bat")
print("\nOr run full verification:")
print("  python verify_system.py")
print("="*70)
