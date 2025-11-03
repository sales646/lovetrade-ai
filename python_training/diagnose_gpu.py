#!/usr/bin/env python3
"""GPU Diagnostics - Check if PyTorch can see and use GPUs"""

import torch
import torch.distributed as dist

print("="*70)
print("GPU DIAGNOSTICS")
print("="*70)

print("\n1. CUDA Availability:")
print(f"   torch.cuda.is_available(): {torch.cuda.is_available()}")
print(f"   torch.cuda.device_count(): {torch.cuda.device_count()}")

if torch.cuda.is_available():
    print("\n2. GPU Details:")
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        print(f"\n   GPU {i}:")
        print(f"     Name: {props.name}")
        print(f"     Memory: {props.total_memory / 1024**3:.1f} GB")
        print(f"     Compute Capability: {props.major}.{props.minor}")
        print(f"     Multi Processors: {props.multi_processor_count}")
    
    print("\n3. Testing GPU Tensor Creation:")
    try:
        for i in range(torch.cuda.device_count()):
            torch.cuda.set_device(i)
            x = torch.randn(1000, 1000).cuda(i)
            y = x @ x
            print(f"   ✅ GPU {i}: Successfully created and multiplied tensors")
            del x, y
            torch.cuda.empty_cache()
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    print("\n4. NCCL Backend:")
    print(f"   NCCL Available: {dist.is_nccl_available()}")
    
    print("\n5. Current CUDA Device:")
    print(f"   Current device: {torch.cuda.current_device()}")
    print(f"   Device name: {torch.cuda.get_device_name(torch.cuda.current_device())}")
    
    print("\n6. Memory Info:")
    for i in range(torch.cuda.device_count()):
        torch.cuda.set_device(i)
        allocated = torch.cuda.memory_allocated(i) / 1024**3
        reserved = torch.cuda.memory_reserved(i) / 1024**3
        total = torch.cuda.get_device_properties(i).total_memory / 1024**3
        print(f"   GPU {i}: {allocated:.2f}GB allocated / {reserved:.2f}GB reserved / {total:.1f}GB total")

else:
    print("\n❌ CUDA is NOT available!")
    print("\nPossible issues:")
    print("  1. PyTorch was installed without CUDA support")
    print("  2. CUDA drivers not installed correctly")
    print("  3. GPU not detected by system")
    print("\nTo fix:")
    print("  - Check: nvidia-smi")
    print("  - Reinstall PyTorch with CUDA: pip install torch --index-url https://download.pytorch.org/whl/cu121")

print("\n" + "="*70)
