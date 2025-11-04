# Fresh Start - Simple GPU Training

This is a complete rebuild of the training system - simple, clear, and working.

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements_simple.txt

# 2. Verify GPU
python check_setup.py

# 3. Start training
python train.py
```

## What Changed

- ✅ Single, simple training script
- ✅ Clear progress at every step
- ✅ Works with 1 GPU (scales to 4 later)
- ✅ No complex orchestration
- ✅ Fast data loading
- ✅ Proper error handling

## Files

- `requirements_simple.txt` - Minimal dependencies
- `check_setup.py` - Verify GPU and data access
- `train.py` - Main training script
- `START.bat` - Windows launcher
