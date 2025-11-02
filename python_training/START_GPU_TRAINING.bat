@echo off
REM Quick start script for GPU-accelerated training
REM This will train a single policy using your GPU

echo ========================================
echo GPU-Accelerated RL Training
echo ========================================
echo.
echo Checking GPU availability...
python -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"
echo.

if errorlevel 1 (
    echo ERROR: PyTorch not found! Please install: pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
    pause
    exit /b 1
)

echo Starting GPU training...
echo Press Ctrl+C to stop
echo.

python train_rl_policy.py

pause
