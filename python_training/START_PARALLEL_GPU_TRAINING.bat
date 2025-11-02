@echo off
REM Parallel GPU training - runs 5 training processes simultaneously
REM Each process will train independently and log to Supabase

echo ========================================
echo Parallel GPU Training (5 Workers)
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

echo Starting 5 parallel training workers...
echo Each worker will run continuously and restart after completion
echo Press Ctrl+C to stop all workers
echo.
echo Results will appear in your training dashboard at:
echo https://7b040b8f-dffe-48c1-aedd-fecc6dccf027.lovableproject.com/training
echo.

python run_continuous_training.py

pause
