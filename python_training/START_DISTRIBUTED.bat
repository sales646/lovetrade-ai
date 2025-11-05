@echo off
echo ======================================================================
echo   DISTRIBUTED RL TRAINING - Drop-in ^& Go
echo ======================================================================
echo.
echo This script will:
echo   1. Install all dependencies
echo   2. Fetch real market data (Polygon S3 + Binance + Yahoo)
echo   3. Store data to Supabase
echo   4. Start distributed GPU training
echo.
echo Requirements: NVIDIA GPUs with CUDA
echo.
pause

REM Install dependencies
echo.
echo [1/3] Installing Python dependencies...
pip install -r requirements_production.txt
pip install -r requirements_distributed.txt

REM Check GPU
echo.
echo [2/3] Checking GPU availability...
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}'); print(f'GPUs: {torch.cuda.device_count()}')"

REM Start training
echo.
echo [3/3] Starting distributed training...
echo.
python distributed_orchestrator.py

echo.
echo ======================================================================
echo Training complete! Check Supabase for metrics and Strategies page
echo ======================================================================
pause
