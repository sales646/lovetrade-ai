@echo off
echo ============================================================
echo   CONTINUOUS TRAINING - 4x A100 Maximum Power
echo ============================================================
echo.
echo Configuration:
echo   - 4 GPUs with 512 envs each = 2048 parallel environments
echo   - Large Transformer: 2048-dim, 12 layers, 32 heads
echo   - Batch size: 65,536
echo   - Data sources: Polygon S3 + Binance + Yahoo Finance
echo   - Auto-repeats when complete
echo.
echo GPU Check:
python -c "import torch; print(f'  CUDA: {torch.cuda.is_available()}'); print(f'  GPUs: {torch.cuda.device_count()}'); [print(f'  GPU {i}: {torch.cuda.get_device_name(i)}') for i in range(torch.cuda.device_count())]"
echo.
echo Press Ctrl+C to stop training
echo ============================================================
echo.

python continuous_training.py

pause
