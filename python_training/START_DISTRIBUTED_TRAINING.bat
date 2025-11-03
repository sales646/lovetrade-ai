@echo off
echo ============================================================
echo   DISTRIBUTED RL TRAINING - 8 GPUs with PBT
echo ============================================================
echo.
echo Starting advanced distributed training with:
echo   - 8 GPUs with 80 parallel environments (10 per GPU)
echo   - Population-Based Training (PBT)
echo   - Transformer policy networks
echo   - BF16 mixed precision
echo   - Advanced profit-optimized rewards
echo   - Will scale up after successful test
echo.
echo Press Ctrl+C to stop training
echo ============================================================
echo.

python distributed_orchestrator.py

pause
