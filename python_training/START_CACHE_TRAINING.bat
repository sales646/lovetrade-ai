@echo off
echo ============================================================
echo   TRAIN FROM CACHED DATA
echo ============================================================
echo.
echo Starting RL training from cached market data:
echo   - Reads from .cache_market/symbols/
echo   - Trains Transformer policy with PPO
echo   - Saves metrics to Supabase in real-time
echo   - Saves trained model to Supabase storage
echo.
echo Press Ctrl+C to stop training
echo ============================================================
echo.

python train_from_cache.py

pause
