@echo off
REM Windows batch script to run continuous parallel training
REM Press Ctrl+C to stop

echo ========================================
echo Continuous Training Runner
echo ========================================
echo.
echo Starting 5 parallel training workers...
echo Press Ctrl+C to stop
echo.

python run_continuous_training.py

pause
