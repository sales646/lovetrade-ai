@echo off
echo ============================================================
echo   SIMPLE GPU TRAINING
echo ============================================================
echo.
echo Checking setup...
python check_setup.py
echo.
if errorlevel 1 (
    echo Setup check failed! Fix the issues above.
    pause
    exit /b 1
)
echo.
echo Starting training...
echo Press Ctrl+C to stop
echo ============================================================
echo.

python train.py

pause
