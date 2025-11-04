@echo off
REM Complete Setup Script - Installs everything from scratch

echo ============================================================
echo   COMPLETE SETUP - Installing Everything
echo ============================================================

REM Check Python
echo.
echo [1/5] Checking Python...
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python not found! Install Python 3.9+ first
    pause
    exit /b 1
)
python --version
echo OK: Python found

REM Check pip
echo.
echo [2/5] Checking pip...
pip --version >nul 2>&1
if errorlevel 1 (
    echo Installing pip...
    python -m ensurepip --default-pip
)
pip --version
echo OK: pip found

REM Install PyTorch with CUDA
echo.
echo [3/5] Installing PyTorch with CUDA support...
echo    This may take a few minutes...
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
echo OK: PyTorch installed

REM Install other dependencies
echo.
echo [4/5] Installing other dependencies...
pip install numpy pandas boto3 python-dotenv supabase gymnasium tqdm
echo OK: Dependencies installed

REM Verify GPU
echo.
echo [5/5] Verifying GPU access...
python -c "import torch; print('CUDA Available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')"

echo.
echo ============================================================
echo OK: SETUP COMPLETE!
echo ============================================================
echo.
echo Now run: python train.py
pause
