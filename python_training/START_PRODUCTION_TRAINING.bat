@echo off
echo ====================================================================
echo   PRODUCTION TRAINING - Real Market Data Only
echo ====================================================================
echo.
echo This will train the RL agent on REAL data from:
echo   - Polygon S3 (Massive) for US Stocks
echo   - Binance for Crypto
echo   - Yahoo Finance for supplementary data + news sentiment
echo.
echo NO SIMULATIONS - Only actual market outcomes
echo.
pause

REM Install requirements
echo.
echo [1/2] Installing dependencies...
pip install -r requirements_production.txt

REM Run training
echo.
echo [2/2] Starting production training...
python production_train.py

echo.
echo ====================================================================
pause
