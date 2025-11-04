@echo off
echo ================================================
echo PNU - Polygon S3 Data Downloader
echo ================================================
echo.

REM Install S3 dependencies
echo [1/2] Installing S3 dependencies...
pip install -r requirements_s3.txt

echo.
echo [2/2] Starting S3 download...
python fetch_polygon_s3.py

echo.
echo ================================================
echo Download complete!
echo Data stored in: ./polygon_data/
echo ================================================
pause
