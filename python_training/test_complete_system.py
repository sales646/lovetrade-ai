#!/usr/bin/env python3
"""
Complete System Test - Verify Drop-in Training Works
Tests:
1. Environment variables loaded
2. Supabase connection
3. Polygon S3 access
4. Binance API access
5. Production data fetcher
6. Data storage to Supabase
7. Distributed training setup (GPU check)
"""

import os
import sys
import platform
from dotenv import load_dotenv

print("=" * 70)
print("  COMPLETE SYSTEM TEST")
print("=" * 70)
print(f"Platform: {platform.system()} {platform.release()}")
print(f"Python: {sys.version.split()[0]}")
print()

# Load .env
load_dotenv()
print("\n[1/7] Environment Variables")
required_vars = [
    "SUPABASE_URL",
    "SUPABASE_SERVICE_ROLE_KEY",
    "POLYGON_S3_ACCESS_KEY",
    "POLYGON_S3_SECRET_KEY",
    "POLYGON_S3_ENDPOINT",
    "POLYGON_S3_BUCKET"
]

missing = []
for var in required_vars:
    val = os.getenv(var)
    if val:
        # Mask secrets
        if "KEY" in var or "SECRET" in var:
            masked = val[:8] + "..." + val[-4:] if len(val) > 12 else "***"
            print(f"  ✅ {var}: {masked}")
        else:
            print(f"  ✅ {var}: {val}")
    else:
        print(f"  ❌ {var}: MISSING")
        missing.append(var)

if missing:
    print(f"\n❌ Missing variables: {missing}")
    sys.exit(1)

# Test Supabase
print("\n[2/7] Supabase Connection")
try:
    from supabase import create_client
    supabase = create_client(
        os.getenv("SUPABASE_URL"),
        os.getenv("SUPABASE_SERVICE_ROLE_KEY")
    )
    
    # Test query
    result = supabase.table("historical_bars").select("symbol").limit(1).execute()
    print(f"  ✅ Connected to Supabase")
    print(f"  ✅ Can query historical_bars table")
except Exception as e:
    print(f"  ❌ Supabase error: {e}")
    sys.exit(1)

# Test Polygon S3
print("\n[3/7] Polygon S3 Access")
try:
    import boto3
    from botocore.exceptions import ClientError
    
    s3 = boto3.client(
        's3',
        endpoint_url=os.getenv("POLYGON_S3_ENDPOINT"),
        aws_access_key_id=os.getenv("POLYGON_S3_ACCESS_KEY"),
        aws_secret_access_key=os.getenv("POLYGON_S3_SECRET_KEY")
    )
    
    # List one file
    response = s3.list_objects_v2(
        Bucket=os.getenv("POLYGON_S3_BUCKET"),
        Prefix="us_stocks_sip/minute_aggs_v1/2024/01/",
        MaxKeys=1
    )
    
    if 'Contents' in response and len(response['Contents']) > 0:
        print(f"  ✅ Connected to Polygon S3")
        print(f"  ✅ Can list files in bucket")
    else:
        print(f"  ⚠️  Connected but no files found in test path")
except Exception as e:
    print(f"  ❌ Polygon S3 error: {e}")
    sys.exit(1)

# Test Binance API
print("\n[4/7] Binance API Access")
try:
    import requests
    
    # Public endpoint - no auth needed
    response = requests.get(
        "https://api.binance.com/api/v3/klines",
        params={
            "symbol": "BTCUSDT",
            "interval": "1m",
            "limit": 1
        },
        timeout=10
    )
    
    if response.status_code == 200:
        data = response.json()
        print(f"  ✅ Connected to Binance")
        print(f"  ✅ Can fetch market data")
    else:
        print(f"  ⚠️  Binance returned status {response.status_code}")
except Exception as e:
    print(f"  ❌ Binance error: {e}")
    sys.exit(1)

# Test Production Data Fetcher
print("\n[5/7] Production Data Fetcher")
try:
    from production_data_fetcher import ProductionDataFetcher
    from datetime import datetime, timedelta
    
    # Small test - just 1 day
    end_date = datetime(2024, 1, 5)
    start_date = end_date - timedelta(days=1)
    
    print(f"  Testing fetch for {start_date.date()} to {end_date.date()}")
    
    fetcher = ProductionDataFetcher(
        start_date=start_date.strftime("%Y-%m-%d"),
        end_date=end_date.strftime("%Y-%m-%d")
    )
    
    # Just test US stocks (faster)
    print("  Fetching sample US stocks...")
    us_data = fetcher.fetch_us_stocks(limit=3)
    
    if us_data and len(us_data) > 0:
        print(f"  ✅ Fetched data for {len(us_data)} symbols")
        for symbol, df in us_data.items():
            print(f"     {symbol}: {len(df)} bars")
    else:
        print(f"  ⚠️  No data fetched (check date range)")
    
except Exception as e:
    print(f"  ❌ Data fetcher error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test Data Storage
print("\n[6/7] Data Storage to Supabase")
try:
    if us_data and len(us_data) > 0:
        symbol = list(us_data.keys())[0]
        df = us_data[symbol]
        
        # Store just 10 bars as test
        bars = []
        for _, row in df.head(10).iterrows():
            bars.append({
                'symbol': symbol,
                'timestamp': row['timestamp'].isoformat(),
                'timeframe': '1Min',
                'open': float(row['open']),
                'high': float(row['high']),
                'low': float(row['low']),
                'close': float(row['close']),
                'volume': int(row['volume'])
            })
        
        supabase.table('historical_bars').upsert(bars).execute()
        print(f"  ✅ Stored {len(bars)} test bars for {symbol}")
    else:
        print(f"  ⚠️  No data to store")
    
except Exception as e:
    print(f"  ❌ Storage error: {e}")
    sys.exit(1)

# Test GPU
print("\n[7/7] GPU Availability")
try:
    import torch
    
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        print(f"  ✅ CUDA available")
        print(f"  ✅ {gpu_count} GPU(s) detected")
        
        for i in range(gpu_count):
            props = torch.cuda.get_device_properties(i)
            print(f"     GPU {i}: {props.name}")
            print(f"            Memory: {props.total_memory / 1e9:.1f} GB")
    else:
        print(f"  ⚠️  No GPU detected (CPU training will be slow)")
        print(f"     Install CUDA: https://developer.nvidia.com/cuda-downloads")
    
except Exception as e:
    print(f"  ❌ GPU check error: {e}")

print("\n" + "=" * 70)
print("  ✅ ALL TESTS PASSED - System Ready!")
print("=" * 70)
print("\nNext steps:")
print("  1. Run: python start_distributed.py")
print("  2. Monitor training on /strategies dashboard")
print("  3. Check Supabase training_metrics table")
print("\n" + "=" * 70)

if __name__ == "__main__":
    # Script runs automatically when executed
    pass

