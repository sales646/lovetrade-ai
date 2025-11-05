#!/usr/bin/env python3
"""
Comprehensive Data Source Testing
Tests each data source independently before training
"""

import os
import sys
from datetime import datetime, timedelta
from dotenv import load_dotenv

load_dotenv()

def test_polygon_s3():
    """Test Polygon S3 access and data fetching"""
    print("\n" + "="*70)
    print("TEST 1: POLYGON S3 (Massive)")
    print("="*70)
    
    try:
        import boto3
        from botocore.exceptions import ClientError
        
        # Check credentials
        access_key = os.getenv("POLYGON_S3_ACCESS_KEY")
        secret_key = os.getenv("POLYGON_S3_SECRET_KEY")
        endpoint = os.getenv("POLYGON_S3_ENDPOINT")
        bucket = os.getenv("POLYGON_S3_BUCKET")
        
        print(f"Access Key: {access_key[:8]}...{access_key[-4:]}")
        print(f"Endpoint: {endpoint}")
        print(f"Bucket: {bucket}")
        
        # Test connection
        s3 = boto3.client(
            's3',
            endpoint_url=endpoint,
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key
        )
        
        # Test listing files (try recent date)
        test_date = datetime.now() - timedelta(days=7)
        year = test_date.strftime("%Y")
        month = test_date.strftime("%m")
        
        prefix = f"us_stocks_sip/minute_aggs_v1/{year}/{month}/"
        print(f"\nTesting prefix: {prefix}")
        
        response = s3.list_objects_v2(
            Bucket=bucket,
            Prefix=prefix,
            MaxKeys=5
        )
        
        if 'Contents' in response and len(response['Contents']) > 0:
            print(f"✅ Connected to Polygon S3")
            print(f"✅ Found {len(response['Contents'])} files")
            for obj in response['Contents'][:3]:
                print(f"   - {obj['Key']} ({obj['Size'] / 1024:.1f} KB)")
            
            # Try downloading one file
            test_file = response['Contents'][0]['Key']
            print(f"\nTesting download: {test_file}")
            
            import gzip
            import io
            import pandas as pd
            
            obj_response = s3.get_object(Bucket=bucket, Key=test_file)
            compressed = obj_response['Body'].read()
            decompressed = gzip.decompress(compressed)
            df = pd.read_csv(io.BytesIO(decompressed))
            
            print(f"✅ Downloaded and parsed file")
            print(f"   Columns: {list(df.columns)}")
            print(f"   Rows: {len(df):,}")
            print(f"   Sample data:")
            print(df.head(2))
            
            return True
        else:
            print(f"⚠️  No files found at {prefix}")
            print("   Trying different date ranges...")
            
            # Try older dates
            for days_back in [30, 90, 180, 365]:
                test_date = datetime.now() - timedelta(days=days_back)
                year = test_date.strftime("%Y")
                month = test_date.strftime("%m")
                prefix = f"us_stocks_sip/minute_aggs_v1/{year}/{month}/"
                
                response = s3.list_objects_v2(Bucket=bucket, Prefix=prefix, MaxKeys=1)
                if 'Contents' in response and len(response['Contents']) > 0:
                    print(f"✅ Found data at {prefix}")
                    return True
            
            print("❌ No data found in any tested date range")
            return False
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_binance():
    """Test Binance API access"""
    print("\n" + "="*70)
    print("TEST 2: BINANCE CRYPTO")
    print("="*70)
    
    try:
        import requests
        
        # Test public API endpoint
        url = "https://api.binance.com/api/v3/klines"
        params = {
            "symbol": "BTCUSDT",
            "interval": "1m",
            "limit": 5
        }
        
        print(f"Testing: {url}")
        response = requests.get(url, params=params, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Binance API connected")
            print(f"✅ Retrieved {len(data)} candles for BTCUSDT")
            print(f"   Sample: {data[0]}")
            
            # Test historical data download
            print("\nTesting historical data download...")
            test_url = "https://data.binance.vision/data/spot/monthly/klines/BTCUSDT/1m/BTCUSDT-1m-2024-01.zip"
            
            response = requests.head(test_url, timeout=10)
            if response.status_code == 200:
                print(f"✅ Historical data archive accessible")
                print(f"   Size: {int(response.headers.get('content-length', 0)) / 1024 / 1024:.1f} MB")
            else:
                print(f"⚠️  Historical archive returned {response.status_code}")
            
            return True
        else:
            print(f"❌ Binance returned {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_yahoo_finance():
    """Test Yahoo Finance access"""
    print("\n" + "="*70)
    print("TEST 3: YAHOO FINANCE")
    print("="*70)
    
    try:
        import yfinance as yf
        
        # Test stock data
        print("Testing stock: AAPL")
        ticker = yf.Ticker("AAPL")
        
        # Try recent 7 days (minute data limited by Yahoo)
        end = datetime.now()
        start = end - timedelta(days=7)
        
        print(f"Fetching minute data: {start.date()} to {end.date()}")
        df_minute = ticker.history(start=start, end=end, interval="1m")
        
        if not df_minute.empty:
            print(f"✅ Minute data retrieved: {len(df_minute):,} bars")
            print(f"   Columns: {list(df_minute.columns)}")
            print(f"   Date range: {df_minute.index[0]} to {df_minute.index[-1]}")
            print(f"   Sample:")
            print(df_minute.head(2))
        else:
            print(f"⚠️  No minute data (trying daily...)")
        
        # Test daily data (longer history)
        print(f"\nFetching daily data: 2020-01-01 to {end.date()}")
        df_daily = ticker.history(start="2020-01-01", end=end, interval="1d")
        
        if not df_daily.empty:
            print(f"✅ Daily data retrieved: {len(df_daily):,} bars")
            print(f"   Date range: {df_daily.index[0]} to {df_daily.index[-1]}")
            return True
        else:
            print(f"❌ No daily data retrieved")
            return False
            
    except ImportError:
        print("❌ yfinance not installed")
        print("   Install: pip install yfinance")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_supabase():
    """Test Supabase connection"""
    print("\n" + "="*70)
    print("TEST 4: SUPABASE DATABASE")
    print("="*70)
    
    try:
        from supabase import create_client
        
        url = os.getenv("SUPABASE_URL")
        key = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
        
        print(f"URL: {url}")
        print(f"Key: {key[:8]}...{key[-4:]}")
        
        supabase = create_client(url, key)
        
        # Test query
        result = supabase.table("historical_bars").select("symbol").limit(1).execute()
        
        print(f"✅ Connected to Supabase")
        print(f"✅ Can query historical_bars table")
        
        # Check row count
        count_result = supabase.table("historical_bars").select("id", count="exact").limit(1).execute()
        if hasattr(count_result, 'count'):
            print(f"   Current rows: {count_result.count:,}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    print("="*70)
    print("  COMPREHENSIVE DATA SOURCE TESTING")
    print("="*70)
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    results = {
        "Polygon S3": test_polygon_s3(),
        "Binance": test_binance(),
        "Yahoo Finance": test_yahoo_finance(),
        "Supabase": test_supabase()
    }
    
    print("\n" + "="*70)
    print("  TEST SUMMARY")
    print("="*70)
    
    for source, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status}  {source}")
    
    all_passed = all(results.values())
    
    print("\n" + "="*70)
    if all_passed:
        print("✅ ALL TESTS PASSED - Ready for training!")
    else:
        print("⚠️  SOME TESTS FAILED - Fix issues before training")
    print("="*70)
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
