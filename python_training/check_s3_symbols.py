#!/usr/bin/env python3
"""Check what symbols are available in a specific year"""

import os
import boto3
from dotenv import load_dotenv

load_dotenv()

s3_client = boto3.client(
    's3',
    aws_access_key_id=os.getenv("POLYGON_S3_ACCESS_KEY"),
    aws_secret_access_key=os.getenv("POLYGON_S3_SECRET_KEY"),
    endpoint_url=os.getenv("POLYGON_S3_ENDPOINT")
)

bucket = os.getenv("POLYGON_S3_BUCKET", "flatfiles")

print("📊 Exploring S3 structure...\n")

# List actual files in stock path
prefix = "us_stocks_sip/minute_aggs_v1/2024/01/"
print(f"Checking: {prefix}")
response = s3_client.list_objects_v2(
    Bucket=bucket,
    Prefix=prefix,
    MaxKeys=20
)

if 'Contents' in response:
    print(f"✅ Found {len(response['Contents'])} files")
    for obj in response['Contents'][:10]:
        print(f"   {obj['Key']}")
else:
    print("❌ No files found")

# Try looking at actual file structure
print("\n📊 Sampling recent stock data files...\n")
prefix = "us_stocks_sip/minute_aggs_v1/"
response = s3_client.list_objects_v2(
    Bucket=bucket,
    Prefix=prefix,
    MaxKeys=50
)

if 'Contents' in response:
    files = [obj['Key'] for obj in response['Contents']]
    print(f"✅ Found {len(files)} sample files:")
    for f in files[:20]:
        print(f"   {f}")
        
    # Extract symbols from filenames
    symbols = set()
    for f in files:
        parts = f.split('/')
        if len(parts) > 3:
            # Files are like: us_stocks_sip/minute_aggs_v1/2024/01/AAPL.csv
            filename = parts[-1]
            if filename.endswith('.csv'):
                symbol = filename.replace('.csv', '')
                symbols.add(symbol)
    
    if symbols:
        print(f"\n✅ Extracted {len(symbols)} stock symbols:")
        print(f"   {', '.join(sorted(symbols)[:50])}")
else:
    print("❌ No files found")

# Check crypto
print("\n📊 Sampling crypto data files...\n")
prefix = "global_crypto/minute_aggs_v1/"
response = s3_client.list_objects_v2(
    Bucket=bucket,
    Prefix=prefix,
    MaxKeys=50
)

if 'Contents' in response:
    files = [obj['Key'] for obj in response['Contents']]
    print(f"✅ Found {len(files)} sample files:")
    for f in files[:20]:
        print(f"   {f}")
        
    # Extract symbols from filenames
    symbols = set()
    for f in files:
        parts = f.split('/')
        if len(parts) > 3:
            filename = parts[-1]
            if filename.endswith('.csv'):
                symbol = filename.replace('.csv', '')
                symbols.add(symbol)
    
    if symbols:
        print(f"\n✅ Extracted {len(symbols)} crypto pairs:")
        print(f"   {', '.join(sorted(symbols)[:50])}")
else:
    print("❌ No crypto files found")
