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
year = "2024"
month = "01"  # January 2024

print(f"📊 Checking symbols in {year}/{month}...\n")

# Check stocks - need to go into month folder
prefix = f"us_stocks_sip/minute_aggs_v1/{year}/{month}/"
response = s3_client.list_objects_v2(
    Bucket=bucket,
    Prefix=prefix,
    Delimiter='/',
    MaxKeys=1000
)

if 'CommonPrefixes' in response:
    symbols = [p['Prefix'].split('/')[-2] for p in response['CommonPrefixes']]
    print(f"✅ Found {len(symbols)} stock symbols in {year}/{month}")
    print(f"First 50: {', '.join(symbols[:50])}")
else:
    print("❌ No stock symbols found")

# Check crypto - need to go into month folder
print(f"\n📊 Checking crypto in {year}/{month}...\n")
prefix = f"global_crypto/minute_aggs_v1/{year}/{month}/"
response = s3_client.list_objects_v2(
    Bucket=bucket,
    Prefix=prefix,
    Delimiter='/',
    MaxKeys=1000
)

if 'CommonPrefixes' in response:
    symbols = [p['Prefix'].split('/')[-2] for p in response['CommonPrefixes']]
    print(f"✅ Found {len(symbols)} crypto pairs in {year}/{month}")
    print(f"First 50: {', '.join(symbols[:50])}")
else:
    print("❌ No crypto symbols found")
