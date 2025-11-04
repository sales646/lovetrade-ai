#!/usr/bin/env python3
"""Check what's actually in Polygon S3 bucket"""

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

print("=" * 70)
print("🔍 Checking Polygon S3 Bucket Structure")
print("=" * 70)
print(f"\nBucket: {bucket}")
print(f"Endpoint: {os.getenv('POLYGON_S3_ENDPOINT')}\n")

# List top-level prefixes
response = s3_client.list_objects_v2(
    Bucket=bucket,
    Delimiter='/',
    MaxKeys=100
)

if 'CommonPrefixes' in response:
    print("📂 Top-level folders:")
    for prefix in response['CommonPrefixes']:
        print(f"   {prefix['Prefix']}")
    
    # Check what's inside us_stocks_sip
    print("\n📂 Inside us_stocks_sip/:")
    response = s3_client.list_objects_v2(
        Bucket=bucket,
        Prefix='us_stocks_sip/',
        Delimiter='/',
        MaxKeys=100
    )
    
    if 'CommonPrefixes' in response:
        for prefix in response['CommonPrefixes']:
            print(f"   {prefix['Prefix']}")
    
    # Check crypto folder if it exists
    print("\n📂 Inside crypto/:")
    try:
        response = s3_client.list_objects_v2(
            Bucket=bucket,
            Prefix='crypto/',
            Delimiter='/',
            MaxKeys=100
        )
        
        if 'CommonPrefixes' in response:
            for prefix in response['CommonPrefixes']:
                print(f"   {prefix['Prefix']}")
        else:
            print("   (empty or doesn't exist)")
    except:
        print("   (doesn't exist)")
    
    # Sample: list a few stock symbols
    print("\n📊 Sample stock symbols in minute_aggs_v1:")
    response = s3_client.list_objects_v2(
        Bucket=bucket,
        Prefix='us_stocks_sip/minute_aggs_v1/',
        Delimiter='/',
        MaxKeys=20
    )
    
    if 'CommonPrefixes' in response:
        for i, prefix in enumerate(response['CommonPrefixes'][:10]):
            symbol = prefix['Prefix'].split('/')[-2]
            print(f"   {symbol}")
        
        total = len(response['CommonPrefixes'])
        if total > 10:
            print(f"   ... and {total - 10} more")
    
else:
    print("❌ No data found or access denied")

print("\n" + "=" * 70)
