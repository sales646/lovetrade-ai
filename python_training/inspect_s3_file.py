#!/usr/bin/env python3
"""Download and inspect an S3 data file to understand the schema"""

import os
import boto3
import gzip
import io
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

s3_client = boto3.client(
    's3',
    aws_access_key_id=os.getenv("POLYGON_S3_ACCESS_KEY"),
    aws_secret_access_key=os.getenv("POLYGON_S3_SECRET_KEY"),
    endpoint_url=os.getenv("POLYGON_S3_ENDPOINT")
)

bucket = os.getenv("POLYGON_S3_BUCKET", "flatfiles")

# Download a recent stock file
file_key = "us_stocks_sip/minute_aggs_v1/2024/01/2024-01-02.csv.gz"
print(f"📥 Downloading: {file_key}")

response = s3_client.get_object(Bucket=bucket, Key=file_key)
compressed_data = response['Body'].read()

print(f"✅ Downloaded {len(compressed_data):,} bytes (compressed)")

# Decompress
decompressed = gzip.decompress(compressed_data)
print(f"✅ Decompressed to {len(decompressed):,} bytes")

# Parse CSV
df = pd.read_csv(io.BytesIO(decompressed), nrows=100)
print(f"\n📊 Schema ({len(df)} rows shown):")
print(df.info())
print(f"\n📊 First 10 rows:")
print(df.head(10))
print(f"\n📊 Unique symbols in this sample: {df['ticker'].nunique() if 'ticker' in df.columns else 'N/A'}")
print(f"📊 Column names: {list(df.columns)}")
