#!/usr/bin/env python3
"""Polygon S3 Flat Files Downloader - Bulk download for training"""

import os
import boto3
from pathlib import Path
from datetime import datetime
from typing import List, Dict
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

class PolygonS3Downloader:
    def __init__(self):
        self.access_key = os.getenv("POLYGON_S3_ACCESS_KEY")
        self.secret_key = os.getenv("POLYGON_S3_SECRET_KEY")
        self.endpoint = os.getenv("POLYGON_S3_ENDPOINT", "https://files.massive.com")
        self.bucket = os.getenv("POLYGON_S3_BUCKET", "flatfiles")
        
        if not self.access_key or not self.secret_key:
            raise ValueError("❌ Missing S3 credentials. Set POLYGON_S3_ACCESS_KEY and POLYGON_S3_SECRET_KEY")
        
        # Initialize S3 client
        self.s3 = boto3.client(
            's3',
            endpoint_url=self.endpoint,
            aws_access_key_id=self.access_key,
            aws_secret_access_key=self.secret_key
        )
        
        self.local_dir = Path("./polygon_data")
        self.local_dir.mkdir(exist_ok=True)
        
    def list_available_files(self, prefix: str = "") -> List[Dict]:
        """List all available files in S3 bucket"""
        print(f"🔍 Scanning S3 bucket: {self.bucket}/{prefix}")
        
        files = []
        paginator = self.s3.get_paginator('list_objects_v2')
        
        for page in paginator.paginate(Bucket=self.bucket, Prefix=prefix):
            if 'Contents' in page:
                for obj in page['Contents']:
                    # Filter for minute aggregates (stocks and crypto)
                    if 'us_stocks_sip/minute_aggs_v1' in obj['Key'] or 'crypto/minute_aggs_v1' in obj['Key']:
                        files.append({
                            'key': obj['Key'],
                            'size': obj['Size'],
                            'modified': obj['LastModified']
                        })
        
        print(f"✅ Found {len(files)} minute aggregate files")
        return files
    
    def download_file(self, s3_key: str) -> bool:
        """Download single file from S3"""
        try:
            # Create local path structure
            local_path = self.local_dir / s3_key
            local_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Skip if already exists
            if local_path.exists():
                return True
            
            # Download
            self.s3.download_file(self.bucket, s3_key, str(local_path))
            return True
            
        except Exception as e:
            print(f"❌ Failed to download {s3_key}: {e}")
            return False
    
    def download_bulk(self, file_list: List[Dict], max_workers: int = 8):
        """Download multiple files in parallel"""
        print(f"\n📥 Downloading {len(file_list)} files with {max_workers} workers...")
        
        downloaded = 0
        failed = 0
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(self.download_file, f['key']): f for f in file_list}
            
            with tqdm(total=len(file_list), desc="Downloading") as pbar:
                for future in as_completed(futures):
                    if future.result():
                        downloaded += 1
                    else:
                        failed += 1
                    pbar.update(1)
        
        print(f"\n✅ Downloaded: {downloaded}")
        print(f"❌ Failed: {failed}")
        
        # Save manifest
        manifest = {
            'downloaded_at': datetime.utcnow().isoformat(),
            'total_files': len(file_list),
            'successful': downloaded,
            'failed': failed,
            'files': [f['key'] for f in file_list]
        }
        
        with open(self.local_dir / "download_manifest.json", "w") as f:
            json.dump(manifest, f, indent=2)
    
    def download_latest_month(self, market: str = "stocks", symbols: List[str] = None):
        """Download latest month of data for specific symbols"""
        prefix = f"us_stocks_sip/minute_aggs_v1/" if market == "stocks" else "crypto/minute_aggs_v1/"
        
        all_files = self.list_available_files(prefix)
        
        # Filter by symbols if provided
        if symbols:
            filtered = []
            for file in all_files:
                for symbol in symbols:
                    if f"/{symbol}/" in file['key'] or file['key'].endswith(f"{symbol}.parquet"):
                        filtered.append(file)
                        break
            all_files = filtered
            print(f"📊 Filtered to {len(all_files)} files for symbols: {symbols}")
        
        # Sort by date and take latest month
        all_files.sort(key=lambda x: x['modified'], reverse=True)
        latest_month = all_files[:1000]  # Adjust based on needs
        
        self.download_bulk(latest_month)


def main():
    """Main execution"""
    print("🧠 PNU - Polygon S3 Data Downloader")
    print("=" * 50)
    
    downloader = PolygonS3Downloader()
    
    # Example: Download latest stocks and crypto
    print("\n1️⃣ Downloading latest stocks data...")
    downloader.download_latest_month(market="stocks", symbols=["AAPL", "TSLA", "NVDA", "SPY", "QQQ"])
    
    print("\n2️⃣ Downloading latest crypto data...")
    downloader.download_latest_month(market="crypto", symbols=["BTCUSD", "ETHUSD", "SOLUSD"])
    
    print("\n✅ Download complete! Data stored in:", downloader.local_dir)


if __name__ == "__main__":
    main()
