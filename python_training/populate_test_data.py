#!/usr/bin/env python3
"""Populate Supabase with test historical data for training"""

import os
import requests
from datetime import datetime, timedelta
from dotenv import load_dotenv

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")

# Test with a few liquid symbols
SYMBOLS = ["AAPL", "MSFT", "GOOGL", "NVDA", "TSLA", "BTCUSD", "ETHUSD"]

def populate_via_edge_function(symbol: str, days: int = 30):
    """Call the populate-historical-data edge function"""
    url = f"{SUPABASE_URL}/functions/v1/populate-historical-data"
    
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    
    payload = {
        "symbol": symbol,
        "start_date": start_date.strftime("%Y-%m-%d"),
        "end_date": end_date.strftime("%Y-%m-%d"),
        "timeframe": "1Min"
    }
    
    headers = {
        "Authorization": f"Bearer {SUPABASE_KEY}",
        "Content-Type": "application/json"
    }
    
    print(f"📊 Fetching {symbol} data ({days} days)...")
    
    try:
        response = requests.post(url, json=payload, headers=headers, timeout=300)
        
        if response.status_code == 200:
            result = response.json()
            print(f"   ✅ {result.get('bars_inserted', 0)} bars inserted")
            return True
        else:
            print(f"   ❌ Error: {response.status_code} - {response.text}")
            return False
    except Exception as e:
        print(f"   ❌ Exception: {e}")
        return False

def main():
    print("=" * 70)
    print("🧠 PNU - Populate Training Data")
    print("=" * 70)
    print(f"\nPopulating {len(SYMBOLS)} symbols with 30 days of 1-minute data...")
    print("This will take several minutes...\n")
    
    success_count = 0
    
    for symbol in SYMBOLS:
        if populate_via_edge_function(symbol, days=30):
            success_count += 1
    
    print("\n" + "=" * 70)
    print(f"✅ Successfully populated {success_count}/{len(SYMBOLS)} symbols")
    print("=" * 70)
    print("\nNow run: python quick_train.py")

if __name__ == "__main__":
    main()
