#!/usr/bin/env python3
"""Pre-load and cache market data to avoid repeated database queries"""

import os
import pickle
from supabase import create_client
from dotenv import load_dotenv

load_dotenv()

def preload_market_data():
    """Load all market data once and save to disk"""
    print("="*70)
    print("PRE-LOADING MARKET DATA FOR TRAINING")
    print("="*70)
    
    supabase_url = os.getenv("SUPABASE_URL")
    supabase_key = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
    
    supabase = create_client(supabase_url, supabase_key)
    
    symbols = ["AAPL", "MSFT", "GOOGL", "TSLA", "NVDA", "AMZN"]
    timeframe = "5m"
    
    print(f"\n📊 Loading historical bars for {symbols}...")
    all_bars = []
    
    for symbol in symbols:
        print(f"   {symbol}...", end=" ")
        response = supabase.table("historical_bars") \
            .select("*") \
            .eq("symbol", symbol) \
            .eq("timeframe", timeframe) \
            .order("timestamp", desc=False) \
            .limit(500000) \
            .execute()
        
        if response.data:
            all_bars.extend(response.data)
            print(f"✅ {len(response.data)} bars")
    
    historical_bars = sorted(all_bars, key=lambda x: x['timestamp'])
    
    print(f"\n📈 Loading technical indicators...")
    indicators = {}
    
    for symbol in symbols:
        print(f"   {symbol}...", end=" ")
        response = supabase.table("technical_indicators") \
            .select("*") \
            .eq("symbol", symbol) \
            .eq("timeframe", timeframe) \
            .order("timestamp", desc=False) \
            .limit(50000) \
            .execute()
        
        if response.data:
            indicators[symbol] = response.data
            print(f"✅ {len(response.data)} indicators")
    
    # Save to disk
    cache_file = "python_training/.market_data_cache.pkl"
    print(f"\n💾 Saving to {cache_file}...")
    
    with open(cache_file, 'wb') as f:
        pickle.dump({
            'bars': historical_bars,
            'indicators': indicators,
            'symbols': symbols,
            'timeframe': timeframe
        }, f)
    
    print(f"✅ Saved {len(historical_bars):,} bars and {len(indicators)} indicator sets")
    print(f"   File size: {os.path.getsize(cache_file) / 1024 / 1024:.1f} MB")
    print("\n" + "="*70)
    print("✅ DATA PRELOAD COMPLETE")
    print("="*70)
    
    return cache_file

if __name__ == "__main__":
    preload_market_data()
