#!/usr/bin/env python3
"""Automatic data discovery from all available sources"""

import os
from pathlib import Path
from typing import Dict, List, Set, Tuple
from datetime import datetime
import json

def discover_symbols_from_supabase() -> Dict[str, List[str]]:
    """Discover symbols from Supabase historical_bars table"""
    from supabase import create_client
    
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
    
    if not url or not key:
        print("⚠️  Supabase credentials not found")
        return {"stocks": [], "crypto": []}
    
    supabase = create_client(url, key)
    
    # Get unique symbols
    response = supabase.table("historical_bars").select("symbol").execute()
    
    symbols = set(row["symbol"] for row in response.data)
    
    # Classify as stock or crypto
    stocks = []
    crypto = []
    
    for symbol in sorted(symbols):
        # Crypto typically ends with USD, USDT, or are known pairs
        if any(symbol.endswith(suffix) for suffix in ["USD", "USDT", "BUSD", "BTC"]):
            crypto.append(symbol)
        else:
            stocks.append(symbol)
    
    print(f"📊 Discovered {len(stocks)} stocks, {len(crypto)} crypto from Supabase")
    return {"stocks": stocks, "crypto": crypto}


def discover_symbols_from_polygon_files(polygon_dir: str = "./polygon_data") -> Dict[str, List[str]]:
    """Discover symbols from Polygon flat files (if available)"""
    path = Path(polygon_dir)
    
    if not path.exists():
        print(f"⚠️  Polygon directory not found: {polygon_dir}")
        return {"stocks": [], "crypto": []}
    
    stocks = set()
    crypto = set()
    
    # Scan for parquet/csv files
    for file_path in path.rglob("*.parquet"):
        # Extract symbol from filename (e.g., "AAPL_2024_01.parquet")
        symbol = file_path.stem.split("_")[0]
        
        if "crypto" in str(file_path).lower() or "btc" in symbol.lower():
            crypto.add(symbol)
        else:
            stocks.add(symbol)
    
    print(f"📂 Discovered {len(stocks)} stocks, {len(crypto)} crypto from Polygon files")
    return {"stocks": sorted(stocks), "crypto": sorted(crypto)}


def discover_all_symbols() -> Dict[str, List[str]]:
    """Discover all available symbols from all sources"""
    print("🔍 Discovering all available market data...")
    
    # Collect from all sources
    supabase_symbols = discover_symbols_from_supabase()
    polygon_symbols = discover_symbols_from_polygon_files()
    
    # Merge (union)
    all_stocks = sorted(set(supabase_symbols["stocks"] + polygon_symbols["stocks"]))
    all_crypto = sorted(set(supabase_symbols["crypto"] + polygon_symbols["crypto"]))
    
    result = {
        "stocks": all_stocks,
        "crypto": all_crypto,
        "total_stocks": len(all_stocks),
        "total_crypto": len(all_crypto),
        "total_symbols": len(all_stocks) + len(all_crypto),
        "discovered_at": datetime.utcnow().isoformat()
    }
    
    print(f"✅ Total discovered: {result['total_symbols']} symbols")
    print(f"   Stocks: {result['total_stocks']}")
    print(f"   Crypto: {result['total_crypto']}")
    
    # Save to cache
    with open("discovered_symbols.json", "w") as f:
        json.dump(result, f, indent=2)
    
    return result


def load_discovered_symbols() -> Dict[str, List[str]]:
    """Load previously discovered symbols from cache"""
    cache_file = Path("discovered_symbols.json")
    
    if cache_file.exists():
        with open(cache_file) as f:
            data = json.load(f)
            print(f"📋 Loaded {data['total_symbols']} symbols from cache")
            return data
    
    # Discover if not cached
    return discover_all_symbols()


if __name__ == "__main__":
    result = discover_all_symbols()
    
    print("\n📈 Stocks:")
    for stock in result["stocks"][:10]:
        print(f"  - {stock}")
    if len(result["stocks"]) > 10:
        print(f"  ... and {len(result['stocks']) - 10} more")
    
    print("\n🪙 Crypto:")
    for crypto in result["crypto"][:10]:
        print(f"  - {crypto}")
    if len(result["crypto"]) > 10:
        print(f"  ... and {len(result['crypto']) - 10} more")
