#!/usr/bin/env python3
"""Fetch macro economic indicators to enhance trading signals"""

import os
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List
import json

class MacroDataFetcher:
    """Fetch economic indicators from FRED (Federal Reserve Economic Data)"""
    
    def __init__(self):
        # FRED API is free and doesn't require a key for basic access
        self.fred_base = "https://api.stlouisfed.org/fred/series/observations"
        
    def fetch_fred_series(self, series_id: str, start_date: str = "2020-01-01") -> pd.DataFrame:
        """Fetch a FRED economic series"""
        try:
            params = {
                'series_id': series_id,
                'api_key': 'demo',  # Use demo key for testing
                'file_type': 'json',
                'observation_start': start_date
            }
            
            response = requests.get(self.fred_base, params=params, timeout=10)
            data = response.json()
            
            if 'observations' in data:
                df = pd.DataFrame(data['observations'])
                df['date'] = pd.to_datetime(df['date'])
                df['value'] = pd.to_numeric(df['value'], errors='coerce')
                df = df[['date', 'value']].dropna()
                df.columns = ['date', series_id]
                return df
            
            return pd.DataFrame()
        except Exception as e:
            print(f"⚠️ Error fetching {series_id}: {e}")
            return pd.DataFrame()
    
    def get_all_macro_indicators(self, start_date: str = "2020-01-01") -> pd.DataFrame:
        """
        Fetch key macro indicators that affect markets
        
        Indicators:
        - FEDFUNDS: Federal Funds Rate (monetary policy)
        - DGS10: 10-Year Treasury Rate (risk-free rate)
        - T10Y2Y: 10Y-2Y Treasury Spread (recession indicator)
        - UNRATE: Unemployment Rate (economic health)
        - CPIAUCSL: CPI Inflation (purchasing power)
        - DEXUSEU: USD/EUR Exchange Rate (dollar strength)
        - VIXCLS: VIX Volatility Index (market fear)
        """
        
        indicators = {
            'FEDFUNDS': 'fed_rate',
            'DGS10': 'treasury_10y',
            'T10Y2Y': 'yield_curve',
            'UNRATE': 'unemployment',
            'CPIAUCSL': 'inflation',
            'DEXUSEU': 'usd_eur',
            'VIXCLS': 'vix'
        }
        
        print("📊 Fetching macro economic indicators...")
        
        dfs = []
        for series_id, name in indicators.items():
            df = self.fetch_fred_series(series_id, start_date)
            if not df.empty:
                df.columns = ['date', name]
                dfs.append(df)
                print(f"  ✅ {name}: {len(df)} observations")
        
        if not dfs:
            print("❌ No macro data fetched")
            return pd.DataFrame()
        
        # Merge all indicators
        macro_df = dfs[0]
        for df in dfs[1:]:
            macro_df = pd.merge(macro_df, df, on='date', how='outer')
        
        # Forward fill missing values (macro data is often monthly/quarterly)
        macro_df = macro_df.sort_values('date')
        macro_df = macro_df.fillna(method='ffill')
        
        # Add derived features
        macro_df['inflation_mom'] = macro_df['inflation'].pct_change()
        macro_df['rate_trend'] = macro_df['fed_rate'].diff()
        macro_df['recession_signal'] = (macro_df['yield_curve'] < 0).astype(int)
        
        return macro_df
    
    def get_macro_state_for_date(self, macro_df: pd.DataFrame, target_date: datetime) -> Dict[str, float]:
        """Get macro state vector for a specific date"""
        if macro_df.empty:
            return {}
        
        # Find closest date
        closest_idx = (macro_df['date'] - target_date).abs().idxmin()
        row = macro_df.loc[closest_idx]
        
        return {
            'fed_rate': row.get('fed_rate', 0.0),
            'treasury_10y': row.get('treasury_10y', 0.0),
            'yield_curve': row.get('yield_curve', 0.0),
            'unemployment': row.get('unemployment', 0.0),
            'inflation': row.get('inflation', 0.0),
            'vix': row.get('vix', 20.0),
            'recession_signal': row.get('recession_signal', 0.0),
        }


class NewsDataFetcher:
    """Fetch and analyze news sentiment for market context"""
    
    def __init__(self):
        # Using free news APIs
        self.newsapi_base = "https://newsapi.org/v2/everything"
        
    def fetch_market_news(self, query: str = "stock market", days_back: int = 7) -> List[Dict]:
        """
        Fetch recent market news
        
        Note: NewsAPI requires a free API key from newsapi.org
        Returns empty list if no key configured
        """
        api_key = os.getenv("NEWS_API_KEY")
        if not api_key:
            print("⚠️ NEWS_API_KEY not set, skipping news data")
            return []
        
        try:
            from_date = (datetime.now() - timedelta(days=days_back)).strftime("%Y-%m-%d")
            
            params = {
                'q': query,
                'from': from_date,
                'sortBy': 'relevancy',
                'apiKey': api_key,
                'language': 'en'
            }
            
            response = requests.get(self.newsapi_base, params=params, timeout=10)
            data = response.json()
            
            if data.get('status') == 'ok':
                articles = data.get('articles', [])
                print(f"✅ Fetched {len(articles)} news articles")
                return articles
            
            return []
        except Exception as e:
            print(f"⚠️ Error fetching news: {e}")
            return []
    
    def analyze_sentiment_batch(self, texts: List[str]) -> List[float]:
        """
        Analyze sentiment of text batch
        Returns sentiment scores [-1 to 1]
        
        This is a placeholder - in production use Lovable AI or another NLP service
        """
        # Simple keyword-based sentiment for now
        positive_words = ['rally', 'gain', 'surge', 'profit', 'growth', 'bullish', 'up', 'high', 'beat']
        negative_words = ['fall', 'drop', 'loss', 'decline', 'recession', 'bearish', 'down', 'low', 'miss']
        
        sentiments = []
        for text in texts:
            text_lower = text.lower()
            pos_count = sum(1 for word in positive_words if word in text_lower)
            neg_count = sum(1 for word in negative_words if word in text_lower)
            
            if pos_count + neg_count == 0:
                sentiments.append(0.0)
            else:
                sentiments.append((pos_count - neg_count) / (pos_count + neg_count))
        
        return sentiments


if __name__ == "__main__":
    print("🧪 Testing Macro Data Fetcher\n")
    
    # Test macro data
    fetcher = MacroDataFetcher()
    macro_df = fetcher.get_all_macro_indicators(start_date="2023-01-01")
    
    print(f"\n📊 Macro Data Summary:")
    print(f"   Date range: {macro_df['date'].min()} to {macro_df['date'].max()}")
    print(f"   Total observations: {len(macro_df)}")
    print(f"\n   Sample data:")
    print(macro_df.tail())
    
    # Test news data
    print("\n\n🧪 Testing News Data Fetcher\n")
    news_fetcher = NewsDataFetcher()
    articles = news_fetcher.fetch_market_news(query="stock market", days_back=3)
    
    if articles:
        print(f"\n📰 Sample headlines:")
        for article in articles[:5]:
            print(f"   - {article.get('title', 'N/A')}")
