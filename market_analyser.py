# src/market_analyzer.py
import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta
import logging
from requests.adapters import HTTPAdapter, Retry
from pathlib import Path
import json
import time
from typing import Optional

class MarketAnalyzer:
    def __init__(self, fred_api_key: str):
        self.fred_api_key = fred_api_key
        self.rate_limit_delay = 2  # seconds
        self.cache_dir = Path('cache')
        self.cache_dir.mkdir(exist_ok=True)
        self.setup_requests_session()
        
    def setup_requests_session(self):
        """Set up a requests session with retry mechanism"""
        retry_strategy = Retry(
            total=5,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["GET", "POST"]
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session = requests.Session()
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)
        
    def _get_cached_data(self, series_id: str) -> Optional[float]:
        """Get cached FRED data if available and not expired"""
        cache_file = self.cache_dir / f"fred_{series_id}.json"
        if not cache_file.exists():
            return None
        
        try:
            cache_age = time.time() - cache_file.stat().st_mtime
            if cache_age > 86400:  # 24 hours cache
                cache_file.unlink()
                return None
                
            with open(cache_file, 'r') as f:
                data = json.load(f)
            return float(data['value'])
        except (json.JSONDecodeError, OSError, KeyError, ValueError) as e:
            logging.error(f"Cache error for {series_id}: {str(e)}")
            try:
                cache_file.unlink()
            except OSError:
                pass
            return None
    
    def _save_cache(self, series_id: str, value: float):
        """Save FRED data to cache"""
        cache_file = self.cache_dir / f"fred_{series_id}.json"
        try:
            with open(cache_file, 'w') as f:
                json.dump({'value': value, 'timestamp': time.time()}, f)
        except OSError as e:
            logging.error(f"Error saving cache for {series_id}: {str(e)}")
    
    def _fetch_vix_data(self) -> float:
        """Fetch VIX data from FRED API with caching"""
        cached_value = self._get_cached_data('VIXCLS')
        if cached_value is not None:
            return cached_value
            
        url = "https://api.stlouisfed.org/fred/series/observations"
        params = {
            'series_id': 'VIXCLS',
            'file_type': 'json',
            'api_key': self.fred_api_key,
            'limit': 1,
            'sort_order': 'desc'
        }
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = self.session.get(url, params=params)
                response.raise_for_status()
                data = response.json()
                observations = data['observations']
                if not observations:
                    logging.error("No VIX data returned")
                    return None
                vix_value = float(observations[0]['value'])
                self._save_cache('VIXCLS', vix_value)
                return vix_value
            except Exception as e:
                if attempt < max_retries - 1:
                    logging.warning(f"Failed to fetch VIX data (attempt {attempt + 1}/{max_retries}). Retrying...")
                    time.sleep(self.rate_limit_delay)
                else:
                    logging.error(f"Failed to fetch VIX data after {max_retries} attempts: {str(e)}")
                    return None
    
    def _fetch_fred_yield(self, series_id: str) -> float:
        """Fetch yield curve data from FRED API"""
        url = f"https://api.stlouisfed.org/fred/series/observations?series_id={series_id}&api_key={self.fred_api_key}&file_type=json"
        try:
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    response = self.session.get(url)
                    response.raise_for_status()
                    data = response.json()
                    values = [float(obs['value']) for obs in data['observations'] if obs['value'] != '.']
                    return values[-1] if values else 4.0
                except Exception as e:
                    if attempt < max_retries - 1:
                        logging.warning(f"Failed to fetch FRED data (attempt {attempt + 1}/{max_retries}). Retrying...")
                        time.sleep(self.rate_limit_delay)
                    else:
                        logging.error(f"Failed to fetch FRED data after {max_retries} attempts: {str(e)}")
                        return 4.0  # Default value
        except:
            logging.warning("Using default yield value")
            return 4.0
    
    def get_market_regime(self, price_data: pd.DataFrame) -> str:
        """Calculate market regime with enhanced technical indicators and adaptive thresholds"""
        try:
            if price_data.empty:
                logging.warning("No price data available for regime detection")
                return 'Neutral'
            
            # Enhanced volatility calculation with exponential weighting
            returns = price_data.pct_change().fillna(0)
            vol_short = returns.ewm(span=21).std() * np.sqrt(252)  # ~1 month trading days
            vol_long = returns.ewm(span=63).std() * np.sqrt(252)   # ~3 months trading days
            current_vol = vol_short.iloc[-1].mean()
            vol_ratio = vol_short.iloc[-1].mean() / vol_long.iloc[-1].mean()
            
            # Fetch VIX data using FRED API
            vix_value = self._fetch_vix_data()
            if vix_value is None:
                logging.warning("Using default VIX value of 20")
                vix_value = 20
            
            # Fetch yield curve data
            ten_year = self._fetch_fred_yield('DGS10')
            two_year = self._fetch_fred_yield('DGS2')
            spread = ten_year - two_year
            
            # Adaptive volatility thresholds
            vol_threshold_high = np.percentile(vol_long, 75)
            vol_threshold_low = np.percentile(vol_long, 25)
            
            # Enhanced regime determination
            if (vix_value > 30 or spread < -0.1) and current_vol > vol_threshold_high:
                return 'Bearish'
            elif vix_value < 20 and spread > 0.5 and current_vol < vol_threshold_low:
                return 'Bullish'
            elif current_vol > vol_threshold_high or vol_ratio > 1.2:
                return 'High Volatility'
            elif vix_value > 25 or spread < 0:
                return 'Moderately Bearish'
            elif vix_value < 15 and spread > 0.3:
                return 'Moderately Bullish'
            else:
                return 'Neutral'
        except Exception as e:
            logging.error(f"Market regime detection failed: {str(e)}")
            return 'Neutral'