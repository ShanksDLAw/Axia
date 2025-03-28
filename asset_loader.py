# src/asset_loader.py
import pandas as pd
import yfinance as yf
import re
import logging
from datetime import datetime, timedelta
from collections import defaultdict
from requests.adapters import HTTPAdapter, Retry
import time
import requests
import random
from threading import Semaphore
from functools import lru_cache
from typing import Dict, Any, Optional
import threading
from queue import Queue
import json
import os
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Tuple

class TokenBucket:
    def __init__(self, tokens: int, fill_rate: float):
        self.capacity = tokens
        self.tokens = tokens
        self.fill_rate = fill_rate
        self.last_update = time.time()
        self._lock = threading.Lock()
    
    def get_token(self, block: bool = True) -> bool:
        with self._lock:
            now = time.time()
            elapsed = now - self.last_update
            self.tokens = min(self.capacity, self.tokens + elapsed * self.fill_rate)
            self.last_update = now
            
            if self.tokens >= 1:
                self.tokens -= 1
                return True
            elif not block:
                return False
            else:
                time.sleep(1.0 / self.fill_rate)
                return self.get_token(block)

class RequestQueue:
    def __init__(self, max_size: int = 100):
        self.queue = Queue(maxsize=max_size)
        self.results = {}
        self._lock = threading.Lock()
    
    def add_request(self, symbol: str, request_func: callable):
        self.queue.put((symbol, request_func))
    
    def get_result(self, symbol: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            return self.results.get(symbol)
    
    def set_result(self, symbol: str, result: Dict[str, Any]):
        with self._lock:
            self.results[symbol] = result

class AssetLoader:
    def __init__(self):
        self.index_map = {
            'US Large Cap': ('S&P 500', 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies', 0, 'Symbol'),
            'US Tech': ('NASDAQ-100', 'https://en.wikipedia.org/wiki/Nasdaq-100', 4, 'Ticker'),
            'Bonds': ['BND', 'AGG', 'TLT', 'IEF', 'LQD', 'HYG'],
            'Commodities': ['GLD', 'SLV', 'USO', 'UNG']
        }
        self.cache_dir = Path('cache')
        self.cache_dir.mkdir(exist_ok=True)
        self.token_bucket = TokenBucket(tokens=100, fill_rate=2.0)  # 100 requests per minute
        self.request_queue = RequestQueue(max_size=200)  # Increased queue size
        self.base_delay = 30  # Increased delay between batches for better rate limiting
        self.max_pool_size = 4  # Reduced parallel processing to avoid overwhelming the API
        self.batch_size = 10  # Reduced batch size for better stability
        self.proxy_pool = [
            'https://finance.yahoo.com',
            'https://query1.finance.yahoo.com',
            'https://query2.finance.yahoo.com'
        ]
        self.rate_limit_delay = 5  # Increased rate limit delay
        self.setup_requests_session()
        self._setup_cache_cleanup()  # New cache management

    def setup_requests_session(self):
        retry_strategy = Retry(
            total=10,
            backoff_factor=2.0,
            allowed_methods=["GET"],
            status_forcelist=[429, 500, 502, 503, 504],
            respect_retry_after_header=True,
            connect=10,
            read=10
        )
        adapter = HTTPAdapter(
            max_retries=retry_strategy,
            pool_connections=self.max_pool_size,
            pool_maxsize=self.max_pool_size,
            pool_block=True
        )
        self.session = requests.Session()
        self.session.timeout = (60, 120)  # Increased timeouts (connect timeout, read timeout)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)
        yf.set_tz_cache_location(str(self.cache_dir / "yfinance_tz_cache"))

    @lru_cache(maxsize=10000)  # Further increased cache size for better performance
    def _get_cached_data(self, symbol: str, allow_stale: bool = False) -> Optional[Dict[str, Any]]:
        cache_file = self.cache_dir / f"{symbol}.json"
        if not cache_file.exists():
            return None

        try:
            cache_age = time.time() - cache_file.stat().st_mtime
            # Extended cache duration with asset-specific timeouts
            if symbol in self.index_map['Bonds']:
                max_age = 1209600  # 14 days for bonds
            elif symbol in self.index_map['Commodities']:
                max_age = 432000  # 5 days for commodities
            else:
                max_age = 259200  # 3 days for stocks
            
            if cache_age > max_age and not allow_stale:
                cache_file.unlink()
                return None

            with open(cache_file, 'r') as f:
                data = json.load(f)

            # Optimized validation with minimal checks
            if not all(key in data for key in ('info', 'sector')):
                logging.warning(f"Invalid cache data format for {symbol}")
                cache_file.unlink()
                return None

            return data

        except (json.JSONDecodeError, OSError) as e:
            logging.error(f"Cache error for {symbol}: {str(e)}")
            try:
                cache_file.unlink()
            except OSError:
                pass
            return None

    def _save_cache(self, symbol: str, data: Dict[str, Any]):
        cache_file = self.cache_dir / f"{symbol}.json"
        try:
            # Validate data structure before saving
            if not isinstance(data, dict) or 'info' not in data or 'sector' not in data:
                logging.error(f"Invalid data format for {symbol}, skipping cache save")
                return
            
            # Use temporary file to ensure atomic writes
            temp_file = cache_file.with_suffix('.tmp')
            with open(temp_file, 'w') as f:
                json.dump(data, f, ensure_ascii=False)
            temp_file.replace(cache_file)  # Atomic replace
        except (OSError, TypeError) as e:
            logging.error(f"Error saving cache for {symbol}: {str(e)}")
            try:
                temp_file.unlink()
            except OSError:
                pass

    def _clean_symbol(self, symbol: str) -> str:
        """Clean symbol by removing special characters and replacing dots with hyphens"""
        return re.sub(r'[^a-zA-Z0-9-]', '', str(symbol)).replace('.', '-')

    def _fetch_symbol_data(self, symbol: str, retry_count: int = 0) -> Optional[Dict[str, Any]]:
        cached_data = self._get_cached_data(symbol)
        if cached_data:
            return cached_data

        if retry_count >= 3:
            logging.error(f"Max retries reached for {symbol}, using stale cache if available")
            return self._get_cached_data(symbol, allow_stale=True)

        if not self.token_bucket.get_token():
            time.sleep(self.rate_limit_delay)
            return self._fetch_symbol_data(symbol, retry_count + 1)

        try:
            proxy = random.choice(self.proxy_pool)
            ticker = yf.Ticker(symbol, session=self.session)
            ticker._base_url = proxy
            
            # Batch fetch info and history in parallel with enhanced error handling
            with ThreadPoolExecutor(max_workers=2) as executor:
                info_future = executor.submit(lambda: ticker.info)
                hist_future = executor.submit(lambda: ticker.history(period="1mo", prepost=False))
                
                try:
                    info = info_future.result(timeout=15)
                    hist = hist_future.result(timeout=15)
                except (TimeoutError, Exception) as e:
                    logging.warning(f"Timeout fetching data for {symbol}: {str(e)}")
                    # Try to use stale cache data as fallback
                    stale_data = self._get_cached_data(symbol, allow_stale=True)
                    if stale_data:
                        logging.info(f"Using stale cache data for {symbol}")
                        return stale_data
                    return None
                
                if not info or info.get('quoteType') == 'CRYPTOCURRENCY':
                    return None
                if hist.empty or len(hist) < 5:
                    return None
                    
                # Enhanced sector classification with industry fallback
                sector = info.get('sector')
                if not sector or sector == 'Unknown':
                    # Try to get sector from industry or industryKey
                    sector = info.get('industry')
                    if not sector:
                        sector = info.get('industryKey', 'Unknown').replace('-', ' ').title()
                
                # Map certain sectors for better classification
                sector_mapping = {
                    'Financial': 'Financial Services',
                    'Technology': 'Information Technology',
                    'Healthcare': 'Health Care',
                    'Consumer Defensive': 'Consumer Staples',
                    'Consumer Cyclical': 'Consumer Discretionary',
                    'Basic Materials': 'Materials'
                }
                sector = sector_mapping.get(sector, sector)
                
                # Enhanced sector classification with industry fallback
                sector = info.get('sector')
                if not sector or sector == 'Unknown':
                    # Try to get sector from industry or industryKey
                    sector = info.get('industry')
                    if not sector:
                        sector = info.get('industryKey', 'Unknown').replace('-', ' ').title()
                
                # Enhanced data processing with optimized sector mapping
                data = {
                    'info': info,
                    'sector': sector,
                    'last_updated': time.time(),
                    'market_cap': info.get('marketCap', 0),
                    'volume': info.get('volume', 0),
                    'price': hist['Close'].iloc[-1] if not hist.empty else None
                }
                
                # Save cache within the executor context to prevent shutdown issues
                executor.submit(self._save_cache, symbol, data)
                return data
            return data
            
        except Exception as e:
            logging.error(f"Error fetching data for {symbol}: {str(e)}")
            return None

    def _process_batch(self, symbols: List[str]) -> Tuple[List[str], Dict[str, str], List[Tuple[str, str]]]:
        """Process a batch of symbols with enhanced parallel processing and adaptive rate limiting"""
        valid_assets = []
        sector_map = {}
        failed_symbols = []
        
        # Pre-filter symbols and group by data source
        cached_symbols = []
        to_fetch = []
        for symbol in symbols:
            cached_data = self._get_cached_data(symbol)
            if cached_data:
                valid_assets.append(symbol)
                sector_map[symbol] = cached_data['sector']
            else:
                to_fetch.append(symbol)
        
        if not to_fetch:
            return valid_assets, sector_map, failed_symbols
        
        # Implement adaptive batch size based on rate limits
        batch_size = min(len(to_fetch), max(5, int(self.token_bucket.tokens)))
        batches = [to_fetch[i:i + batch_size] for i in range(0, len(to_fetch), batch_size)]
        
        for batch in batches:
            with ThreadPoolExecutor(max_workers=min(len(batch), self.max_pool_size)) as executor:
                futures = {executor.submit(self._fetch_symbol_data, symbol): symbol for symbol in batch}
                
                # Process completed futures with timeout
                for future in as_completed(futures):
                    symbol = futures[future]
                    try:
                        # Adjust timeout for known slow-responding symbols
                        symbol_timeout = 60 if symbol in ['LEN'] else 30
                        data = future.result(timeout=symbol_timeout)
                        if data:
                            valid_assets.append(symbol)
                            sector_map[symbol] = data['sector']
                        else:
                            # Try to use stale cache as fallback
                            stale_data = self._get_cached_data(symbol, allow_stale=True)
                            if stale_data:
                                valid_assets.append(symbol)
                                sector_map[symbol] = stale_data['sector']
                                logging.info(f"Using stale cache for {symbol}")
                            else:
                                failed_symbols.append((symbol, "Failed to fetch data"))
                    except TimeoutError:
                        logging.warning(f"Timeout fetching data for {symbol}")
                        # Try to use stale cache as fallback for timeout errors
                        stale_data = self._get_cached_data(symbol, allow_stale=True)
                        if stale_data:
                            valid_assets.append(symbol)
                            sector_map[symbol] = stale_data['sector']
                            logging.info(f"Using stale cache for {symbol} after timeout")
                        else:
                            failed_symbols.append((symbol, "Timeout fetching data"))
                    except Exception as e:
                        logging.error(f"Error processing {symbol}: {str(e)}")
                        failed_symbols.append((symbol, str(e)))
                
            # Adaptive rate limiting
            remaining_tokens = self.token_bucket.tokens
            if remaining_tokens < self.token_bucket.capacity * 0.3:
                delay = max(1, self.rate_limit_delay * (1 - remaining_tokens/self.token_bucket.capacity))
                time.sleep(delay)
        
        return valid_assets, sector_map, failed_symbols

    def _setup_cache_cleanup(self):
        """Setup periodic cache cleanup to remove old files"""
        def cleanup_old_cache():
            current_time = time.time()
            for cache_file in self.cache_dir.glob("*.json"):
                try:
                    if current_time - cache_file.stat().st_mtime > 604800:  # 7 days
                        cache_file.unlink()
                except OSError:
                    continue

        cleanup_old_cache()  # Initial cleanup

    def fetch_assets(self) -> tuple[list[str], dict[str, str]]:
        """Fetch and validate assets with optimized parallel processing"""
        assets = []
        all_valid_assets = []
        all_sector_map = {}
        all_failed_symbols = []

        # Parallel fetch of equity indices
        def fetch_index(index):
            try:
                config = self.index_map[index]
                tables = pd.read_html(config[1])
                df = tables[config[2]]
                return [self._clean_symbol(s) for s in df[config[3]].tolist()]
            except Exception as e:
                logging.error(f"Error fetching {index}: {str(e)}")
                return []

        with ThreadPoolExecutor(max_workers=2) as executor:
            index_futures = {executor.submit(fetch_index, index): index 
                           for index in ['US Large Cap', 'US Tech']}
            
            for future in as_completed(index_futures):
                assets.extend(future.result())

        # Add fixed income and commodities
        assets.extend(self.index_map['Bonds'])
        assets.extend(self.index_map['Commodities'])
        unique_assets = list(set(assets))

        # Process assets in optimized parallel batches
        batch_size = self.batch_size * self.max_pool_size
        batches = [unique_assets[i:i + batch_size] for i in range(0, len(unique_assets), batch_size)]
        
        for batch in batches:
            valid_batch, sector_batch, failed_batch = self._process_batch(batch)
            all_valid_assets.extend(valid_batch)
            all_sector_map.update(sector_batch)
            all_failed_symbols.extend(failed_batch)

        logging.info(f"Loaded {len(all_valid_assets)} valid symbols")
        if all_failed_symbols:
            logging.warning(f"Failed to load {len(all_failed_symbols)} symbols")
        return all_valid_assets, all_sector_map