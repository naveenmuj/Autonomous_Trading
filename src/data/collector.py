import os
import sys
import json
import time
import yaml
import pyotp
import random
import logging
import requests
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from functools import wraps
from SmartApi import SmartConnect
from .websocket import MarketDataWebSocket
import threading
import talib

logger = logging.getLogger(__name__)

def with_rate_limit(max_retries=5, initial_delay=2.0):
    """Decorator for rate-limited API calls with exponential backoff and better handling of rate errors."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            delay = initial_delay
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if 'rate' in str(e).lower() or 'access denied' in str(e).lower():
                        logger.warning(f"Rate limit or access error, retrying in {delay}s... ({attempt + 1}/{max_retries})")
                        time.sleep(delay)
                        delay *= 2
                    else:
                        raise
            logger.error(f"Exceeded max retries for rate-limited function {func.__name__}")
            return func(*args, **kwargs)
        return wrapper
    return decorator

class RateLimiter:
    """Simple rate limiter for API calls"""
    def __init__(self, rate=3, burst=5):
        self.rate = rate  # requests per second
        self.burst = burst
        self.tokens = burst
        self.last_update = time.time()

    def acquire(self):
        """Try to acquire a token. Returns True if successful."""
        now = time.time()
        time_passed = now - self.last_update
        self.tokens = min(self.burst, self.tokens + time_passed * self.rate)
        self.last_update = now

        if self.tokens >= 1:
            self.tokens -= 1
            return True
        return False

class DataCollector:
    # --- Rate limiting state for historical data API ---
    _hist_api_calls_sec = []  # timestamps of calls in last second
    _hist_api_calls_min = []  # timestamps of calls in last minute
    _hist_api_calls_hr = []   # timestamps of calls in last hour
    _hist_api_lock = threading.Lock()

    def get_historical_data(self, symbol, days=10, interval='ONE_DAY'):
        import time
        from datetime import datetime, timedelta
        # --- Rate limit constants (official) ---
        HIST_API_LIMIT_SEC = 3
        HIST_API_LIMIT_MIN = 180
        HIST_API_LIMIT_HR = 5000

        def _rate_limit_wait():
            now = time.time()
            with self._hist_api_lock:
                # Clean up old timestamps
                self._hist_api_calls_sec = [t for t in self._hist_api_calls_sec if t > now - 1]
                self._hist_api_calls_min = [t for t in self._hist_api_calls_min if t > now - 60]
                self._hist_api_calls_hr = [t for t in self._hist_api_calls_hr if t > now - 3600]
                # If any limit exceeded, calculate wait time
                wait_times = []
                if len(self._hist_api_calls_sec) >= HIST_API_LIMIT_SEC:
                    wait_times.append(1 - (now - self._hist_api_calls_sec[0]))
                if len(self._hist_api_calls_min) >= HIST_API_LIMIT_MIN:
                    wait_times.append(60 - (now - self._hist_api_calls_min[0]))
                if len(self._hist_api_calls_hr) >= HIST_API_LIMIT_HR:
                    wait_times.append(3600 - (now - self._hist_api_calls_hr[0]))
                if wait_times:
                    wait_time = max(wait_times)
                    logger.warning(f"[RATE LIMIT] Waiting {wait_time:.2f}s to comply with Angel One historical data API limits.")
                    time.sleep(max(wait_time, 0.01))
                # Register this call
                self._hist_api_calls_sec.append(now)
                self._hist_api_calls_min.append(now)
                self._hist_api_calls_hr.append(now)

        """
        Fetch historical OHLCV data for a symbol, with candlestick pattern detection.
        This method should be called by all model/data consumers.
        """
        try:
            if self.angel_api is None:
                logger.error("Angel One API not initialized.")
                return pd.DataFrame()
            end_date = datetime.now()
            min_date = datetime(2000, 1, 1)
            if days < 1:
                logger.warning(f"Requested days < 1, adjusting to 1.")
                days = 1
            start_date = end_date - timedelta(days=days)
            if start_date < min_date:
                logger.warning(f"Requested start_date {start_date.date()} before 2000-01-01, adjusting to 2000-01-01.")
                start_date = min_date
            if end_date > datetime.now():
                logger.warning(f"Requested end_date {end_date.date()} is in the future, adjusting to today.")
                end_date = datetime.now()
            if interval == 'ONE_DAY':
                start_str = start_date.strftime('%Y-%m-%d') + ' 09:15'
                end_str = end_date.strftime('%Y-%m-%d') + ' 15:30'
            else:
                start_str = start_date.strftime('%Y-%m-%d %H:%M')
                end_str = end_date.strftime('%Y-%m-%d %H:%M')
            logger.info(f"Requesting historical data for {symbol} from {start_str} to {end_str} (interval: {interval})")
            token = self.symbol_token_map.get(symbol)
            if token is None:
                token = self.symbol_token_map.get(symbol.replace('.NS', ''))
            if token is None:
                token = self.symbol_token_map.get(symbol.replace('.NS', '-EQ'))
            if token is None:
                logger.error(f"No token found for symbol {symbol} (tried direct, base, and -EQ)")
                return pd.DataFrame()
            params = {
                "exchange": "NSE",
                "symboltoken": token,
                "interval": interval,
                "fromdate": start_str,
                "todate": end_str,
            }
            max_retries = 5
            retry_wait = self.config.get('api_rate_limit_wait', 2)
            for attempt in range(max_retries):
                _rate_limit_wait()
                response = self.angel_api.getCandleData(params)
                if response and 'data' in response:
                    candles = response['data']
                    df = pd.DataFrame(candles, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                    df['timestamp'] = pd.to_datetime(df['timestamp'])
                    # --- DIAGNOSTIC: OHLCV completeness check ---
                    expected_rows = self._calculate_expected_rows(start_date, end_date, interval)
                    actual_rows = len(df)
                    missing_timestamps = []
                    if 'timestamp' in df.columns:
                        df_sorted = df.sort_values('timestamp')
                        all_expected = pd.date_range(start=start_date, end=end_date, freq=self._interval_to_pandas_freq(interval))
                        actual_set = set(pd.to_datetime(df_sorted['timestamp']))
                        missing_timestamps = [ts for ts in all_expected if ts not in actual_set]
                    logger.info(f"[DIAGNOSTIC] {symbol}: Expected rows={expected_rows}, Actual rows={actual_rows}, Missing timestamps={len(missing_timestamps)}")
                    if missing_timestamps:
                        logger.warning(f"[DIAGNOSTIC] {symbol}: Missing timestamps: {missing_timestamps[:10]}{'...' if len(missing_timestamps)>10 else ''}")
                    if actual_rows < expected_rows:
                        logger.warning(f"[DIAGNOSTIC] {symbol}: Fetched fewer rows than expected. Possible API/data issue.")
                    for col in ['open', 'high', 'low', 'close', 'volume']:
                        df[col] = pd.to_numeric(df[col], errors='coerce')
                    # Verification log: check required columns
                    required_cols = ['open', 'high', 'low', 'close', 'volume']
                    missing_cols = [col for col in required_cols if col not in df.columns]
                    if missing_cols:
                        logger.error(f"Missing columns {missing_cols} in historical data for {symbol}. Failing fetch.")
                        return pd.DataFrame()
                    if df.isnull().any().any():
                        logger.warning(f"Null values found in historical data for {symbol}. Shape: {df.shape}")
                    if len(df) < 60:
                        logger.error(f"Insufficient data points ({len(df)}) for {symbol}. Failing fetch.")
                        return pd.DataFrame()
                    df = self.detect_candlestick_patterns(df)
                    if 'pattern' not in df.columns:
                        logger.warning(f"Pattern detection failed for {symbol}, adding default 'None' values.")
                        df['pattern'] = 'None'
                    else:
                        logger.info(f"Pattern detection completed for {symbol}. Example patterns: {df['pattern'].value_counts().to_dict()}")
                    return df
                else:
                    if response and b'access rate' in str(response).lower():
                        logger.warning(f"Rate limit exceeded for {symbol}. Waiting {retry_wait} seconds before retry.")
                        time.sleep(retry_wait)
                    else:
                        logger.error(f"No data returned for {symbol} from API. Attempt {attempt+1}/{max_retries}")
                        time.sleep(retry_wait)
            logger.error(f"Failed to fetch historical data for {symbol} after {max_retries} retries.")
            return pd.DataFrame()
        except Exception as e:
            logger.error(f"Error fetching historical data for {symbol}: {e}")
            return pd.DataFrame()

    def _interval_to_pandas_freq(self, interval):
        # Helper to convert API interval to pandas frequency string
        mapping = {
            'ONE_DAY': 'B',  # business day
            'ONE_MINUTE': 'T',
            'FIVE_MINUTE': '5T',
            'FIFTEEN_MINUTE': '15T',
            'THIRTY_MINUTE': '30T',
            'ONE_HOUR': 'H',
        }
        return mapping.get(interval, 'B')

    def _calculate_expected_rows(self, start_date, end_date, interval):
        # Helper to estimate expected number of rows for diagnostics
        freq = self._interval_to_pandas_freq(interval)
        rng = pd.date_range(start=start_date, end=end_date, freq=freq)
        return len(rng)

    # Websocket modes
    MODE_LTP = 1
    MODE_QUOTE = 2
    MODE_SNAPQUOTE = 3
    # ...existing code...
    # Websocket modes
    MODE_LTP = 1
    MODE_QUOTE = 2
    MODE_SNAPQUOTE = 3

    """Enhanced data collection class with rate limiting"""
    def __init__(self, config):
        print("[DEBUG] DataCollector.__init__() - start")
        logger.info("[DEBUG] DataCollector.__init__() - start")
        # Commented out Streamlit debug logs (st is not defined)
        # try:
        #     st.info("[DEBUG] DataCollector.__init__() - entered constructor")
        # except Exception:
        #     pass
        config = self.validate_config(config)
        self.config = config
        self.angel_api = None
        self.auth_token = None 
        self.feed_token = None
        self.websocket = None
        self.symbol_token_map = {}
        self.token_symbol_map = {}
        self.session_timer = None
        self._watchlist_cache = None
        self._watchlist_cache_time = 0
        self._watchlist_cache_ttl = 300  # 5 minutes
        self._historical_cache = {}  # (symbol, interval, days) -> DataFrame
        # --- LIVE QUOTES PATCH ---
        self._live_quotes = {}  # symbol -> latest tick dict
        self._live_quotes_lock = threading.Lock()
        # Rate limiting (configurable)
        rate_limit_cfg = config.get('api', {}).get('rate_limit', {})
        rate = rate_limit_cfg.get('rate', 3)
        burst = rate_limit_cfg.get('burst', 5)
        self._rate_limiter = RateLimiter(rate=rate, burst=burst)  # More conservative values
        # try:
        #     st.info("[DEBUG] DataCollector.__init__() - before _initialize_api")
        # except Exception:
        #     pass
        # Initialize API
        self._initialize_api()
        # try:
        #     st.info("[DEBUG] DataCollector.__init__() - after _initialize_api, before _initialize_token_mapping")
        # except Exception:
        #     pass
        self._initialize_token_mapping()
        # try:
        #     st.info("[DEBUG] DataCollector.__init__() - after _initialize_token_mapping, before _initialize_websocket")
        # except Exception:
        #     pass
        try:
            self._initialize_websocket()
        except Exception as e:
            # Surface websocket init errors to logs only
            logger.error(f"[ERROR] DataCollector.__init__() - _initialize_websocket failed: {e}")
            raise
        # try:
        #     st.info("[DEBUG] DataCollector.__init__() - end of constructor")
        # except Exception:
        #     pass
        
    @staticmethod
    def validate_config(config):
        """Validate and auto-correct config values for dashboard compatibility and UI limits."""
        # Ensure risk values are <= 1.0
        risk = config.get('trading', {}).get('risk', {})
        changed = False
        for key in ['position_size', 'stop_loss', 'risk_per_trade']:
            if key in risk:
                if risk[key] > 1.0:
                    logger.warning(f"Risk config value for {key} ({risk[key]}) exceeds 1.0, capping to 1.0.")
                    risk[key] = 1.0
                    changed = True
                elif risk[key] < 0:
                    logger.warning(f"Risk config value for {key} ({risk[key]}) below 0, capping to 0.0.")
                    risk[key] = 0.0
                    changed = True
        if changed:
            config['trading']['risk'] = risk
            logger.warning(f"Config risk values after auto-correction: {risk}")
        else:
            logger.info(f"Config risk values used: {risk}")
        return config

    def _initialize_api(self):
        """Initialize Angel One API connection with enhanced error handling and session management"""
        try:
            angel_config = self.config['apis']['angel_one']
            max_retries = 5
            retry_delay = 10
            retry_count = 0
            last_error = None
            
            while retry_count < max_retries:
                try:
                    logger.info(f"Attempting API connection (attempt {retry_count + 1}/{max_retries})")
                    
                    # Initialize session with better timeout and retry settings
                    session = requests.Session()
                    adapter = requests.adapters.HTTPAdapter(
                        max_retries=3,
                        pool_connections=10,
                        pool_maxsize=10
                    )
                    session.mount('https://', adapter)
                    session.timeout = (30, 60)  # (connect timeout, read timeout)
                    
                    # Initialize SmartConnect
                    self.angel_api = SmartConnect(api_key=angel_config['api_key'])
                    self.angel_api._SmartConnect__http_session = session
                    logger.info("SmartAPI instance created with extended timeout")
                    
                    # Generate TOTP
                    totp = pyotp.TOTP(angel_config['totp_secret'])
                    current_totp = totp.now()
                    logger.debug("Generated TOTP successfully")
                    
                    # Authenticate
                    data = self.angel_api.generateSession(
                        angel_config['client_id'],
                        angel_config['mpin'],
                        current_totp
                    )
                    
                    if data.get('status'):
                        self.auth_token = data['data']['jwtToken']
                        self.refresh_token = data['data']['refreshToken']
                        self.feed_token = self.angel_api.getfeedToken()
                        
                        # Verify connection
                        profile = self.angel_api.getProfile(self.refresh_token)
                        if profile.get('status'):
                            logger.info(f"Successfully authenticated with Angel One API")
                            logger.info(f"Successfully retrieved user profile for {profile['data'].get('name', 'Unknown')}")
                            return
                    
                    retry_count += 1
                    if retry_count < max_retries:
                        wait_time = retry_delay * (2 ** (retry_count - 1))  # Exponential backoff
                        logger.warning(f"Authentication failed, retrying in {wait_time} seconds...")
                        time.sleep(wait_time)
                    
                except requests.exceptions.Timeout as e:
                    last_error = e
                    retry_count += 1
                    if retry_count < max_retries:
                        wait_time = retry_delay * (2 ** (retry_count - 1))
                        logger.warning(f"Connection timeout, retrying in {wait_time} seconds...")
                        time.sleep(wait_time)
                    
                except Exception as e:
                    last_error = e
                    retry_count += 1
                    if retry_count < max_retries:
                        wait_time = retry_delay * (2 ** (retry_count - 1))
                        logger.warning(f"Error during authentication: {str(e)}, retrying in {wait_time} seconds...")
                        time.sleep(wait_time)
            
            # If we get here, all retries failed
            error_msg = f"Failed to initialize Angel One API after {max_retries} attempts"
            if last_error:
                error_msg += f": {str(last_error)}"
            logger.error(error_msg)
            raise Exception(error_msg)
            
        except Exception as e:
            logger.error(f"Error initializing Angel One API: {str(e)}")
            raise
            
    def _start_session_renewal_timer(self):
        """Start a timer to handle session renewal before expiry"""
        import threading
        
        def renew_session():
            while True:
                try:
                    # Sleep for 6 hours (token validity is 8 hours)
                    time.sleep(6 * 60 * 60)
                    
                    # Generate new TOTP
                    totp = pyotp.TOTP(self.config['apis']['angel_one']['totp_secret'])
                    current_totp = totp.now()
                    
                    # Request new session
                    data = self.angel_api.generateSession(
                        self.config['apis']['angel_one']['client_id'],
                        self.config['apis']['angel_one']['mpin'],
                        current_totp
                    )
                    
                    if data.get('status'):
                        # Update tokens
                        self.auth_token = data['data']['jwtToken']
                        self.refresh_token = data['data']['refreshToken']
                        self.feed_token = self.angel_api.getfeedToken()
                        
                        # Verify profile after token renewal
                        profile = self.angel_api.getProfile(self.refresh_token)
                        if profile.get('status'):
                            logger.info(f"Successfully renewed Angel One session for {profile['data'].get('name', 'Unknown')}")
                            
                            # Update WebSocket if it exists
                            if hasattr(self, 'websocket') and self.websocket:
                                self.websocket.update_tokens(
                                    auth_token=self.auth_token,
                                    feed_token=self.feed_token
                                )
                        else:
                            logger.error(f"Failed to verify profile after renewal: {profile.get('message', 'Unknown error')}")
                    else:
                        logger.error(f"Failed to renew session: {data.get('message', 'Unknown error')}")
                        
                except Exception as e:
                    logger.error(f"Error during session renewal: {str(e)}")
                    # Sleep for shorter duration before retry
                    time.sleep(300)  # 5 minutes
        
        # Start renewal thread
        self.session_timer = threading.Thread(target=renew_session, daemon=True)
        self.session_timer.start()
        logger.info("Session renewal timer started")
        
    def _initialize_token_mapping(self):
        """Initialize token mapping for configured symbols with robust symbol variant matching"""
        try:
            logger.info("Initializing token mapping...")
            self.symbol_mappings = {
                'HDFC.NS': 'HDFCBANK.NS',
                'ICICI.NS': 'ICICIBANK.NS'
            }
            mapped_symbols = self.get_symbols_from_config()
            mapped_symbols = [self.symbol_mappings.get(s, s) for s in mapped_symbols]
            mapped_symbols = list(dict.fromkeys(mapped_symbols))
            instrument_url = "https://margincalculator.angelbroking.com/OpenAPI_File/files/OpenAPIScripMaster.json"
            for attempt in range(3):
                try:
                    logger.info(f"Downloading instrument file (attempt {attempt + 1}/3)...")
                    response = requests.get(instrument_url, timeout=30)
                    if response.status_code == 200:
                        data = response.json()
                        logger.info("Successfully downloaded instrument file")
                        break
                except Exception as e:
                    logger.error(f"Error downloading instrument file: {str(e)}")
                    if attempt < 2:
                        time.sleep(2 ** attempt)
            else:
                raise Exception("Failed to download instrument file after 3 attempts")
            self.symbol_token_map = {}
            mapped_count = 0
            unmapped_symbols = []
            for symbol in mapped_symbols:
                base_symbol = symbol.replace('.NS', '')
                variants = [
                    base_symbol,
                    f"{base_symbol}-EQ",
                    symbol,
                    symbol.replace('.NS', '-EQ') if symbol.endswith('.NS') else symbol
                ]
                token = None
                for instrument in data:
                    if instrument['symbol'] in variants:
                        token = instrument['token']
                        self.symbol_token_map[symbol] = token
                        mapped_count += 1
                        logger.debug(f"Mapped {symbol} to token {token}")
                        break
                if not token:
                    unmapped_symbols.append(symbol)
            if mapped_count == 0:
                raise ValueError("No symbols could be mapped to tokens")
            logger.info(f"Successfully mapped {mapped_count}/{len(mapped_symbols)} symbols to tokens")
            if unmapped_symbols:
                logger.warning(f"Unmapped symbols: {', '.join(unmapped_symbols)}")
            trading_config = self.config.get('trading', {})
            trading_config['symbols'] = mapped_symbols
            self.config['trading'] = trading_config
            # --- PATCH: Populate token_symbol_map for reverse lookup ---
            self.token_symbol_map = {v: k for k, v in self.symbol_token_map.items()}
            # Ensure all tokens are strings for mapping consistency
            self.symbol_token_map = {k: str(v) for k, v in self.symbol_token_map.items()}
            self.token_symbol_map = {str(v): k for k, v in self.symbol_token_map.items()}
            logger.info(f"[PATCH] symbol_token_map: {self.symbol_token_map}")
            logger.info(f"[PATCH] token_symbol_map: {self.token_symbol_map}")
        except Exception as e:
            logger.error(f"Error initializing token mapping: {str(e)}")
            raise
            
    def _initialize_websocket(self):
        """Initialize WebSocket connection for live market data in a non-blocking background thread"""
        def websocket_connect_and_subscribe():
            max_retries = 3
            for attempt in range(1, max_retries + 1):
                try:
                    if not self.angel_api:
                        logger.error("Angel API not initialized, cannot start WebSocket")
                        return
                    angel_config = self.config['apis']['angel_one']
                    websocket_config = {
                        'retry': {
                            'max_attempts': 5,
                            'initial_delay': 10,
                            'multiplier': 2,
                            'max_duration': 60
                        },
                        'mode': self.config.get('data', {}).get('websocket', {}).get('mode', 'QUOTE')
                    }
                    self.websocket = MarketDataWebSocket(
                        auth_token=self.auth_token,
                        api_key=angel_config['api_key'],
                        client_code=angel_config['client_id'],
                        feed_token=self.feed_token,
                        config=websocket_config
                    )
                    self.websocket.add_tick_callback(self._on_market_data)

                    # --- Add subscription response logging ---
                    if hasattr(self.websocket, 'add_subscription_response_callback'):
                        def _on_subscription_response(response):
                            logger.info(f"WebSocket subscription response: {response}")
                        self.websocket.add_subscription_response_callback(_on_subscription_response)
                    # If not supported, log after subscribe() below

                    # --- PATCH: Add generic message callback for diagnostics ---
                    if hasattr(self.websocket, 'add_message_callback'):
                        def _on_any_message(msg):
                            logger.info(f"[WS DIAG] Received raw WS message: {msg}")
                        self.websocket.add_message_callback(_on_any_message)
                    else:
                        logger.warning("MarketDataWebSocket does not support add_message_callback; only tick callbacks will be logged.")

                    logger.info(f"Initiating WebSocket connection (attempt {attempt}/{max_retries})...")
                    self.websocket.connect()
                    # --- PATCH: Aggressive diagnostics for live tick debugging ---
                    symbol_tokens = list(self.symbol_token_map.values())
                    logger.info(f"[PATCH] Subscribing to tokens for watchlist: {symbol_tokens}")
                    if symbol_tokens:
                        logger.info(f"Subscribing to {len(symbol_tokens)} symbols...")
                        mode_map = {
                            'LTP': MarketDataWebSocket.MODE_LTP,
                            'QUOTE': MarketDataWebSocket.MODE_QUOTE,
                            'SNAPQUOTE': MarketDataWebSocket.MODE_SNAPQUOTE
                        }
                        mode_str = self.config.get('data', {}).get('websocket', {}).get('mode', 'QUOTE')
                        if isinstance(mode_str, str):
                            mode_key = mode_str.upper()
                        elif isinstance(mode_str, int):
                            reverse_map = {v: k for k, v in mode_map.items()}
                            mode_key = reverse_map.get(mode_str, 'QUOTE')
                            logger.warning(f"WebSocket mode is int: {mode_str}, mapped to '{mode_key}'")
                        else:
                            logger.warning(f"WebSocket mode is not a string or int: {mode_str} (type: {type(mode_str)}), defaulting to 'QUOTE'")
                            mode_key = 'QUOTE'
                        mode = mode_map.get(mode_key, MarketDataWebSocket.MODE_QUOTE)
                        # --- PATCH: Log subscription request and response in detail ---
                        batch_size = 200
                        for i in range(0, len(symbol_tokens), batch_size):
                            batch = symbol_tokens[i:i+batch_size]
                            logger.info(f"[PATCH] Subscribing batch: {batch} in mode {mode} (type: {type(mode)})")
                            try:
                                response = self.websocket.subscribe(tokens=batch, mode=mode)
                                logger.info(f"[PATCH] Subscription request sent for {len(batch)} symbols in mode {mode}. Response: {response}")
                                # --- PATCH: Force subscribe to RELIANCE.NS for diagnostics ---
                                if 'RELIANCE.NS' not in batch and 'RELIANCE.NS' in self.symbol_token_map:
                                    logger.info("[PATCH] Forcing subscription to RELIANCE.NS for tick diagnostics")
                                    self.websocket.subscribe(tokens=[self.symbol_token_map['RELIANCE.NS']], mode=mode)
                            except Exception as sub_err:
                                logger.error(f"[PATCH] Exception during subscribe(): {sub_err}")
                        # --- PATCH: Start a watchdog thread to log if no ticks received ---
                        def tick_watchdog():
                            time.sleep(15)
                            if hasattr(self.websocket, 'live_feed') and len(self.websocket.live_feed) == 0:
                                logger.warning("[PATCH] No live ticks received in 15 seconds after subscription! Check broker status, token validity, and network.")
                        threading.Thread(target=tick_watchdog, daemon=True).start()
                    else:
                        logger.warning("No tokens to subscribe to!")
                    return  # Success
                except Exception as e:
                    logger.error(f"Error initializing WebSocket (attempt {attempt}/{max_retries}): {str(e)}")
                    if attempt == max_retries:
                        logger.error("WebSocket connection failed after maximum retries. Please check your network, credentials, and broker status.")
                        return
                    else:
                        wait_time = 5 * attempt
                        logger.info(f"Retrying WebSocket connection in {wait_time} seconds...")
                        time.sleep(wait_time)
        # Start the websocket connection in a daemon thread
        t = threading.Thread(target=websocket_connect_and_subscribe, daemon=True)
        t.start()
        logger.info("WebSocket connection started in background thread (non-blocking)")
            
    def _on_market_data(self, tick_data: Dict[str, Any]):
        """Handle incoming market data ticks"""
        try:
            token = tick_data.get('token')
            logger.info(f"[PATCH] Received tick for token: {token}, tick_data: {tick_data}")
            if not token:
                logger.warning("Received market data without token")
                return
            # Convert token to symbol if we have the mapping
            symbol = self.token_symbol_map.get(token)
            logger.info(f"[PATCH] token {token} maps to symbol: {symbol}")
            if not symbol:
                logger.warning(f"Received data for unknown token: {token}")
                return
            # Log market data with proper formatting
            log_msg = (
                f"Market Data - {symbol}: "
                f"LTP: ₹{tick_data.get('ltp', 0):.2f}, "
                f"Time: {tick_data.get('timestamp', datetime.now()).strftime('%H:%M:%S')}"
            )
            if 'volume' in tick_data:
                log_msg += f", Vol: {tick_data['volume']:,}"
            if 'open_interest' in tick_data:
                log_msg += f", OI: {tick_data['open_interest']:,}"
            logger.info(log_msg)
            # Print full tick data for debug
            logger.debug(f"Full tick data: {tick_data}")
            # --- PATCH: Store latest tick in self._live_quotes with lock ---
            if hasattr(self, '_live_quotes') and hasattr(self, '_live_quotes_lock'):
                with self._live_quotes_lock:
                    self._live_quotes[symbol] = tick_data.copy()
        except Exception as e:
            logger.error(f"Error processing market data: {str(e)}")

    @with_rate_limit(max_retries=3, initial_delay=1.0)
    def get_ltp(self, symbol: str) -> Optional[dict]:
        """Get Last Traded Price for a symbol with enhanced error handling"""
        try:
            # Handle .NS extension and get token
            base_symbol = symbol.replace('.NS', '')
            token = self.symbol_token_map.get(symbol) or self.symbol_token_map.get(base_symbol)
            
            if not token:
                logger.error(f"No token found for symbol {symbol}")
                return None
            
            # Try to get quote data
            quote_data = self.angel_api.ltpData("NSE", base_symbol + "-EQ", str(token))
            
            if quote_data and quote_data.get('data'):
                data = quote_data['data']
                return {
                    'ltp': float(data.get('ltp', 0)),
                    'volume': int(data.get('volume', 0)) if 'volume' in data else 0,
                    'change': float(data.get('netPrice', 0)) if 'netPrice' in data else 0,
                }
            else:
                logger.warning(f"No LTP data available for {symbol}")
                return None
                
        except Exception as e:
            logger.error(f"Error fetching LTP data for {symbol}: {str(e)}")
            return None
            
    @with_rate_limit(max_retries=3, initial_delay=1.0)
    def get_quote(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get detailed quote for a symbol"""
        try:
            token = self.symbol_token_map.get(symbol)
            if not token:
                raise ValueError(f"No token found for symbol {symbol}")
                
            quote_data = self.angel_api.ltpData("NSE", symbol.replace('.NS', '-EQ'), token)
            
            if quote_data and quote_data.get('data'):
                # Clean and format data
                data = quote_data['data']
                return {
                    'symbol': symbol,
                    'ltp': float(data.get('ltp', 0)),
                    'volume': int(data.get('volume', 0)),
                    'change': float(data.get('change', 0)),
                    'change_percent': float(data.get('change_percent', 0)),
                    'timestamp': datetime.fromtimestamp(int(data.get('exchange_time', time.time() * 1000)) / 1000),                }
                
                logger.warning(f"No quote data available for {symbol}")
                return None
            
        except Exception as e:
            logger.error(f"Error fetching quote data for {symbol}: {str(e)}")
            raise
    @with_rate_limit(max_retries=3, initial_delay=1.0)
    def get_market_status(self) -> Dict[str, bool]:
        """Get market status with enhanced error handling and rate limit awareness"""
        try:
            if not self.angel_api:
                logger.error("Angel API not initialized")
                return {'NSE': False}
            
            # Get current time in IST
            ist_now = datetime.now()
            
            # Define market hours (IST)
            market_open = ist_now.replace(hour=9, minute=15, second=0, microsecond=0)
            market_close = ist_now.replace(hour=15, minute=30, second=0, microsecond=0)
            
            # Check if it's a weekday
            is_weekday = ist_now.weekday() < 5
            
            # Check if within market hours
            within_hours = market_open <= ist_now <= market_close
            
            # Market is open if it's a weekday and within trading hours
            is_open = is_weekday and within_hours
            
            if not is_open:
                reason = "weekend" if not is_weekday else "outside trading hours"
                logger.info(f"Market is closed ({reason})")
            
            return {'NSE': is_open}
            if not market_status:
                logger.warning("No market status data in response")
                return {'NSE': False}
            
            status = {}
            for segment in market_status:
                if segment.get('exchange') == 'NSE':
                    is_open = segment.get('marketStatus', '').lower() == 'open'
                    status['NSE'] = is_open
                    logger.info(f"NSE market status: {'Open' if is_open else 'Closed'}")
                    break
            
            return status or {'NSE': False}
            
        except Exception as e:
            logger.error(f"Error fetching market status: {str(e)}")
            return {'NSE': False}  # Assume closed on error

    def get_market_data(self, symbols=None, days=10):
        """Fetch historical market data for specified symbols or all configured symbols.
        
        Args:
            symbols (str or list, optional): Specific symbol(s) to fetch data for. 
                                          If None, fetches for all configured symbols.
            days (int): Number of days of historical data to fetch.
            
        Returns:
            pd.DataFrame: Historical market data
        """
        try:
            # Handle single symbol case
            if isinstance(symbols, str):
                return self.get_historical_data(symbols, days)
            
            # Get symbols list
            if not symbols:
                trading_config = self.config.get('trading', {})
                data_config = trading_config.get('data', {})
                
                if data_config.get('mode') == 'manual':
                    symbols = data_config.get('manual_symbols', [])
                else:
                    symbols = data_config.get('manual_list', [])
                
            if not symbols:
                logger.warning("No symbols configured for data collection")
                return pd.DataFrame()
            
            # Fetch data for each symbol
            all_data = []
            for sym in symbols:
                try:
                    df = self.get_historical_data(sym, days)
                    if not df.empty:
                        df['symbol'] = sym  # Add symbol column
                        all_data.append(df)
                except Exception as e:
                    logger.warning(f"Error fetching data for {sym}: {e}")
            
            if not all_data:
                logger.warning("No data available for any configured symbols")
                return pd.DataFrame()
            
            # Combine all data frames
            result = pd.concat(all_data, axis=0)
            logger.info(f"Retrieved market data for {len(all_data)} symbols")
            return result
            
        except Exception as e:
            logger.error(f"Error in get_market_data: {e}")
            return pd.DataFrame()
        
    def get_historical_data(self, symbol, days=10, interval='ONE_DAY'):
        import time
        from datetime import datetime, timedelta
        # --- Rate limit constants (official) ---
        HIST_API_LIMIT_SEC = 3
        HIST_API_LIMIT_MIN = 180
        HIST_API_LIMIT_HR = 5000

        def _rate_limit_wait():
            now = time.time()
            with self._hist_api_lock:
                # Clean up old timestamps
                self._hist_api_calls_sec = [t for t in self._hist_api_calls_sec if t > now - 1]
                self._hist_api_calls_min = [t for t in self._hist_api_calls_min if t > now - 60]
                self._hist_api_calls_hr = [t for t in self._hist_api_calls_hr if t > now - 3600]
                # If any limit exceeded, calculate wait time
                wait_times = []
                if len(self._hist_api_calls_sec) >= HIST_API_LIMIT_SEC:
                    wait_times.append(1 - (now - self._hist_api_calls_sec[0]))
                if len(self._hist_api_calls_min) >= HIST_API_LIMIT_MIN:
                    wait_times.append(60 - (now - self._hist_api_calls_min[0]))
                if len(self._hist_api_calls_hr) >= HIST_API_LIMIT_HR:
                    wait_times.append(3600 - (now - self._hist_api_calls_hr[0]))
                if wait_times:
                    wait_time = max(wait_times)
                    logger.warning(f"[RATE LIMIT] Waiting {wait_time:.2f}s to comply with Angel One historical data API limits.")
                    time.sleep(max(wait_time, 0.01))
                # Register this call
                self._hist_api_calls_sec.append(now)
                self._hist_api_calls_min.append(now)
                self._hist_api_calls_hr.append(now)

        """
        Fetch historical OHLCV data for a symbol, with candlestick pattern detection.
        This method should be called by all model/data consumers.
        """
        try:
            if self.angel_api is None:
                logger.error("Angel One API not initialized.")
                return pd.DataFrame()
            end_date = datetime.now()
            min_date = datetime(2000, 1, 1)
            if days < 1:
                logger.warning(f"Requested days < 1, adjusting to 1.")
                days = 1
            start_date = end_date - timedelta(days=days)
            if start_date < min_date:
                logger.warning(f"Requested start_date {start_date.date()} before 2000-01-01, adjusting to 2000-01-01.")
                start_date = min_date
            if end_date > datetime.now():
                logger.warning(f"Requested end_date {end_date.date()} is in the future, adjusting to today.")
                end_date = datetime.now()
            if interval == 'ONE_DAY':
                start_str = start_date.strftime('%Y-%m-%d') + ' 09:15'
                end_str = end_date.strftime('%Y-%m-%d') + ' 15:30'
            else:
                start_str = start_date.strftime('%Y-%m-%d %H:%M')
                end_str = end_date.strftime('%Y-%m-%d %H:%M')
            logger.info(f"Requesting historical data for {symbol} from {start_str} to {end_str} (interval: {interval})")
            token = self.symbol_token_map.get(symbol)
            if token is None:
                token = self.symbol_token_map.get(symbol.replace('.NS', ''))
            if token is None:
                token = self.symbol_token_map.get(symbol.replace('.NS', '-EQ'))
            if token is None:
                logger.error(f"No token found for symbol {symbol} (tried direct, base, and -EQ)")
                return pd.DataFrame()
            params = {
                "exchange": "NSE",
                "symboltoken": token,
                "interval": interval,
                "fromdate": start_str,
                "todate": end_str,
            }
            max_retries = 5
            retry_wait = self.config.get('api_rate_limit_wait', 2)
            for attempt in range(max_retries):
                _rate_limit_wait()
                response = self.angel_api.getCandleData(params)
                if response and 'data' in response:
                    candles = response['data']
                    df = pd.DataFrame(candles, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                    df['timestamp'] = pd.to_datetime(df['timestamp'])
                    # --- DIAGNOSTIC: OHLCV completeness check ---
                    expected_rows = self._calculate_expected_rows(start_date, end_date, interval)
                    actual_rows = len(df)
                    missing_timestamps = []
                    if 'timestamp' in df.columns:
                        df_sorted = df.sort_values('timestamp')
                        all_expected = pd.date_range(start=start_date, end=end_date, freq=self._interval_to_pandas_freq(interval))
                        actual_set = set(pd.to_datetime(df_sorted['timestamp']))
                        missing_timestamps = [ts for ts in all_expected if ts not in actual_set]
                    logger.info(f"[DIAGNOSTIC] {symbol}: Expected rows={expected_rows}, Actual rows={actual_rows}, Missing timestamps={len(missing_timestamps)}")
                    if missing_timestamps:
                        logger.warning(f"[DIAGNOSTIC] {symbol}: Missing timestamps: {missing_timestamps[:10]}{'...' if len(missing_timestamps)>10 else ''}")
                    if actual_rows < expected_rows:
                        logger.warning(f"[DIAGNOSTIC] {symbol}: Fetched fewer rows than expected. Possible API/data issue.")
                    for col in ['open', 'high', 'low', 'close', 'volume']:
                        df[col] = pd.to_numeric(df[col], errors='coerce')
                    # Verification log: check required columns
                    required_cols = ['open', 'high', 'low', 'close', 'volume']
                    missing_cols = [col for col in required_cols if col not in df.columns]
                    if missing_cols:
                        logger.error(f"Missing columns {missing_cols} in historical data for {symbol}. Failing fetch.")
                        return pd.DataFrame()
                    if df.isnull().any().any():
                        logger.warning(f"Null values found in historical data for {symbol}. Shape: {df.shape}")
                    if len(df) < 60:
                        logger.error(f"Insufficient data points ({len(df)}) for {symbol}. Failing fetch.")
                        return pd.DataFrame()
                    df = self.detect_candlestick_patterns(df)
                    if 'pattern' not in df.columns:
                        logger.warning(f"Pattern detection failed for {symbol}, adding default 'None' values.")
                        df['pattern'] = 'None'
                    else:
                        logger.info(f"Pattern detection completed for {symbol}. Example patterns: {df['pattern'].value_counts().to_dict()}")
                    return df
                else:
                    if response and b'access rate' in str(response).lower():
                        logger.warning(f"Rate limit exceeded for {symbol}. Waiting {retry_wait} seconds before retry.")
                        time.sleep(retry_wait)
                    else:
                        logger.error(f"No data returned for {symbol} from API. Attempt {attempt+1}/{max_retries}")
                        time.sleep(retry_wait)
            logger.error(f"Failed to fetch historical data for {symbol} after {max_retries} retries.")
            return pd.DataFrame()
        except Exception as e:
            logger.error(f"Error fetching historical data for {symbol}: {e}")
            return pd.DataFrame()

    def _interval_to_pandas_freq(self, interval):
        # Helper to convert API interval to pandas frequency string
        mapping = {
            'ONE_DAY': 'B',  # business day
            'ONE_MINUTE': 'T',
            'FIVE_MINUTE': '5T',
            'FIFTEEN_MINUTE': '15T',
            'THIRTY_MINUTE': '30T',
            'ONE_HOUR': 'H',
        }
        return mapping.get(interval, 'B')

    def _calculate_expected_rows(self, start_date, end_date, interval):
        # Helper to estimate expected number of rows for diagnostics
        freq = self._interval_to_pandas_freq(interval)
        rng = pd.date_range(start=start_date, end=end_date, freq=freq)
        return len(rng)

    # Websocket modes
    MODE_LTP = 1
    MODE_QUOTE = 2
    MODE_SNAPQUOTE = 3
        
    def _convert_ltp_data(self, data: Any) -> Optional[dict]:
        """Safely convert LTP data to required format"""
        try:
            if isinstance(data, (int, float)):
                return {'ltp': float(data), 'change': 0.0, 'volume': 0}
            elif isinstance(data, dict):
                return {
                    'ltp': float(data.get('ltp', 0)),
                    'change': float(data.get('change', 0)),
                    'volume': int(data.get('volume', 0))
                }
            return None
        except (TypeError, ValueError) as e:
            logger.error(f"Error converting LTP data: {e}")
            return None
        
    def _validate_token_mapping(self, symbol: str) -> Optional[str]:
        """Validate and get token for symbol with better error handling"""
        try:
            if not symbol:
                logger.error("Empty symbol provided")
                return None
                
            token = self.symbol_token_map.get(symbol)
            if not token:
                # Try alternative symbol formats
                alt_symbol = symbol.replace('.NS', '') if '.NS' in symbol else f"{symbol}.NS"
                token = self.symbol_token_map.get(alt_symbol)
                
            if not token:
                logger.error(f"No token found for symbol {symbol}")
                return None
                
            return str(token)
            
        except Exception as e:
            logger.error(f"Error validating token for {symbol}: {e}")
            return None
        
    def get_symbols_from_config(self) -> List[str]:
        """Get list of symbols based on trading mode from config, with robust auto-discovery for 'auto' mode using all NSE EQ stocks for swing trading."""
        import requests
        trading_config = self.config.get('trading', {})
        data_config = trading_config.get('data', {})
        mode = data_config.get('mode', 'manual')
        symbols = []
        if mode == 'manual':
            # Get manually configured symbols
            symbols = data_config.get('manual_symbols', [])
        elif mode == 'auto':
            # --- ROBUST: Use all NSE EQ stocks from ScripMaster for swing trading, log structure if fails ---
            try:
                instrument_url = "https://margincalculator.angelbroking.com/OpenAPI_File/files/OpenAPIScripMaster.json"
                instr_response = requests.get(instrument_url, timeout=30)
                instr_response.raise_for_status()
                instruments = instr_response.json()
                logger.debug(f"Loaded {len(instruments)} instruments from ScripMaster file")
                # Log all unique instrumenttype values
                unique_types = set()
                for inst in instruments:
                    inst_type = (inst.get('instrumenttype') or inst.get('instrumentType') or '').upper()
                    unique_types.add(inst_type)
                logger.warning(f"Unique instrumenttype values in ScripMaster: {sorted(unique_types)}")
                # Log first 20 non-AMXIDX records
                non_index = [inst for inst in instruments if (inst.get('instrumenttype') or '').upper() != 'AMXIDX']
                logger.warning(f"First 20 non-AMXIDX records: {non_index[:20]}")
                eligible = []
                for inst in instruments:
                    exch_seg = (inst.get('exch_seg') or inst.get('exchSeg') or '').upper()
                    inst_type = (inst.get('instrumenttype') or inst.get('instrumentType') or '')
                    symbol = (inst.get('symbol') or inst.get('Symbol') or '')
                    if exch_seg == 'NSE' and inst_type == '' and symbol.endswith('-EQ'):
                        eligible.append(symbol + '.NS')
                logger.debug(f"Found {len(eligible)} NSE EQ stocks for swing trading. Example: {eligible[:10]}")
                if not eligible:
                    logger.warning("Auto-discovery found no eligible stocks, using manual_symbols as fallback.")
                    logger.warning(f"First 5 ScripMaster records: {instruments[:5]}")
                    unique_keys = set()
                    for rec in instruments[:50]:
                        unique_keys.update(rec.keys())
                    logger.warning(f"Unique keys in first 50 records: {sorted(unique_keys)}")
                    symbols = data_config.get('manual_symbols', self._get_default_symbols())
                else:
                    symbols = eligible
            except Exception as e:
                logger.error(f"Failed to fetch NSE EQ stocks for auto-discovery: {e}")
                symbols = data_config.get('manual_symbols', self._get_default_symbols())
        # Always add configured indices (for live, not for training)
        indices = data_config.get('indices', [])
        # Only add indices if not in training context (let training scripts filter them out)
        if not self.config.get('skip_indices', False):
            symbols.extend(indices)
        # Remove any excluded symbols
        excluded = data_config.get('exclude_symbols', [])
        symbols = [s for s in symbols if s not in excluded]
        # Remove duplicates while preserving order
        return list(dict.fromkeys(symbols))
        
    def _get_default_symbols(self) -> List[str]:
        """Get default symbols list when no specific symbols are configured"""
        return [
            'RELIANCE.NS',
            'TCS.NS',
            'HDFCBANK.NS',
            'INFY.NS',
            'ICICIBANK.NS'
        ]
    
    def get_last_known_ltp(self, symbol: str) -> Optional[dict]:
        """Get last known LTP for a symbol from historical data as fallback"""
        try:
            # Try to get the most recent close price from historical data
            df = self.get_historical_data(symbol, interval='1d', days=1)
            if df is not None and not df.empty:
                last_row = df.iloc[-1]
                return {
                    'ltp': last_row['close'],
                    'timestamp': last_row['timestamp']
                }
        except Exception as e:
            logger.error(f"Error fetching last known LTP for {symbol}: {str(e)}")
        return None

    def _check_angel_one_credentials(self):
        # Log the config path and values for debugging
        config_path = 'self.config["apis"]["angel_one"]'
        angel_config = self.config.get('apis', {}).get('angel_one', {})
        logger.info(f"Checking Angel One credentials at {config_path}: {angel_config}")
        required = ['api_key', 'client_id', 'mpin', 'totp_secret']
        missing = [k for k in required if not angel_config.get(k)]
        if missing:
            logger.error(f"Missing Angel One credentials: {', '.join(missing)} (from {config_path})")
            return False
        # Mask sensitive info for logging
        safe_config = {k: (v[:3] + '***' if isinstance(v, str) and len(v) > 3 else '***') for k, v in angel_config.items()}
        logger.info(f"Angel One credentials loaded: {safe_config}")
        return True

    @with_rate_limit()
    def get_broker_portfolio(self):
        """
        Fetch live portfolio from Angel One broker for portfolio sync and dashboard.
        Uses robust fallback logic as in test_portfolio.py.
        Returns a normalized list of holdings with standard keys for dashboard display.
        """
        import pkg_resources
        import pyotp
        import requests
        import socket
        import uuid
        try:
            angel_config = self.config['apis']['angel_one']
            from SmartApi import SmartConnect
            smart_api = SmartConnect(angel_config['api_key'])
            totp = pyotp.TOTP(angel_config['totp_secret']).now()
            session_resp = smart_api.generateSession(angel_config['client_id'], angel_config['mpin'], totp)
            logger.info(f"generateSession response: {session_resp}")
            if not session_resp or not session_resp.get('status'):
                logger.error(f"Failed to authenticate with Angel One: {session_resp}")
                return []
            holdings = []
            if hasattr(smart_api, 'getAllHolding'):
                holdings_raw = smart_api.getAllHolding()
                logger.info(f"Raw getAllHolding API response: {holdings_raw}")
                if not holdings_raw or 'data' not in holdings_raw or 'holdings' not in holdings_raw['data']:
                    logger.warning("No holdings data returned from Angel One getAllHolding. Trying getHolding() if available.")
                    if hasattr(smart_api, 'getHolding'):
                        try:
                            holdings_raw = smart_api.getHolding()
                            logger.info(f"Raw getHolding API response: {holdings_raw}")
                            if holdings_raw and 'data' in holdings_raw and 'holdings' in holdings_raw['data']:
                                holdings = holdings_raw['data']['holdings']
                            else:
                                holdings = []
                        except Exception as e:
                            logger.error(f"Exception calling getHolding: {e}")
                            holdings = []
                    else:
                        holdings = []
                else:
                    holdings = holdings_raw['data']['holdings']
            else:
                # Fallback: Use HTTP API directly as per Angel One documentation
                logger.info("SDK does not support getAllHolding(). Using HTTP fallback.")
                try:
                    jwt_token = session_resp.get('data', {}).get('jwtToken')
                    if not jwt_token:
                        logger.error("No jwtToken found in session response. Cannot fetch holdings.")
                        holdings = []
                    else:
                        # Remove 'Bearer ' prefix if present
                        if jwt_token.startswith('Bearer '):
                            jwt_token_clean = jwt_token[len('Bearer '):]
                        else:
                            jwt_token_clean = jwt_token
                        # Get real local IP
                        try:
                            local_ip = socket.gethostbyname(socket.gethostname())
                        except Exception:
                            local_ip = "127.0.0.1"
                        # Get public IP
                        try:
                            public_ip = requests.get('https://api.ipify.org').text
                        except Exception:
                            public_ip = "127.0.0.1"
                        # Get MAC address
                        try:
                            mac = uuid.getnode()
                            mac_str = ':'.join(['{:02x}'.format((mac >> ele) & 0xff) for ele in range(40, -1, -8)])
                        except Exception:
                            mac_str = "00:00:00:00:00:00"
                        url = "https://apiconnect.angelone.in/rest/secure/angelbroking/portfolio/v1/getHolding"
                        headers = {
                            "Authorization": f"Bearer {jwt_token_clean}",
                            "X-UserType": "USER",
                            "X-SourceID": "WEB",
                            "X-ClientLocalIP": local_ip,
                            "X-ClientPublicIP": public_ip,
                            "X-MACAddress": mac_str,
                            "X-PrivateKey": angel_config['api_key'],
                            "Accept": "application/json",
                            "Content-Type": "application/json",
                            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36"
                        }
                        logger.info(f"DEBUG: Using jwtToken: {jwt_token_clean}")
                        logger.info(f"DEBUG: HTTP headers: {headers}")
                        resp = requests.get(url, headers=headers)
                        logger.info(f"HTTP Holdings API response: {resp.status_code} {resp.text}")
                        try:
                            data = resp.json()
                        except Exception as e:
                            logger.error(f"ERROR: Could not parse JSON from holdings API response: {e}")
                            holdings = []
                        else:
                            # Angel One sometimes returns data as a list, sometimes as a dict with 'holdings' key
                            if data.get('status') and 'data' in data:
                                if isinstance(data['data'], list):
                                    holdings = data['data']
                                elif isinstance(data['data'], dict) and 'holdings' in data['data']:
                                    holdings = data['data']['holdings']
                                else:
                                    logger.warning("No holdings found in HTTP API response.")
                                    holdings = []
                            else:
                                logger.warning("No holdings found in HTTP API response.")
                                holdings = []
                except Exception as e:
                    logger.error(f"Exception during HTTP fallback for holdings: {e}")
                    holdings = []
            logger.info(f"Your Angel One Holdings (raw): {holdings}")
            # Optionally, show normalized output as in project code
            normalized = []
            for h in holdings:
                try:
                    qty = float(h.get('quantity', 0))
                    if qty <= 0:
                        continue
                    normalized.append({
                        'symbol': h.get('tradingsymbol') or h.get('symbol'),
                        'exchange': h.get('exchange'),
                        'quantity': qty,
                        'avg_price': float(h.get('averageprice', 0)),
                        'ltp': float(h.get('ltp', 0)),
                        'current_value': float(h.get('currentvalue', 0)),
                        'pnl': float(h.get('pnl', 0)),
                        'day_change': float(h.get('daychange', 0)),
                        'day_change_percent': float(h.get('daychangepercentage', 0)),
                        'isin': h.get('isin'),
                    })
                except Exception as e:
                    logger.error(f"Error normalizing holding: {h} | {e}")
            if not normalized:
                logger.warning("No normalized holdings found.")
            else:
                logger.info(f"Normalized Holdings for Dashboard: {normalized}")
            return normalized
        except Exception as e:
            logger.error(f"Error fetching holdings from Angel One: {e}")
            return []
    @with_rate_limit()
    def get_broker_balance(self):
        """
        Fetch live trading balance (funds/margins) from Angel One broker for dashboard and trading logic.
        Uses robust fallback logic as in test_portfolio.py.
        Returns a normalized dict with standard keys for dashboard display.
        """
        import pkg_resources
        import pyotp
        import requests
        import socket
        import uuid
        try:
            angel_config = self.config['apis']['angel_one']
            from SmartApi import SmartConnect
            smart_api = SmartConnect(angel_config['api_key'])
            totp = pyotp.TOTP(angel_config['totp_secret']).now()
            session_resp = smart_api.generateSession(angel_config['client_id'], angel_config['mpin'], totp)
            logger.info(f"generateSession response (balance): {session_resp}")
            if not session_resp or not session_resp.get('status'):
                logger.error(f"Failed to authenticate with Angel One for balance: {session_resp}")
                return {'error': 'Authentication failed', 'errorcode': session_resp.get('errorcode', 'AUTH'), 'message': session_resp.get('message', 'Authentication failed')}
            # Remove 'Bearer ' prefix if present
            jwt_token = session_resp.get('data', {}).get('jwtToken')
            if not jwt_token:
                logger.error("No jwtToken found in session response. Cannot fetch balance.")
                return {'error': 'No jwtToken found', 'errorcode': 'NO_TOKEN', 'message': 'No jwtToken found in session response.'}
            if jwt_token.startswith('Bearer '):
                jwt_token_clean = jwt_token[len('Bearer '):]
            else:
                jwt_token_clean = jwt_token
            # Get real local IP
            try:
                local_ip = socket.gethostbyname(socket.gethostname())
            except Exception:
                local_ip = "127.0.0.1"
            # Get public IP
            try:
                public_ip = requests.get('https://api.ipify.org').text
            except Exception:
                public_ip = "127.0.0.1"
            # Get MAC address
            try:
                mac = uuid.getnode()
                mac_str = ':'.join(['{:02x}'.format((mac >> ele) & 0xff) for ele in range(40, -1, -8)])
            except Exception:
                mac_str = "00:00:00:00:00:00"
            url = "https://apiconnect.angelone.in/rest/secure/angelbroking/user/v1/getRMS"
            headers = {
                "Authorization": f"Bearer {jwt_token_clean}",
                "X-UserType": "USER",
                "X-SourceID": "WEB",
                "X-ClientLocalIP": local_ip,
                "X-ClientPublicIP": public_ip,
                "X-MACAddress": mac_str,
                "X-PrivateKey": angel_config['api_key'],
                "Accept": "application/json",
                "Content-Type": "application/json",
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36"
            }
            logger.info(f"DEBUG (balance): Using jwtToken: {jwt_token_clean}")
            logger.info(f"DEBUG (balance): HTTP headers: {headers}")
            resp = requests.get(url, headers=headers)
            logger.info(f"HTTP RMS API response: {resp.status_code} {resp.text}")
            try:
                data = resp.json()
            except Exception as e:
                logger.error(f"ERROR: Could not parse JSON from RMS API response: {e}")
                return {'error': 'Invalid JSON', 'errorcode': 'PARSE', 'message': str(e)}
            if data.get('status') and 'data' in data:
                funds = data['data']
                normalized = {
                    'available_cash': funds.get('availablecash'),
                    'net': funds.get('net'),
                    'utilised_debits': funds.get('utiliseddebits'),
                    'collateral': funds.get('collateral'),
                    'span': funds.get('span'),
                    'exposure': funds.get('exposure'),
                    'turnover': funds.get('turnover'),
                    'payin': funds.get('payin'),
                    'payout': funds.get('payout'),
                    'unrealizedpnl': funds.get('unrealizedpnl'),
                    'realizedpnl': funds.get('realizedpnl'),
                }
                logger.info(f"Normalized RMS balance for Dashboard: {normalized}")
                return normalized
            else:
                # Return error code and message from API response if present
                logger.warning(f"No balance found in RMS API response: {data}")
                return {
                    'error': data.get('message', 'No balance found'),
                    'errorcode': data.get('errorcode', 'NO_DATA'),
                    'message': data.get('message', 'No balance found')
                }
        except Exception as e:
            logger.error(f"Error fetching trading balance from Angel One: {e}")
            return {'error': str(e), 'errorcode': 'EXCEPTION', 'message': str(e)}
    @with_rate_limit()
    def get_broker_orders(self):
        """
        Fetch live order book from Angel One broker for dashboard and trading logic.
        Returns a normalized list of orders with standard keys for dashboard display.
        """
        import pkg_resources
        import pyotp
        import requests
        import socket
        import uuid
        try:
            angel_config = self.config['apis']['angel_one']
            from SmartApi import SmartConnect
            smart_api = SmartConnect(angel_config['api_key'])
            totp = pyotp.TOTP(angel_config['totp_secret']).now()
            session_resp = smart_api.generateSession(angel_config['client_id'], angel_config['mpin'], totp)
            logger.info(f"generateSession response (orders): {session_resp}")
            if not session_resp or not session_resp.get('status'):
                logger.error(f"Failed to authenticate with Angel One for orders: {session_resp}")
                return []
            # Remove 'Bearer ' prefix if present
            jwt_token = session_resp.get('data', {}).get('jwtToken')
            if not jwt_token:
                logger.error("No jwtToken found in session response. Cannot fetch orders.")
                return []
            if jwt_token.startswith('Bearer '):
                jwt_token_clean = jwt_token[len('Bearer '):]
            else:
                jwt_token_clean = jwt_token
            # Get real local IP
            try:
                local_ip = socket.gethostbyname(socket.gethostname())
            except Exception:
                local_ip = "127.0.0.1"
            # Get public IP
            try:
                public_ip = requests.get('https://api.ipify.org').text
            except Exception:
                public_ip = "127.0.0.1"
            # Get MAC address
            try:
                mac = uuid.getnode()
                mac_str = ':'.join(['{:02x}'.format((mac >> ele) & 0xff) for ele in range(40, -1, -8)])
            except Exception:
                mac_str = "00:00:00:00:00:00"
            url = "https://apiconnect.angelone.in/rest/secure/angelbroking/order/v1/getOrderBook"
            headers = {
                "Authorization": f"Bearer {jwt_token_clean}",
                "X-UserType": "USER",
                "X-SourceID": "WEB",
                "X-ClientLocalIP": local_ip,
                "X-ClientPublicIP": public_ip,
                "X-MACAddress": mac_str,
                "X-PrivateKey": angel_config['api_key'],
                "Accept": "application/json",
                "Content-Type": "application/json",
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36"
            }
            logger.info(f"DEBUG (orders): Using jwtToken: {jwt_token_clean}")
            logger.info(f"DEBUG (orders): HTTP headers: {headers}")
            resp = requests.get(url, headers=headers)
            logger.info(f"HTTP OrderBook API response: {resp.status_code} {resp.text}")
            try:
                data = resp.json()
            except Exception as e:
                logger.error(f"ERROR: Could not parse JSON from OrderBook API response: {e}")
                return []
            orders = []
            if data.get('status') and 'data' in data:
                raw_orders = data['data']
                for o in raw_orders:
                    try:
                        orders.append({
                            'order_id': o.get('orderid'),
                            'symbol': o.get('tradingsymbol'),
                            'exchange': o.get('exchange'),
                            'transaction_type': o.get('transactiontype'),
                            'quantity': o.get('quantity'),
                            'filled_quantity': o.get('filledquantity'),
                            'pending_quantity': o.get('pendingquantity'),
                            'order_status': o.get('orderstatus'),
                            'order_type': o.get('ordertype'),
                            'product_type': o.get('producttype'),
                            'price': o.get('price'),
                            'product_type': o.get('producttype'),
                            'price': o.get('price'),
                            'trigger_price': o.get('triggerprice'),
                            'average_price': o.get('averageprice'),
                            'order_time': o.get('orderentrytime'),
                        })
                    except Exception as e:
                        logger.error(f"Error normalizing order: {o} | {e}")
            else:
                logger.warning("No orders found in OrderBook API response.")
            logger.info(f"Normalized Orders for Dashboard: {orders}")
            return orders
        except Exception as e:
            logger.error(f"Error fetching order book from Angel One: {e}")
            return []
    
    def get_latest_price(self, symbol: str, exchange: str = 'NSE') -> float:
        """
        Return the latest price for the given symbol. Tries WebSocket (if live), then REST API, then historical data as fallback.
        Args:
            symbol (str): Trading symbol (e.g., 'RELIANCE.NS')
            exchange (str): Exchange name (default 'NSE')
        Returns:
            float: Latest price (LTP/close), or None if unavailable
        """
        # Try WebSocket live data first
        try:
            if hasattr(self, 'websocket') and self.websocket and self.websocket.is_connected:
                token = self.symbol_token_map.get(symbol)
                if token:
                    ws_data = self.websocket.get_market_data(str(token))
                    if ws_data and 'ltp' in ws_data:
                        return float(ws_data['ltp'])
        except Exception as e:
            logger.warning(f"WebSocket price fetch failed for {symbol}: {e}")
        # Try REST API (LTP)
        try:
            ltp_data = self.get_ltp(symbol)
        except Exception as e:
            logger.warning(f"REST LTP fetch failed for {symbol}: {e}")
        # Fallback: use most recent close from historical data
        try:
            hist = self.get_historical_data(symbol, interval='ONE_DAY', days=1)
            if not hist.empty:
                return float(hist.iloc[-1]['close'])
        except Exception as e:
            logger.warning(f"Historical price fetch failed for {symbol}: {e}")
        logger.error(f"No price available for {symbol}")
        return None

    def detect_candlestick_patterns(self, df):
        """
        Detect common candlestick patterns using TA-Lib and annotate the DataFrame.
        Adds a 'pattern' column with the detected pattern name (or '' if none).
        """
        import talib
        all_patterns = {
            'CDL2CROWS': talib.CDL2CROWS,
            'CDL3BLACKCROWS': talib.CDL3BLACKCROWS,
            'CDL3INSIDE': talib.CDL3INSIDE,
            'CDL3LINESTRIKE': talib.CDL3LINESTRIKE,
            'CDL3OUTSIDE': talib.CDL3OUTSIDE,
            'CDL3STARSINSOUTH': talib.CDL3STARSINSOUTH,
            'CDL3WHITESOLDIERS': talib.CDL3WHITESOLDIERS,
            'CDLABANDONEDBABY': talib.CDLABANDONEDBABY,
            'CDLADVANCEBLOCK': talib.CDLADVANCEBLOCK,
            'CDLBELTHOLD': talib.CDLBELTHOLD,
            'CDLBREAKAWAY': talib.CDLBREAKAWAY,
            'CDLCLOSINGMARUBOZU': talib.CDLCLOSINGMARUBOZU,
            'CDLCONCEALBABYSWALL': talib.CDLCONCEALBABYSWALL,
            'CDLCOUNTERATTACK': talib.CDLCOUNTERATTACK,
            'CDLDARKCLOUDCOVER': talib.CDLDARKCLOUDCOVER,
            'CDLDOJI': talib.CDLDOJI,
            'CDLDOJISTAR': talib.CDLDOJISTAR,
            'CDLDRAGONFLYDOJI': talib.CDLDRAGONFLYDOJI,
            'CDLENGULFING': talib.CDLENGULFING,
            'CDLEVENINGDOJISTAR': talib.CDLEVENINGDOJISTAR,
            'CDLEVENINGSTAR': talib.CDLEVENINGSTAR,
            'CDLGAPSIDESIDEWHITE': talib.CDLGAPSIDESIDEWHITE,
            'CDLGRAVESTONEDOJI': talib.CDLGRAVESTONEDOJI,
            'CDLHAMMER': talib.CDLHAMMER,
            'CDLHANGINGMAN': talib.CDLHANGINGMAN,
            'CDLHARAMI': talib.CDLHARAMI,
            'CDLHARAMICROSS': talib.CDLHARAMICROSS,
            'CDLHIGHWAVE': talib.CDLHIGHWAVE,
            'CDLHIKKAKE': talib.CDLHIKKAKE,
            'CDLHIKKAKEMOD': talib.CDLHIKKAKEMOD,
            'CDLHOMINGPIGEON': talib.CDLHOMINGPIGEON,
            'CDLIDENTICAL3CROWS': talib.CDLIDENTICAL3CROWS,
            'CDLINNECK': talib.CDLINNECK,
            'CDLINVERTEDHAMMER': talib.CDLINVERTEDHAMMER,
            'CDLKICKING': talib.CDLKICKING,
            'CDLKICKINGBYLENGTH': talib.CDLKICKINGBYLENGTH,
            'CDLLADDERBOTTOM': talib.CDLLADDERBOTTOM,
            'CDLLONGLEGGEDDOJI': talib.CDLLONGLEGGEDDOJI,
            'CDLLONGLINE': talib.CDLLONGLINE,
            'CDLMARUBOZU': talib.CDLMARUBOZU,
            'CDLMATCHINGLOW': talib.CDLHMATCHINGLOW,
            'CDLMATHOLD': talib.CDLHMATHOLD,
            'CDLMORNINGDOJISTAR': talib.CDLMORNINGDOJISTAR,
            'CDLMORNINGSTAR': talib.CDLMORNINGSTAR,
            'CDLONNECK': talib.CDLONNECK,
            'CDLPPIERCING': talib.CDLPIERCING,
            'CDLRICKSHAWMAN': talib.CDLRICKSHAWMAN,
            'CDLRISEFALL3METHODS': talib.CDLRISEFALL3METHODS,
            'CDLSEPARATINGLINES': talib.CDLSEPARATINGLINES,
            'CDLSHOOTINGSTAR': talib.CDLSHOOTINGSTAR,
            'CDLSHORTLINE': talib.CDLSHORTLINE,
            'CDLSPINNINGTOP': talib.CDLSPINNINGTOP,
            'CDLSTALLEDPATTERN': talib.CDLSTALLEDPATTERN,
            'CDLSTICKSANDWICH': talib.CDLSTICKSANDWICH,
            'CDLTAKURI': talib.CDLTAKURI,
            'CDLTASUKIGAP': talib.CDLTASUKIGAP,
            'CDLTHRUSTING': talib.CDLTHRUSTING,
            'CDLTRISTAR': talib.CDLTRISTAR,
            'CDLUNIQUE3RIVER': talib.CDLUNIQUE3RIVER,
            'CDLUPSIDEGAP2CROWS': talib.CDLUPSIDEGAP2CROWS,
            'CDLXSIDEGAP3METHODS': talib.CDLXSIDEGAP3METHODS,
        }
        if df is None or df.empty:
            logger.warning("Input DataFrame is empty or None for pattern detection.")
            # Always return a DataFrame with a 'pattern' column
            if df is None:
                import pandas as pd
                df = pd.DataFrame()
            df = df.copy()
            df['pattern'] = [''] * len(df)
            return df
        pattern_result = []
        for i in range(len(df)):
            found = ''
            for name, func in all_patterns.items():
                try:
                    res = func(
                        df['open'].values[max(0, i-4):i+1],
                        df['high'].values[max(0, i-4):i+1],
                        df['low'].values[max(0, i-4):i+1],
                        df['close'].values[max(0, i-4):i+1]
                    )
                    if res[-1] != 0:
                        found = name
                        logger.info(f"Pattern detected at index {i}: {name}")
                        break
                except Exception as e:
                    logger.debug(f"Pattern detection error for {name} at index {i}: {e}")
                    continue
            pattern_result.append(found)
        df = df.copy()
        df['pattern'] = pattern_result
        # Log summary statistics for detected patterns
        pattern_counts = df['pattern'].value_counts().to_dict()
        logger.info(f"Candlestick pattern detection summary: {pattern_counts}")
        return df
