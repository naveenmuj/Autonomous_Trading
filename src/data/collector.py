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

logger = logging.getLogger(__name__)

def with_rate_limit(max_retries=3, initial_delay=1.0):
    """Decorator for rate-limited API calls with exponential backoff."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            delay = initial_delay
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if 'rate' in str(e).lower():
                        logger.warning(f"Rate limit hit, retrying in {delay}s... ({attempt + 1}/{max_retries})")
                        time.sleep(delay)
                        delay *= 2
                    else:
                        raise
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
    # Websocket modes
    MODE_LTP = 1
    MODE_QUOTE = 2
    MODE_SNAPQUOTE = 3

    """Enhanced data collection class with rate limiting"""
    def __init__(self, config):
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
        # Rate limiting (configurable)
        rate_limit_cfg = config.get('api', {}).get('rate_limit', {})
        rate = rate_limit_cfg.get('rate', 3)
        burst = rate_limit_cfg.get('burst', 5)
        self._rate_limiter = RateLimiter(rate=rate, burst=burst)  # More conservative values
        
        # Initialize API
        self._initialize_api()
        self._initialize_token_mapping()
        self._initialize_websocket()
        
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
        except Exception as e:
            logger.error(f"Error initializing token mapping: {str(e)}")
            raise
            
    def _initialize_websocket(self):
        """Initialize WebSocket connection for live market data"""
        try:
            if not self.angel_api:
                logger.error("Angel API not initialized, cannot start WebSocket")
                return
                
            angel_config = self.config['apis']['angel_one']
            
            # Configure WebSocket with enhanced settings
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
            
            # Add market data callback
            self.websocket.add_tick_callback(self._on_market_data)            # Start connection and wait for it to be ready
            logger.info("Initiating WebSocket connection...")
            self.websocket.connect()
            
            # Wait for connection with increased timeout
            if self.websocket.wait_for_connection(timeout=30):  # Increased from 10 to 30 seconds                # Format tokens according to WebSocket 2.0 spec
                symbol_tokens = [str(token) for token in self.symbol_token_map.values()]
                if symbol_tokens:
                    logger.info(f"Subscribing to {len(symbol_tokens)} symbols...")
                    
                    # Map mode string to numeric value
                    mode_map = {
                        'LTP': MarketDataWebSocket.MODE_LTP,
                        'QUOTE': MarketDataWebSocket.MODE_QUOTE,
                        'SNAPQUOTE': MarketDataWebSocket.MODE_SNAPQUOTE
                    }
                    mode_str = self.config.get('data', {}).get('websocket', {}).get('mode', 'QUOTE')
                    if isinstance(mode_str, str):
                        mode_key = mode_str.upper()
                    elif isinstance(mode_str, int):
                        # If already int, map to string key
                        reverse_map = {v: k for k, v in mode_map.items()}
                        mode_key = reverse_map.get(mode_str, 'QUOTE')
                        logger.warning(f"WebSocket mode is int: {mode_str}, mapped to '{mode_key}'")
                    else:
                        logger.warning(f"WebSocket mode is not a string or int: {mode_str} (type: {type(mode_str)}), defaulting to 'QUOTE'")
                        mode_key = 'QUOTE'
                    mode = mode_map.get(mode_key, MarketDataWebSocket.MODE_QUOTE)
                    
                    # Subscribe to all tokens
                    self.websocket.subscribe(
                        tokens=symbol_tokens, 
                        mode=mode
                    )
                    logger.info(f"Subscription request sent for {len(symbol_tokens)} symbols in mode {mode}")
            else:
                raise Exception("WebSocket connection timeout")
                
        except Exception as e:
            logger.error(f"Error initializing WebSocket: {str(e)}")
            raise
            
    def _on_market_data(self, tick_data: Dict[str, Any]):
        """Handle incoming market data ticks"""
        try:
            token = tick_data.get('token')
            if not token:
                logger.warning("Received market data without token")
                return
                
            # Convert token to symbol if we have the mapping
            symbol = self.token_symbol_map.get(token)
            if not symbol:
                logger.warning(f"Received data for unknown token: {token}")
                return
                
            # Log market data with proper formatting
            log_msg = (
                f"Market Data - {symbol}: "
                f"LTP: ₹{tick_data['ltp']:.2f}, "
                f"Time: {tick_data['timestamp'].strftime('%H:%M:%S')}"
            )
            
            if 'volume' in tick_data:
                log_msg += f", Vol: {tick_data['volume']:,}"
            if 'open_interest' in tick_data:
                log_msg += f", OI: {tick_data['open_interest']:,}"                
            logger.info(log_msg)
            # Store data for further processing if needed
            # You can add more processing logic here
            
        except Exception as e:            logger.error(f"Error processing market data: {str(e)}")

    def cleanup(self):
        """Cleanup resources"""
        try:
            if hasattr(self, 'websocket') and self.websocket:
                self.websocket.close()
                logger.info("WebSocket connection closed")
        except Exception as e:
            logger.error(f"Error during cleanup: {str(e)}")

    def get_historical_data(self, symbol: str, interval: str = 'ONE_DAY', days: int = 10) -> pd.DataFrame:
        """Get historical data for a symbol with robust error handling
        
        Args:
            symbol: Trading symbol (e.g., 'RELIANCE.NS')
            interval: Time interval ('1m', '5m', '15m', '30m', '1h', '1d')
            days: Number of days of historical data
            
        Returns:
            pd.DataFrame: Historical data with columns timestamp, open, high, low, close, volume
            Returns empty DataFrame on error
        """
        empty_df = pd.DataFrame(columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        
        cache_key = (symbol, interval, days)
        if cache_key in self._historical_cache:
            logger.info(f"Historical data cache hit for {symbol} {interval} {days}d")
            return self._historical_cache[cache_key].copy()
        
        try:
            # Get token for symbol
            token = self.symbol_token_map.get(symbol)
            if not token:
                logger.error(f"No token found for symbol {symbol}")
                return empty_df
            
            # Map interval to Angel One format
            interval_map = {
                '1m': 'ONE_MINUTE',
                '5m': 'FIVE_MINUTE',
                '15m': 'FIFTEEN_MINUTE',
                '30m': 'THIRTY_MINUTE',
                '1h': 'ONE_HOUR',
                '1d': 'ONE_DAY'
            }
            angel_interval = interval_map.get(interval, interval)
            
            # Calculate date range
            end_date = datetime.now().replace(hour=15, minute=30, second=0, microsecond=0)
            if end_date.hour < 9 or (end_date.hour == 9 and end_date.minute < 15):
                end_date = end_date - timedelta(days=1)
            start_date = end_date - timedelta(days=days)
            
            # Prepare request parameters
            params = {
                "exchange": "NSE",
                "symboltoken": str(token),
                "interval": angel_interval,
                "fromdate": start_date.strftime("%Y-%m-%d 09:15"),
                "todate": end_date.strftime("%Y-%m-%d 15:30")
            }
            
            # Try to get data with retries
            for attempt in range(3):
                try:
                    response = self.angel_api.getCandleData(params)
                    
                    if not response:
                        logger.warning(f"Empty response for {symbol} on attempt {attempt + 1}")
                        if attempt < 2:
                            time.sleep(2 ** attempt)
                        continue
                    
                    if isinstance(response, str):
                        response = json.loads(response)
                    
                    data = response.get('data', [])
                    if not data:
                        logger.warning(f"No data for {symbol} on attempt {attempt + 1}")
                        if attempt < 2:
                            time.sleep(2 ** attempt)
                        continue
                    
                    # Create DataFrame with proper types
                    df = pd.DataFrame(data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                    df['timestamp'] = pd.to_datetime(df['timestamp'])
                    for col in ['open', 'high', 'low', 'close']:
                        df[col] = pd.to_numeric(df[col], errors='coerce')
                    df['volume'] = pd.to_numeric(df['volume'], errors='coerce').fillna(0).astype(int)
                    
                    # Clean up the data
                    df = df.sort_values('timestamp')
                    df = df.drop_duplicates(subset=['timestamp'], keep='last')
                    df = df.reset_index(drop=True)
                    
                    logger.info(f"Successfully fetched {len(df)} records for {symbol}")
                    
                    # Cache the result
                    self._historical_cache[cache_key] = df.copy()
                    
                    return df
                    
                except Exception as e:
                    logger.error(f"Error on attempt {attempt + 1} for {symbol}: {str(e)}")
                    if attempt < 2:
                        time.sleep(2 ** attempt)
            
            logger.error(f"Failed to fetch data for {symbol} after 3 attempts")
            return empty_df
            
        except Exception as e:
            logger.error(f"Unexpected error for {symbol}: {str(e)}")
            return empty_df

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
        
    def get_watchlist(self) -> List[Dict[str, Any]]:
        """Get the current watchlist with latest prices and indicators"""
        try:
            watchlist = []
            symbols = self.config.get('trading', {}).get('data', {}).get('manual_symbols', [])
            
            for symbol in symbols:
                try:
                    # Get latest data
                    data = self.get_historical_data(symbol=symbol, days=1, interval='ONE_DAY')
                    
                    if not data.empty:
                        latest = data.iloc[-1]
                        watchlist.append({
                            'symbol': symbol,
                            'ltp': latest['close'],
                            'change': round((latest['close'] - latest['open']) / latest['open'] * 100, 2),
                            'volume': latest['volume'],
                            'high': latest['high'],
                            'low': latest['low']
                        })
                    else:
                        logger.warning(f"No data available for {symbol}")
                except Exception as e:
                    logger.error(f"Error getting data for {symbol}: {str(e)}")
                    continue
            
            return watchlist
        except Exception as e:
            logger.error(f"Error getting watchlist: {str(e)}")
            return []

    @with_rate_limit(max_retries=5, initial_delay=2.0)
    def get_live_quote(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get real-time quote for a symbol with enhanced error handling
        
        Args:
            symbol (str): Stock symbol (e.g. 'RELIANCE.NS')
        
        Returns:
            Dict[str, Any]: Quote data including last traded price, change, volume etc.
                          or None if not available
        """
        try:
            token = self.symbol_token_map.get(symbol)
            if not token:
                raise ValueError(f"No token found for symbol {symbol}")
            
            # Try to get data from WebSocket first if available
            ws_data = None
            if self.websocket and self.websocket.is_connected:
                try:
                    ws_data = self.websocket.get_market_data(str(token))
                except Exception as ws_err:
                    logger.warning(f"WebSocket error for {symbol}: {str(ws_err)}, falling back to REST API")
                
                if ws_data:
                    try:
                        # Format the quote with required fields
                        quote = {
                            'symbol': symbol,
                            'exchange': 'NSE',
                            'timestamp': ws_data.get('timestamp', datetime.now()),
                            'ltp': float(ws_data.get('ltp', 0)),
                            'volume': int(ws_data.get('volume', 0)),
                            'change': 0.0,
                            'change_percent': 0.0
                        }
                        
                        # Add OHLC if available
                        if all(k in ws_data for k in ['open', 'high', 'low', 'close']):
                            quote.update({
                                'open': float(ws_data['open']),
                                'high': float(ws_data['high']),
                                'low': float(ws_data['low']),
                                'close': float(ws_data['close'])
                            })
                            # Calculate change and percentage if open price is available
                            if quote['open'] > 0:
                                quote['change'] = quote['ltp'] - quote['open']
                                quote['change_percent'] = (quote['change'] / quote['open']) * 100
                        
                        # Add market depth if available
                        if all(k in ws_data for k in ['best_bid_price', 'best_bid_quantity', 'best_ask_price', 'best_ask_quantity']):
                            quote.update({
                                'bid_price': float(ws_data['best_bid_price']),
                                'bid_quantity': int(ws_data['best_bid_quantity']),
                                'ask_price': float(ws_data['best_ask_price']),
                                'ask_quantity': int(ws_data['best_ask_quantity'])
                            })
                        
                        return quote
                    except Exception as fmt_err:
                        logger.warning(f"Error formatting WebSocket data for {symbol}: {str(fmt_err)}")
                        # Continue to REST API fallback
            
            # Fallback to REST API with retries
            retry_attempt = 0
            max_local_retries = 3
            
            while retry_attempt < max_local_retries:
                try:
                    wait_time = self._rate_limiter.acquire()
                    if wait_time > 0:
                        logger.info(f"Rate limited for {symbol}, waiting {wait_time:.2f}s before request")
                        time.sleep(wait_time)
                    
                    ltp_data = self.angel_api.ltpData("NSE", symbol.replace('.NS', '-EQ'), token)
                    
                    if ltp_data and ltp_data.get('data'):
                        data = ltp_data['data']
                        return {
                            'symbol': symbol,
                            'exchange': 'NSE',
                            'timestamp': datetime.fromtimestamp(int(data.get('exchange_time', time.time() * 1000)) / 1000),
                            'ltp': float(data.get('ltp', 0)),
                            'volume': int(data.get('volume', 0)),
                            'change': float(data.get('priceChange', 0)),
                            'change_percent': float(data.get('perChange', 0))
                        }
                    else:
                        error_msg = ltp_data.get('message', 'Unknown error') if ltp_data else 'No response'
                        if any(x in str(error_msg).lower() for x in ['rate', 'limit', 'access denied', 'ab1004']):
                            retry_attempt += 1
                            if retry_attempt < max_local_retries:
                                wait = 2.0 * (2 ** retry_attempt)
                                logger.warning(f"Rate/access error for {symbol}: {error_msg}, retry {retry_attempt}/{max_local_retries} in {wait:.1f}s")
                                time.sleep(wait)
                                continue
                        logger.warning(f"Invalid response format for {symbol}: {error_msg}")
                except Exception as e:
                    retry_attempt += 1
                    if retry_attempt < max_local_retries:
                        wait = 2.0 * (2 ** retry_attempt)
                        logger.warning(f"Error for {symbol}: {str(e)}, retry {retry_attempt}/{max_local_retries} in {wait:.1f}s")
                        time.sleep(wait)
                        continue
                    logger.error(f"Error fetching quote for {symbol}: {e}")
                    break
            
            logger.warning(f"No quote available for {symbol} after {retry_attempt} attempts")
            return None
                
        except Exception as e:
            logger.error(f"Error getting live quote for {symbol}: {e}")
            return None
        
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
