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
        
        # Rate limiting
        self._rate_limiter = RateLimiter(rate=3, burst=5)  # More conservative values
        
        # Initialize API
        self._initialize_api()
        self._initialize_token_mapping()
        self._initialize_websocket()
        
    def _initialize_api(self):
        """Initialize Angel One API connection with enhanced error handling and session management"""
        try:
            angel_config = self.config['apis']['angel_one']
            max_retries = 5  # Increased from 3
            retry_delay = 10  # Increased initial delay
            retry_count = 0
            
            while retry_count < max_retries:
                try:
                    if not hasattr(self, 'angel_api') or self.angel_api is None:
                        # Initialize SmartConnect with increased timeout
                        session = requests.Session()
                        session.timeout = 30  # Increased from default 7 seconds
                        self.angel_api = SmartConnect(api_key=angel_config['api_key'])
                        self.angel_api._SmartConnect__http_session = session
                        logger.info("SmartAPI instance created with extended timeout")
                    
                    # Generate TOTP
                    totp = pyotp.TOTP(angel_config['totp_secret'])
                    current_totp = totp.now()
                    logger.debug(f"Generated TOTP for authentication")
                    
                    # Try authentication
                    data = self.angel_api.generateSession(
                        angel_config['client_id'],
                        angel_config['mpin'],
                        current_totp
                    )
                    
                    if data.get('status'):
                        # Store tokens for session management
                        self.auth_token = data['data']['jwtToken']
                        self.refresh_token = data['data']['refreshToken']
                        self.feed_token = self.angel_api.getfeedToken()
                        
                        # Verify connection with profile fetch using refresh token
                        profile = self.angel_api.getProfile(self.refresh_token)
                        if profile.get('status'):
                            logger.info(f"Successfully authenticated as {profile['data'].get('name', 'Unknown')}")
                            
                            # Set up session auto-renewal
                            self._start_session_renewal_timer()
                            return  # Success, exit the retry loop
                        else:
                            error_msg = profile.get('message', 'Unknown error')
                            logger.error(f"Failed to get user profile: {error_msg}")
                            
                    else:
                        error_msg = data.get('message', 'Unknown error')
                        logger.error(f"Failed to authenticate with Angel One API: {error_msg}")
                        if 'rate' in str(error_msg).lower():
                            # Handle rate limiting with exponential backoff
                            retry_count += 1
                            if retry_count < max_retries:
                                retry_delay *= 2  # Exponential backoff
                                logger.warning(f"Rate limited, retrying in {retry_delay}s ({retry_count}/{max_retries})")
                                time.sleep(retry_delay)
                                continue
                        raise Exception(f"Authentication failed: {error_msg}")
                        
                except Exception as e:
                    error_msg = str(e)
                    logger.error(f"Error connecting to Angel One API: {error_msg}")
                    if 'rate' in error_msg.lower():
                        retry_count += 1
                        if retry_count < max_retries:
                            retry_delay *= 2  # Exponential backoff
                            logger.warning(f"Rate limited, retrying in {retry_delay}s ({retry_count}/{max_retries})")
                            time.sleep(retry_delay)
                            continue
                    raise  # Re-raise if not rate limited or out of retries
            
            raise Exception("Maximum retry attempts reached")
            
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
        """Initialize token mapping for configured symbols"""
        try:
            logger.info("Initializing token mapping...")
            instrument_url = "https://margincalculator.angelbroking.com/OpenAPI_File/files/OpenAPIScripMaster.json"
            
            # Retry logic for rate limit with timeout
            for attempt in range(3):
                try:
                    logger.info(f"Downloading instrument file (attempt {attempt + 1}/3)...")
                    response = requests.get(instrument_url, timeout=30)  # 30 seconds timeout
                    if response.status_code == 200:
                        data = response.json()
                        logger.info("Successfully downloaded instrument file")
                        break
                    else:
                        logger.error(f"Failed to download instrument file: HTTP {response.status_code}")
                        time.sleep(2 ** attempt)
                except requests.Timeout:
                    logger.error(f"Timeout downloading instrument file (attempt {attempt + 1}/3)")
                    time.sleep(2 ** attempt)
                except Exception as e:
                    logger.error(f"Error downloading instrument file: {str(e)} (attempt {attempt + 1}/3)")
                    time.sleep(2 ** attempt)
            else:
                logger.error("Failed to download instrument file after all retries")
                raise ConnectionError("Could not download instrument file")

            # Get trading symbols from config
            trading_cfg = self.config.get('trading', {})
            data_cfg = trading_cfg.get('data', {})
            
            symbols = []
            if data_cfg.get('mode') == 'manual':
                symbols = data_cfg.get('manual_symbols', [])
            elif data_cfg.get('mode') == 'auto':
                symbols = data_cfg.get('manual_list', [])

            if not symbols:
                logger.warning("No trading symbols configured")
                return

            # Map symbols to tokens
            mapped_count = 0
            unmapped_symbols = []
            
            for symbol in symbols:
                if not isinstance(symbol, str):
                    logger.warning(f"Skipping invalid symbol: {symbol} (not a string)")
                    continue
                    
                # Convert NSE symbol format to Angel One format
                if '-EQ' in symbol:
                    angel_symbol = symbol
                elif '.NS' in symbol:
                    angel_symbol = symbol.replace('.NS', '-EQ')
                else:
                    angel_symbol = f"{symbol}-EQ"
                    
                found = False
                
                for instrument in data:
                    if (instrument['symbol'] == angel_symbol and 
                        instrument['exch_seg'] == 'NSE'):
                        self.symbol_token_map[symbol] = instrument['token']
                        self.token_symbol_map[instrument['token']] = symbol
                        mapped_count += 1
                        found = True
                        logger.debug(f"Mapped {symbol} to token {instrument['token']}")
                        break
                
                if not found:
                    unmapped_symbols.append(symbol)
                    logger.warning(f"Could not find token for symbol: {symbol}")
            
            if mapped_count == 0:
                raise ValueError("No symbols could be mapped to tokens")
            
            logger.info(f"Successfully mapped {mapped_count}/{len(symbols)} symbols to tokens")
            if unmapped_symbols:
                logger.warning(f"Unmapped symbols: {', '.join(unmapped_symbols)}")
                
        except Exception as e:
            logger.error(f"Error initializing token mapping: {str(e)}")
            raise  # Re-raise to prevent continuing with incomplete mapping
            
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
                    mode = mode_map.get(
                        self.config.get('data', {}).get('websocket', {}).get('mode', 'QUOTE').upper(),
                        MarketDataWebSocket.MODE_QUOTE
                    )
                    
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
            
        except Exception as e:
            logger.error(f"Error processing market data: {str(e)}")

    def cleanup(self):
        """Cleanup resources"""
        try:
            if self.websocket:
                self.websocket.close()
                logger.info("WebSocket connection closed")
        except Exception as e:
            logger.error(f"Error during cleanup: {str(e)}")    
    @with_rate_limit(max_retries=5, initial_delay=2.0)
    def get_historical_data(self, symbol: str, days: int = 10, interval: str = 'ONE_DAY') -> pd.DataFrame:
        """Get historical data with enhanced error handling and market hours consideration"""
        try:
            token = self.symbol_token_map.get(symbol)
            if not token:
                raise ValueError(f"No token found for symbol {symbol}")
                
            # Enhanced rate limiting with token-specific backoff
            retry_attempt = 0
            max_local_retries = 5  # Increased retries for rate limit recovery
            initial_delay = 2.0
            
            # Initialize token-specific backoff dict
            token_backoff = getattr(self, '_token_backoff', {})
            if not hasattr(self, '_token_backoff'):
                self._token_backoff = token_backoff
                
            if token not in token_backoff:
                token_backoff[token] = {
                    'error_count': 0,
                    'last_error': None,
                    'backoff_multiplier': 1.0,
                    'cooldown_until': 0
                }
            
            while retry_attempt < max_local_retries:
                now = time.time()
                
                # Check cooldown period
                if token_backoff[token]['cooldown_until'] > now:
                    wait_time = token_backoff[token]['cooldown_until'] - now
                    logger.info(f"Token {token} ({symbol}) in cooldown, waiting {wait_time:.2f}s")
                    time.sleep(wait_time)
                
                # Apply rate limiting with token awareness
                backoff_mul = token_backoff[token]['backoff_multiplier']
                wait_time = self._rate_limiter.acquire() * backoff_mul
                if wait_time > 0:
                    logger.info(f"Rate limited for {symbol} (token {token}), waiting {wait_time:.2f}s (backoff: {backoff_mul:.1f}x)")
                    time.sleep(wait_time)
                
                # Check market status before proceeding
                market_status = self.get_market_status()
                if not market_status.get('NSE', False):
                    logger.warning("Market is closed, historical data may be delayed")
                    time.sleep(random.uniform(1.0, 3.0))
                
                # Calculate date range with market hours consideration
                to_date = datetime.now()
                if to_date.hour < 9 or (to_date.hour == 9 and to_date.minute < 15):
                    to_date = to_date.replace(hour=15, minute=30, second=0, microsecond=0) - timedelta(days=1)
                elif to_date.hour > 15 or (to_date.hour == 15 and to_date.minute > 30):
                    to_date = to_date.replace(hour=15, minute=30, second=0, microsecond=0)
                
                from_date = to_date - timedelta(days=days)
                
                params = {
                    "exchange": "NSE",
                    "symboltoken": token,
                    "interval": interval,
                    "fromdate": from_date.strftime("%Y-%m-%d %H:%M"),
                    "todate": to_date.strftime("%Y-%m-%d %H:%M")
                }
                
                try:
                    response = self.angel_api.getCandleData(params)
                    
                    if response and response.get('data'):
                        data = response['data']
                        df = pd.DataFrame(
                            data,
                            columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
                        )
                        df['timestamp'] = pd.to_datetime(df['timestamp'])
                        
                        # Reset error count on success
                        token_backoff[token]['error_count'] = max(0, token_backoff[token]['error_count'] - 1)
                        token_backoff[token]['backoff_multiplier'] = max(1.0, token_backoff[token]['backoff_multiplier'] * 0.5)
                        
                        return df
                    else:
                        error_msg = response.get('message', 'Unknown error')
                        error_code = response.get('errorcode', '')
                        
                        # Check for various error conditions
                        is_rate_limit = any(x in str(error_msg).lower() for x in ['rate', 'limit', 'access denied', 'ab1004'])
                        is_auth_error = 'authentication' in str(error_msg).lower() or error_code == 'AG8001'
                        
                        if is_rate_limit or error_code == 'AB1004':
                            # Update token-specific backoff
                            token_backoff[token]['error_count'] += 1
                            token_backoff[token]['last_error'] = now
                            
                            # Exponential backoff with token-specific multiplier
                            token_backoff[token]['backoff_multiplier'] = min(
                                8.0,  # Cap multiplier
                                1.0 + (token_backoff[token]['error_count'] * 0.5)
                            )
                            
                            # Set cooldown period for severe rate limiting
                            if token_backoff[token]['error_count'] >= 3:
                                cooldown = 60.0 * (2 ** min(token_backoff[token]['error_count'] - 3, 3))
                                token_backoff[token]['cooldown_until'] = now + cooldown
                                logger.warning(f"Token {token} ({symbol}) in cooldown for {cooldown:.1f}s due to repeated rate limits")
                            
                            retry_attempt += 1
                            if retry_attempt < max_local_retries:
                                # Calculate backoff with jitter
                                base_wait = initial_delay * (2 ** retry_attempt) * token_backoff[token]['backoff_multiplier']
                                jitter = random.uniform(0.8, 1.2)  # ±20% jitter
                                wait = base_wait * jitter
                                
                                logger.warning(
                                    f"Rate limit/AB1004 for {symbol} (token {token}): {error_msg}, "
                                    f"retry {retry_attempt}/{max_local_retries} in {wait:.1f}s "
                                    f"(backoff: {token_backoff[token]['backoff_multiplier']:.1f}x)"
                                )
                                time.sleep(wait)
                                continue
                        elif is_auth_error:
                            logger.error(f"Authentication error for {symbol}: {error_msg}")
                            break
                        else:
                            logger.warning(f"Invalid response format: {response}")
                            retry_attempt += 1
                            if retry_attempt < max_local_retries:
                                wait = initial_delay * (2 ** retry_attempt)
                                time.sleep(wait)
                                continue
                
                except Exception as e:
                    error_msg = str(e)
                    retry_attempt += 1
                    
                    # Check if exception indicates rate limiting
                    if any(x in error_msg.lower() for x in ['rate', 'limit', 'access denied', 'ab1004']):
                        # Update token-specific backoff
                        token_backoff[token]['error_count'] += 1
                        token_backoff[token]['last_error'] = now
                        if retry_attempt < max_local_retries:
                            wait = initial_delay * (2 ** retry_attempt) * token_backoff[token]['backoff_multiplier']
                            logger.warning(f"Rate limit error for {symbol}: {error_msg}, retry {retry_attempt}/{max_local_retries} in {wait:.1f}s")
                            time.sleep(wait)
                            continue
                    elif retry_attempt < max_local_retries:
                        wait = initial_delay * (2 ** retry_attempt)
                        logger.warning(f"Error for {symbol}: {error_msg}, retry {retry_attempt}/{max_local_retries} in {wait:.1f}s")
                        time.sleep(wait)
                        continue
                    
                    logger.error(f"Error fetching data for {symbol}: {e}")
                    break
            
            logger.warning(f"No data available for {symbol} after {retry_attempt} attempts")
            return pd.DataFrame()
            
        except Exception as e:
            logger.error(f"Error fetching historical data for {symbol}: {e}")
            return pd.DataFrame()

    @with_rate_limit(max_retries=3, initial_delay=1.0)
    def get_ltp(self, symbol: str) -> float:
        """Get last traded price for a symbol"""
        try:
            token = self.symbol_token_map.get(symbol)
            if not token:
                raise ValueError(f"No token found for symbol {symbol}")
                
            ltp_data = self.angel_api.ltpData("NSE", symbol.replace('.NS', '-EQ'), token)
            
            if ltp_data and ltp_data.get('data', {}).get('ltp'):
                return float(ltp_data['data']['ltp'])
            else:
                logger.warning(f"No LTP data available for {symbol}")
                return 0.0
                
        except Exception as e:
            logger.error(f"Error fetching LTP data for {symbol}: {str(e)}")
            raise
            
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
