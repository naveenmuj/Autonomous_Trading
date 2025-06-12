import os
import sys
import json
import logging
import threading
import time
import requests
from typing import Dict, List, Optional, Any, Union, Tuple
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd
import numpy as np
import pyotp
from SmartApi import SmartConnect
import yfinance as yf
import pandas_ta as ta
from functools import wraps
from .websocket import MarketDataWebSocket
import contextlib
import codecs

logger = logging.getLogger(__name__)

# Configure logger for UTF-8 encoding
handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
handler.stream.reconfigure(encoding='utf-8')  # Set UTF-8 encoding
logger.addHandler(handler)

# Session renewal interval (8 hours in seconds)
SESSION_RENEWAL_INTERVAL = 8 * 60 * 60

@contextlib.contextmanager
def utf8_stdout():
    """Context manager that temporarily sets stdout to use UTF-8 encoding"""
    old_stdout = sys.stdout
    try:
        sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer)
        yield
    finally:
        sys.stdout = old_stdout

def with_timeout(timeout_seconds: int):
    """Windows-compatible timeout decorator"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            import threading
            import queue

            result = queue.Queue()
            def worker():
                try:
                    r = func(*args, **kwargs)
                    result.put(('success', r))
                except Exception as e:
                    result.put(('error', e))

            thread = threading.Thread(target=worker)
            thread.daemon = True
            thread.start()
            thread.join(timeout_seconds)

            if thread.is_alive():
                thread.join(0)  # Non-blocking join
                raise TimeoutError(f"Function call timed out after {timeout_seconds} seconds")

            status, value = result.get()
            if status == 'error':
                raise value
            return value

        return wrapper
    return decorator

class DataCollector:
    """Data collection class for market data"""
    
    _instance = None  # Singleton instance
    INSTRUMENT_MASTER_URL = "https://margincalculator.angelbroking.com/OpenAPI_File/files/OpenAPIScripMaster.json"
    NSE_STOCKS_URL = "https://archives.nseindia.com/content/equities/EQUITY_L.csv"
    NIFTY_50_URL = "https://archives.nseindia.com/content/indices/ind_nifty50list.csv"
    NIFTY_100_URL = "https://archives.nseindia.com/content/indices/ind_nifty100list.csv"
    
    def __new__(cls, config: Optional[Dict] = None) -> 'DataCollector':
        if cls._instance is None:
            cls._instance = super(DataCollector, cls).__new__(cls)
        return cls._instance
        
    def __init__(self, config: Optional[Dict] = None):
        if not hasattr(self, 'initialized'):
            self.config = config if config else {}
            self.angel_api = None
            self.websocket = None
            self.token_mapping = {}
            self.exchange_manager = None
            self.live_data = {}  # Store real-time data
            self.initialized = True
            self.session_timer = None
            self.is_running = True
            
            # Initialize Angel One API
            self._initialize_angel_api()
            # Initialize token mapping for NSE symbols
            self._initialize_token_mapping()
            # Start WebSocket connection
            self._initialize_websocket()
            
    def _start_session_renewal_timer(self):
        """Start timer to renew Angel One session"""
        def renewal_task():
            while self.is_running:
                time.sleep(SESSION_RENEWAL_INTERVAL)
                if self.is_running:
                    try:
                        self._renew_session()
                    except Exception as e:
                        logger.error(f"Error in session renewal: {str(e)}")
                        
        self.session_timer = threading.Thread(target=renewal_task, daemon=True)
        self.session_timer.start()
        logger.info("Session renewal timer started")
        
    def _renew_session(self):
        """Renew Angel One API session"""
        try:
            # Generate new TOTP
            totp = pyotp.TOTP(self.config['apis']['angel_one']['totp_secret'])
            totp_token = totp.now()
            
            # Re-authenticate with Angel One
            self.angel_api = SmartConnect(
                api_key=self.config['apis']['angel_one']['api_key']
            )
            
            data = self.angel_api.generateSession(
                self.config['apis']['angel_one']['client_id'],
                self.config['apis']['angel_one']['mpin'],
                totp_token
            )
            
            if data['status'] and data['message'] == 'SUCCESS':
                logger.info("Session renewed successfully")
                
                # Reconnect WebSocket with new tokens if needed
                if self.websocket:
                    self.websocket.close()
                    self._initialize_websocket()
            else:
                raise Exception(f"Session renewal failed: {data.get('message', 'Unknown error')}")
                
        except Exception as e:
            logger.error(f"Error renewing session: {str(e)}")
            raise
    def _initialize_websocket(self):
        """Initialize WebSocket connection for real-time data"""
        try:
            # Validate API initialization
            if not self.angel_api:
                logger.error("Angel One API not initialized, cannot create WebSocket")
                return False

            # Validate required tokens
            if not hasattr(self.angel_api, 'feed_token') or not self.angel_api.feed_token:
                logger.error("Missing feed token, cannot create WebSocket")
                return False
                
            if not hasattr(self.angel_api, 'auth_token') or not self.angel_api.auth_token:
                logger.error("Missing auth token, cannot create WebSocket")
                return False

            # Get API credentials
            credentials = self.config.get('apis', {}).get('angel_one', {})
            if not credentials.get('api_key') or not credentials.get('client_id'):
                logger.error("Missing API credentials in config")
                return False

            # Close existing WebSocket if any
            if self.websocket:
                try:
                    self.websocket.close()
                except Exception as e:
                    logger.warning(f"Error closing existing WebSocket: {str(e)}")
                self.websocket = None

            # Create new WebSocket instance with retry parameters
            logger.info("Creating new WebSocket instance...")
            self.websocket = MarketDataWebSocket(
                auth_token=self.angel_api.auth_token,
                api_key=credentials['api_key'],
                client_code=credentials['client_id'],
                feed_token=self.angel_api.feed_token,
                config=self.config
            )

            # Register tick callback
            self.websocket.add_tick_callback(self._on_tick_data)

            # Connect WebSocket
            logger.info("Connecting to WebSocket...")
            self.websocket.connect()

            # Wait for connection to establish
            timeout = 10
            start_time = time.time()
            while not self.websocket.is_connected and time.time() - start_time < timeout:
                time.sleep(0.1)

            if not self.websocket.is_connected:
                raise Exception("WebSocket connection timed out")

            # Subscribe to default symbols
            logger.info("Subscribing to default symbols...")
            self._subscribe_default_symbols()

            logger.info("WebSocket initialization completed successfully")
            return True

        except Exception as e:
            logger.error(f"Error initializing WebSocket: {str(e)}")
            if self.websocket:
                try:
                    self.websocket.close()
                except:
                    pass  # Ignore errors during cleanup
                self.websocket = None
            return False
    
    def _on_tick_data(self, tick_data: Dict[str, Any]) -> None:
        """Handle incoming tick data"""
        try:
            # Store tick data by token
            token = tick_data['token']
            self.live_data[token] = tick_data
            
            # Log tick data at debug level
            logger.debug(f"Received tick: {tick_data}")
            
        except Exception as e:
            logger.error(f"Error processing tick data: {str(e)}")
    
    def _subscribe_default_symbols(self) -> None:
        """Subscribe to default symbols"""
        try:
            if not self.websocket:
                logger.warning("WebSocket not initialized, cannot subscribe to symbols")
                return

            if not self.websocket.is_connected:
                logger.warning("WebSocket not connected, cannot subscribe to symbols")
                return

            if not self.token_mapping:
                logger.warning("Token mapping not initialized")
                return

            # Get trading symbols
            symbols = self._get_trading_symbols()
            if not symbols:
                logger.warning("No trading symbols configured")
                return

            logger.info(f"Subscribing to {len(symbols)} symbols...")
            
            # Convert symbols to tokens and build symbol mapping
            tokens = []
            for symbol in symbols:
                # Try both formats (with and without .NS)
                if symbol.endswith('.NS'):
                    base_symbol = symbol.replace('.NS', '')
                    token = self.token_mapping.get(base_symbol)
                else:
                    token = self.token_mapping.get(symbol)
                    if not token:
                        token = self.token_mapping.get(symbol + '.NS')

                if token:
                    tokens.append(str(token))

            if tokens:
                # Subscribe to the tokens
                self.websocket.subscribe(tokens)
                
                # Verify subscription
                time.sleep(1)  # Give some time for subscription to process
                active_feeds = len(self.websocket.live_feed) if self.websocket else 0
                logger.info(f"Active market data feeds: {active_feeds}")

        except Exception as e:
            logger.error(f"Error subscribing to default symbols: {str(e)}")
            logger.debug("Subscription error details", exc_info=True)
    
    def _initialize_angel_api(self):
        """Initialize Angel One API connection"""
        # Check if Angel One configuration exists
        if 'apis' not in self.config or 'angel_one' not in self.config['apis']:
            logger.warning("Missing Angel One configuration")
            return
                
        credentials = self.config['apis']['angel_one']
        api_key = credentials.get('api_key')
        client_id = credentials.get('client_id')
        mpin = credentials.get('mpin')
        totp_secret = credentials.get('totp_secret', '')

        if not all([api_key, client_id, mpin, totp_secret]):
            logger.error("Missing required Angel One credentials")
            raise ValueError("Incomplete Angel One credentials in config")
        
        # Retry configuration
        max_retries = 3
        retry_delay = 5  # seconds
        retry_count = 0
        
        while retry_count < max_retries:
            try:
                # Initialize SmartAPI with rate limiting
                if not hasattr(self, 'angel_api') or self.angel_api is None:
                    self.angel_api = SmartConnect(api_key=api_key)
                    logger.info("SmartAPI instance created")
                
                # Generate TOTP
                totp = pyotp.TOTP(totp_secret)
                current_totp = totp.now()
                logger.info("Generated TOTP successfully")
                
                # Authenticate with Angel One
                data = self.angel_api.generateSession(client_id, mpin, current_totp)
                
                if data.get('status'):
                    logger.info("Successfully authenticated with Angel One API")
                    session_data = data.get('data', {})
                    
                    # Store tokens in both angel_api instance and self
                    self.angel_api.auth_token = session_data.get('jwtToken', '')
                    self.angel_api.refresh_token = session_data.get('refreshToken', '')
                    self.angel_api.feed_token = session_data.get('feedToken', '')
                    self.auth_token = self.angel_api.auth_token
                    self.refresh_token = self.angel_api.refresh_token
                    self.feed_token = self.angel_api.feed_token
                    
                    profile = self.angel_api.getProfile(self.refresh_token)
                    
                    if profile.get('status'):
                        logger.info(f"Successfully retrieved user profile for {profile.get('data', {}).get('name', 'Unknown')}")
                        self._start_session_renewal_timer()  # Start session renewal
                        return  # Success, exit the retry loop
                    else:
                        raise Exception(f"Failed to get user profile: {profile.get('message', 'Unknown error')}")
                else:
                    error_msg = data.get('message', 'Unknown error')
                    logger.error(f"Failed to authenticate with Angel One API: {error_msg}")
                    if 'rate' in str(error_msg).lower():
                        retry_count += 1
                        if retry_count < max_retries:
                            logger.warning(f"Rate limited, waiting {retry_delay} seconds before retry {retry_count}/{max_retries}")
                            time.sleep(retry_delay)
                            retry_delay *= 2  # Exponential backoff
                            continue
                    raise Exception(f"Angel One authentication failed: {error_msg}")
                    
            except Exception as e:
                error_msg = str(e)
                logger.error(f"Error connecting to Angel One API: {error_msg}")
                if 'rate' in error_msg.lower():
                    retry_count += 1
                    if retry_count < max_retries:
                        logger.warning(f"Rate limited, waiting {retry_delay} seconds before retry {retry_count}/{max_retries}")
                        time.sleep(retry_delay)
                        retry_delay *= 2  # Exponential backoff
                        continue
                raise  # Re-raise if not rate limited or out of retries
                
    def _initialize_token_mapping(self):
        """Initialize token mapping for NSE symbols"""
        try:
            logger.info("Initializing token mapping...")
            
            # Download instrument master file
            response = requests.get(self.INSTRUMENT_MASTER_URL)
            response.raise_for_status()
            
            instruments = response.json()
            logger.info(f"Downloaded {len(instruments)} instruments from master file")
            
            # Filter and build mapping
            self.token_mapping = {}
            nse_symbols_found = 0
            
            for instrument in instruments:
                try:
                    exchange = instrument.get('exch_seg')
                    if exchange != 'NSE':
                        continue
                        
                    token = instrument.get('token')
                    trading_symbol = instrument.get('symbol')
                    
                    if not trading_symbol or not token:
                        continue

                    # Only include NSE equity symbols (ending with -EQ)
                    if trading_symbol.endswith('-EQ'):
                        # Store both formats of the symbol
                        base_symbol = trading_symbol.replace('-EQ', '')
                        yf_symbol = f"{base_symbol}.NS"  # Yahoo Finance format
                        angel_symbol = trading_symbol    # Angel One format
                        
                        self.token_mapping[yf_symbol] = token
                        self.token_mapping[angel_symbol] = token
                        nse_symbols_found += 1
                        
                        logger.debug(f"Mapped symbol {base_symbol}: YF={yf_symbol}, Angel={angel_symbol}, Token={token}")
                        
                except Exception as e:
                    logger.debug(f"Error processing instrument {instrument}: {str(e)}")
                    continue
            
            if not self.token_mapping:
                raise Exception("No valid instruments found in master file")
                
            logger.info(f"Successfully mapped {nse_symbols_found} NSE symbols (with both formats)")
            
            # Validate mapping for configured symbols
            symbols = self._get_trading_symbols()
            for symbol in symbols:
                base_symbol = symbol.replace('.NS', '')
                angel_symbol = f"{base_symbol}-EQ"
                if angel_symbol not in self.token_mapping and symbol not in self.token_mapping:
                    logger.warning(f"No token mapping found for configured symbol: {symbol} (Angel format: {angel_symbol})")
            
        except requests.RequestException as e:
            logger.error(f"Error downloading instrument master: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Error initializing token mapping: {str(e)}")
            raise
    def _discover_nse_stocks(self) -> List[str]:
        """Discover NSE stocks based on configuration criteria"""
        try:
            # Get all NSE stocks
            nse_stocks_df = pd.read_csv(self.NSE_STOCKS_URL)
            logger.info(f"Found {len(nse_stocks_df)} total NSE stocks")
            
            # Get configuration
            auto_config = self.config.get('trading', {}).get('data', {}).get('auto', {})
            market_caps = auto_config.get('market_caps', ['large'])  # Default to large cap if not specified
            min_volume = auto_config.get('min_volume', 100000)
            sectors = auto_config.get('sectors', [])
            exclude_symbols = set(auto_config.get('exclude_symbols', []))
            
            # Initialize an empty DataFrame for filtered stocks
            filtered_stocks = pd.DataFrame()
            
            # Apply market cap filters
            logger.info(f"Filtering for market caps: {market_caps}")
            for cap in market_caps:
                if cap == 'large':
                    cap_stocks = nse_stocks_df[nse_stocks_df['MARKET_CAP'] >= 20000]
                elif cap == 'mid':
                    cap_stocks = nse_stocks_df[
                        (nse_stocks_df['MARKET_CAP'] >= 5000) & 
                        (nse_stocks_df['MARKET_CAP'] < 20000)
                    ]
                elif cap == 'small':
                    cap_stocks = nse_stocks_df[
                        (nse_stocks_df['MARKET_CAP'] >= 500) & 
                        (nse_stocks_df['MARKET_CAP'] < 5000)
                    ]
                filtered_stocks = pd.concat([filtered_stocks, cap_stocks])
            
            logger.info(f"Found {len(filtered_stocks)} stocks matching market cap criteria")
            
            # Apply volume filter
            if min_volume > 0:
                filtered_stocks = filtered_stocks[filtered_stocks['AVG_VOLUME'] >= min_volume]
                logger.info(f"Found {len(filtered_stocks)} stocks with minimum volume of {min_volume}")
            
            # Get sector-specific stocks if specified
            selected_stocks = set()
            if sectors:
                logger.info(f"Filtering by sectors: {sectors}")
                for sector in sectors:
                    if sector == 'NIFTY50':
                        try:
                            nifty50_df = pd.read_csv(self.NIFTY_50_URL)
                            selected_stocks.update(nifty50_df['Symbol'].tolist())
                            logger.info(f"Added {len(nifty50_df)} NIFTY50 stocks")
                        except Exception as e:
                            logger.error(f"Error loading NIFTY50 stocks: {e}")
                    
                    elif sector == 'NIFTY100':
                        try:
                            nifty100_df = pd.read_csv(self.NIFTY_100_URL)
                            selected_stocks.update(nifty100_df['Symbol'].tolist())
                            logger.info(f"Added {len(nifty100_df)} NIFTY100 stocks")
                        except Exception as e:
                            logger.error(f"Error loading NIFTY100 stocks: {e}")
            else:
                # If no sectors specified, take all filtered stocks
                selected_stocks.update(filtered_stocks['SYMBOL'].tolist())
                logger.info("No sector filter applied, using all stocks that meet criteria")
            
            # Remove duplicates and excluded symbols
            selected_stocks = set(selected_stocks) - set(exclude_symbols)
            
            # Add -EQ suffix and create final list
            final_symbols = sorted([f"{sym}-EQ" for sym in selected_stocks])
            
            logger.info(f"Final selection: {len(final_symbols)} NSE stocks")
            if final_symbols:
                market_cap_stats = filtered_stocks.groupby('CAP_GROUP').size()
                logger.info(f"Market cap distribution:\n{market_cap_stats}")
                logger.info(f"Sample symbols: {final_symbols[:5]}")
            
            return final_symbols
            
        except Exception as e:
            logger.error(f"Error discovering NSE stocks: {str(e)}")
            # Fallback to default symbols in case of error
            default_symbols = self.config.get('trading', {}).get('data', {}).get('manual_symbols', [])
            logger.warning(f"Falling back to {len(default_symbols)} manual symbols")
            return default_symbols
    
    def _get_trading_symbols(self) -> List[str]:
        """Get the list of symbols to trade based on configuration mode"""
        try:
            data_config = self.config.get('trading', {}).get('data', {})
            mode = data_config.get('mode', 'manual')
            
            if mode == 'manual':
                # Get symbols from manual_symbols config
                symbols = data_config.get('manual_symbols', [])
                if not symbols:
                    logger.warning("No symbols configured in manual mode")
                return symbols
            else:  # auto mode
                # Discover NSE stocks based on criteria
                return self._discover_nse_stocks()
                
        except Exception as e:
            logger.error(f"Error getting trading symbols: {str(e)}")
            return []

    def get_live_quote(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get real-time quote for a symbol
        
        Args:
            symbol: Stock symbol (e.g. 'RELIANCE.NS')
        
        Returns:
            Dict with real-time quote data or None if not available
        """
        try:
            # Convert symbol format
            angel_symbol = symbol.replace('.NS', '-EQ')
            token = self.token_mapping.get(angel_symbol)
            
            if not token:
                logger.warning(f"Token not found for symbol {symbol}")
                return None
            
            # Get quote from live data
            quote = self.websocket.get_live_feed(token) if self.websocket else None
            
            if quote:
                # Add symbol info
                quote['symbol'] = symbol
                quote['exchange'] = 'NSE'
                
            return quote
            
        except Exception as e:
            logger.error(f"Error getting live quote for {symbol}: {str(e)}")
            return None
    
    def cleanup(self):
        """Cleanup resources and reset singleton instance"""
        try:
            self.is_running = False
            
            # Close WebSocket connection if exists
            if self.websocket:
                self.websocket.close()
                self.websocket = None
            
            # Wait for session timer to stop
            if self.session_timer and self.session_timer.is_alive():
                self.session_timer.join(timeout=5)
            
            # Reset instance
            DataCollector._instance = None
            self.initialized = False
            logger.info("DataCollector cleanup completed")
            
        except Exception as e:
            logger.error(f"Error during cleanup: {str(e)}")
            raise
            
    def __del__(self):
        """Destructor to ensure cleanup"""
        self.cleanup()
            
    def get_market_data(self) -> pd.DataFrame:
        """Get real-time market data for configured symbols
        
        If market is closed, attempts to fetch last known prices.
        Returns an empty DataFrame if no data is available.
        """
        try:
            # Get symbols to monitor
            symbols = self._get_trading_symbols()
            if not symbols:
                logger.warning("No symbols configured for monitoring")
                return pd.DataFrame()

            # Convert symbols to tokens and build token mapping
            tokens = []
            token_map = {}

            for symbol in symbols:
                angel_symbol = symbol.replace('.NS', '-EQ')
                token = self.token_mapping.get(angel_symbol)
                if token:
                    token_str = str(token)
                    tokens.append(token_str)
                    token_map[token_str] = symbol
                else:
                    logger.warning(f"Token not found for symbol {symbol}")

            if not tokens:
                logger.error("No valid tokens found for any configured symbols")
                return pd.DataFrame()

            # Initialize data structure
            data = []
            current_time = datetime.now()

            # Check if within market hours (9:15 AM to 3:30 PM IST on weekdays)
            market_open = (
                current_time.weekday() < 5 and  # Monday to Friday
                datetime.strptime("09:15:00", "%H:%M:%S").time() <= current_time.time() <= 
                datetime.strptime("15:30:00", "%H:%M:%S").time()
            )

            if market_open:
                # Ensure subscription for real-time data
                try:
                    current_feeds = set(self.websocket.live_feed.keys() if self.websocket else set())
                    missing_tokens = [t for t in tokens if t not in current_feeds]
                    
                    if missing_tokens and self.websocket and self.websocket.is_connected:
                        logger.info(f"Subscribing to {len(missing_tokens)} missing tokens...")
                        self.websocket.subscribe(missing_tokens)

                except Exception as e:
                    logger.error(f"Error managing WebSocket subscriptions: {str(e)}")

            # Collect data
            for token in tokens:
                symbol = token_map[token]
                tick_data = None

                if market_open and self.websocket and token in self.websocket.live_feed:
                    # Get real-time data if available
                    tick_data = self.websocket.live_feed[token]
                else:
                    # Market is closed or no real-time data, try to get last traded price
                    try:
                        ltp_data = self.angel_api.ltpData("NSE", symbol.replace(".NS", "-EQ"), token)
                        if ltp_data and ltp_data.get("status"):
                            tick_data = {
                                'token': token,
                                'ltp': float(ltp_data['data']['ltp']),
                                'volume': float(ltp_data['data'].get('trading_volume', 0)),
                                'timestamp': datetime.now()
                            }
                    except Exception as e:
                        logger.error(f"Error fetching LTP for {symbol}: {str(e)}")

                if tick_data:
                    data.append({
                        'symbol': symbol,
                        'token': token,
                        'price': tick_data.get('ltp', 0),
                        'volume': tick_data.get('volume', 0),
                        'bid_price': tick_data.get('bid_price', 0),
                        'ask_price': tick_data.get('ask_price', 0),
                        'timestamp': tick_data.get('timestamp', current_time)
                    })

            # Convert to DataFrame
            df = pd.DataFrame(data) if data else pd.DataFrame(columns=[
                'symbol', 'token', 'price', 'volume', 'bid_price', 'ask_price', 'timestamp'
            ])

            if df.empty:
                logger.warning("No market data available")
            else:
                logger.debug(f"Retrieved market data for {len(df)} symbols")

            return df

        except Exception as e:
            logger.error(f"Error getting market data: {str(e)}")
            return pd.DataFrame()
    @with_timeout(30)
    def get_historical_data(self, symbol: str, start_date: Optional[datetime] = None, 
                        end_date: Optional[datetime] = None, interval: str = '1d') -> pd.DataFrame:
        """Get historical market data using Angel One's REST API

        Args:
            symbol: Stock symbol (base symbol without exchange suffix)
            start_date: Start date for data
            end_date: End date for data
            interval: Data interval ('1d', '1h', '15m', '5m', '1m')

        Returns:
            DataFrame with historical market data
        """
        try:
            if not self.angel_api:
                logger.error("Angel One API not initialized")
                return pd.DataFrame()

            if not start_date:
                start_date = datetime.now() - timedelta(days=365)  # Default to 1 year
            if not end_date:
                end_date = datetime.now()

            # Force market hours
            from_date = start_date.replace(hour=9, minute=15, second=0, microsecond=0)  # Market opening time
            to_date = end_date.replace(hour=15, minute=30, second=0, microsecond=0)  # Market closing time

            # Convert symbol format
            base_symbol = symbol.split('.')[0].split('^')[0]  # Handle both normal and index symbols
            angel_symbol = base_symbol + '-EQ'
            token = self.token_mapping.get(angel_symbol)

            if not token:
                logger.warning(f"Token not found for symbol {symbol}")
                return pd.DataFrame()

            # Map intervals to Angel One format
            interval_map = {
                '1d': 'ONE_DAY',
                '1h': 'ONE_HOUR',
                '15m': 'FIFTEEN_MINUTE',
                '5m': 'FIVE_MINUTE',
                '1m': 'ONE_MINUTE'
            }
            angel_interval = interval_map.get(interval, 'ONE_DAY')

            # Prepare parameters for historical data
            params = {
                "exchange": "NSE",
                "symboltoken": token,
                "interval": angel_interval,
                "fromdate": from_date.strftime('%Y-%m-%d %H:%M'),
                "todate": to_date.strftime('%Y-%m-%d %H:%M')
            }

            # Get historical data with retry logic
            max_retries = 3
            retry_delay = 1  # Initial delay in seconds

            for attempt in range(max_retries):
                try:
                    # Rate limiting
                    time.sleep(retry_delay)

                    # Fetch historical data
                    response = self.angel_api.getCandleData(params)

                    if response.get('status'):
                        # Convert to DataFrame
                        df = pd.DataFrame(response['data'],
                                        columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])

                        # Convert timestamp to datetime
                        df['timestamp'] = pd.to_datetime(df['timestamp'])
                        df.set_index('timestamp', inplace=True)

                        # Validate data
                        if df.empty:
                            logger.warning(f"No data received for {symbol}")
                            return pd.DataFrame()

                        # Add metadata
                        df['symbol'] = symbol
                        df['exchange'] = 'NSE'
                        df['data_source'] = 'angel_one'

                        # Calculate statistics
                        stats = {
                            'records': len(df),
                            'date_range': f"{df.index.min()} to {df.index.max()}",
                            'min_price': df['low'].min(),
                            'max_price': df['high'].max(),
                            'avg_volume': df['volume'].mean(),
                            'total_volume': df['volume'].sum(),
                            'price_change': df['close'][-1] - df['close'][0],
                            'price_change_pct': ((df['close'][-1] - df['close'][0]) / df['close'][0]) * 100
                        }

                        # Log statistics using UTF-8 context manager
                        with utf8_stdout():
                            logger.info(f"\nData Statistics for {symbol}:")
                            logger.info(f"Total Records: {stats['records']}")
                            logger.info(f"Date Range: {stats['date_range']}")
                            logger.info(f"Price Range: Rs.{stats['min_price']:.2f} - Rs.{stats['max_price']:.2f}")
                            logger.info(f"Average Volume: {stats['avg_volume']:,.0f}")
                            logger.info(f"Total Volume: {stats['total_volume']:,.0f}")
                            logger.info(f"Price Change: Rs.{stats['price_change']:.2f} ({stats['price_change_pct']:.2f}%)")

                        # Add basic technical indicators
                        df['SMA_20'] = df['close'].rolling(window=20).mean()
                        df['Daily_Return'] = df['close'].pct_change()

                        # Sort by timestamp
                        df.sort_index(inplace=True)
                        return df

                    else:
                        error_msg = response.get('message', 'Unknown error')
                        logger.error(f"Error on attempt {attempt + 1}: {error_msg}")
                        if 'rate limit' in error_msg.lower():
                            retry_delay *= 2  # Exponential backoff
                            continue
                        else:
                            break

                except Exception as e:
                    logger.error(f"Error on attempt {attempt + 1}: {str(e)}")
                    retry_delay *= 2
                    continue

            logger.error(f"Failed to fetch historical data for {symbol} after {max_retries} attempts")
            return pd.DataFrame()

        except Exception as e:
            logger.error(f"Error fetching historical data for {symbol}: {str(e)}")
            return pd.DataFrame()
    
    def initialize_symbols(self):
        """Initialize trading symbols based on configuration"""
        data_config = self.config.get('trading', {}).get('data', {})
        mode = data_config.get('mode', 'manual')
        
        if mode == 'manual':
            self.symbols = data_config.get('manual_symbols', [])
        else:  # auto mode
            self.symbols = self._discover_nse_stocks()
            
        logger.info(f"Initialized {len(self.symbols)} symbols in {mode} mode")
    
    def get_training_data(self, start_date: Optional[datetime] = None, 
                         end_date: Optional[datetime] = None) -> pd.DataFrame:
        """Get historical data for all configured symbols for training"""
        if not start_date:
            start_date = datetime.now() - timedelta(days=365)  # 1 year of data
        if not end_date:
            end_date = datetime.now()
        
        # Get symbols based on configuration mode
        symbols = self._get_trading_symbols()
        logger.info(f"Collecting training data for {len(symbols)} symbols")
            
        all_data = []
        with ThreadPoolExecutor(max_workers=5) as executor:
            future_to_symbol = {
                executor.submit(self.get_historical_data, symbol, start_date, end_date): symbol 
                for symbol in symbols
            }
            
            for future in as_completed(future_to_symbol):
                symbol = future_to_symbol[future]
                try:
                    data = future.result()
                    if not data.empty:
                        data['symbol'] = symbol
                        all_data.append(data)
                except Exception as e:
                    logger.error(f"Error fetching data for {symbol}: {str(e)}")
        
        if all_data:
            combined_data = pd.concat(all_data, ignore_index=True)
            logger.info(f"Collected {len(combined_data)} data points across {len(symbols)} symbols")
            return combined_data
        else:
            raise ValueError("No training data could be collected")
