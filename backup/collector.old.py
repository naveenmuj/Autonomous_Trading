import os
import sys
import json
import logging
import requests
import pandas as pd
import numpy as np
import threading
import time
from typing import Dict, List, Optional, Union, Tuple
from datetime import datetime, timedelta
from SmartApi import SmartConnect  # Using the official import style
import pyotp
import yfinance as yf
import pandas_ta as ta
from functools import wraps

# Setup logging
logger = logging.getLogger(__name__)

def with_timeout(timeout_seconds: int):
    """Windows-compatible timeout decorator"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            result = []
            error = []
            
            def target():
                try:
                    result.append(func(*args, **kwargs))
                except Exception as e:
                    error.append(e)
            
            thread = threading.Thread(target=target)
            thread.daemon = True
            thread.start()
            thread.join(timeout_seconds)
            
            if thread.is_alive():
                thread.join(1)  # Give it one more second to cleanup
                raise TimeoutError(f"Function {func.__name__} timed out after {timeout_seconds} seconds")
            
            if error:
                raise error[0]
            
            return result[0] if result else None
        return wrapper
    return decorator

class DataCollector:
    """Data collection class for market data"""
    
    _instance = None  # Singleton instance
    INSTRUMENT_MASTER_URL = "https://margincalculator.angelbroking.com/OpenAPI_File/files/OpenAPIScripMaster.json"
    
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
            
            # Initialize Angel One API
            self._initialize_angel_api()
            # Initialize token mapping for NSE symbols
            self._initialize_token_mapping()
            # Start WebSocket connection
            self._initialize_websocket()
            
    def _generate_otp(self, totp_secret: str) -> str:
        """Generate TOTP for Angel One authentication
        
        Args:
            totp_secret (str): TOTP secret key from Angel One
            
        Returns:
            str: Generated 6-digit TOTP
        """
        if not totp_secret:
            raise ValueError("TOTP secret is required for Angel One authentication")
            
        try:
            totp = pyotp.TOTP(totp_secret)
            return totp.now()
        except Exception as e:
            logger.error(f"Error generating TOTP: {str(e)}")
            raise

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
                    self.refresh_token = data.get('data', {}).get('refreshToken', '')
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
        # Retry configuration
        max_retries = 3
        retry_delay = 5  # seconds
        retry_count = 0
        
        while retry_count < max_retries:
            try:
                # Check if we already have the mapping
                if hasattr(self, 'token_mapping') and self.token_mapping:
                    logger.info("Using existing token mapping")
                    return
                    
                # Get the instrument master file from Angel One
                response = requests.get(
                    'https://margincalculator.angelbroking.com/OpenAPI_File/files/OpenAPIScripMaster.json',
                    timeout=10
                )
                
                if response.status_code == 200:
                    try:
                        data = response.json()
                        # Create mapping for NSE equity symbols
                        self.token_mapping = {
                            item['symbol']: item['token']
                            for item in data
                            if item['exch_seg'] == 'NSE' and item['symbol'].endswith('-EQ')
                        }
                        logger.info(f"Token mapping initialized with {len(self.token_mapping)} symbols")
                        return  # Success, exit the retry loop
                    except json.JSONDecodeError:
                        logger.error("Error parsing instrument master JSON")
                        raise
                elif response.status_code == 429 or 'rate' in response.text.lower():
                    # Rate limited
                    retry_count += 1
                    if retry_count < max_retries:
                        logger.warning(f"Rate limited while fetching token mapping, waiting {retry_delay} seconds before retry {retry_count}/{max_retries}")
                        time.sleep(retry_delay)
                        retry_delay *= 2  # Exponential backoff
                        continue
                    else:
                        raise Exception("Max retries exceeded while fetching token mapping")
                else:
                    raise Exception(f"Failed to fetch instrument master. Status code: {response.status_code}")
                    
            except Exception as e:
                error_msg = str(e)
                logger.error(f"Error initializing token mapping: {error_msg}")
                if 'rate' in error_msg.lower() or isinstance(e, requests.exceptions.RequestException):
                    retry_count += 1
                    if retry_count < max_retries:
                        logger.warning(f"Network error, waiting {retry_delay} seconds before retry {retry_count}/{max_retries}")
                        time.sleep(retry_delay)
                        retry_delay *= 2
                        continue
                raise

    def _start_session_renewal_timer(self):
        """Start timer to renew Angel One session"""
        def renew_session():
            while True:
                time.sleep(3600)  # Sleep for 1 hour
                self._renew_session()
                
        thread = threading.Thread(target=renew_session, daemon=True)
        thread.start()
        
    def _renew_session(self):
        """Renew Angel One API session"""
        try:
            if self.angel_api:
                credentials = self.config.get('credentials', {}).get('angel_one', {})
                totp_key = credentials.get('totp_key')
                totp = pyotp.TOTP(totp_key)
                data = self.angel_api.generateSession(
                    clientcode=credentials.get('client_id'),
                    password=credentials.get('pin'),
                    totp=totp.now()
                )
                
                if not data['status']:
                    logger.error("Failed to renew Angel One session")
                    
        except Exception as e:
            logger.error(f"Error renewing Angel One session: {str(e)}")
            
    def _initialize_websocket(self):
        """Initialize WebSocket connection for real-time data"""
        try:
            if not self.angel_api:
                logger.warning("Angel One API not initialized, skipping WebSocket connection")
                return
            
            from .websocket import MarketDataWebSocket
            
            # Get required tokens from authenticated session
            credentials = self.config.get('apis', {}).get('angel_one', {})
            self.websocket = MarketDataWebSocket(
                auth_token=self.angel_api.auth_token,  # From authenticated session
                api_key=credentials.get('api_key'),
                client_code=credentials.get('client_id'),
                feed_token=self.angel_api.feed_token  # From authenticated session
            )
            
            # Register tick callback
            self.websocket.add_tick_callback(self._on_tick_data)
            
            # Connect WebSocket
            self.websocket.connect()
            
            # Subscribe to default symbols
            self._subscribe_default_symbols()
            
            logger.info("WebSocket connection initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing WebSocket: {str(e)}")
    
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
            if not self.websocket or not self.token_mapping:
                return
            
            # Get trading symbols
            symbols = self._get_trading_symbols()
            
            # Convert symbols to tokens
            tokens = []
            for symbol in symbols:
                # Convert NSE symbol format to Angel One format
                angel_symbol = symbol.replace('.NS', '-EQ')
                token = self.token_mapping.get(angel_symbol)
                if token:
                    tokens.append(token)
            
            if tokens:
                # Subscribe to tokens
                self.websocket.subscribe(tokens)
                logger.info(f"Subscribed to {len(tokens)} default symbols")
            
        except Exception as e:
            logger.error(f"Error subscribing to default symbols: {str(e)}")
    
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
            # Close WebSocket connection
            if self.websocket:
                self.websocket.close()
            
            # Close Angel One API session
            if self.angel_api:
                self.angel_api.terminate()
                
            DataCollector._instance = None
            self.initialized = False
            
            logger.info("DataCollector resources cleaned up")
            
        except Exception as e:
            logger.error(f"Error cleaning up resources: {str(e)}")
    
    @with_timeout(30)
    def get_historical_data(self, symbol: str, start_date: Optional[datetime] = None, 
                       end_date: Optional[datetime] = None, interval: str = '1d') -> pd.DataFrame:
        """Get historical market data with fallback to yfinance
        
        Args:
            symbol: Stock symbol (base symbol without exchange suffix)
            start_date: Start date for data
            end_date: End date for data
            interval: Data interval ('1d', '1h', etc.)
            
        Returns:
            DataFrame with market data
        """
        if not start_date:
            start_date = datetime.now() - timedelta(days=365)
        if not end_date:
            end_date = datetime.now()
        
        # Remove any exchange suffix if present
        base_symbol = symbol.split('.')[0].split('^')[0]  # Handle both normal and index symbols
        
        def _fetch_with_timeout():
            errors = []
            
            # Try Angel One first if available
            if self.angel_api and not symbol.startswith('^'):  # Don't try Angel One for indices
                try:
                    data = self._get_angel_one_data(symbol, start_date, end_date, interval)
                    if not data.empty:
                        return self._process_market_data(data, symbol)
                except Exception as e:
                    errors.append(f"Angel One error: {str(e)}")
            
            # Try yfinance with both exchanges and index format
            if symbol.startswith('^'):
                # For indices, try direct symbol
                try_symbols = [symbol]
            else:
                # For stocks, try both exchanges
                try_symbols = [
                    f"{base_symbol}.NS",  # NSE
                    f"{base_symbol}.BO",  # BSE
                ]
            
            best_data = None
            best_symbol = None
            
            for try_symbol in try_symbols:
                try:
                    logger.info(f"Attempting to fetch data for {try_symbol}")
                    data = yf.download(
                        try_symbol,
                        start=start_date,
                        end=end_date,
                        interval=interval,
                        progress=False,
                        auto_adjust=True  # Use adjusted data
                    )
                    
                    if not data.empty:
                        # Validate data quality
                        if (
                            len(data) > 10 and  # Require minimum number of data points
                            not data.isnull().all().all() and
                            ('Volume' not in data.columns or not (data['Volume'] == 0).all()) and
                            # Check for large date gaps (more than 5 days for daily data)
                            (interval != '1d' or data.index.to_series().diff().max().days <= 5)
                        ):
                            if best_data is None or len(data) > len(best_data):
                                # Compare data quality scores
                                current_quality = len(data) - data.isnull().sum().sum() / len(data.columns)
                                if best_data is not None:
                                    best_quality = len(best_data) - best_data.isnull().sum().sum() / len(best_data.columns)
                                else:
                                    best_quality = -1
                                    
                                if current_quality > best_quality:
                                    best_data = data
                                    best_symbol = try_symbol
                                    logger.info(f"Found better quality data from {try_symbol} (quality score: {current_quality:.2f})")
                        else:
                            logger.warning(f"Data quality check failed for {try_symbol}")
                            
                except Exception as e:
                    errors.append(f"{try_symbol} error: {str(e)}")
                    continue
            
            if best_data is not None:
                logger.info(f"Using data from {best_symbol}")
                result = self._process_market_data(best_data, symbol)
                result['data_source'] = 'yfinance'
                result['exchange'] = 'NSE' if '.NS' in best_symbol else 'BSE' if '.BO' in best_symbol else 'INDEX'
                return result
            
            # If all attempts failed, try test data as last resort
            if symbol in ['RELIANCE.NS', 'RELIANCE.BO', 'TCS.NS', 'TCS.BO', '^NSEI', '^BSESN']:
                logger.warning(f"Falling back to test data for {symbol}")
                test_data = self._generate_test_data(
                    symbols=[symbol],
                    start_date=start_date,
                    end_date=end_date,
                    interval=interval
                )
                test_data['data_source'] = 'test'
                test_data['exchange'] = 'TEST'
                return test_data
                
            error_msg = "; ".join(errors)
            raise ValueError(f"Failed to fetch data for {symbol}: {error_msg}")
            
        return _fetch_with_timeout()
        
    def _process_market_data(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """Process market data and add technical indicators
        
        Args:
            df: DataFrame with OHLCV data
            symbol: Stock symbol
            
        Returns:
            DataFrame with added technical indicators
        """
        try:
            # Ensure required columns exist
            required_cols = ['open', 'high', 'low', 'close', 'volume']
            df.columns = df.columns.str.lower()
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                raise ValueError(f"Missing required columns: {missing_cols}")
        
            # Add symbol column
            df['symbol'] = symbol
            
            # Calculate technical indicators if configured
            if self.config and 'model' in self.config:
                features = self.config['model'].get('features', {})
                if features.get('technical_indicators'):
                    # SMA
                    df['sma_10'] = df['close'].rolling(window=10).mean()
                    df['sma_20'] = df['close'].rolling(window=20).mean()
                    df['sma_50'] = df['close'].rolling(window=50).mean()
                    
                    # RSI
                    try:
                        rsi = df.ta.rsi(close='close', length=14)
                        df['rsi'] = rsi
                    except:
                        # Fallback calculation if pandas_ta fails
                        delta = df['close'].diff()
                        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
                        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
                        rs = gain / loss
                        df['rsi'] = 100 - (100 / (1 + rs))
                    
                    # MACD
                    try:
                        macd = df.ta.macd(close='close')
                        if isinstance(macd, pd.DataFrame):
                            df['macd'] = macd.iloc[:, 0]
                            df['macd_signal'] = macd.iloc[:, 1]
                            df['macd_hist'] = macd.iloc[:, 2]
                        else:
                            df['macd'] = macd
                    except:
                        # Fallback calculation
                        exp1 = df['close'].ewm(span=12, adjust=False).mean()
                        exp2 = df['close'].ewm(span=26, adjust=False).mean()
                        df['macd'] = exp1 - exp2
                        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
                        df['macd_hist'] = df['macd'] - df['macd_signal']
                    
                    # Bollinger Bands
                    try:
                        bb = df.ta.bbands(close='close', length=20)
                        if isinstance(bb, pd.DataFrame):
                            df['bb_upper'] = bb.iloc[:, 0]
                            df['bb_middle'] = bb.iloc[:, 1]
                            df['bb_lower'] = bb.iloc[:, 2]
                    except:
                        # Fallback calculation
                        sma = df['close'].rolling(window=20).mean()
                        std = df['close'].rolling(window=20).std()
                        df['bb_upper'] = sma + (std * 2)
                        df['bb_middle'] = sma
                        df['bb_lower'] = sma - (std * 2)
        
            # Forward fill any NaN values
            return df.ffill()
        
        except Exception as e:
            logger.error(f"Error processing market data for {symbol}: {str(e)}")
            raise

    def _get_trading_symbols(self) -> List[str]:
        """Get the list of symbols to trade based on configuration mode"""
        try:
            data_config = self._validate_trading_config()
            mode = data_config['mode']  # Safe after validation
            symbols = []
            
            if mode == 'manual':
                # Get manually configured symbols - safe after validation
                raw_symbols = data_config['manual_symbols']
                # Strip any exchange suffixes from manual symbols
                symbols = [s.split('.')[0] for s in raw_symbols]
                logger.info(f"Using manually configured symbols: {len(symbols)} symbols")
            
            elif mode == 'auto':
                auto_config = data_config['auto']  # Safe after validation
                market_cap = auto_config['market_cap']
                min_volume = auto_config['min_volume']
                sectors = auto_config.get('sectors', [])
                exclude = set(auto_config.get('exclude_symbols', []))
                
                # Validate token mapping is initialized
                if not self.token_mapping:
                    raise ValueError("Token mapping not initialized. Cannot fetch symbols in auto mode")
                
                # Get all symbols from token mapping
                all_symbols = [
                    symbol.split('.')[0]  # Remove any exchange suffix
                    for symbol in self.token_mapping.keys()
                    if symbol.endswith('-EQ') and symbol not in exclude
                ]
                
                if sectors:
                    # Filter by sectors if specified
                    symbols = self._filter_symbols_by_sectors(all_symbols, sectors)
                else:
                    symbols = all_symbols
                
                # Apply volume filter using Angel One API
                symbols = self._filter_symbols_by_volume(symbols, min_volume)
                logger.info(f"Auto mode found {len(symbols)} symbols matching criteria")
            
            # Always include configured indices (without any exchange suffix)
            indices = data_config.get('indices', [])
            symbols.extend([idx.replace('.NS', '').replace('.BO', '') for idx in indices])
            
            # Final validation
            symbols = list(set(symbols))  # Remove duplicates
            if not symbols:
                raise ValueError(
                    f"No symbols found based on current configuration (mode: {mode}). "
                    "Please check your configuration and ensure it will result in at least one symbol."
                )
            
            logger.info(f"Final symbol list contains {len(symbols)} unique symbols")
            return symbols
            
        except Exception as e:
            logger.error(f"Error in _get_trading_symbols: {str(e)}")
            raise
            
    def get_market_data(self) -> pd.DataFrame:
        """Get market data for configured symbols"""
        try:
            # Get base symbols without exchange suffixes
            symbols = self._get_trading_symbols()
            logger.info(f"Fetching data for {len(symbols)} symbols")
            
            data_frames = []
            failed_symbols = []
            
            # Fetch data for each symbol using the exchange manager
            end_date = datetime.now()
            start_date = end_date - timedelta(days=365)  # Default to 1 year of data
            
            for symbol in symbols:
                try:
                    data, used_symbol = self._exchange_manager.get_data(symbol, start_date, end_date)
                    if data is not None and not data.empty:
                        # Add symbol and exchange info
                        data['symbol'] = symbol
                        data['exchange'] = 'NSE' if '.NS' in used_symbol else 'BSE'
                        data_frames.append(data)
                    else:
                        failed_symbols.append(symbol)
                except Exception as e:
                    logger.error(f"Error fetching data for {symbol}: {str(e)}")
                    failed_symbols.append(symbol)
            
            if not data_frames:
                raise ValueError("No data available for any configured symbols")
            
            # Combine all data frames
            combined_data = pd.concat(data_frames, axis=0)
            
            if failed_symbols:
                logger.warning(f"Failed to fetch data for symbols: {failed_symbols}")
            
            return combined_data
            
        except Exception as e:
            logger.error(f"Error in get_market_data: {str(e)}")
            raise
          
    def _generate_test_data(self, days: int = 100, symbols: Optional[List[str]] = None,
                            start_date: Optional[datetime] = None, 
                            end_date: Optional[datetime] = None,
                            interval: str = '1d') -> pd.DataFrame:
        """Generate synthetic market data for testing
        
        Args:
            days: Number of days to generate
            symbols: List of symbols to generate for
            start_date: Start date for data
            end_date: End date for data
            interval: Data interval ('1d', '1h', etc.)
            
        Returns:
            DataFrame with synthetic market data
        """
        if symbols is None:
            # Include both NSE and BSE symbols for testing
            symbols = [
                'RELIANCE.NS', 'RELIANCE.BO',
                'TCS.NS', 'TCS.BO',
                '^NSEI', '^BSESN'
            ]
            
        if start_date is None:
            start_date = datetime.now() - timedelta(days=days)
        if end_date is None:
            end_date = datetime.now()
            
        # Convert interval to pandas frequency
        freq_map = {
            '1m': 'T', '5m': '5T', '15m': '15T', '30m': '30T',
            '1h': 'H', '1d': 'D', '1wk': 'W', '1mo': 'M'
        }
        freq = freq_map.get(interval, 'D')
        
        dfs = []
        for symbol in symbols:
            # Generate dates
            dates = pd.date_range(start=start_date, end=end_date, freq=freq)
            if len(dates) == 0:
                continue
                
            # Generate base price and trend
            base_price = 1000.0
            trend = np.random.normal(0, 0.0002, size=len(dates)).cumsum()
            volatility = np.random.uniform(0.01, 0.02)
            
            # Generate OHLCV data
            df = pd.DataFrame({
                'datetime': dates,
                'open': base_price * (1 + trend + np.random.normal(0, volatility, size=len(dates))),
                'close': base_price * (1 + trend + np.random.normal(0, volatility, size=len(dates)))
            })
            
            # Ensure high/low maintain proper relationship
            df['high'] = df[['open', 'close']].max(axis=1) + base_price * np.random.uniform(0, volatility, size=len(dates))
            df['low'] = df[['open', 'close']].min(axis=1) - base_price * np.random.uniform(0, volatility, size=len(dates))
              # Generate volume (higher for NSE, lower for BSE)
            is_nse = '.NS' in symbol
            volume_base = 500000 if is_nse else 200000
            volume_range = 500000 if is_nse else 200000
            df['volume'] = np.random.randint(volume_base, volume_base + volume_range, size=len(dates))
            
            # Set symbol, index and exchange info
            df['symbol'] = symbol.split('.')[0].split('^')[0]  # Base symbol without exchange
            df['exchange'] = 'NSE' if '.NS' in symbol else 'BSE' if '.BO' in symbol else 'INDEX'
            df['data_source'] = 'test'
            df.set_index('datetime', inplace=True)
            
            # Add technical indicators
            df = self._process_market_data(df, symbol)
            dfs.append(df)
            
        if not dfs:
            raise ValueError("No test data generated")
            
        result = pd.concat(dfs)
        return result.sort_index()
        
    def get_nse_stocks(self) -> List[str]:
        """Get list of NSE stocks in both formats (with and without .NS suffix)"""
        try:
            stocks = set()
            if self.angel_api:
                all_symbols = self.angel_api.getAllSymbols()
                for item in all_symbols:
                    if item['exch_seg'] == 'NSE' and item['instrumenttype'] == 'EQ':
                        symbol = item['symbol']
                        stocks.add(symbol)  # Without .NS
                        stocks.add(f"{symbol}.NS")  # With .NS
            else:
                # Fallback to default NSE-50 symbols for testing
                default_symbols = [
                    "RELIANCE", "TCS", "HDFCBANK", "INFY", "HDFC",
                    "ICICIBANK", "ITC", "KOTAKBANK", "LT", "AXISBANK"
                ]
                for symbol in default_symbols:
                    stocks.add(symbol)  # Without .NS
                    stocks.add(f"{symbol}.NS")  # With .NS
                
            return sorted(list(stocks))
        
        except Exception as e:
            logger.error(f"Error fetching NSE stocks: {str(e)}")
            raise

    def _validate_trading_config(self) -> Dict:
        """Validate trading configuration and return validated config
        
        Raises:
            ValueError: If configuration is invalid or missing required fields
        """
        if not self.config:
            raise ValueError("Configuration is not initialized")
            
        trading_config = self.config.get('trading')
        if not trading_config:
            raise ValueError("Trading configuration section is missing")
            
        data_config = trading_config.get('data')
        if not data_config:
            raise ValueError("Trading data configuration section is missing")
            
        mode = data_config.get('mode')
        if not mode:
            raise ValueError("Trading mode must be explicitly set in config")
        if mode not in ['manual', 'auto']:
            raise ValueError(f"Invalid trading mode '{mode}'. Must be 'manual' or 'auto'")
            
        if mode == 'manual':
            manual_symbols = data_config.get('manual_symbols')
            if not manual_symbols:
                raise ValueError("Manual mode requires 'manual_symbols' list in config")
            if not isinstance(manual_symbols, list) or not manual_symbols:
                raise ValueError("manual_symbols must be a non-empty list")
                
        elif mode == 'auto':
            auto_config = data_config.get('auto')
            if not auto_config:
                raise ValueError("Auto mode requires 'auto' configuration section")
            if 'min_volume' not in auto_config:
                raise ValueError("Auto mode requires 'min_volume' parameter")
            if 'market_cap' not in auto_config:
                raise ValueError("Auto mode requires 'market_cap' parameter")
                
        return data_config

    def _get_trading_symbols(self) -> List[str]:
        """Get the list of symbols to trade based on configuration mode
        
        Returns:
            List[str]: List of trading symbols
            
        Raises:
            ValueError: If configuration is invalid or no symbols are found
        """
        try:
            data_config = self._validate_trading_config()
            mode = data_config['mode']  # Safe after validation
            symbols = []
            
            if mode == 'manual':
                # Get manually configured symbols - safe after validation
                raw_symbols = data_config['manual_symbols']
                # Strip any exchange suffixes from manual symbols
                symbols = [s.split('.')[0] for s in raw_symbols]
                logger.info(f"Using manually configured symbols: {len(symbols)} symbols")
            
            elif mode == 'auto':
                auto_config = data_config['auto']  # Safe after validation
                market_cap = auto_config['market_cap']
                min_volume = auto_config['min_volume']
                sectors = auto_config.get('sectors', [])
                exclude = set(auto_config.get('exclude_symbols', []))
                
                # Validate token mapping is initialized
                if not self.token_mapping:
                    raise ValueError("Token mapping not initialized. Cannot fetch symbols in auto mode")
                
                # Get all symbols from token mapping
                all_symbols = [
                    symbol.split('.')[0]  # Remove any exchange suffix
                    for symbol in self.token_mapping.keys()
                    if symbol.endswith('-EQ') and symbol not in exclude
                ]
                
                if sectors:
                    # Filter by sectors if specified
                    symbols = self._filter_symbols_by_sectors(all_symbols, sectors)
                else:
                    symbols = all_symbols
                
                # Apply volume filter using Angel One API
                symbols = self._filter_symbols_by_volume(symbols, min_volume)
                logger.info(f"Auto mode found {len(symbols)} symbols matching criteria")
            
            # Always include configured indices (without any exchange suffix)
            indices = data_config.get('indices', [])
            symbols.extend([idx.replace('.NS', '').replace('.BO', '') for idx in indices])
            
            # Final validation
            symbols = list(set(symbols))  # Remove duplicates
            if not symbols:
                raise ValueError(
                    f"No symbols found based on current configuration (mode: {mode}). "
                    "Please check your configuration and ensure it will result in at least one symbol."
                )
            
            logger.info(f"Final symbol list contains {len(symbols)} unique symbols")
            return symbols
            
        except Exception as e:
            logger.error(f"Error in _get_trading_symbols: {str(e)}")
            raise

class ExchangeDataManager:
    """Manages data fetching from multiple exchanges with caching and fallback"""
    
    def __init__(self):
        self._symbol_cache = {}  # Cache of working symbols
        self._exchange_priority = ['NS', 'BO']  # Default to trying NSE first
        
    def _get_exchange_symbols(self, base_symbol: str) -> List[str]:
        """Generate exchange-specific symbols to try"""
        if base_symbol.startswith('^'):  # Index symbol
            return [base_symbol]
        return [f"{base_symbol}.{exchange}" for exchange in self._exchange_priority]
    
    def _validate_data_quality(self, data: pd.DataFrame, symbol: str) -> bool:
        """Check if the data meets quality standards"""
        try:
            if data is None or data.empty:
                return False
            
            # Check for missing values
            if data.isnull().sum().sum() > len(data) * 0.1:  # More than 10% missing
                logger.warning(f"Poor data quality for {symbol}: Too many missing values")
                return False
                
            # Check for zero volume (might indicate poor data)
            if 'Volume' in data.columns and (data['Volume'] == 0).all():
                logger.warning(f"Poor data quality for {symbol}: All volumes are zero")
                return False
                
            # Check for duplicate indices
            if data.index.duplicated().any():
                logger.warning(f"Poor data quality for {symbol}: Duplicate timestamps")
                return False
                
            return True
            
        except Exception as e:
            logger.error(f"Error validating data quality for {symbol}: {str(e)}")
            return False
    
    @with_timeout(30)  # 30 second timeout for data fetch
    def get_data(self, base_symbol: str, start_date: datetime, end_date: datetime) -> Tuple[pd.DataFrame, str]:
        """
        Fetch data for a symbol, trying multiple exchanges if necessary
        
        Args:
            base_symbol: Base symbol without exchange suffix
            start_date: Start date for data
            end_date: End date for data
            
        Returns:
            Tuple of (DataFrame with market data, successful symbol used)
        """
        # Check cache first
        if base_symbol in self._symbol_cache:
            try:
                cached_symbol = self._symbol_cache[base_symbol]
                data = yf.download(cached_symbol, start=start_date, end=end_date, progress=False, auto_adjust=True)
                if self._validate_data_quality(data, cached_symbol):
                    return data, cached_symbol
            except Exception as e:
                logger.warning(f"Cached symbol {cached_symbol} failed: {str(e)}")
                del self._symbol_cache[base_symbol]
        
        # Try each exchange in priority order
        errors = []
        best_data = None
        best_symbol = None
        
        for symbol in self._get_exchange_symbols(base_symbol):
            try:
                logger.info(f"Attempting to fetch data for {symbol}")
                data = yf.download(symbol, start=start_date, end=end_date, progress=False, auto_adjust=True)
                
                if self._validate_data_quality(data, symbol):
                    if best_data is None or len(data) > len(best_data):
                        best_data = data
                        best_symbol = symbol
                        
            except Exception as e:
                errors.append(f"{symbol}: {str(e)}")
                continue
        
        if best_data is not None and best_symbol is not None:
            # Cache the successful symbol
            self._symbol_cache[base_symbol] = best_symbol
            return best_data, best_symbol
            
        raise ValueError(f"Failed to fetch data for {base_symbol} from any exchange: {'; '.join(errors)}")
