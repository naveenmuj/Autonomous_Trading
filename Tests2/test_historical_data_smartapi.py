import yaml
import pyotp
import json
import time
from datetime import datetime, timedelta
from SmartApi import SmartConnect
import logging
import pandas as pd

# Set up logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

def load_config():
    """Load configuration from config.yaml"""
    try:
        with open('config.yaml', 'r') as file:
            config = yaml.safe_load(file)
            return config['apis']['angel_one']
    except Exception as e:
        logger.error(f"Error loading config: {str(e)}")
        raise

class HistoricalDataTester:
    """Test class for historical data fetching from Angel One SmartAPI"""
    
    EXCHANGE_MAP = {
        'NSE': 'NSE',
        'BSE': 'BSE',
        'NFO': 'NFO'
    }

    # Common token mapping for testing
    TOKEN_MAP = {
        'SUZLON': '12018',  # NSE
        'SBIN': '3045',     # NSE SBIN
        'TCS': '11536',     # NSE TCS
        'INFY': '1594',     # NSE INFOSYS
        'RELIANCE': '2885'  # NSE RELIANCE
    }

    # API rate limits
    RATE_LIMIT_SLEEP = 1  # seconds between requests
    MAX_RETRIES = 3

    def __init__(self):
        """Initialize the tester"""
        self.smart_api = None
        self.auth_token = None
        self.refresh_token = None
        self.feed_token = None
        
        # Load credentials
        creds = load_config()
        self.api_key = creds['api_key']
        self.client_id = creds['client_id']
        self.mpin = creds['mpin']
        self.totp_secret = creds['totp_secret']
        
        # Track request times for rate limiting
        self.last_request_time = 0

    def login(self):
        """Login to Angel One"""
        try:
            logger.info("Initializing SmartAPI login...")
            self.smart_api = SmartConnect(api_key=self.api_key)
            
            # Generate TOTP
            totp = pyotp.TOTP(self.totp_secret)
            totp_value = totp.now()
            
            # Login
            login = self.smart_api.generateSession(self.client_id, self.mpin, totp_value)
            
            if login.get('status'):
                self.auth_token = login['data']['jwtToken']
                self.refresh_token = login['data']['refreshToken']
                self.feed_token = self.smart_api.getfeedToken()
                logger.info("Login successful")
                return True
            else:
                logger.error(f"Login failed: {login.get('message')}")
                return False
                
        except Exception as e:
            logger.error(f"Error in login: {str(e)}")
            return False

    def _wait_for_rate_limit(self):
        """Implements rate limiting"""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        if time_since_last < self.RATE_LIMIT_SLEEP:
            time.sleep(self.RATE_LIMIT_SLEEP - time_since_last)
        self.last_request_time = time.time()

    def _validate_data(self, df):
        """Validate the fetched data"""
        if df is None or len(df) == 0:
            return False, "No data received"
            
        # Check for missing values
        if df.isnull().any().any():
            missing = df.isnull().sum()
            logger.warning(f"Missing values found:\n{missing}")
            
        # Check for duplicate timestamps
        duplicates = df[df.duplicated(subset=['timestamp'])]
        if not duplicates.empty:
            logger.warning(f"Found {len(duplicates)} duplicate timestamps")
            
        # Check for out-of-order timestamps
        if not df['timestamp'].is_monotonic_increasing:
            logger.warning("Timestamps are not in chronological order")
            
        # Basic value validation
        if (df['high'] < df['low']).any():
            return False, "Invalid high/low values found"
            
        if (df['volume'] < 0).any():
            return False, "Negative volumes found"
            
        return True, "Data validation passed"

    def fetch_historical_data(self, symbol, exchange, from_date, to_date, interval="ONE_MINUTE"):
        """
        Fetch historical data for a given symbol with retry logic and validation
        
        Parameters:
        - symbol: Trading symbol (e.g., 'SBIN-EQ') or token
        - exchange: Exchange (NSE/BSE/NFO)
        - from_date: Start date (format: 'YYYY-MM-DD HH:mm')
        - to_date: End date (format: 'YYYY-MM-DD HH:mm')
        - interval: Candle interval (ONE_MINUTE/THREE_MINUTE/FIVE_MINUTE/TEN_MINUTE/
                   FIFTEEN_MINUTE/THIRTY_MINUTE/ONE_HOUR/ONE_DAY)
        """
        for attempt in range(self.MAX_RETRIES):
            try:
                # Rate limiting
                self._wait_for_rate_limit()
                
                # Convert dates to expected format
                from_datetime = datetime.strptime(from_date, '%Y-%m-%d %H:%M')
                to_datetime = datetime.strptime(to_date, '%Y-%m-%d %H:%M')
                
                # If symbol is in token map, use the token
                if symbol in self.TOKEN_MAP:
                    symbol = self.TOKEN_MAP[symbol]
                
                logger.info(f"Fetching historical data for {symbol} from {from_date} to {to_date} (Attempt {attempt + 1})")
                
                # Fetch historical data
                historicParam = {
                    "exchange": exchange,
                    "symboltoken": symbol,
                    "interval": interval,
                    "fromdate": from_datetime.strftime('%Y-%m-%d %H:%M'),
                    "todate": to_datetime.strftime('%Y-%m-%d %H:%M')
                }
                
                resp = self.smart_api.getCandleData(historicParam)
                
                if resp['status']:
                    # Convert to DataFrame for better analysis
                    data = pd.DataFrame(resp['data'], 
                                      columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                    data['timestamp'] = pd.to_datetime(data['timestamp'])
                    
                    # Validate data
                    is_valid, message = self._validate_data(data)
                    if not is_valid:
                        raise ValueError(f"Data validation failed: {message}")
                        
                    return data
                else:
                    error_msg = resp.get('message', 'Unknown error')
                    if 'rate limit' in error_msg.lower():
                        # If rate limited, wait longer before retry
                        time.sleep(self.RATE_LIMIT_SLEEP * 2)
                        continue
                    logger.error(f"Error fetching historical data: {error_msg}")
                    return None
                    
            except Exception as e:
                logger.error(f"Error on attempt {attempt + 1}: {str(e)}")
                if attempt < self.MAX_RETRIES - 1:
                    time.sleep(self.RATE_LIMIT_SLEEP * (attempt + 1))
                    continue
                return None
        
        logger.error(f"Failed to fetch data after {self.MAX_RETRIES} attempts")
        return None

def main():
    """Main test function with comprehensive test cases"""
    try:
        # Initialize tester
        tester = HistoricalDataTester()
        
        # Login
        if not tester.login():
            logger.error("Failed to login. Exiting...")
            return

        # Test cases
        test_cases = [
            {
                'symbol': 'SUZLON',
                'exchange': 'NSE',
                'days': 1,
                'intervals': ['ONE_MINUTE']
            },
            {
                'symbol': 'SBIN',
                'exchange': 'NSE',
                'days': 5,
                'intervals': ['FIVE_MINUTE', 'FIFTEEN_MINUTE']
            },
            {
                'symbol': 'RELIANCE',
                'exchange': 'NSE',
                'days': 30,
                'intervals': ['ONE_HOUR', 'ONE_DAY']
            }
        ]

        for case in test_cases:
            logger.info(f"\nTesting {case['symbol']} data...")
            
            # Calculate date range
            end_date = datetime.now()
            start_date = end_date - timedelta(days=case['days'])
            
            # Format dates
            from_date = start_date.strftime('%Y-%m-%d 09:15')  # Market opening time
            to_date = end_date.strftime('%Y-%m-%d %H:%M')

            for interval in case['intervals']:
                logger.info(f"\nFetching {interval} data for {case['symbol']}...")
                df = tester.fetch_historical_data(
                    case['symbol'],
                    case['exchange'],
                    from_date,
                    to_date,
                    interval
                )
                
                if df is not None:
                    # Basic statistics
                    logger.info(f"\n{interval} Data Summary:")
                    logger.info(f"Total records: {len(df)}")
                    logger.info(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
                    
                    # Market statistics
                    logger.info("\nMarket Statistics:")
                    logger.info(f"Highest price: {df['high'].max():.2f}")
                    logger.info(f"Lowest price: {df['low'].min():.2f}")
                    logger.info(f"Average volume: {df['volume'].mean():.2f}")
                    
                    # Daily summary if interval is minutes
                    if 'MINUTE' in interval:
                        daily_summary = df.resample('D', on='timestamp').agg({
                            'open': 'first',
                            'high': 'max',
                            'low': 'min',
                            'close': 'last',
                            'volume': 'sum'
                        }).dropna()
                        
                        logger.info("\nDaily Summary:")
                        logger.info(f"\n{daily_summary.head()}")
                    
                    # Calculate basic technical indicators
                    df['SMA_20'] = df['close'].rolling(window=20).mean()
                    df['Daily_Return'] = df['close'].pct_change()
                    
                    logger.info("\nTechnical Indicators:")
                    logger.info(f"Average Daily Return: {df['Daily_Return'].mean():.4f}")
                    logger.info(f"Return Volatility: {df['Daily_Return'].std():.4f}")
                    
                    logger.info("\nFirst few records:")
                    logger.info(f"\n{df.head()}")
                else:
                    logger.error(f"Failed to fetch {interval} data for {case['symbol']}")
                
                # Add delay between requests
                time.sleep(1)
        
    except Exception as e:
        logger.error(f"Error in main: {str(e)}")
    finally:
        # Cleanup
        if hasattr(tester, 'smart_api'):
            try:
                tester.smart_api.terminateSession(tester.client_id)
                logger.info("Session terminated")
            except Exception as e:
                logger.error(f"Error during cleanup: {str(e)}")
                pass

if __name__ == "__main__":
    main()
