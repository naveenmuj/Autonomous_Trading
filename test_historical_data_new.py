import logging
import yaml
import pyotp
import json
import time
import pandas as pd
from datetime import datetime, timedelta
from SmartApi import SmartConnect

# Set up logging with debug level
logging.basicConfig(
    level=logging.INFO,  # Changed to INFO to reduce noise
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('historical_data_test.log', encoding='utf-8')
    ]
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

class HistoricalDataFetcher:
    """Class for fetching historical data from Angel One SmartAPI"""
    
    INTERVALS = {
        "1min": "ONE_MINUTE",
        "5min": "FIVE_MINUTE",
        "15min": "FIFTEEN_MINUTE",
        "30min": "THIRTY_MINUTE",
        "1hour": "ONE_HOUR",
        "1day": "ONE_DAY"
    }
    
    # Common stock tokens for testing
    STOCK_TOKENS = {
        'SUZLON': '12018',
        'SBIN': '3045',
        'TCS': '11536',
        'RELIANCE': '2885'
    }
    
    def __init__(self):
        """Initialize the fetcher"""
        self.smart_api = None
        self.client_id = None
        self.last_request_time = 0
        self.rate_limit_delay = 1  # seconds between requests

    def _wait_for_rate_limit(self):
        """Implement rate limiting"""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        if time_since_last < self.rate_limit_delay:
            time.sleep(self.rate_limit_delay - time_since_last)
        self.last_request_time = time.time()

    def connect(self):
        """Connect to Angel One API"""
        try:
            # Load credentials
            creds = load_config()
            self.client_id = creds['client_id']
            
            # Generate TOTP
            totp = pyotp.TOTP(creds['totp_secret'])
            totp_value = totp.now()
            
            # Initialize SmartConnect
            self.smart_api = SmartConnect(api_key=creds['api_key'])
            
            # Login
            login = self.smart_api.generateSession(creds['client_id'], creds['mpin'], totp_value)
            
            if login.get('status'):
                logger.info("Connected to Angel One API successfully")
                return True
            else:
                logger.error(f"Connection failed: {login.get('message')}")
                return False
                
        except Exception as e:
            logger.error(f"Error connecting to API: {str(e)}")
            return False

    def get_historical_data(self, symbol, token=None, interval="1day", days=30):
        """
        Fetch historical data for a symbol
        
        Parameters:
        - symbol: Stock symbol (e.g., 'SUZLON')
        - token: Trading token (optional, will be looked up if not provided)
        - interval: Time interval (1min/5min/15min/30min/1hour/1day)
        - days: Number of days of historical data to fetch
        """
        try:
            self._wait_for_rate_limit()
            
            # Get token if not provided
            if token is None:
                token = self.STOCK_TOKENS.get(symbol)
                if token is None:
                    raise ValueError(f"Trading token not found for symbol {symbol}")
            
            # Calculate date range
            to_date = datetime.now()
            from_date = to_date - timedelta(days=days)
            
            # Adjust for market hours
            from_date = from_date.replace(hour=9, minute=15, second=0, microsecond=0)
            to_date = to_date.replace(hour=15, minute=30, second=0, microsecond=0)
            
            # Format dates for API
            from_date_str = from_date.strftime('%Y-%m-%d %H:%M')
            to_date_str = to_date.strftime('%Y-%m-%d %H:%M')
            
            api_interval = self.INTERVALS.get(interval, interval)
            
            logger.info(f"Fetching {interval} data for {symbol} from {from_date_str} to {to_date_str}")
            
            # Call API
            hist_data = self.smart_api.getCandleData({
                "exchange": "NSE",
                "symboltoken": token,
                "interval": api_interval,
                "fromdate": from_date_str,
                "todate": to_date_str
            })
            
            if hist_data and isinstance(hist_data, dict) and hist_data.get('data'):
                # Convert to DataFrame
                df = pd.DataFrame(hist_data['data'],
                                columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df.set_index('timestamp', inplace=True)
                
                # Calculate statistics
                stats = {
                    'records': len(df),
                    'date_range': f"{df.index.min()} to {df.index.max()}",
                    'price_range': f"₹{df['low'].min():.2f} - ₹{df['high'].max():.2f}",
                    'avg_volume': df['volume'].mean(),
                    'total_volume': df['volume'].sum(),
                    'price_change': df['close'][-1] - df['close'][0],
                    'price_change_pct': ((df['close'][-1] - df['close'][0]) / df['close'][0]) * 100
                }
                
                # Print statistics
                logger.info(f"\nData Statistics for {symbol}:")
                logger.info(f"Total Records: {stats['records']}")
                logger.info(f"Date Range: {stats['date_range']}")
                logger.info(f"Price Range: {stats['price_range']}")
                logger.info(f"Average Volume: {stats['avg_volume']:,.0f}")
                logger.info(f"Total Volume: {stats['total_volume']:,.0f}")
                logger.info(f"Price Change: ₹{stats['price_change']:.2f} ({stats['price_change_pct']:.2f}%)")
                
                return df
            else:
                logger.error(f"No data received for {symbol}")
                if isinstance(hist_data, dict):
                    logger.error(f"Error message: {hist_data.get('message')}")
                return None
                
        except Exception as e:
            logger.error(f"Error fetching data for {symbol}: {str(e)}")
            return None

    def disconnect(self):
        """Disconnect from Angel One API"""
        if self.smart_api and self.client_id:
            try:
                self.smart_api.terminateSession(self.client_id)
                logger.info("Disconnected from Angel One API")
            except Exception as e:
                logger.error(f"Error during disconnect: {str(e)}")

def main():
    """Main test function"""
    # Test data
    test_data = [
        {"symbol": "SUZLON", "token": "12018"},
        {"symbol": "SBIN", "token": "3045"},
        {"symbol": "TCS", "token": "11536"},
        {"symbol": "RELIANCE", "token": "2885"}
    ]
    
    # Initialize fetcher
    fetcher = HistoricalDataFetcher()
    
    try:
        # Connect
        if not fetcher.connect():
            return
        
        # Test different intervals for SUZLON
        intervals = ["1min", "5min", "15min", "1hour", "1day"]
        days_map = {"1min": 1, "5min": 2, "15min": 5, "1hour": 7, "1day": 30}
        
        # First test SUZLON with all intervals
        symbol_data = test_data[0]  # SUZLON
        for interval in intervals:
            days = days_map[interval]
            logger.info(f"\nTesting {interval} data for {symbol_data['symbol']}")
            
            df = fetcher.get_historical_data(
                symbol_data['symbol'],
                token=symbol_data['token'],
                interval=interval,
                days=days
            )
            
            if df is not None:
                print(f"\nFirst few records for {symbol_data['symbol']} ({interval}):")
                print(df.head().to_string())
                print("\nBasic statistics:")
                print(df.describe().to_string())
            
            time.sleep(1)  # Rate limiting
        
        # Then test daily data for all symbols
        logger.info("\nTesting daily data for all symbols...")
        for symbol_data in test_data[1:]:  # Skip SUZLON as it was already tested
            df = fetcher.get_historical_data(
                symbol_data['symbol'],
                token=symbol_data['token'],
                interval="1day",
                days=30
            )
            
            if df is not None:
                print(f"\nFirst few records for {symbol_data['symbol']} (daily):")
                print(df.head().to_string())
            
            time.sleep(1)  # Rate limiting
    
    except KeyboardInterrupt:
        logger.info("\nTest interrupted by user")
    except Exception as e:
        logger.error(f"Test failed: {str(e)}")
    finally:
        fetcher.disconnect()

if __name__ == "__main__":
    main()
