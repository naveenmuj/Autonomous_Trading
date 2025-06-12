import logging
import yaml
import pyotp
import json
import time
import pandas as pd
from datetime import datetime, timedelta
from SmartApi import SmartConnect

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
    
    def __init__(self):
        """Initialize the tester"""
        self.smart_api = None
        self.last_request_time = 0
        self.rate_limit_delay = 1  # seconds between requests

    def _wait_for_rate_limit(self):
        """Implement rate limiting"""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        if time_since_last < self.rate_limit_delay:
            time.sleep(self.rate_limit_delay - time_since_last)
        self.last_request_time = time.time()

    def authenticate(self):
        """Authenticate with Angel One"""
        try:
            # Load credentials
            creds = load_config()
            self.api_key = creds['api_key']
            self.client_id = creds['client_id']
            self.mpin = creds['mpin']
            totp_secret = creds['totp_secret']

            # Generate TOTP
            totp = pyotp.TOTP(totp_secret)
            totp_value = totp.now()

            # Initialize Smart Connect
            self.smart_api = SmartConnect(api_key=self.api_key)

            # Login
            logger.info("Attempting to authenticate...")
            login = self.smart_api.generateSession(self.client_id, self.mpin, totp_value)

            if login.get('status'):
                self.feed_token = login['data']['feedToken']
                logger.info("Authentication successful!")
                return True
            else:
                logger.error(f"Login failed: {login.get('message')}")
                return False

        except Exception as e:
            logger.error(f"Error in authentication: {str(e)}")
            return False

    def fetch_historical_data(self, symbol_token, from_date, to_date, interval="ONE_MINUTE"):
        """
        Fetch historical data using SmartAPI's getCandleData method
        
        Parameters:
        - symbol_token: Trading symbol token (e.g., '12018' for SUZLON)
        - from_date: Start date (format: 'YYYY-MM-DD HH:mm')
        - to_date: End date (format: 'YYYY-MM-DD HH:mm')
        - interval: ONE_MINUTE/FIVE_MINUTE/FIFTEEN_MINUTE/THIRTY_MINUTE/ONE_HOUR/ONE_DAY
        """
        try:
            self._wait_for_rate_limit()

            logger.info(f"Fetching {interval} data for token {symbol_token} from {from_date} to {to_date}")

            # Prepare parameters for historical data
            params = {
                "exchange": "NSE",
                "symboltoken": symbol_token,
                "interval": interval,
                "fromdate": from_date,
                "todate": to_date
            }

            # Fetch historical data using the official method
            response = self.smart_api.getCandleData(params)

            if response.get('status'):
                # Convert to DataFrame for better analysis
                data = pd.DataFrame(response['data'],
                                  columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                data['timestamp'] = pd.to_datetime(data['timestamp'])
                
                # Print statistics
                logger.info("\nData Summary:")
                logger.info(f"Total Records: {len(data)}")
                logger.info(f"Date Range: {data['timestamp'].min()} to {data['timestamp'].max()}")
                logger.info(f"Average Volume: {data['volume'].mean():,.0f}")
                logger.info(f"Price Range: ₹{data['low'].min():.2f} - ₹{data['high'].max():.2f}")
                
                return data
            else:
                logger.error(f"Error in response: {response.get('message')}")
                return None

        except Exception as e:
            logger.error(f"Error fetching historical data: {str(e)}")
            return None

    def cleanup(self):
        """Cleanup and terminate session"""
        if self.smart_api:
            try:
                self.smart_api.terminateSession(self.client_id)
                logger.info("Session terminated")
            except Exception as e:
                logger.error(f"Error during cleanup: {str(e)}")

def test_historical_data():
    """Test historical data retrieval"""
    try:
        # Initialize tester
        tester = HistoricalDataTester()
        
        # Authenticate
        if not tester.authenticate():
            logger.error("Failed to authenticate. Exiting...")
            return

        # Test parameters
        symbol_token = "12018"  # SUZLON token
        now = datetime.now()
        end_date = now.strftime('%Y-%m-%d 15:30')  # Market closing time
        start_date = (now - timedelta(days=5)).strftime('%Y-%m-%d 09:15')  # Market opening time

        # Test different intervals
        intervals = ["ONE_MINUTE", "FIVE_MINUTE", "FIFTEEN_MINUTE", "ONE_HOUR", "ONE_DAY"]

        for interval in intervals:
            logger.info(f"\nTesting {interval} interval...")
            df = tester.fetch_historical_data(symbol_token, start_date, end_date, interval)

            if df is not None:
                # Display basic analysis
                logger.info("\nFirst few records:")
                print(df.head().to_string())
                
                if len(df) > 0:
                    # Calculate basic indicators
                    logger.info("\nBasic Technical Indicators:")
                    df['SMA_20'] = df['close'].rolling(window=20).mean()
                    df['Daily_Return'] = df['close'].pct_change()
                    
                    logger.info(f"Latest SMA(20): ₹{df['SMA_20'].iloc[-1]:.2f}")
                    logger.info(f"Average Daily Return: {df['Daily_Return'].mean()*100:.2f}%")
                    logger.info(f"Return Volatility: {df['Daily_Return'].std()*100:.2f}%")
            else:
                logger.error(f"Failed to fetch {interval} data")

            # Rate limiting
            time.sleep(1)

    except Exception as e:
        logger.error(f"Error in test: {str(e)}")
    finally:
        # Cleanup
        if 'tester' in locals():
            tester.cleanup()

if __name__ == "__main__":
    test_historical_data()
