import logging
import yaml
import pyotp
import json
import requests
import time
from datetime import datetime, timedelta
from SmartApi import SmartConnect

# Set up logging
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_credentials():
    """Load and validate Angel One credentials from config file."""
    try:
        with open('config.yaml', 'r') as file:
            config = yaml.safe_load(file)
        
        angel_config = config['apis']['angel_one']
        credentials = {
            'api_key': angel_config['api_key'],
            'client_id': angel_config['client_id'],
            'mpin': angel_config['mpin'],
            'totp_secret': angel_config['totp_secret']
        }
        
        # Validate credentials
        if not all(credentials.values()):
            raise ValueError("Missing required credentials in config.yaml")
        
        return credentials
    except Exception as e:
        logger.error(f"Error loading credentials: {str(e)}")
        raise

def get_token_and_exchange(smart_api, symbol: str, exchange: str = "NSE") -> tuple:
    """Get the token and correct exchange for a symbol."""
    try:
        # First try to get the symbol mapping
        instrument_url = "https://margincalculator.angelbroking.com/OpenAPI_File/files/OpenAPIScripMaster.json"
        response = requests.get(instrument_url)
        data = response.json()
        
        for instrument in data:
            if instrument['symbol'] == symbol and instrument['exch_seg'] == exchange:
                return instrument['token'], instrument['exch_seg'], instrument['symbol']
                
        logger.error(f"Symbol {symbol} not found in instrument list")
        return None, None, None
        
    except Exception as e:
        logger.error(f"Error getting token: {str(e)}")
        return None, None, None

def fetch_stock_data(smart_api, symbol: str):
    """Fetch stock data for a given symbol."""
    try:
        logger.info(f"\nFetching data for {symbol}...")
        
        # Get the correct token and exchange
        token, exchange, trading_symbol = get_token_and_exchange(smart_api, symbol)
        if not token:
            raise ValueError(f"Could not find token for {symbol}")
            
        logger.info(f"Using token: {token}, exchange: {exchange}, trading symbol: {trading_symbol}")
        
        # Fetching real-time LTP data
        try:
            logger.info("Fetching LTP data...")
            ltp_data = smart_api.ltpData(exchange, trading_symbol, token)
            
            if ltp_data and ltp_data.get('data'):
                logger.info("\nLTP Data:")
                logger.info(f"Symbol: {symbol}")
                logger.info(f"LTP: ₹{ltp_data['data'].get('ltp', 'N/A')}")
                logger.info(f"Exchange: {ltp_data['data'].get('exchange', 'N/A')}")
                logger.info(f"Trading Symbol: {ltp_data['data'].get('tradingsymbol', 'N/A')}")
                logger.info(f"Last traded time: {datetime.fromtimestamp(ltp_data['data'].get('tradetime', 0)/1000).strftime('%Y-%m-%d %H:%M:%S') if ltp_data['data'].get('tradetime') else 'N/A'}")
            else:
                logger.warning(f"No LTP data available for {symbol}")
                
        except Exception as e:
            logger.error(f"Error fetching LTP data: {str(e)}")
        
        # Fetching historical data
        try:
            logger.info("\nFetching historical data...")
            
            # Get to_date as today's date at 15:30 IST (market closing time)
            to_date = datetime.now().replace(hour=15, minute=30, second=0, microsecond=0)
            
            # If current time is before market open (9:15), use previous day
            if datetime.now().hour < 9 or (datetime.now().hour == 9 and datetime.now().minute < 15):
                to_date = to_date - timedelta(days=1)
            
            # Get from_date as 5 trading days before to_date
            from_date = to_date - timedelta(days=10)  # Go back 10 calendar days to ensure we get 5 trading days
            from_date = from_date.replace(hour=9, minute=15, second=0, microsecond=0)
            
            historic_params = {
                "exchange": exchange,
                "symboltoken": token,
                "interval": "ONE_DAY",  # Changed to daily candles for more reliable data
                "fromdate": from_date.strftime("%Y-%m-%d %H:%M"),
                "todate": to_date.strftime("%Y-%m-%d %H:%M")
            }
            
            logger.info(f"Requesting data from {from_date.strftime('%Y-%m-%d %H:%M')} to {to_date.strftime('%Y-%m-%d %H:%M')}")
            hist_data = smart_api.getCandleData(historic_params)
            
            if hist_data and isinstance(hist_data.get('data', []), list) and hist_data.get('data'):
                logger.info(f"\nCandle Data (Daily):")
                # Show all candles since we're using daily data
                for candle in hist_data['data']:
                    timestamp, open_price, high, low, close, volume = candle
                    logger.info(f"Date: {timestamp}")
                    logger.info(f"Open: ₹{open_price:.2f}, High: ₹{high:.2f}, Low: ₹{low:.2f}, Close: ₹{close:.2f}")
                    logger.info(f"Volume: {volume:,}")
                    logger.info("-" * 50)
            else:
                logger.warning(f"No historical data available for {symbol}")
                if isinstance(hist_data, dict):
                    logger.warning(f"API Response: {json.dumps(hist_data, indent=2)}")
                
        except Exception as e:
            logger.error(f"Error fetching historical data: {str(e)}")
            if 'hist_data' in locals() and isinstance(hist_data, dict):
                logger.error(f"API Response: {json.dumps(hist_data, indent=2)}")
        
        return True
        
    except Exception as e:
        logger.error(f"Error fetching {symbol} data: {str(e)}")
        return False

def test_angel_one_auth():
    """Test Angel One authentication and profile fetch."""
    logger.info("Testing Angel One Authentication")
    logger.info("=" * 50)
    
    try:
        # Load credentials
        credentials = load_credentials()
        logger.info("\nCredentials loaded from config:")
        logger.info(f"API Key: {credentials['api_key']}")
        logger.info(f"Client ID: {credentials['client_id']}")
        logger.info(f"MPIN: {'*' * len(credentials['mpin'])}")
        logger.info(f"TOTP Secret: {'*' * len(credentials['totp_secret'])}")
        
        # Initialize Smart API
        logger.info("\nInitializing SmartConnect...")
        smart_api = SmartConnect(api_key=credentials['api_key'])
        
        # Generate TOTP
        try:
            totp = pyotp.TOTP(credentials['totp_secret']).now()
            logger.info(f"\nGenerated TOTP: {totp}")
        except Exception as e:
            logger.error("Invalid TOTP secret")
            raise e
        
        # Try to authenticate
        logger.info("\nAttempting to authenticate...")
        data = smart_api.generateSession(credentials['client_id'], credentials['mpin'], totp)
        
        if not data['status']:
            logger.error(f"\n× Authentication failed: {data}")
            return False
        
        logger.info("\n✓ Authentication successful!")
        
        # Extract tokens from response
        auth_token = data['data']['jwtToken']
        refresh_token = data['data']['refreshToken']
        
        logger.info("\nSession details:")
        logger.info(f"JWT Token length: {len(auth_token)}")
        logger.info(f"Refresh Token length: {len(refresh_token)}")
        
        # Get feed token
        logger.info("\nFetching feed token...")
        feed_token = smart_api.getfeedToken()
        logger.info(f"Feed Token: {feed_token}")
        
        # Fetch user profile using refresh token
        logger.info("\nFetching user profile...")
        try:
            profile = smart_api.getProfile(refresh_token)
            if profile['status']:
                profile_data = profile['data']
                logger.info("\nUser Profile:")
                logger.info(f"Name: {profile_data.get('name', 'N/A')}")
                logger.info(f"Email: {profile_data.get('email', 'N/A')}")
                logger.info(f"Broker ID: {profile_data.get('broker_id', 'N/A')}")
                
                # Get allowed exchanges
                exchanges = profile_data.get('exchanges', [])
                logger.info(f"Allowed Exchanges: {', '.join(exchanges)}")
            else:
                logger.warning(f"Profile fetch failed: {profile.get('message', 'Unknown error')}")
        except Exception as e:
            logger.error(f"Error fetching profile: {str(e)}")
        
        # Fetch SUZLON data
        fetch_stock_data(smart_api, "SUZLON-EQ")
        
        # Generate new token
        logger.info("\nGenerating new token...")
        try:
            token_result = smart_api.generateToken(refresh_token)
            logger.info(f"Token generation result: {token_result}")
        except Exception as e:
            logger.error(f"Error generating new token: {str(e)}")
        
        # Test session termination
        logger.info("\nTesting logout...")
        try:
            logout = smart_api.terminateSession(credentials['client_id'])
            logger.info("Logout successful")
        except Exception as e:
            logger.error(f"Logout failed: {str(e)}")
        
        return True
        
    except Exception as e:
        logger.error(f"\n× Error during authentication test: {str(e)}")
        return False

if __name__ == "__main__":
    success = test_angel_one_auth()
    if success:
        logger.info("\nAuthentication test completed successfully")
    else:
        logger.error("\nAuthentication test failed")
