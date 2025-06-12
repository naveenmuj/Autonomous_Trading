import logging
import yaml
import pyotp
import json
import time
from datetime import datetime
from SmartApi import SmartConnect
from src.data.websocket import MarketDataWebSocket

# Set up logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - [%(name)s] - %(levelname)s - %(message)s',
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

def on_tick(tick_data):
    """Callback function for tick data"""
    logger.info(f"Tick data received: {json.dumps(tick_data, default=str)}")

def main():
    try:
        # Load configuration
        config = load_config()
        
        # Initialize SmartAPI
        api = SmartConnect(api_key=config['api_key'])
          # Generate TOTP for authentication
        logger.debug("Generating TOTP...")
        totp_gen = pyotp.TOTP(config['totp_secret'])
        current_totp = totp_gen.now()
        logger.debug(f"TOTP generated: {current_totp}")
        
        # Generate session
        logger.debug("Generating session...")
        session = api.generateSession(
            config['client_id'],
            config['mpin'],
            current_totp
        )
        
        # Get feed token
        feed_token = api.getfeedToken()
        
        # Print connection details
        logger.info(f"Connected successfully. Feed token: {feed_token}")
        
        # Initialize WebSocket
        ws = MarketDataWebSocket(
            auth_token=session['data']['jwtToken'],
            api_key=config['api_key'],
            client_code=config['client_id'],
            feed_token=feed_token,
            config=config
        )
        
        # Add tick callback
        ws.add_tick_callback(on_tick)
        
        # Connect to WebSocket
        ws.connect()
        
        # Subscribe to SUZLON (token: 12018)
        ws.subscribe(['12018'], mode=ws.MODE_QUOTE)
        
        # Keep the script running
        logger.info("Waiting for market data...")
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            logger.info("Closing WebSocket connection...")
            ws.close()
            
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        raise

if __name__ == "__main__":
    main()
