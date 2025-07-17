import logging
import yaml
import pyotp
import json
import time
from datetime import datetime, timedelta
from SmartApi import SmartConnect
from SmartApi.smartWebSocketV2 import SmartWebSocketV2

# Set up logging with debug level
logging.basicConfig(
    level=logging.DEBUG,  # Changed to DEBUG to see more details
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

class SmartWebSocket:
    EXCHANGE_TYPE_MAP = {
        'NSE': 1,
        'BSE': 2,
        'NFO': 3,
        'BFO': 4,
        'MCX': 5,
        'NCDEX': 7
    }

    MODE_LTP = 1           # Limited tick data
    MODE_QUOTE = 2        # Market depth data
    MODE_SNAPQUOTE = 3    # Market depth and other details

    def __init__(self):
        self.client_code = None
        self.feed_token = None
        self.api_key = None
        self.auth_token = None
        self.ws = None
        self.subscribed_tokens = {}
        self.is_connected = False
        self.last_price = None
        self.stopping = False
        self.live_feed = {}  # Store live market data
        
        # Clear the terminal
        print("\033[2J\033[H", end="")

    def on_data(self, wsapp, message):
        """Process incoming market data"""
        try:
            # Log raw message for debugging
            logger.debug(f"Received message: {message}")
            
            # Convert to dict if it's a string
            data = message if isinstance(message, dict) else json.loads(message)
            
            # Handle market data
            if isinstance(data, dict) and 'token' in data:
                token = str(data.get('token', ''))
                
                # Store tick data with proper conversions
                tick_data = {
                    'token': token,
                    'ltp': float(data.get('last_traded_price', 0)) / 100,  # Convert price
                    'volume': int(data.get('volume_trade_for_the_day', 0)),  # Total traded volume
                    'bid_price': float(data.get('best_5_buy_data', [{}])[0].get('price', 0)) / 100,  # Best bid price
                    'ask_price': float(data.get('best_5_sell_data', [{}])[0].get('price', 0)) / 100,  # Best ask price
                    'oi': int(data.get('open_interest', 0)),  # Open Interest
                    'timestamp': datetime.fromtimestamp(int(data.get('exchange_timestamp', time.time())) / 1000)  # Convert timestamp
                }
                
                # Store in live feed
                self.live_feed[token] = tick_data
                
                # Print formatted output for SUZLON
                if token == "12018":  # SUZLON token
                    # Calculate price change indicator
                    price_indicator = "⟿"  # Default no change
                    if self.last_price is not None:
                        if tick_data['ltp'] > self.last_price:
                            price_indicator = "↑"  # Price increased
                        elif tick_data['ltp'] < self.last_price:
                            price_indicator = "↓"  # Price decreased
                    self.last_price = tick_data['ltp']
                    
                    # Format and print the output
                    print("\033[H\033[2J", end="")  # Clear screen
                    print("\n" + "═" * 50)
                    print(f"  SUZLON ENERGY LIVE MARKET DATA")
                    print("═" * 50)
                    print(f"  Time: {tick_data['timestamp'].strftime('%H:%M:%S')}")
                    print(f"  LTP: {price_indicator} ₹{tick_data['ltp']:,.2f}")
                    print(f"  Bid/Ask: ₹{tick_data['bid_price']:,.2f} / ₹{tick_data['ask_price']:,.2f}")
                    print(f"  Volume: {tick_data['volume']:,}")
                    if tick_data['oi'] > 0:
                        print(f"  Open Interest: {tick_data['oi']:,}")
                    print("═" * 50)
                    print("\nPress Ctrl+C to exit")
                    
        except Exception as e:
            logger.error(f"Error processing message: {str(e)}")
            logger.debug(f"Raw message: {message}")

    def subscribe(self, tokens):
        """Subscribe to market tokens"""
        try:
            if not self.ws:
                logger.error("WebSocket not connected")
                return

            # Store tokens for reconnection
            self.subscribed_tokens = tokens

            # Format tokens for SmartWebSocketV2
            token_list = []
            for exchange, exchange_tokens in tokens.items():
                exchange_type = self.EXCHANGE_TYPE_MAP.get(exchange)
                if not exchange_type:
                    logger.error(f"Unknown exchange type: {exchange}")
                    continue
                
                # Format: [{"exchangeType": 1, "tokens": ["12018"]}]
                token_list = [{
                    "exchangeType": exchange_type,
                    "tokens": [str(token) for token in exchange_tokens]
                }]

            if token_list:
                try:
                    # Log the subscription request
                    logger.info(f"Subscribing to tokens: {token_list}")
                    
                    action = 1  # 1 for subscribe, 0 for unsubscribe
                    mode = self.MODE_SNAPQUOTE
                    correlation_id = "websocket_" + str(int(time.time()))
                      # Subscribe using the official method
                    self.ws.subscribe(
                        correlation_id=correlation_id, 
                        mode=mode, 
                        token_list=token_list
                    )
                    
                    logger.info("Subscription request sent successfully")
                except Exception as e:
                    logger.error(f"Error in subscribe call: {str(e)}")
                    logger.debug(f"Token list at error: {token_list}")

        except Exception as e:
            logger.error(f"Error preparing subscription: {str(e)}")
            logger.debug(f"Current tokens: {tokens}")

    def connect(self, client_code, feed_token, api_key, auth_token):
        """Connect to Angel One WebSocket"""
        try:
            self.client_code = client_code
            self.feed_token = feed_token
            self.api_key = api_key
            self.auth_token = auth_token
            self.stopping = False
            
            # Initialize SmartWebSocketV2 with retry parameters
            self.ws = SmartWebSocketV2(
                auth_token=auth_token,
                api_key=api_key,
                client_code=client_code,
                feed_token=feed_token,
                max_retry_attempt=3,
                retry_strategy=1,  # 1 for exponential backoff
                retry_delay=10,
                retry_multiplier=2,
                retry_duration=30
            )

            # Assign callbacks
            self.ws.on_open = self.on_open
            self.ws.on_data = self.on_data
            self.ws.on_error = self.on_error
            self.ws.on_close = self.on_close
            self.ws.on_control_message = self.on_control_message

            # Connect to WebSocket
            self.ws.connect()

        except Exception as e:
            logger.error(f"Error connecting to WebSocket: {str(e)}")
            raise

    def on_open(self, wsapp):
        """Handle WebSocket connection open"""
        logger.info("\nWebSocket Connection Opened")
        self.is_connected = True
        
        # Automatically subscribe after connection
        self.subscribe({"NSE": ["12018"]})  # SUZLON-EQ

    def on_error(self, ws, error):
        """Handle WebSocket errors"""
        logger.error(f"WebSocket error: {str(error)}")
        
        if not self.stopping and self.ws:
            logger.info("Connection lost, attempting to reconnect...")
            self.ws = None
            self.is_connected = False
            self.connect(self.client_code, self.feed_token, self.api_key, self.auth_token)

    def on_close(self, ws, close_status_code, close_msg):
        """Handle WebSocket connection close"""
        logger.info(f"WebSocket connection closed: {close_status_code} - {close_msg}")
        self.ws = None
        self.is_connected = False
        
        if not self.stopping:
            logger.info("Connection closed unexpectedly, attempting to reconnect...")
            self.connect(self.client_code, self.feed_token, self.api_key, self.auth_token)

    def on_control_message(self, wsapp, message):
        """Handle WebSocket control messages"""
        try:
            logger.info(f"Control Message: {message}")
            data = json.loads(message)
            
            # Handle subscription responses
            if data.get('type') == 'success' and data.get('message') == 'subscribed':
                logger.info("Successfully subscribed to market data")
            elif data.get('type') == 'error':
                logger.error(f"WebSocket error message: {data}")
                
                # Attempt resubscribe on certain errors
                if "not subscribed" in str(data.get('message', '')).lower():
                    logger.info("Attempting to resubscribe...")
                    self.subscribe({"NSE": ["12018"]})
        except Exception as e:
            logger.error(f"Error processing control message: {str(e)}")
            logger.debug(f"Raw message: {message}")

    def close(self):
        """Close WebSocket connection"""
        self.stopping = True
        if self.ws:
            self.ws.close()

def test_websocket():
    """Test WebSocket connection and real-time data"""
    try:
        logger.info("Testing Angel One WebSocket")
        logger.info("=" * 50)

        # Load credentials
        creds = load_config()
        api_key = creds['api_key']
        client_id = creds['client_id']
        mpin = creds['mpin']
        totp_secret = creds['totp_secret']

        logger.info("\nCredentials loaded from config:")
        logger.info(f"API Key: {api_key}")
        logger.info(f"Client ID: {client_id}")
        logger.info(f"MPIN: {'*' * len(mpin)}")
        logger.info(f"TOTP Secret: {'*' * len(totp_secret)}")

        # Generate TOTP
        totp = pyotp.TOTP(totp_secret)
        totp_value = totp.now()
        logger.info(f"\nGenerated TOTP: {totp_value}")

        # Initialize Smart Connect
        logger.info("\nInitializing SmartConnect...")
        smart_api = SmartConnect(api_key=api_key)

        # Login
        logger.info("\nAttempting to authenticate...")
        login = smart_api.generateSession(client_id, mpin, totp_value)
        
        if login.get('status'):
            auth_token = login['data']['jwtToken']
            refresh_token = login['data']['refreshToken']
            feed_token = smart_api.getfeedToken()

            logger.info("\n✓ Authentication successful!")
            logger.info("\nSession details:")
            logger.info(f"JWT Token length: {len(auth_token)}")
            logger.info(f"Refresh Token length: {len(refresh_token)}")
            logger.info(f"\nFetching feed token...")
            logger.info(f"Feed Token: {feed_token}")

            # Initialize and connect WebSocket
            logger.info("\nInitializing WebSocket...")
            smart_socket = SmartWebSocket()
            smart_socket.connect(client_id, feed_token, api_key, auth_token)
            
            # Wait for connection to establish
            time.sleep(1)
              # Subscribe to SUZLON-EQ (NSE: 12018)
            time.sleep(1)  # Wait for connection to be ready
            logger.info("Subscribing to SUZLON...")
            smart_socket.subscribe({"NSE": ["12018"]})
            
            # Keep the script running
            try:
                while True:
                    time.sleep(1)
            except KeyboardInterrupt:
                logger.info("\nStopping WebSocket connection...")
                smart_socket.close()

        else:
            logger.error(f"Login failed: {login.get('message')}")

    except KeyboardInterrupt:
        logger.info("\nStopping WebSocket connection...")
        if 'smart_socket' in locals():
            smart_socket.close()
        
    except Exception as e:
        logger.error(f"Error in WebSocket test: {str(e)}")
        raise

    finally:
        # Cleanup
        try:
            smart_api.terminateSession(client_id)
            logger.info("Session terminated")
        except:
            pass

if __name__ == "__main__":
    test_websocket()
