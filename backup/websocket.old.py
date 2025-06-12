import json
import logging
import threading
import time
from typing import Dict, List, Optional, Any
from datetime import datetime
from SmartApi.smartWebSocketV2 import SmartWebSocketV2

logger = logging.getLogger(__name__)

class MarketDataWebSocket:
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

    def __init__(self, auth_token: str, api_key: str, client_code: str, feed_token: str, config: Dict[str, Any]):
        """Initialize WebSocket connection with Angel One"""
        self.client_code = client_code
        self.feed_token = feed_token
        self.api_key = api_key
        self.auth_token = auth_token
        self.config = config
        self.ws = None
        self.subscribed_tokens = {}  # Map of token -> symbol
        self.live_feed = {}  # Store live market data per token
        self.is_connected = False
        self.on_tick_callbacks = []  # List of callback functions to be called on each tick
        self.last_price = None
        
        # Log initialization
        logger.info("Initializing WebSocket connection")
    def _on_message(self, *args) -> None:
        """Handle incoming WebSocket messages"""
        try:
            # Extract message from args
            message = args[1] if len(args) > 1 else None
            if not message:
                return
                  # Handle different message types
            if isinstance(message, dict):
                data = message
            elif isinstance(message, str):
                data = json.loads(message)
            elif isinstance(message, bytes):
                data = json.loads(message.decode('utf-8'))
            else:
                logger.warning(f"Unknown message type: {type(message)}")
                return
                
            # Handle market data messages
            if isinstance(data, dict) and 'token' in data:
                token = str(data.get('token', ''))
                
                # Format tick data with proper conversions
                tick_data = {
                    'token': token,
                    'symbol': self.subscribed_tokens.get(token, 'UNKNOWN'),
                    'ltp': float(data.get('last_traded_price', 0)) / 100,
                    'volume': int(data.get('volume_trade_for_the_day', 0)),
                    'bid_price': float(data.get('best_5_buy_data', [{}])[0].get('price', 0)) / 100,
                    'ask_price': float(data.get('best_5_sell_data', [{}])[0].get('price', 0)) / 100,
                    'oi': int(data.get('open_interest', 0)),
                    'timestamp': datetime.fromtimestamp(int(data.get('exchange_timestamp', time.time())) / 1000)
                }
                
                # Store in live feed
                self.live_feed[token] = tick_data
                
                # Call registered callbacks
                for callback in self.on_tick_callbacks:
                    try:
                        callback(tick_data)
                    except Exception as e:
                        logger.error(f"Error in tick callback: {str(e)}")
                
        except Exception as e:
            logger.error(f"Error processing message: {str(e)}")
            logger.debug(f"Raw message: {message}")
    def _on_open(self, *args) -> None:
        """Handle WebSocket connection open"""
        logger.info("WebSocket connection established")
        self.is_connected = True

    def _on_error(self, *args) -> None:
        """Handle WebSocket errors"""
        error = args[1] if len(args) > 1 else "Unknown error"
        logger.error(f"WebSocket error: {str(error)}")
        self.is_connected = False    def _on_close(self, *args) -> None:
        """Handle WebSocket connection close"""
        status_code = args[1] if len(args) > 1 else None
        close_msg = args[2] if len(args) > 2 else None
        logger.info(f"WebSocket connection closed. Status: {status_code}, Message: {close_msg}")
        self.is_connected = False
        
    def _on_control_message(self, wsapp, message_type, message) -> None:
        """Handle WebSocket control messages"""
        logger.debug(f"Control message received - Type: {message_type}, Message: {message}")
        
    def subscribe(self, tokens: List[str], mode: int = None) -> None:
        """Subscribe to market data for given tokens"""
        try:
            if not self.is_connected or not self.ws:
                logger.warning("WebSocket not connected, cannot subscribe")
                return

            if not mode:
                mode = self.config.get('data', {}).get('websocket', {}).get('mode', self.MODE_SNAPQUOTE)

            # Format tokens for SmartWebSocketV2
            token_list = [{
                "exchangeType": self.EXCHANGE_TYPE_MAP['NSE'],
                "tokens": [str(token) for token in tokens]
            }]

            # Generate correlation ID
            correlation_id = "websocket_" + str(int(time.time()))

            # Log the subscription request
            logger.info(f"Subscribing to tokens: {token_list}")

            # Subscribe using the official method
            logger.debug(f"Sending subscription request - correlation_id: {correlation_id}, mode: {mode}, tokens: {tokens}")
            self.ws.subscribe(
                correlation_id=correlation_id,
                mode=mode,
                token_list=token_list
            )
            logger.info("Subscription request sent successfully")

            # Store subscribed tokens
            for token in tokens:
                self.subscribed_tokens[str(token)] = str(token)

        except Exception as e:
            logger.error(f"Error in subscription: {str(e)}")
            logger.debug(f"Token list at error: {token_list}")
            raise
        close_msg = args[2] if len(args) > 2 else None
        logger.info(f"WebSocket connection closed. Status: {status_code}, Message: {close_msg}")
        self.is_connected = False
    
    def add_tick_callback(self, callback):
        """Add a callback function to be called for each tick"""
        if callback not in self.on_tick_callbacks:
            self.on_tick_callbacks.append(callback)
            
    def remove_tick_callback(self, callback):
        """Remove a callback function"""
        if callback in self.on_tick_callbacks:
            self.on_tick_callbacks.remove(callback)
            
    def connect(self) -> None:
        """Connect to Angel One WebSocket"""
        try:
            if self.ws:
                logger.warning("WebSocket already exists, closing first")
                self.close()
                  # Create new SmartWebSocketV2 instance with retry parameters
            self.ws = SmartWebSocketV2(
                auth_token=self.auth_token,
                api_key=self.api_key,
                client_code=self.client_code,
                feed_token=self.feed_token,
                max_retry_attempt=3,
                retry_strategy=1,  # 1 for exponential backoff
                retry_delay=10,
                retry_multiplier=2,
                retry_duration=30
            )
            
            # Set callbacks
            self.ws.on_open = self._on_open
            self.ws.on_error = self._on_error
            self.ws.on_close = self._on_close
            self.ws.on_message = self._on_message
            
            # Start WebSocket connection in a separate thread
            self.ws_thread = threading.Thread(target=self.ws.connect)
            self.ws_thread.daemon = True  # Thread will close when main program exits
            self.ws_thread.start()
            
            # Wait for connection to establish
            timeout = 5
            start_time = time.time()
            while not self.is_connected and time.time() - start_time < timeout:
                time.sleep(0.1)
                
            if not self.is_connected:
                raise Exception("WebSocket connection timed out")
                
            logger.info("WebSocket connection thread started")
            
        except Exception as e:
            logger.error(f"Error connecting to WebSocket: {str(e)}")
            raise    def subscribe(self, tokens: List[str], mode: int = None) -> None:
        """Subscribe to market data for given tokens"""
        try:
            if not self.is_connected or not self.ws:
                logger.warning("WebSocket not connected, cannot subscribe")
                return

            if not mode:
                mode = self.config.get('data', {}).get('websocket', {}).get('mode', self.MODE_SNAPQUOTE)

            # Format tokens for SmartWebSocketV2
            token_list = [{
                "exchangeType": self.EXCHANGE_TYPE_MAP['NSE'],
                "tokens": [str(token) for token in tokens]
            }]

            # Generate correlation ID
            correlation_id = "websocket_" + str(int(time.time()))

            # Log the subscription request
            logger.info(f"Subscribing to tokens: {token_list}")

            # Subscribe using the official method
            logger.debug(f"Sending subscription request - correlation_id: {correlation_id}, mode: {mode}, tokens: {tokens}")
            self.ws.subscribe(
                correlation_id=correlation_id,
                mode=mode,
                token_list=token_list
            )
            logger.info("Subscription request sent successfully")

            # Store subscribed tokens
            for token in tokens:
                self.subscribed_tokens[str(token)] = str(token)

        except Exception as e:
            logger.error(f"Error in subscription: {str(e)}")
            logger.debug(f"Token list at error: {token_list}")
            raise
            
    def unsubscribe(self, tokens: List[str]) -> None:
        """Unsubscribe from market data for given tokens"""
        if not self.is_connected:
            logger.warning("WebSocket not connected, cannot unsubscribe")
            return
            
        try:
            # Remove from subscribed tokens
            for token in tokens:
                self.subscribed_tokens.pop(str(token), None)
                
            # Unsubscribe using SmartAPI WebSocket
            self.ws.unsubscribe(tokens)
            logger.info(f"Unsubscribed from {len(tokens)} tokens")
            
        except Exception as e:
            logger.error(f"Error unsubscribing from tokens: {str(e)}")
            
    def close(self) -> None:
        """Close the WebSocket connection"""
        try:
            if self.ws:
                self.ws.close()
                self.ws = None
            self.is_connected = False
            self.subscribed_tokens.clear()
            self.live_feed.clear()
            logger.info("WebSocket connection closed")
        except Exception as e:
            logger.error(f"Error closing WebSocket: {str(e)}")
