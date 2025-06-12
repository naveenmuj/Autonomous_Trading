import json
import logging
import threading
import time
from typing import Dict, List, Optional, Any
from datetime import datetime
from SmartApi.smartWebSocketV2 import SmartWebSocketV2
from .market_utils import is_market_open, format_time_until_market_open

logger = logging.getLogger(__name__)

class MarketDataWebSocket:
    # Exchange type mapping
    EXCHANGE_TYPE_MAP = {
        'NSE': 1,
        'BSE': 2,
        'NFO': 3,
        'BFO': 4,
        'MCX': 5,
        'NCDEX': 7
    }

    # Mode constants
    MODE_LTP = 1        # Limited tick data
    MODE_QUOTE = 2      # Market depth data
    MODE_SNAPQUOTE = 3  # Market depth and other details

    def __init__(self, auth_token: str, api_key: str, client_code: str, feed_token: str, config: dict = None):
        """Initialize WebSocket connection parameters"""
        # Existing parameters
        self.auth_token = auth_token
        self.api_key = api_key
        self.client_code = client_code
        self.feed_token = feed_token
        self.config = config or {}
        
        # Connection state tracking
        self.ws = None
        self.is_connected = False
        self.is_connecting = False
        self.last_heartbeat = None
        self.heartbeat_interval = 25  # seconds
        self.heartbeat_timeout = 30   # seconds
        self.max_reconnect_attempts = 5
        self.reconnect_delay = 5      # seconds
        self.current_reconnect_attempt = 0
        
        # Data storage
        self.subscribed_tokens = {}
        self.live_feed = {}
        self.on_tick_callbacks = []
        self.stopping = False
        
        # Connection lock for thread safety
        self._connection_lock = threading.Lock()
        
        logger.info("Initializing WebSocket connection")

    def _on_message(self, wsapp, message):
        """Handle incoming WebSocket messages with improved error handling"""
        try:
            # Convert message to dict if it's a string
            if isinstance(message, str):
                data = json.loads(message)
            elif isinstance(message, dict):
                data = message
            elif isinstance(message, bytes):
                data = json.loads(message.decode('utf-8'))
            else:
                logger.warning(f"Unknown message type: {type(message)}")
                return

            # Update last heartbeat time for any valid message
            self.last_heartbeat = time.time()

            # Handle market data messages
            if isinstance(data, dict):
                if 'type' in data:  # Control message
                    self._handle_control_message(data)
                elif 'token' in data:  # Market data message
                    self._handle_market_data(data)
                else:
                    logger.debug(f"Unhandled message format: {data}")
                    
        except json.JSONDecodeError as e:
            logger.warning(f"Invalid JSON message received: {str(e)}")
        except Exception as e:
            logger.error(f"Error processing message: {str(e)}")
            logger.debug(f"Raw message: {message}")

    def _handle_market_data(self, data):
        """Process market data messages"""
        try:
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
            logger.error(f"Error processing market data: {str(e)}")

    def _handle_control_message(self, data):
        """Process WebSocket control messages"""
        try:
            msg_type = data.get('type')
            message = data.get('message', '')
            
            if msg_type == 'success':
                if message == 'subscribed':
                    logger.info("Successfully subscribed to market data")
            elif msg_type == 'error':
                logger.error(f"WebSocket error message: {message}")
                
                # Handle specific error cases
                if "not subscribed" in message.lower():
                    logger.info("Attempting to resubscribe...")
                    if self.subscribed_tokens:
                        tokens_dict = {"NSE": list(self.subscribed_tokens.keys())}
                        self.subscribe(tokens_dict)
                elif "authentication" in message.lower():
                    logger.error("Authentication error, need to reconnect with new credentials")
                    self.disconnect()
                    # Signal authentication error to client
                    raise Exception("WebSocket authentication failed")
                    
        except Exception as e:
            logger.error(f"Error processing control message: {str(e)}")

    def _on_open(self, wsapp):
        """Handle WebSocket connection open"""
        with self._connection_lock:
            logger.info("WebSocket connection established")
            self.is_connected = True
            self.last_heartbeat = time.time()
            
            # Resubscribe to tokens if any were previously subscribed
            if self.subscribed_tokens:
                logger.info("Resubscribing to previous tokens...")
                tokens_dict = {"NSE": list(self.subscribed_tokens.keys())}
                self.subscribe(tokens_dict)

    def _on_error(self, wsapp, error):
        """Handle WebSocket errors"""
        logger.error(f"WebSocket error: {str(error)}")
        
        with self._connection_lock:
            was_connected = self.is_connected
            self.is_connected = False
            
            if not self.stopping and was_connected:
                logger.info("Connection lost, initiating reconnection...")
                # Use a separate thread for reconnection to avoid blocking
                threading.Thread(target=self.connect, daemon=True).start()

    def _on_close(self, wsapp, close_status_code, close_msg):
        """Handle WebSocket connection close"""
        logger.info(f"WebSocket connection closed. Status: {close_status_code}, Message: {close_msg}")
        
        with self._connection_lock:
            was_connected = self.is_connected
            self.is_connected = False
            
            if not self.stopping and was_connected:
                if close_status_code in [1000, 1001]:  # Normal closure
                    logger.info("WebSocket closed normally")
                else:
                    logger.warning(f"WebSocket closed unexpectedly (code: {close_status_code})")
                    # Use a separate thread for reconnection to avoid blocking
                    threading.Thread(target=self.connect, daemon=True).start()

    def connect(self) -> None:
        """Connect to Angel One WebSocket with improved error handling and heartbeat"""
        with self._connection_lock:
            if self.is_connecting:
                logger.warning("Connection attempt already in progress")
                return
                
            if self.is_connected:
                logger.warning("Already connected to WebSocket")
                return
                
            self.is_connecting = True
            
        try:
            # Initialize SmartWebSocketV2 with proper headers and configuration
            self.ws = SmartWebSocketV2(
                auth_token=self.auth_token,
                api_key=self.api_key,
                client_code=self.client_code,
                feed_token=self.feed_token,
                max_retry_attempt=self.max_reconnect_attempts,
                retry_strategy=1,  # Progressive delay
                retry_delay=self.reconnect_delay,
                retry_multiplier=2,
                retry_duration=30
            )            # Register callbacks - Control messages are handled in _on_message
            self.ws.on_open = self._on_open
            self.ws.on_data = self._on_message
            self.ws.on_error = self._on_error
            self.ws.on_close = self._on_close
            
            # Connect to WebSocket
            logger.info("Starting WebSocket connection...")
            self.ws.connect()

            # Wait for connection to establish with timeout
            max_wait = 10
            start_time = time.time()
            while not self.is_connected and time.time() - start_time < max_wait:
                time.sleep(0.1)

            if not self.is_connected:
                raise Exception("WebSocket connection failed to establish within timeout")

            # Reset reconnection counter on successful connection
            self.current_reconnect_attempt = 0
            
            # Start heartbeat monitoring
            self._start_heartbeat_monitor()
            
        except Exception as e:
            logger.error(f"Connection error: {str(e)}")
            self.is_connected = False
            self.current_reconnect_attempt += 1
            
            # Implement exponential backoff for reconnection
            if self.current_reconnect_attempt < self.max_reconnect_attempts:
                delay = self.reconnect_delay * (2 ** (self.current_reconnect_attempt - 1))
                logger.info(f"Retrying connection in {delay} seconds...")
                time.sleep(delay)
                self.connect()
            else:
                logger.error("Max reconnection attempts reached")
                
        finally:
            self.is_connecting = False

    def _start_heartbeat_monitor(self):
        """Start the heartbeat monitoring thread"""
        def heartbeat_monitor():
            while not self.stopping and self.is_connected:
                try:
                    # Send heartbeat
                    if hasattr(self.ws, '_ws_app') and self.ws._ws_app:
                        self.ws._ws_app.send('ping')
                        
                        # Wait for pong response with timeout
                        start_time = time.time()
                        while time.time() - start_time < 5:  # 5 second timeout for pong
                            try:
                                response = self.ws._ws_app.recv()
                                if response == 'pong':
                                    self.last_heartbeat = time.time()
                                    break
                            except Exception:
                                time.sleep(0.1)
                                continue
                        
                        # Check if heartbeat was successful
                        if not self.last_heartbeat or time.time() - self.last_heartbeat > self.heartbeat_timeout:
                            logger.warning("Heartbeat timeout, initiating reconnection...")
                            self.is_connected = False
                            self.connect()
                            break
                            
                    time.sleep(self.heartbeat_interval)
                    
                except Exception as e:
                    logger.error(f"Heartbeat error: {str(e)}")
                    if not self.stopping:
                        self.is_connected = False
                        self.connect()
                    break

        # Start heartbeat monitor in a daemon thread
        heartbeat_thread = threading.Thread(target=heartbeat_monitor, daemon=True)
        heartbeat_thread.start()

    def subscribe(self, tokens_dict: Dict[str, List[str]], mode: int = None) -> None:
        """Subscribe to market data for given tokens"""
        try:
            if not self.is_connected or not self.ws:
                logger.warning("WebSocket not connected, cannot subscribe")
                return

            if not mode:
                mode = self.config.get('data', {}).get('websocket', {}).get('mode', self.MODE_SNAPQUOTE)

            # Format tokens for SmartWebSocketV2
            token_list = []
            for exchange, exchange_tokens in tokens_dict.items():
                exchange_type = self.EXCHANGE_TYPE_MAP.get(exchange)
                if not exchange_type:
                    logger.error(f"Unknown exchange type: {exchange}")
                    continue
                
                token_list.append({
                    "exchangeType": exchange_type,
                    "tokens": [str(token) for token in exchange_tokens]
                })

            if token_list:
                # Generate correlation ID
                correlation_id = "websocket_" + str(int(time.time()))
                
                # Log the subscription request
                logger.debug(f"Sending subscription request - correlation_id: {correlation_id}, mode: {mode}, token_list: {token_list}")

                # Subscribe using the official method
                self.ws.subscribe(
                    correlation_id=correlation_id,
                    mode=mode,
                    token_list=token_list
                )
                
                # Store subscribed tokens
                for exchange_tokens in tokens_dict.values():
                    for token in exchange_tokens:
                        self.subscribed_tokens[str(token)] = str(token)

                logger.info("Subscription request sent successfully")

        except Exception as e:
            logger.error(f"Error in subscription: {str(e)}")
            logger.debug(f"Token list at error: {token_list}")
            raise

    def unsubscribe(self, tokens_dict: Dict[str, List[str]]) -> None:
        """Unsubscribe from market data for given tokens"""
        if not self.is_connected:
            logger.warning("WebSocket not connected, cannot unsubscribe")
            return
            
        try:
            # Format tokens for SmartWebSocketV2
            token_list = []
            for exchange, exchange_tokens in tokens_dict.items():
                exchange_type = self.EXCHANGE_TYPE_MAP.get(exchange)
                if not exchange_type:
                    logger.error(f"Unknown exchange type: {exchange}")
                    continue
                
                token_list.append({
                    "exchangeType": exchange_type,
                    "tokens": [str(token) for token in exchange_tokens]
                })

            if token_list:
                # Generate correlation ID
                correlation_id = "unsub_" + str(int(time.time()))
                
                # Get default mode or use QUOTE mode
                mode = self.config.get('data', {}).get('websocket', {}).get('mode', self.MODE_QUOTE)
                
                # Unsubscribe using SmartAPI WebSocket
                self.ws.unsubscribe(
                    correlation_id=correlation_id,
                    mode=mode,
                    token_list=token_list
                )
                
                # Remove from subscribed tokens
                for exchange_tokens in tokens_dict.values():
                    for token in exchange_tokens:
                        self.subscribed_tokens.pop(str(token), None)
                        self.live_feed.pop(str(token), None)

                logger.info(f"Unsubscribed from {sum(len(tokens) for tokens in tokens_dict.values())} tokens")
            
        except Exception as e:
            logger.error(f"Error unsubscribing from tokens: {str(e)}")
            raise

    def close(self) -> None:
        """Close the WebSocket connection"""
        try:
            self.stopping = True
            
            # First unsubscribe from all tokens if connected
            if self.is_connected and self.subscribed_tokens:
                try:
                    tokens_dict = {"NSE": list(self.subscribed_tokens.keys())}
                    self.unsubscribe(tokens_dict)
                except Exception as e:
                    logger.warning(f"Error during unsubscribe: {str(e)}")

            # Close WebSocket connection if exists
            if self.ws:
                try:
                    self.ws.close()
                except Exception as e:
                    logger.warning(f"Error during WebSocket close: {str(e)}")

            # Reset state
            self.ws = None
            self.is_connected = False
            self.subscribed_tokens.clear()
            self.live_feed.clear()
            logger.info("WebSocket connection closed and cleaned up")

        except Exception as e:
            logger.error(f"Error in WebSocket cleanup: {str(e)}")
            # Don't re-raise as this is cleanup code

    def disconnect(self):
        """Properly disconnect from WebSocket"""
        logger.info("Disconnecting from WebSocket...")
        self.stopping = True
        
        if self.ws:
            try:
                self.ws.close()
            except Exception as e:
                logger.error(f"Error closing WebSocket: {str(e)}")
            
        self.is_connected = False
        self.last_heartbeat = None
        logger.info("WebSocket disconnected")

    def __del__(self):
        """Cleanup when object is destroyed"""
        if not self.stopping:
            self.disconnect()

    def reset(self):
        """Reset the WebSocket connection state"""
        self.disconnect()
        self.stopping = False
        self.current_reconnect_attempt = 0
        self.subscribed_tokens.clear()
        self.live_feed.clear()
        self.last_heartbeat = None

    def add_tick_callback(self, callback):
        """Add a callback function to be called for each tick"""
        if callback not in self.on_tick_callbacks:
            self.on_tick_callbacks.append(callback)
            
    def remove_tick_callback(self, callback):
        """Remove a callback function"""
        if callback in self.on_tick_callbacks:
            self.on_tick_callbacks.remove(callback)
