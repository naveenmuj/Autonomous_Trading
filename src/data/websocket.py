import json
import logging
import threading
import time
import random
import math
from typing import Dict, List, Optional, Any
from datetime import datetime
from .market_utils import is_market_open, format_time_until_market_open
import pandas as pd  # Import pandas for timestamp conversion
from SmartApi.smartWebSocketV2 import SmartWebSocketV2
import os  # Import os for directory creation

logger = logging.getLogger(__name__)

# Add app_logger for app.log
app_logger = logging.getLogger('app_logger')
app_log_path = 'logs/2025-07-09/app.log'  # You may want to make this dynamic
app_log_dir = os.path.dirname(app_log_path)
os.makedirs(app_log_dir, exist_ok=True)
if not any(isinstance(h, logging.FileHandler) and getattr(h, 'baseFilename', '').endswith('app.log') for h in app_logger.handlers):
    app_log_handler = logging.FileHandler(app_log_path)
    app_log_handler.setLevel(logging.INFO)
    app_log_handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)s %(message)s'))
    app_logger.addHandler(app_log_handler)
    app_logger.propagate = False

# Global WebSocket instance to prevent multiple connections
_global_websocket_instance = None
_websocket_lock = threading.Lock()

class MarketDataWebSocket(SmartWebSocketV2):
    def add_message_callback(self, callback):
        """Add a callback function to be called on every raw WebSocket message (text or binary)."""
        if not hasattr(self, '_message_callbacks'):
            self._message_callbacks = []
        self._message_callbacks.append(callback)

    def remove_message_callback(self, callback):
        """Remove a previously added message callback function."""
        if hasattr(self, '_message_callbacks') and callback in self._message_callbacks:
            self._message_callbacks.remove(callback)
    # Exchange type constants
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

    # Connection states
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    RECONNECTING = "reconnecting"

    # WebSocket 2.0 endpoint
    WEBSOCKET_URL = "wss://smartapisocket.angelone.in/smart-stream"

    MAX_RECONNECT_ATTEMPTS = 5  # Maximum number of reconnection attempts
    INITIAL_BACKOFF = 1.0    # Initial backoff delay in seconds
    MAX_BACKOFF = 300.0      # Maximum backoff delay in seconds

    # DEBUG: Set to True to force single-token subscription for debugging
    DEBUG_SINGLE_TOKEN_SUB = False  # PATCH: Set to False to allow real subscriptions
    DEBUG_TOKEN = '12018'  # SUZLON NSE token

    def __init__(self, auth_token: str, api_key: str, client_code: str, feed_token: str, config: Dict[str, Any] = None, debug: bool = False):
        """Initialize WebSocket with improved state tracking"""
        self.auth_token = auth_token
        self.api_key = api_key
        self.client_code = client_code
        self.feed_token = feed_token
        self.config = config or {}
        self.debug = debug
        
        # State tracking
        self._connection_state = self.DISCONNECTED
        self.stopping = False
        self.pending_subscriptions = []
        self._connection_ready = False
        self._connection_event = threading.Event()
        self._reconnect_count = 0
        self._reconnect_delay = self.INITIAL_BACKOFF
        
        # Data storage
        self.subscribed_tokens = {}
        self.live_feed = {}
        self.tick_callbacks = []
        
        # Heartbeat
        self._last_heartbeat = 0
        self._heartbeat_interval = 30  # Send heartbeat every 30 seconds
        self._heartbeat_thread = None
        
        # Thread safety
        self._callback_lock = threading.Lock()
        self._state_lock = threading.Lock()
        self._reconnect_lock = threading.Lock()
        
        # WebSocket instance: use SmartWebSocketV2 as in test script
        self.ws = SmartWebSocketV2(
            auth_token=self.auth_token,
            api_key=self.api_key,
            feed_token=self.feed_token,
            client_code=self.client_code
        )
        # Assign event/callback handlers as in the test script
        self.ws.on_open = self._on_open  # Use correct handler
        self.ws.on_data = self._on_data  # Use correct handler for data messages
        self.ws.on_error = self._on_error  # Use correct handler
        self.ws.on_close = self._on_close  # Use correct handler
        
        # Initialize logging
        logger.info("Initializing Market Data WebSocket with enhanced retry settings...")

    @property
    def connection_state(self) -> str:
        """Get the current connection state"""
        with self._state_lock:
            return self._connection_state

    @connection_state.setter
    def connection_state(self, state: str):
        """Set the current connection state"""
        with self._state_lock:
            self._connection_state = state
            
    @property
    def is_connected(self) -> bool:
        """Check if WebSocket is currently connected"""
        return self.connection_state == self.CONNECTED

    def update_tokens(self, auth_token: str = None, feed_token: str = None):
        """Update authentication tokens and reconnect if needed"""
        if auth_token:
            self.auth_token = auth_token
        if feed_token:
            self.feed_token = feed_token
            
        # If already connected, reconnect with new tokens
        if self.is_connected:
            self.reconnect()

    def reconnect(self):
        """Reconnect the WebSocket with exponential backoff"""
        with self._reconnect_lock:
            if self.stopping or self._reconnect_count >= self.MAX_RECONNECT_ATTEMPTS:
                logger.error("Maximum reconnection attempts reached or stopping requested")
                self.connection_state = self.DISCONNECTED
                return False
            
            try:
                self.connection_state = self.RECONNECTING
                self._reconnect_count += 1
                
                # Calculate backoff with jitter
                jitter = random.uniform(0, 0.1) * self._reconnect_delay
                wait_time = min(self._reconnect_delay + jitter, self.MAX_BACKOFF)
                
                logger.info(f"Attempting reconnection {self._reconnect_count}/{self.MAX_RECONNECT_ATTEMPTS} "
                          f"after {wait_time:.1f}s delay")
                
                time.sleep(wait_time)
                
                # Close existing connection if any
                if self.ws:
                    try:
                        self.ws.close()
                    except:
                        pass
                
                # Connect with current tokens
                self.connect()
                
                # Double the reconnection delay for next attempt
                self._reconnect_delay = min(self._reconnect_delay * 2, self.MAX_BACKOFF)
                
                # Reset counter on successful connection
                if self.wait_for_connection(timeout=10):
                    logger.info("Reconnection successful")
                    self._reconnect_count = 0
                    self._reconnect_delay = self.INITIAL_BACKOFF
                    return True
                    
                return False
                
            except Exception as e:
                logger.error(f"Error during reconnection attempt: {str(e)}")
                self.connection_state = self.DISCONNECTED
                return False

    def _start_heartbeat(self):
        """Start heartbeat thread"""
        def send_heartbeat():
            while not self.stopping:
                try:
                    # SmartWebSocketV2 does not expose a public send() method for manual ping.
                    # If ping/heartbeat is needed, use the library's built-in keepalive or add support in SmartWebSocketV2.
                    # self.ws.send('ping')  # <-- Removed: Not supported by SmartWebSocketV2
                    # logger.debug("Sent heartbeat ping")
                    self._last_heartbeat = time.time()
                    time.sleep(self._heartbeat_interval)
                except Exception as e:
                    logger.error(f"Error in heartbeat: {str(e)}")
                    time.sleep(5)

        self._heartbeat_thread = threading.Thread(target=send_heartbeat, daemon=True)
        self._heartbeat_thread.start()
        logger.debug("Started heartbeat thread")

    def wait_for_connection(self, timeout: int = 10) -> bool:
        """Wait for WebSocket connection to be established."""
        return self._connection_event.wait(timeout=timeout)
        
    def connect(self) -> bool:
        """Connect using SmartWebSocketV2 logic with enhanced error handling"""
        try:
            # Check market hours
            if not is_market_open():
                wait_time = format_time_until_market_open()
                logger.warning(f"Market is currently closed. Next market opening in {wait_time}")
                # Still try to connect for after-hours data
            else:
                logger.info("[DIAGNOSTIC] WebSocket is running in LIVE mode.")

            # Set up connection state
            self.connection_state = self.CONNECTING
            self._connection_ready = False
            self._connection_event.clear()
            self.stopping = False

            # Call SmartWebSocketV2 connect
            logger.info("About to call SmartWebSocketV2.connect()")
            
            # Start connect in a separate thread to avoid blocking
            def connect_websocket():
                try:
                    logger.info("Starting SmartWebSocketV2.connect() in thread...")
                    self.ws.connect()
                    logger.info("SmartWebSocketV2.connect() completed")
                except Exception as e:
                    logger.error(f"Exception in SmartWebSocketV2.connect(): {e}")
                    self.connection_state = self.DISCONNECTED
            
            connect_thread = threading.Thread(target=connect_websocket, daemon=True)
            connect_thread.start()
            logger.info("WebSocket connection thread started")
            
            # Wait for connection to establish
            logger.info("Waiting for connection event...")
            success = self.wait_for_connection(timeout=20)  # Increased timeout
            logger.info(f"wait_for_connection returned: {success}")
            logger.info(f"is_connected status: {self.is_connected}")
            logger.info(f"connection_state: {self.connection_state}")
            
            if success and self.is_connected:
                logger.info("WebSocket connection established successfully")
                return True
            else:
                logger.error(f"WebSocket connection failed or timed out - success={success}, is_connected={self.is_connected}")
                self.connection_state = self.DISCONNECTED
                return False
                
        except Exception as e:
            self.connection_state = self.DISCONNECTED
            logger.error(f"Connection error: {str(e)}")
            return False

    def _on_open(self, wsapp):
        """Handle WebSocket connection open with improved state management and diagnostics"""
        logger.info("[HANDLER] _on_open called")
        try:
            logger.info("WebSocket connection opened [DIAGNOSTIC]")
            self.connection_state = self.CONNECTED
            self._connection_ready = True
            self._connection_event.set()
            self._last_heartbeat = time.time()
            self._last_message_time = time.time()
            
            # Start heartbeat monitoring if not already running
            if not self._heartbeat_thread or not self._heartbeat_thread.is_alive():
                self._start_heartbeat()
            
            # Always subscribe to debug token for diagnostics
            if self.DEBUG_SINGLE_TOKEN_SUB:
                logger.info(f"[HANDLER] Subscribing to debug token {self.DEBUG_TOKEN} in _on_open")
                self.subscribe([self.DEBUG_TOKEN], self.MODE_QUOTE)
            
            # Resubscribe to existing tokens
            existing_subscriptions = dict(self.subscribed_tokens)  # Make a copy
            if existing_subscriptions:
                logger.info(f"Resubscribing to {len(existing_subscriptions)} tokens")
                for token, mode in existing_subscriptions.items():
                    try:
                        self.subscribe([token], mode)
                    except Exception as e:
                        logger.error(f"Error resubscribing to {token}: {str(e)}")
            
            # Process any pending subscriptions
            if self.pending_subscriptions:
                logger.info(f"Processing {len(self.pending_subscriptions)} pending subscriptions")
                current_subscriptions = self.pending_subscriptions.copy()
                self.pending_subscriptions.clear()
                for tokens, mode in current_subscriptions:
                    try:
                        self.subscribe(tokens, mode)
                    except Exception as e:
                        logger.error(f"Error processing pending subscription: {str(e)}")
                        self.pending_subscriptions.append((tokens, mode))
                        
        except Exception as e:
            logger.error(f"Error in connection open handler: {str(e)}")
            self.connection_state = self.DISCONNECTED

    def _on_close(self, ws, close_status_code, close_msg):
        logger.info(f"[HANDLER] _on_close called (Status: {close_status_code}, Message: {close_msg})")
        try:
            logger.info(f"WebSocket connection closed (Status: {close_status_code}, Message: {close_msg}) [DIAGNOSTIC]")
            self.connection_state = self.DISCONNECTED
            self._connection_ready = False
            self._connection_event.clear()
            
            # If not intentionally stopping, attempt to reconnect
            if not self.stopping:
                logger.info("Connection lost, attempting to reconnect...")
                self.reconnect()
                
        except Exception as e:
            logger.error(f"Error in connection close handler: {str(e)}")
            self.connection_state = self.DISCONNECTED

    def _on_error(self, ws, error):
        logger.error(f"[HANDLER] _on_error called: {error}")
        logger.error(f"WebSocket error: {error} [DIAGNOSTIC]")

    def _on_data(self, wsapp, message):
        logger.debug(f"[HANDLER] _on_data called with message: {repr(message)}")
        # Aggressive diagnostics: log every raw message
        app_logger = logging.getLogger("app")
        app_logger.info(f"[RAW_WS_MESSAGE] {repr(message)}")
        try:
            logger.debug(f"[WS-RAW] Received message: {repr(message)}")
            self._last_message_time = time.time()
            if not message:
                return
            # Call message callbacks if any
            if hasattr(self, '_message_callbacks'):
                for cb in self._message_callbacks:
                    try:
                        cb(message)
                    except Exception as e:
                        logger.error(f"Error in message callback: {e}")
            if isinstance(message, str):
                if message.lower() == 'pong':
                    self._last_heartbeat = time.time()
                    return
                # Handle text messages (subscription responses, etc)
                try:
                    data = json.loads(message)
                except json.JSONDecodeError:
                    logger.warning(f"Received non-JSON text message: {message}")
                    return
            elif isinstance(message, dict):
                # Message is already a dict (no need to parse JSON)
                data = message
            else:
                # Handle binary data
                if isinstance(message, bytes):
                    # Patch: log every tick received for diagnostics
                    try:
                        tick_token = self._extract_token_from_binary(message)
                        logger.debug(f"Received binary tick for token {tick_token}")
                        app_logger.info(f"[BINARY_TICK_RAW] token={tick_token} bytes={message[:32].hex()}")
                    except Exception:
                        logger.debug("Received binary tick (token extraction failed)")
                        app_logger.info(f"[BINARY_TICK_RAW] token=UNKNOWN bytes={message[:32].hex()}")
                    self._handle_binary_data(message)
                return
            
            # Handle market data in JSON format (like in test case)
            if isinstance(data, dict) and 'token' in data and 'last_traded_price' in data:
                        token = str(data.get('token', ''))
                        
                        # Process market data similar to test case
                        tick_data = {
                            'token': token,
                            'ltp': float(data.get('last_traded_price', 0)) / 100.0,  # Convert price like test case
                            'volume': int(data.get('volume_trade_for_the_day', 0)),
                            'bid_price': float(data.get('best_5_buy_data', [{}])[0].get('price', 0)) / 100.0 if data.get('best_5_buy_data') else 0.0,
                            'ask_price': float(data.get('best_5_sell_data', [{}])[0].get('price', 0)) / 100.0 if data.get('best_5_sell_data') else 0.0,
                            'open': float(data.get('open_price_of_the_day', 0)) / 100.0,
                            'high': float(data.get('high_price_of_the_day', 0)) / 100.0,
                            'low': float(data.get('low_price_of_the_day', 0)) / 100.0,
                            'close': float(data.get('closed_price', 0)) / 100.0,
                            'timestamp': datetime.fromtimestamp(int(data.get('exchange_timestamp', time.time() * 1000)) / 1000),
                            'source': 'websocket_json'
                        }
                        
                        # Store in live feed
                        self.live_feed[token] = tick_data
                        
                        # Log the market data
                        logger.info(f"[JSON_TICK] Token: {token}, LTP: ₹{tick_data['ltp']:.2f}, Volume: {tick_data['volume']:,}")
                        app_logger.info(f"[JSON_LTP] Token: {token}, LTP: {tick_data['ltp']}, Time: {tick_data['timestamp']}")
                        
                        # Notify callbacks
                        with self._callback_lock:
                            for callback in self.tick_callbacks:
                                try:
                                    callback(tick_data)
                                except Exception as e:
                                    logger.error(f"Error in tick callback: {str(e)}")
                        return
                    
            # Handle subscription responses
            if 'action' in data:
                if data['action'] == 'subscribe':
                    if data.get('status', False):
                        logger.info(f"Subscription successful: {data}")
                        app_logger.info(f"[SUBSCRIBE_OK] {data}")
                    else:
                        logger.error(f"Subscription failed: {data.get('message', 'Unknown error')} | Full response: {data}")
                        app_logger.error(f"[SUBSCRIBE_FAIL] {data}")
                elif data['action'] == 'unsubscribe':
                    if data.get('status', False):
                        logger.info(f"Unsubscription successful: {data}")
                    else:
                        logger.error(f"Unsubscription failed: {data.get('message', 'Unknown error')} | Full response: {data}")
                else:
                    logger.info(f"WebSocket message: {data}")
        except Exception as e:
            # Handle binary data
            if isinstance(message, bytes):
                # Patch: log every tick received for diagnostics
                try:
                    tick_token = self._extract_token_from_binary(message)
                    logger.debug(f"Received binary tick for token {tick_token}")
                    app_logger.info(f"[BINARY_TICK_RAW] token={tick_token} bytes={message[:32].hex()}")
                except Exception:
                    logger.debug("Received binary tick (token extraction failed)")
                    app_logger.info(f"[BINARY_TICK_RAW] token=UNKNOWN bytes={message[:32].hex()}")
                self._handle_binary_data(message)
        except Exception as e:
            logger.error(f"Error in raw message handler: {str(e)}")
            app_logger.error(f"[WS_HANDLER_ERROR] {str(e)} | Message: {repr(message)}")

    def _extract_token_from_binary(self, data: bytes) -> str:
        """Extract token from binary data for logging purposes"""
        try:
            if len(data) < 27:
                return "UNKNOWN"
            
            # Extract token (null-terminated string)
            token = ""
            for i in range(2, 27):
                if data[i:i+1] == b'\x00':
                    break
                token += data[i:i+1].decode('utf-8')
            return token if token else "UNKNOWN"
        except Exception:
            return "UNKNOWN"

    def _handle_binary_data(self, data: bytes):
        """Handle binary market data according to WebSocket 2.0 spec"""
        try:
            # Use little endian byte order as per spec
            mode = int.from_bytes(data[0:1], byteorder='little')
            exchange_type = int.from_bytes(data[1:2], byteorder='little')
            
            # Extract token (null-terminated string)
            token = ""
            token_end = 2
            for i in range(2, 27):
                if data[i:i+1] == b'\x00':
                    token_end = i
                    break
                token += data[i:i+1].decode('utf-8')
            
            # Parse core fields
            sequence_number = int.from_bytes(data[27:35], byteorder='little')
            exchange_timestamp = int.from_bytes(data[35:43], byteorder='little')
            last_traded_price = int.from_bytes(data[43:51], byteorder='little') / 100.0  # Convert to rupees
            
            # Format tick data
            tick_data = {
                'token': token,
                'sequence_number': sequence_number,
                'timestamp': datetime.fromtimestamp(exchange_timestamp / 1000),
                'ltp': last_traded_price
            }

            # Add fields for QUOTE and SNAPQUOTE modes
            if mode >= self.MODE_QUOTE:
                tick_data.update({
                    'volume': int.from_bytes(data[67:75], byteorder='little'),
                    'last_traded_quantity': int.from_bytes(data[59:67], byteorder='little'),
                    'total_buy_quantity': int.from_bytes(data[75:83], byteorder='little'),
                    'total_sell_quantity': int.from_bytes(data[83:91], byteorder='little'),
                    'open': int.from_bytes(data[91:99], byteorder='little') / 100.0,
                    'high': int.from_bytes(data[99:107], byteorder='little') / 100.0,
                    'low': int.from_bytes(data[107:115], byteorder='little') / 100.0,
                    'close': int.from_bytes(data[115:123], byteorder='little') / 100.0,
                    'avg_trade_price': int.from_bytes(data[123:131], byteorder='little') / 100.0,
                })

            if mode == self.MODE_SNAPQUOTE:
                tick_data.update({
                    'open_interest': int.from_bytes(data[131:139], byteorder='little'),
                    'open_interest_change': int.from_bytes(data[139:147], byteorder='little'),
                    'best_bid_price': int.from_bytes(data[147:155], byteorder='little') / 100.0,
                    'best_bid_quantity': int.from_bytes(data[155:163], byteorder='little'),
                    'best_ask_price': int.from_bytes(data[247:255], byteorder='little') / 100.0,
                    'best_ask_quantity': int.from_bytes(data[255:263], byteorder='little')
                })

            # Update live feed
            self.live_feed[token] = tick_data

            # Log every received tick for debugging
            logger.info(f"[WebSocketTick] Token: {token}, LTP: {tick_data['ltp']}, Time: {tick_data['timestamp']}")
            logger.debug(f"[TickData] {tick_data}")
            # Log LTP to app.log as well
            app_logger.info(f"[LTP] Token: {token}, LTP: {tick_data['ltp']}, Time: {tick_data['timestamp']}")
            # Notify callbacks with thread safety
            with self._callback_lock:
                for callback in self.tick_callbacks:
                    try:
                        callback(tick_data)
                    except Exception as e:
                        logger.error(f"Error in callback: {str(e)}")

            logger.debug(f"Processed market data for {token}: ₹{tick_data['ltp']:.2f}")

        except Exception as e:
            logger.error(f"Error parsing binary data: {str(e)}")
            if len(data) > 0:
                logger.debug(f"First few bytes: {data[:20].hex()}")

    def add_tick_callback(self, callback):
        """Add a callback function to be called when market data is received"""
        with self._callback_lock:
            self.tick_callbacks.append(callback)
            
    def remove_tick_callback(self, callback):
        """Remove a previously added callback function"""
        with self._callback_lock:
            if callback in self.tick_callbacks:
                self.tick_callbacks.remove(callback)

    def close(self) -> None:
        """Close WebSocket connection gracefully"""
        try:
            self.stopping = True
            if self._heartbeat_thread:
                self._heartbeat_thread.join(timeout=1)
            if self.ws:
                self.ws.close()
            logger.info("WebSocket connection closed")
            
        except Exception as e:
            logger.error(f"Error closing WebSocket: {str(e)}")
            raise

    def subscribe(self, tokens, mode=None):
        """Subscribe to market data for given tokens using SmartWebSocketV2.subscribe() with diagnostics"""
        try:
            # Force test subscription to RELIANCE.NS token for diagnostics
            TEST_TOKEN = None
            for t, s in self.subscribed_tokens.items():
                if str(t).startswith("RELIANCE") or str(t) == "500325":
                    TEST_TOKEN = t
                    break
            if not TEST_TOKEN:
                # Try to add a known token (should be mapped in DataCollector)
                tokens = list(tokens) + ["RELIANCE.NS"]
                logger.warning("[DEBUG] Forcing test subscription to RELIANCE.NS for diagnostics.")
            
            if self.DEBUG_SINGLE_TOKEN_SUB:
                logger.warning(f"[DEBUG] Forcing single-token subscription for debugging: {self.DEBUG_TOKEN}")
                tokens = [self.DEBUG_TOKEN]
            if not self._connection_ready:
                logger.warning("WebSocket not ready, adding to pending subscriptions")
                self.pending_subscriptions.append((tokens, mode))
                return
            if not tokens:
                logger.warning("No tokens provided for subscription")
                return
            if mode is None:
                mode = self.MODE_QUOTE
            token_list = [{
                "exchangeType": 1,  # NSE
                "tokens": [str(token) for token in tokens]
            }]
            correlation_id = f"websocket_{int(time.time())}"
            logger.warning(f"[DEBUG] Subscribing with token_list: {token_list}, mode: {mode}, correlation_id: {correlation_id}")
            app_logger = logging.getLogger("app")
            app_logger.info(f"[SUBSCRIBE_ATTEMPT] tokens={tokens} mode={mode} correlation_id={correlation_id}")
            try:
                self.ws.subscribe(
                    correlation_id=correlation_id,
                    mode=mode,
                    token_list=token_list
                )
                logger.info(f"Sent SmartWebSocketV2 subscription for tokens: {tokens} in mode {mode}")
                app_logger.info(f"[SUBSCRIBE_SENT] tokens={tokens} mode={mode}")
            except Exception as e:
                logger.error(f"Error in SmartWebSocketV2.subscribe: {str(e)} | token_list: {token_list}")
                app_logger.error(f"[SUBSCRIBE_ERROR] {str(e)} | token_list={token_list}")
            for token in tokens:
                self.subscribed_tokens[token] = mode
            self.log_subscription_state()
            import threading
            def warn_if_no_messages():
                time.sleep(10)
                if time.time() - getattr(self, '_last_message_time', 0) > 9:
                    logger.warning("[DIAGNOSTIC] No websocket messages received within 10 seconds of subscription!")
                    app_logger.warning("[NO_TICKS_WARNING] No websocket messages received within 10 seconds of subscription!")
            threading.Thread(target=warn_if_no_messages, daemon=True).start()
        except Exception as e:
            logger.error(f"Error in subscribe: {str(e)} | Tokens: {tokens}")
            app_logger = logging.getLogger("app")
            app_logger.error(f"[SUBSCRIBE_FATAL] {str(e)} | Tokens: {tokens}")
            raise

    def log_subscription_state(self):
        """Log the current state of all subscribed tokens for diagnostics"""
        logger.info(f"[SubscriptionState] Subscribed tokens: {list(self.subscribed_tokens.keys())}")
        logger.info(f"[SubscriptionState] Total: {len(self.subscribed_tokens)} tokens")

    def unsubscribe(self, tokens):
        """Unsubscribe from market data for given tokens
        
        Args:
            tokens (List[str]): List of instrument tokens to unsubscribe from
        """
        try:
            if not self._connection_ready:
                logger.warning("WebSocket not ready")
                return

            if not tokens:
                logger.warning("No tokens provided for unsubscription")
                return

            # Format unsubscription message
            unsubscription_msg = {
                "action": "unsubscribe",
                "key": tokens
            }

            # Send unsubscription request
            self.ws.send(json.dumps(unsubscription_msg))
            
            # Remove from subscribed tokens
            for token in tokens:
                self.subscribed_tokens.pop(token, None)
                
            logger.info(f"Sent unsubscription request for {len(tokens)} tokens")
        except Exception as e:
            logger.error(f"Error in unsubscribe: {str(e)}")
            raise

    def close(self):
        """Close WebSocket connection gracefully"""
        try:
            logger.info("Closing WebSocket connection...")
            self.stopping = True
            
            # Set state to disconnecting
            self.connection_state = self.DISCONNECTING
            
            # Stop heartbeat thread
            if hasattr(self, '_heartbeat_thread') and self._heartbeat_thread:
                self._heartbeat_thread = None
            
            # Close the WebSocket connection
            if hasattr(self, 'ws') and self.ws:
                try:
                    self.ws.close()
                    logger.info("WebSocket connection closed successfully")
                except Exception as e:
                    logger.debug(f"Error closing WebSocket: {e}")
            
            # Clear state
            self.connection_state = self.DISCONNECTED
            self._connection_ready = False
            self._connection_event.clear()
            self.subscribed_tokens.clear()
            self.pending_subscriptions.clear()
            
        except Exception as e:
            logger.error(f"Error closing WebSocket: {e}")

    def get_market_data(self, token: str) -> Optional[Dict[str, Any]]:
        """Get latest market data for a token
        
        Args:
            token (str): Instrument token
            
        Returns:
            Optional[Dict[str, Any]]: Latest market data if available, None otherwise
        """
        try:
            if not token:
                logger.warning("Token not provided for market data lookup")
                return None

            if not self.is_connected:
                logger.warning(f"WebSocket is not connected when requesting market data for token {token}")

            if token not in self.subscribed_tokens:
                logger.warning(f"Token {token} was never subscribed. Subscribed tokens: {list(self.subscribed_tokens.keys())}")

            data = self.live_feed.get(str(token))
            if data:
                # Ensure numeric values are properly converted
                numeric_fields = [
                    'ltp', 'open', 'high', 'low', 'close', 'volume',
                    'best_bid_price', 'best_bid_quantity',
                    'best_ask_price', 'best_ask_quantity'
                ]
                for field in numeric_fields:
                    if field in data:
                        try:
                            data[field] = float(data[field])
                        except (ValueError, TypeError):
                            data[field] = 0.0
                # Ensure timestamp is properly formatted
                if 'timestamp' in data and not isinstance(data['timestamp'], datetime):
                    try:
                        data['timestamp'] = pd.to_datetime(data['timestamp'])
                    except:
                        data['timestamp'] = datetime.now()
                return data
            logger.debug(f"No market data available for token {token}. Connection ready: {self._connection_ready}, Subscribed: {token in self.subscribed_tokens}")
            return None
        except Exception as e:
            logger.error(f"Error getting market data for token {token}: {str(e)}")
            return None

    def get_connection_status(self) -> Dict[str, Any]:
        """Get detailed connection status information
        
        Returns:
            Dict[str, Any]: Connection status information including:
                - state: Current connection state
                - is_connected: Whether socket is connected
                - last_heartbeat: Timestamp of last heartbeat
                - subscriptions: Dictionary of subscribed tokens by exchange
                - feed_type: Type of market data feed
                - feed_token: Current feed token
                - client_code: Client code
        """
        try:
            current_time = time.time()
            heartbeat_age = current_time - self._last_heartbeat if self._last_heartbeat > 0 else 0
            
            # Format subscribed tokens by exchange
            subscriptions = {
                'NSE': []  # Initialize with NSE since that's what we're using
            }
            
            # Add tokens to appropriate exchange
            for token, mode in self.subscribed_tokens.items():
                subscriptions['NSE'].append({
                    'token': str(token),
                    'mode': mode,
                    'mode_name': {
                        self.MODE_LTP: 'LTP',
                        self.MODE_QUOTE: 'QUOTE',
                        self.MODE_SNAPQUOTE: 'SNAPQUOTE'
                    }.get(mode, 'UNKNOWN')
                })
            
            return {
                'state': self.connection_state,
                'is_connected': self.is_connected,
                'last_heartbeat': datetime.fromtimestamp(self._last_heartbeat) if self._last_heartbeat > 0 else None,
                'heartbeat_age': f"{heartbeat_age:.1f}s" if heartbeat_age > 0 else "N/A",
                'feed_type': 'WebSocket 2.0',
                'subscriptions': subscriptions,  # Changed from subscribed_tokens to formatted subscriptions
                'feed_token': f"{self.feed_token[:10]}..." if self.feed_token else None,
                'client_code': self.client_code,
                'exchange': 'NSE',  # Add exchange info
                'total_subscriptions': len(self.subscribed_tokens)
            }
            
        except Exception as e:
            logger.error(f"Error getting connection status: {str(e)}")
            return {
                'state': 'ERROR',
                'is_connected': False,
                'error': str(e),
                'subscriptions': {'NSE': []}  # Return empty subscriptions on error
            }

def get_websocket_instance(auth_token: str, api_key: str, client_code: str, feed_token: str, config: Dict[str, Any] = None, debug: bool = False) -> MarketDataWebSocket:
    """Get or create a singleton WebSocket instance to prevent connection limit errors"""
    global _global_websocket_instance
    
    with _websocket_lock:
        # If instance exists and tokens match, reuse it
        if (_global_websocket_instance and 
            _global_websocket_instance.auth_token == auth_token and
            _global_websocket_instance.feed_token == feed_token):
            logger.info("Reusing existing WebSocket instance")
            return _global_websocket_instance
        
        # Close existing instance if tokens are different
        if _global_websocket_instance:
            logger.info("Closing existing WebSocket instance (token mismatch)")
            try:
                _global_websocket_instance.close()
            except:
                pass
            _global_websocket_instance = None
        
        # Create new instance
        logger.info("Creating new WebSocket instance")
        _global_websocket_instance = MarketDataWebSocket(
            auth_token=auth_token,
            api_key=api_key,
            client_code=client_code,
            feed_token=feed_token,
            config=config,
            debug=debug
        )
        
        return _global_websocket_instance

def close_global_websocket():
    """Close the global WebSocket instance"""
    global _global_websocket_instance
    
    with _websocket_lock:
        if _global_websocket_instance:
            logger.info("Closing global WebSocket instance")
            try:
                _global_websocket_instance.close()
            except:
                pass
            _global_websocket_instance = None
