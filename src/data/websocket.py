import json
import logging
import threading
import time
import random
from typing import Dict, List, Optional, Any
from datetime import datetime
from SmartApi.smartWebSocketV2 import SmartWebSocketV2
from .market_utils import is_market_open, format_time_until_market_open
import pandas as pd  # Import pandas for timestamp conversion

logger = logging.getLogger(__name__)

class MarketDataWebSocket(SmartWebSocketV2):
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

    def __init__(self, auth_token: str, api_key: str, client_code: str, feed_token: str, config: Dict[str, Any] = None):
        """Initialize WebSocket with improved state tracking"""
        self.auth_token = auth_token
        self.api_key = api_key
        self.client_code = client_code
        self.feed_token = feed_token
        self.config = config or {}
        
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
        
        # WebSocket instance
        self.ws = None
        
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
                    if self.ws and time.time() - self._last_heartbeat >= self._heartbeat_interval:
                        self.ws.send('ping')
                        logger.debug("Sent heartbeat ping")
                        self._last_heartbeat = time.time()
                    time.sleep(1)  # Check every second
                except Exception as e:
                    logger.error(f"Error in heartbeat: {str(e)}")
                    # Don't spam logs with the same error
                    time.sleep(5)

        self._heartbeat_thread = threading.Thread(target=send_heartbeat, daemon=True)
        self._heartbeat_thread.start()
        logger.debug("Started heartbeat thread")

    def wait_for_connection(self, timeout: int = 10) -> bool:
        """Wait for WebSocket connection to be established."""
        return self._connection_event.wait(timeout=timeout)
        
    def connect(self) -> None:
        """Connect to WebSocket with enhanced error handling"""
        try:
            # Check market hours
            if not is_market_open():
                wait_time = format_time_until_market_open()
                logger.warning(f"Market is currently closed. Next market opening in {wait_time}")
                logger.info("Proceeding with connection in simulation mode...")
            
            # Set up WebSocket URL with authentication
            url = f"{self.WEBSOCKET_URL}?clientCode={self.client_code}&feedToken={self.feed_token}&apiKey={self.api_key}"
            
            # Create WebSocket instance
            import websocket
            websocket.enableTrace(True)  # Enable trace for debugging
            self.ws = websocket.WebSocketApp(
                url,
                on_message=self._on_message,
                on_error=self._on_error,
                on_close=self._on_close,
                on_open=self._on_open,
                header={
                    'Authorization': f'Bearer {self.auth_token}',
                    'x-api-key': self.api_key,
                    'x-client-code': self.client_code,
                    'x-feed-token': self.feed_token
                }
            )
            
            # Start WebSocket in a thread with proper parameters
            self.ws_thread = threading.Thread(
                target=lambda: self.ws.run_forever(
                    ping_interval=20,  # Send ping every 20 seconds
                    ping_timeout=10,   # Wait 10 seconds for pong response
                    ping_payload='ping'  # Use 'ping' as the ping message
                )
            )
            self.ws_thread.daemon = True
            self.ws_thread.start()
            
            logger.info("WebSocket connection initiated")
            
        except Exception as e:
            self.connection_state = self.DISCONNECTED
            logger.error(f"Connection error: {str(e)}")
            raise

    def _on_open(self, ws=None):
        """Handle WebSocket connection open with improved state management"""
        try:
            logger.info("WebSocket connection opened")
            self.connection_state = self.CONNECTED
            self._connection_ready = True
            self._connection_event.set()
            self._last_heartbeat = time.time()
            
            # Start heartbeat monitoring if not already running
            if not self._heartbeat_thread or not self._heartbeat_thread.is_alive():
                self._start_heartbeat()
            
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

    def _on_close(self, ws=None, close_status_code=None, close_msg=None):
        """Handle WebSocket connection close"""
        try:
            logger.info(f"WebSocket connection closed (Status: {close_status_code}, Message: {close_msg})")
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

    def close(self):
        """Close WebSocket connection cleanly"""
        try:
            self.stopping = True
            if self.ws:
                self.ws.close()
            self.connection_state = self.DISCONNECTED
            logger.info("WebSocket connection closed")
        except Exception as e:
            logger.error(f"Error closing WebSocket: {str(e)}")

    def _on_message(self, ws=None, message=None):
        """Handle raw WebSocket messages"""
        try:
            if not message:
                return
                
            if isinstance(message, str):
                if message.lower() == 'pong':
                    self._last_heartbeat = time.time()
                    return
                    
                # Handle text messages (subscription responses, etc)
                try:
                    data = json.loads(message)
                    if 'action' in data:
                        if data['action'] == 'subscribe':
                            if data.get('status', False):
                                logger.info("Subscription successful")
                            else:
                                logger.error(f"Subscription failed: {data.get('message', 'Unknown error')}")
                        elif data['action'] == 'unsubscribe':
                            if data.get('status', False):
                                logger.info("Unsubscription successful")
                            else:
                                logger.error(f"Unsubscription failed: {data.get('message', 'Unknown error')}")
                except json.JSONDecodeError:
                    logger.warning(f"Received non-JSON text message: {message}")
                return
            
            # Handle binary data
            if isinstance(message, bytes):
                self._handle_binary_data(message)
            
        except Exception as e:
            logger.error(f"Error in raw message handler: {str(e)}")

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
        """Subscribe to market data for given tokens
        
        Args:
            tokens (List[str]): List of instrument tokens to subscribe to
            mode (int, optional): Subscription mode (LTP/QUOTE/SNAPQUOTE). 
                                Defaults to QUOTE if not specified.
        """
        try:
            if not self._connection_ready:
                logger.warning("WebSocket not ready, adding to pending subscriptions")
                self.pending_subscriptions.append((tokens, mode))
                return

            if not tokens:
                logger.warning("No tokens provided for subscription")
                return
                
            # Use default mode if not specified
            if mode is None:
                mode = self.MODE_QUOTE

            # Format subscription message according to SmartWebSocket V2 spec
            subscription_msg = {
                "action": "subscribe",
                "key": tokens,
                "messageType": str(mode)  # Convert mode to string as required by API
            }

            # Send subscription request
            self.ws.send(json.dumps(subscription_msg))
            
            # Store subscribed tokens with their mode
            for token in tokens:
                self.subscribed_tokens[token] = mode
                
            logger.info(f"Sent subscription request for {len(tokens)} tokens in mode {mode}")

        except Exception as e:
            logger.error(f"Error in subscribe: {str(e)}")
            raise

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
                
            logger.debug(f"No market data available for token {token}")
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
