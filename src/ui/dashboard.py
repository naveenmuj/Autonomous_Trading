import streamlit as st
import plotly.graph_objects as go
from datetime import datetime
import pandas as pd
import yaml
import logging
import time
from typing import Dict, Any, Optional, List, Tuple
import threading
from collections import defaultdict
import os
from src.trading.paper_trading import PaperTradingEngine
from src.trading.strategy import EnhancedTradingStrategy
from src.autonomous_trader import AutonomousTrader

logger = logging.getLogger(__name__)
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter('%(asctime)s - [%(name)s] - [%(levelname)s] - %(message)s'))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

class MarketDataCache:
    """Cache for market data to handle market closed scenarios"""
    def __init__(self):
        self._cache = {}
        self._last_update = {}
        
    def update(self, symbol: str, data: dict):
        """Update cache with latest market data"""
        if data and 'ltp' in data:
            self._cache[symbol] = data
            self._last_update[symbol] = datetime.now()
            
    def get(self, symbol: str) -> Optional[dict]:
        """Get cached data for symbol"""
        return self._cache.get(symbol)
    
    def get_last_update(self, symbol: str) -> Optional[datetime]:
        """Get last update time for symbol"""
        return self._last_update.get(symbol)

# Utility for INR formatting
def format_inr(value):
    try:
        return f"₹{value:,.2f}"
    except Exception:
        return f"₹{value}"

class DashboardUI:
    def __init__(self, config: Dict, data_collector=None, trade_manager=None, technical_analyzer=None, ai_model=None):
        """Initialize Dashboard UI with components"""
        # Core components
        self.config = config
        self.data_collector = data_collector
        self.trade_manager = trade_manager
        self.technical_analyzer = technical_analyzer
        self.ai_model = ai_model
        
        # Data management
        self.market_data_cache = MarketDataCache()
        self._update_interval = 5  # seconds
        
        # Autonomous trading
        self.autonomous_trader = None
        self.autonomous_thread = None
        
        # Initialize WebSocket connection for live data
        if self.data_collector and hasattr(self.data_collector, 'ensure_websocket_connected'):
            logger.info("Initializing WebSocket connection for live price data...")
            try:
                self.data_collector.ensure_websocket_connected()
                logger.info("WebSocket connection initialized successfully")
            except Exception as e:
                logger.warning(f"Failed to initialize WebSocket connection: {e}")
        
        logger.info("DashboardUI initialized")
        print("=== DashboardUI.__init__() complete ===")
    
    def _get_symbols(self):
        import streamlit as st
        st.info("[DEBUG] _get_symbols() called")
        """Get the list of symbols to monitor, respecting config and auto mode, and using model/ai_model if available."""
        # Prefer ai_model or technical_analyzer if they have get_symbols
        if hasattr(self.ai_model, 'get_symbols'):
            try:
                st.info("[DEBUG] Getting symbols from ai_model")
                return self.ai_model.get_symbols(self.data_collector)
            except Exception as e:
                st.error(f"[DEBUG] Error getting symbols from ai_model: {e}")
                logger.error(f"Error getting symbols from ai_model: {e}")
        if hasattr(self.technical_analyzer, 'get_symbols'):
            try:
                st.info("[DEBUG] Getting symbols from technical_analyzer")
                return self.technical_analyzer.get_symbols(self.data_collector)
            except Exception as e:
                st.error(f"[DEBUG] Error getting symbols from technical_analyzer: {e}")
                logger.error(f"Error getting symbols from technical_analyzer: {e}")
        if self.data_collector and hasattr(self.data_collector, 'get_symbols_from_config'):
            try:
                st.info("[DEBUG] Getting symbols from data_collector")
                return self.data_collector.get_symbols_from_config()
            except Exception as e:
                st.error(f"[DEBUG] Error getting symbols from data_collector: {e}")
                logger.error(f"Error getting symbols from data_collector: {e}")
        # Fallback to config
        st.info("[DEBUG] Falling back to config for symbols")
        trading_config = self.config.get('trading', {})
        data_config = trading_config.get('data', {})
        if data_config.get('mode') == 'manual':
            return data_config.get('manual_symbols', [])
        elif data_config.get('mode') == 'auto':
            # If auto mode but collector not available, fallback to manual_symbols or default
            return data_config.get('manual_symbols', ['RELIANCE.NS', 'TCS.NS', 'HDFCBANK.NS', 'INFY.NS', 'ICICIBANK.NS'])
        return trading_config.get('symbols', ['RELIANCE.NS', 'TCS.NS', 'HDFCBANK.NS', 'INFY.NS', 'ICICIBANK.NS'])

    def _fetch_and_display_live_prices(self, symbols: List[str] = None):
        """Display live prices with WebSocket data streaming - continuous updates without page refresh"""
        try:
            # Early return if not properly initialized
            if symbols is None or len(symbols) == 0:
                st.info("No symbols selected for monitoring")
                return
                
            if not self.data_collector:
                st.error("Data collector not available!")
                return

            # Check market status 
            current_time = datetime.now()
            market_open_time = current_time.replace(hour=9, minute=15, second=0, microsecond=0)
            market_close_time = current_time.replace(hour=15, minute=30, second=0, microsecond=0)
            is_market_open = market_open_time <= current_time <= market_close_time
            
            # Try to get WebSocket instance
            websocket_instance = None
            try:
                if hasattr(self.data_collector, 'websocket') and self.data_collector.websocket:
                    websocket_instance = self.data_collector.websocket
                else:
                    from data.websocket import get_websocket_instance
                    websocket_instance = get_websocket_instance()
            except Exception as e:
                logger.debug(f"[WEBSOCKET] Could not access websocket: {e}")
            
            # Create the streaming interface
            if websocket_instance:
                st.markdown("### 🔴 Live WebSocket Stream")
                st.caption("Continuous price updates from WebSocket feed")
                self._create_live_streaming_interface(symbols, websocket_instance, is_market_open)
            else:
                st.markdown("### 📊 Market Data (API)")
                st.warning("WebSocket not available - Using API fallback")
                self._display_api_fallback(symbols, is_market_open)
                        
        except Exception as e:
            logger.error(f"Error in live streaming setup: {str(e)}", exc_info=True)
            st.error("Live streaming setup failed")

    def _create_live_streaming_interface(self, symbols: List[str], websocket_instance, is_market_open: bool):
        """Create a live streaming interface that continuously updates from WebSocket data"""
        try:
            # Display status
            status_cols = st.columns([2, 2, 1])
            with status_cols[0]:
                if is_market_open:
                    st.success("🟢 Market OPEN")
                else:
                    st.info("🔴 Market CLOSED")
            
            with status_cols[1]:
                connection_status = getattr(websocket_instance, 'connection_state', 'unknown')
                if connection_status == 'connected':
                    st.success("🔴 LIVE STREAM")
                else:
                    st.warning(f"🔶 {connection_status.upper()}")
            
            with status_cols[2]:
                # This will be updated by the streaming loop
                time_placeholder = st.empty()
            
            # Create streaming containers for each symbol
            cols = st.columns(len(symbols))
            placeholders = {}
            
            for i, symbol in enumerate(symbols):
                with cols[i]:
                    symbol_name = symbol.replace('.NS', '')
                    
                    # Create containers that will be updated continuously
                    header_container = st.container()
                    price_container = st.container()
                    status_container = st.container()
                    time_container = st.container()
                    
                    # Initialize display
                    with header_container:
                        st.subheader(f"🔴 {symbol_name}")
                    
                    # Create empty placeholders for live updates
                    price_placeholder = price_container.empty()
                    status_placeholder = status_container.empty()
                    timestamp_placeholder = time_container.empty()
                    
                    # Store placeholders for updates
                    placeholders[symbol] = {
                        'price': price_placeholder,
                        'status': status_placeholder,
                        'timestamp': timestamp_placeholder
                    }
                    
                    # Initialize with placeholder data
                    price_placeholder.metric("Price", "₹ --")
                    status_placeholder.caption("🔴 Connecting...")
                    timestamp_placeholder.caption("⏰ --:--:--")
            
            # Streaming loop - this will continuously update the placeholders
            st.markdown("---")
            stream_status = st.empty()
            update_counter = st.empty()
            
            # Initialize streaming state
            if 'streaming_active' not in st.session_state:
                st.session_state.streaming_active = True
                st.session_state.update_count = 0
                st.session_state.last_update = time.time()
            
            # Button to control streaming
            col1, col2 = st.columns([1, 4])
            with col1:
                if st.button("⏹️ Stop Stream" if st.session_state.streaming_active else "▶️ Start Stream"):
                    st.session_state.streaming_active = not st.session_state.streaming_active
                    st.rerun()
            
            with col2:
                st.caption("Stream will update continuously while active")
            
            # Main streaming loop
            if st.session_state.streaming_active:
                self._run_streaming_loop(
                    symbols, 
                    websocket_instance, 
                    placeholders, 
                    time_placeholder,
                    stream_status,
                    update_counter
                )
            else:
                stream_status.info("🔴 Streaming paused")
                
        except Exception as e:
            logger.error(f"Error creating live streaming interface: {e}")
            st.error("Failed to create streaming interface")

    def _run_streaming_loop(self, symbols, websocket_instance, placeholders, time_placeholder, stream_status, update_counter):
        """Run the continuous streaming loop that updates UI"""
        try:
            # Check if we have WebSocket data
            if not hasattr(websocket_instance, 'live_feed') or not websocket_instance.live_feed:
                stream_status.warning("🔶 WebSocket connected but no data received yet")
                return
            
            current_time = datetime.now()
            time_placeholder.caption(f"⏰ {current_time.strftime('%H:%M:%S')}")
            
            # Update each symbol
            symbols_updated = 0
            for symbol in symbols:
                try:
                    if symbol not in placeholders:
                        continue
                    
                    # Try to find data for this symbol in WebSocket feed
                    live_data = None
                    symbol_variants = [
                        symbol, 
                        symbol.replace('.NS', ''), 
                        symbol.upper(), 
                        symbol.lower()
                    ]
                    
                    for variant in symbol_variants:
                        if variant in websocket_instance.live_feed:
                            live_data = websocket_instance.live_feed[variant]
                            break
                    
                    if live_data and isinstance(live_data, dict):
                        # Extract price information
                        price = float(live_data.get('ltp', live_data.get('last_price', 0)))
                        
                        if price > 0:
                            # Calculate change from previous price
                            prev_price_key = f"prev_price_{symbol}"
                            change = 0
                            change_pct = 0
                            
                            if prev_price_key in st.session_state:
                                prev_price = st.session_state[prev_price_key]
                                if prev_price > 0:
                                    change = price - prev_price
                                    change_pct = (change / prev_price) * 100
                            
                            # Store current price for next comparison
                            st.session_state[prev_price_key] = price
                            
                            # Format change display
                            delta_color = "normal" if change > 0 else "inverse" if change < 0 else "off"
                            change_text = f"{change:+.2f} ({change_pct:+.2f}%)" if change != 0 else None
                            
                            # Update the price display
                            placeholders[symbol]['price'].metric(
                                "Price",
                                f"₹{price:,.2f}",
                                delta=change_text,
                                delta_color=delta_color
                            )
                            
                            # Update status
                            placeholders[symbol]['status'].caption("🔴 LIVE WebSocket")
                            
                            # Update timestamp
                            timestamp = live_data.get('timestamp')
                            if timestamp and hasattr(timestamp, 'strftime'):
                                time_str = timestamp.strftime('%H:%M:%S')
                            else:
                                time_str = current_time.strftime('%H:%M:%S')
                            
                            placeholders[symbol]['timestamp'].caption(f"⏰ {time_str}")
                            
                            symbols_updated += 1
                            logger.debug(f"[STREAM] Updated {symbol}: ₹{price:.2f}")
                        else:
                            # Price is 0 or invalid
                            placeholders[symbol]['price'].metric("Price", "₹ --")
                            placeholders[symbol]['status'].caption("🔴 LIVE (No Price)")
                    else:
                        # No data for this symbol
                        placeholders[symbol]['status'].caption("🔴 LIVE (Waiting...)")
                        
                except Exception as e:
                    logger.error(f"Error updating {symbol}: {e}")
                    placeholders[symbol]['status'].caption("❌ Update Error")
            
            # Update streaming status
            st.session_state.update_count += 1
            st.session_state.last_update = time.time()
            
            stream_status.success(f"🔴 STREAMING ACTIVE - {symbols_updated}/{len(symbols)} symbols updated")
            update_counter.caption(f"Updates: {st.session_state.update_count} | Last: {current_time.strftime('%H:%M:%S')}")
            
            # Show WebSocket feed debug info
            with st.expander("🔧 WebSocket Feed Debug", expanded=False):
                st.write(f"**Total symbols in feed:** {len(websocket_instance.live_feed)}")
                st.write(f"**Available symbols:** {list(websocket_instance.live_feed.keys())}")
                st.write(f"**Connection state:** {getattr(websocket_instance, 'connection_state', 'unknown')}")
                
                if websocket_instance.live_feed:
                    st.write("**Sample data:**")
                    # Show sample data from first available symbol
                    first_key = list(websocket_instance.live_feed.keys())[0]
                    sample_data = websocket_instance.live_feed[first_key]
                    if isinstance(sample_data, dict):
                        st.json({k: v for k, v in sample_data.items() if k in ['token', 'ltp', 'timestamp', 'volume']})
            
            # Auto-refresh for continuous streaming
            time.sleep(1)  # 1 second interval
            st.rerun()
            
        except Exception as e:
            logger.error(f"Error in streaming loop: {e}")
            stream_status.error(f"❌ Streaming error: {str(e)}")
            st.session_state.streaming_active = False

    def _display_api_fallback(self, symbols: List[str], is_market_open: bool):
        """Display market data using API calls when WebSocket is not available"""
        try:
            # Display status bar
            current_time = datetime.now()
            status_cols = st.columns([2, 2, 1])
            with status_cols[0]:
                if is_market_open:
                    st.success("� Market OPEN")
                else:
                    st.info("🔴 Market CLOSED")
            
            with status_cols[1]:
                st.warning("📡 API Mode")
            
            with status_cols[2]:
                st.caption(f"⏰ {current_time.strftime('%H:%M:%S')}")
            
            # Create price display columns
            cols = st.columns(len(symbols))
            
            for i, symbol in enumerate(symbols):
                with cols[i]:
                    try:
                        if hasattr(self.data_collector, 'get_live_quote'):
                            price_data = self.data_collector.get_live_quote(symbol)
                            st.subheader(symbol.replace('.NS', ''))
                            
                            if price_data and isinstance(price_data, dict) and 'ltp' in price_data and price_data['ltp'] > 0:
                                price = float(price_data['ltp'])
                                change = float(price_data.get('change', 0))
                                change_pct = float(price_data.get('change_percent', 0))
                                
                                delta_color = "normal" if change > 0 else "inverse" if change < 0 else "off"
                                change_text = f"{change:+.2f} ({change_pct:+.2f}%)" if change != 0 else None
                                
                                st.metric(
                                    "Price",
                                    f"₹{price:,.2f}",
                                    delta=change_text,
                                    delta_color=delta_color
                                )
                                
                                st.caption("📊 API Data")
                            else:
                                st.metric("Price", "₹ --")
                                st.caption("⚠️ No data")
                        else:
                            st.subheader(symbol.replace('.NS', ''))
                            st.metric("Price", "₹ --")
                            st.caption("⚠️ No API")
                    except Exception as e:
                        st.subheader(symbol.replace('.NS', ''))
                        st.metric("Price", "₹ --")
                        st.caption("❌ API Error")
        except Exception as e:
            logger.error(f"Error in API fallback display: {e}")
            st.error("Error displaying market data")

    def _setup_live_streaming_for_symbol(self, symbol, containers, websocket_instance):
        """Setup live streaming for a specific symbol"""
        try:
            # Get initial data if available
            live_data = None
            if hasattr(websocket_instance, 'live_feed') and websocket_instance.live_feed:
                symbol_variants = [symbol, symbol.replace('.NS', ''), symbol.upper(), symbol.lower()]
                for variant in symbol_variants:
                    if variant in websocket_instance.live_feed:
                        live_data = websocket_instance.live_feed[variant]
                        break
            
            if live_data and isinstance(live_data, dict):
                # Display live WebSocket data
                containers['header'].subheader(f"🔴 {symbol.replace('.NS', '')} LIVE")
                
                price = float(live_data.get('ltp', live_data.get('last_price', 0)))
                change = float(live_data.get('change', live_data.get('chg', 0)))
                change_pct = float(live_data.get('change_percent', live_data.get('chg_per', 0)))
                
                if price > 0:
                    # Color logic
                    delta_color = "normal" if change > 0 else "inverse" if change < 0 else "off"
                    change_text = f"{change:+.2f} ({change_pct:+.2f}%)" if change != 0 else None
                    
                    with containers['price']:
                        st.metric(
                            "Price",
                            f"₹{price:,.2f}",
                            delta=change_text,
                            delta_color=delta_color
                        )
                    
                    containers['status'].caption("🔴 LIVE WebSocket Stream")
                    
                    # Timestamp
                    timestamp = live_data.get('timestamp', live_data.get('exchange_timestamp'))
                    if timestamp and isinstance(timestamp, (int, float)):
                        dt = datetime.fromtimestamp(timestamp)
                        containers['timestamp'].caption(f"⏰ {dt.strftime('%H:%M:%S')}")
                else:
                    containers['price'].metric("Price", "₹ --")
                    containers['status'].caption("🔴 LIVE Stream (No Price)")
            else:
                # No live data yet, show placeholder
                containers['header'].subheader(f"🔴 {symbol.replace('.NS', '')} LIVE")
                containers['price'].metric("Price", "₹ --")
                containers['status'].caption("🔴 Connecting to stream...")
                
        except Exception as e:
            logger.error(f"Error setting up streaming for {symbol}: {e}")
            containers['header'].subheader(symbol.replace('.NS', ''))
            containers['price'].metric("Price", "₹ --")
            containers['status'].caption("❌ Stream Error")

    def _display_static_price(self, symbol, containers):
        """Display static price data as fallback"""
        try:
            if hasattr(self.data_collector, 'get_live_quote'):
                price_data = self.data_collector.get_live_quote(symbol)
                containers['header'].subheader(symbol.replace('.NS', ''))
                
                if price_data and isinstance(price_data, dict) and 'ltp' in price_data and price_data['ltp'] > 0:
                    price = float(price_data['ltp'])
                    change = float(price_data.get('change', 0))
                    change_pct = float(price_data.get('change_percent', 0))
                    
                    delta_color = "normal" if change > 0 else "inverse" if change < 0 else "off"
                    change_text = f"{change:+.2f} ({change_pct:+.2f}%)" if change != 0 else None
                    
                    with containers['price']:
                        st.metric(
                            "Price",
                            f"₹{price:,.2f}",
                            delta=change_text,
                            delta_color=delta_color
                        )
                    
                    containers['status'].caption("📊 API Data")
                else:
                    containers['price'].metric("Price", "₹ --")
                    containers['status'].caption("⚠️ No data")
            else:
                containers['header'].subheader(symbol.replace('.NS', ''))
                containers['price'].metric("Price", "₹ --")
                containers['status'].caption("⚠️ No API")
        except Exception as e:
            containers['header'].subheader(symbol.replace('.NS', ''))
            containers['price'].metric("Price", "₹ --")
            containers['status'].caption("❌ API Error")

    def _websocket_live_update_callback(self, message):
        """Callback function that updates UI in real-time when WebSocket receives data"""
        try:
            # This function will be called every time WebSocket receives data
            # Parse the message and update the appropriate containers
            if isinstance(message, dict) and 'symbol' in message:
                symbol = message['symbol']
                
                # Check if we have a container for this symbol
                if ('live_price_containers' in st.session_state and 
                    symbol in st.session_state.live_price_containers):
                    
                    containers = st.session_state.live_price_containers[symbol]
                    
                    # Update price in real-time
                    price = float(message.get('ltp', message.get('last_price', 0)))
                    change = float(message.get('change', message.get('chg', 0)))
                    change_pct = float(message.get('change_percent', message.get('chg_per', 0)))
                    
                    if price > 0:
                        delta_color = "normal" if change > 0 else "inverse" if change < 0 else "off"
                        change_text = f"{change:+.2f} ({change_pct:+.2f}%)" if change != 0 else None
                        
                        # Update the price container in real-time
                        with containers['price']:
                            st.metric(
                                "Price",
                                f"₹{price:,.2f}",
                                delta=change_text,
                                delta_color=delta_color
                            )
                        
                        # Update timestamp
                        current_time = datetime.now()
                        containers['timestamp'].caption(f"⏰ {current_time.strftime('%H:%M:%S')}")
                        
                        logger.info(f"[LIVE_STREAM] Updated {symbol}: ₹{price:,.2f}")
                        
        except Exception as e:
            logger.error(f"Error in live update callback: {e}")

    def _create_websocket_streaming_ui(self, symbols: List[str]):
        """Create a true WebSocket streaming UI using continuous data polling"""
        try:
            # Check if WebSocket is available
            websocket_instance = None
            try:
                if hasattr(self.data_collector, 'websocket') and self.data_collector.websocket:
                    websocket_instance = self.data_collector.websocket
                else:
                    from data.websocket import get_websocket_instance
                    websocket_instance = get_websocket_instance()
            except Exception as e:
                logger.debug(f"[WEBSOCKET] Could not access websocket: {e}")
                st.error("WebSocket connection not available for live streaming")
                return
            
            if not websocket_instance:
                st.error("No WebSocket instance available")
                return
            
            # Status indicator
            status_container = st.container()
            with status_container:
                cols = st.columns([1, 1, 2])
                with cols[0]:
                    if hasattr(websocket_instance, 'is_connected') and websocket_instance.is_connected:
                        st.success("🔴 LIVE STREAMING")
                    else:
                        st.warning("🔶 CONNECTING...")
                with cols[1]:
                    current_time = datetime.now()
                    st.caption(f"⏰ {current_time.strftime('%H:%M:%S')}")
                with cols[2]:
                    st.caption("WebSocket live data feed - Updates as data arrives")
            
            # Create price display area with empty placeholders
            price_container = st.container()
            
            with price_container:
                cols = st.columns(len(symbols))
                
                # Create placeholders for each symbol
                placeholders = {}
                for i, symbol in enumerate(symbols):
                    with cols[i]:
                        symbol_key = symbol.replace('.NS', '')
                        
                        # Create placeholders that we'll update
                        header_ph = st.empty()
                        price_ph = st.empty()
                        status_ph = st.empty()
                        time_ph = st.empty()
                        
                        placeholders[symbol] = {
                            'header': header_ph,
                            'price': price_ph,
                            'status': status_ph,
                            'timestamp': time_ph
                        }
                        
                        # Initialize display
                        header_ph.subheader(f"🔴 {symbol_key} LIVE")
                        price_ph.metric("Price", "₹ --")
                        status_ph.caption("🔴 Connecting to stream...")
                        time_ph.caption("⏰ --:--:--")
            
            # Continuous streaming loop using WebSocket data
            streaming_container = st.empty()
            
            def start_streaming():
                """Start the continuous streaming process"""
                try:
                    update_count = 0
                    last_update_time = time.time()
                    
                    while True:
                        current_time = time.time()
                        
                        # Update every 0.5 seconds for near real-time feel
                        if current_time - last_update_time >= 0.5:
                            
                            # Check if we still have WebSocket data
                            if not hasattr(websocket_instance, 'live_feed') or not websocket_instance.live_feed:
                                # No data yet, keep trying
                                for symbol in symbols:
                                    if symbol in placeholders:
                                        placeholders[symbol]['status'].caption("🔴 Waiting for data...")
                                time.sleep(0.5)
                                continue
                            
                            # Update each symbol with latest WebSocket data
                            for symbol in symbols:
                                try:
                                    if symbol not in placeholders:
                                        continue
                                    
                                    # Try to find data for this symbol
                                    live_data = None
                                    symbol_variants = [
                                        symbol, 
                                        symbol.replace('.NS', ''), 
                                        symbol.upper(), 
                                        symbol.lower()
                                    ]
                                    
                                    for variant in symbol_variants:
                                        if variant in websocket_instance.live_feed:
                                            live_data = websocket_instance.live_feed[variant]
                                            break
                                    
                                    if live_data and isinstance(live_data, dict):
                                        # Extract price data
                                        price = float(live_data.get('ltp', live_data.get('last_price', 0)))
                                        
                                        if price > 0:
                                            # Calculate change if we have previous price
                                            change = 0
                                            change_pct = 0
                                            prev_key = f"prev_price_{symbol}"
                                            
                                            if prev_key in st.session_state:
                                                prev_price = st.session_state[prev_key]
                                                change = price - prev_price
                                                change_pct = (change / prev_price) * 100 if prev_price > 0 else 0
                                            
                                            # Store current price for next comparison
                                            st.session_state[prev_key] = price
                                            
                                            # Update UI elements
                                            delta_color = "normal" if change > 0 else "inverse" if change < 0 else "off"
                                            change_text = f"{change:+.2f} ({change_pct:+.2f}%)" if change != 0 else None
                                            
                                            # Update price display
                                            placeholders[symbol]['price'].metric(
                                                "Price",
                                                f"₹{price:,.2f}",
                                                delta=change_text,
                                                delta_color=delta_color
                                            )
                                            
                                            # Update status and timestamp
                                            placeholders[symbol]['status'].caption("🔴 LIVE WebSocket")
                                            
                                            # Use WebSocket timestamp if available
                                            timestamp = live_data.get('timestamp')
                                            if timestamp and hasattr(timestamp, 'strftime'):
                                                time_str = timestamp.strftime('%H:%M:%S')
                                            else:
                                                time_str = datetime.now().strftime('%H:%M:%S')
                                            
                                            placeholders[symbol]['timestamp'].caption(f"⏰ {time_str}")
                                            
                                            update_count += 1
                                            logger.debug(f"[STREAM] Updated {symbol}: ₹{price:.2f}")
                                        else:
                                            # No valid price
                                            placeholders[symbol]['price'].metric("Price", "₹ --")
                                            placeholders[symbol]['status'].caption("🔴 LIVE (No Price)")
                                    else:
                                        # No data for this symbol
                                        placeholders[symbol]['status'].caption("🔴 LIVE (Waiting...)")
                                        
                                except Exception as e:
                                    logger.error(f"Error updating {symbol}: {e}")
                                    placeholders[symbol]['status'].caption("❌ Update Error")
                            
                            last_update_time = current_time
                            
                        # Small sleep to prevent CPU overload
                        time.sleep(0.1)
                        
                        # Break after some updates for demo (remove in production)
                        if update_count > 100:  # Limit for demo purposes
                            break
                            
                except Exception as e:
                    logger.error(f"Error in streaming loop: {e}")
                    st.error("Streaming loop error")
            
            # Show streaming status
            connection_status = getattr(websocket_instance, 'connection_state', 'unknown')
            if connection_status == 'connected':
                st.success("🔗 WebSocket Connected - Starting live stream...")
                
                # Start streaming in background (note: this won't work in Streamlit's normal mode)
                # Instead, we'll use a different approach with session state
                st.info("💡 **Note:** This demonstrates the streaming concept. In practice, you would need to use Streamlit's `st.fragment(run_every)` or external streaming service.")
                
                # Show current data as a snapshot
                self._show_current_websocket_data(symbols, websocket_instance)
                
            else:
                st.warning(f"🔗 WebSocket Status: {connection_status}")
                st.error("Cannot start streaming - WebSocket not connected")
                
        except Exception as e:
            logger.error(f"Error setting up WebSocket streaming UI: {e}")
            st.error("Failed to set up live streaming interface")

    def _show_current_websocket_data(self, symbols: List[str], websocket_instance):
        """Show current WebSocket data as a snapshot"""
        try:
            st.markdown("#### Current WebSocket Data")
            
            if not hasattr(websocket_instance, 'live_feed') or not websocket_instance.live_feed:
                st.info("No live data available yet from WebSocket")
                return
                
            cols = st.columns(len(symbols))
            
            for i, symbol in enumerate(symbols):
                with cols[i]:
                    symbol_key = symbol.replace('.NS', '')
                    
                    # Try to find data for this symbol
                    live_data = None
                    symbol_variants = [symbol, symbol.replace('.NS', ''), symbol.upper(), symbol.lower()]
                    
                    for variant in symbol_variants:
                        if variant in websocket_instance.live_feed:
                            live_data = websocket_instance.live_feed[variant]
                            break
                    
                    if live_data and isinstance(live_data, dict):
                        price = float(live_data.get('ltp', live_data.get('last_price', 0)))
                        
                        if price > 0:
                            st.metric(f"{symbol_key}", f"₹{price:.2f}")
                            st.caption("🔴 Live from WebSocket")
                            
                            # Show timestamp if available
                            timestamp = live_data.get('timestamp')
                            if timestamp and hasattr(timestamp, 'strftime'):
                                st.caption(f"⏰ {timestamp.strftime('%H:%M:%S')}")
                        else:
                            st.metric(f"{symbol_key}", "₹ --")
                            st.caption("🔴 Connected (No Price)")
                    else:
                        st.metric(f"{symbol_key}", "₹ --") 
                        st.caption("⚠️ No data for symbol")
            
            # Show raw WebSocket feed info
            with st.expander("🔧 WebSocket Feed Debug"):
                st.write(f"**Total symbols in feed:** {len(websocket_instance.live_feed)}")
                st.write(f"**Available symbols:** {list(websocket_instance.live_feed.keys())}")
                
                if websocket_instance.live_feed:
                    # Show sample data from first symbol
                    first_symbol = list(websocket_instance.live_feed.keys())[0]
                    sample_data = websocket_instance.live_feed[first_symbol]
                    st.json(sample_data)
                    
        except Exception as e:
            logger.error(f"Error showing current WebSocket data: {e}")
            st.error("Error displaying WebSocket data")
        except Exception as e:
            logger.error(f"Error in fragment: {str(e)}", exc_info=True)
            # Show minimal error in UI to avoid session issues
            st.error("Live data temporarily unavailable")
        """Display live prices with WebSocket streaming - NO UI REFRESH"""
        try:
            if not self.data_collector:
                st.error("Data collector not available!")
                return
            
            if symbols is None:
                symbols = self._get_symbols()
            
            # Check market status (one time only)
            current_time = datetime.now()
            market_open_time = current_time.replace(hour=9, minute=15, second=0, microsecond=0)
            market_close_time = current_time.replace(hour=15, minute=30, second=0, microsecond=0)
            is_market_open = market_open_time <= current_time <= market_close_time
            
            # Display market status
            if is_market_open:
                st.success("🟢 Market is OPEN - Live WebSocket streaming")
            else:
                st.info("🔴 Market is CLOSED - Showing Last Traded Prices (LTP)")
            
            # Initialize WebSocket connection if available
            websocket_data = {}
            has_websocket = False
            
            # Try to get WebSocket data from data collector
            if hasattr(self.data_collector, 'websocket') and self.data_collector.websocket:
                try:
                    # Check if websocket has live_feed attribute
                    if hasattr(self.data_collector.websocket, 'live_feed'):
                        websocket_data = dict(self.data_collector.websocket.live_feed)
                        has_websocket = len(websocket_data) > 0
                        if has_websocket:
                            st.success("🔴 WebSocket data available!")
                            logger.info(f"[WEBSOCKET] Retrieved {len(websocket_data)} symbols from live_feed")
                    else:
                        logger.warning("[WEBSOCKET] live_feed attribute not found on websocket")
                except Exception as e:
                    logger.warning(f"[WEBSOCKET] Error accessing websocket data: {e}")
            
            # If no WebSocket, check for direct instance
            try:
                from data.websocket import get_websocket_instance
                ws_instance = get_websocket_instance()
                if ws_instance and hasattr(ws_instance, 'live_feed'):
                    websocket_data = dict(ws_instance.live_feed)
                    has_websocket = len(websocket_data) > 0
                    if has_websocket:
                        st.success("🔴 Global WebSocket data available!")
                        logger.info(f"[WEBSOCKET] Retrieved {len(websocket_data)} symbols from global instance")
            except Exception as e:
                logger.debug(f"[WEBSOCKET] Could not access global websocket: {e}")
            
            # Create columns for symbols
            cols = st.columns(len(symbols))
            
            for i, symbol in enumerate(symbols):
                with cols[i]:
                    # Check if we have live WebSocket data for this symbol
                    live_data = None
                    is_streaming = False
                    
                    # Try to find WebSocket data for this symbol
                    if has_websocket:
                        # Try different symbol formats
                        symbol_variants = [symbol, symbol.replace('.NS', ''), symbol.upper(), symbol.lower()]
                        for variant in symbol_variants:
                            if variant in websocket_data:
                                live_data = websocket_data[variant]
                                is_streaming = True
                                logger.info(f"[WEBSOCKET] Found live data for {symbol} as {variant}")
                                break
                    
                    if is_streaming and live_data:
                        # Use WebSocket streaming data
                        st.subheader(f"🔴 {symbol.replace('.NS', '')} LIVE")
                        
                        # Extract price data from WebSocket
                        if isinstance(live_data, dict):
                            price = live_data.get('ltp', live_data.get('last_price', 0))
                            change = live_data.get('change', 0)
                            change_pct = live_data.get('change_percent', live_data.get('chg_per', 0))
                        else:
                            price = change = change_pct = 0
                            
                        if price > 0:
                            # Display WebSocket price
                            if change > 0:
                                delta_color = "normal"
                            elif change < 0:
                                delta_color = "inverse"
                            else:
                                delta_color = "off"
                            
                            # Format change text
                            if change != 0:
                                change_text = f"{change:+.2f} ({change_pct:+.2f}%)"
                            else:
                                change_text = "No change"
                            
                            st.metric(
                                "Price",
                                f"₹{price:,.2f}",
                                delta=change_text if change != 0 else None,
                                delta_color=delta_color
                            )
                            
                            st.caption("🔴 LIVE WebSocket Stream")
                            
                            # Show timestamp if available
                            timestamp = live_data.get('timestamp', live_data.get('exchange_timestamp'))
                            if timestamp:
                                if isinstance(timestamp, str):
                                    st.caption(f"⏰ {timestamp}")
                                elif isinstance(timestamp, (int, float)):
                                    dt = datetime.fromtimestamp(timestamp)
                                    st.caption(f"⏰ {dt.strftime('%H:%M:%S')}")
                        else:
                            st.metric("Price", "₹ --")
                            st.caption("� LIVE WebSocket Stream (No Price)")
                    else:
                        # Fallback to API call
                        try:
                            if hasattr(self.data_collector, 'get_live_quote'):
                                price_data = self.data_collector.get_live_quote(symbol)
                            else:
                                # Try direct quote method
                                price_data = None
                                st.subheader(symbol.replace('.NS', ''))
                                st.metric("Price", "₹ --")
                                st.caption("⚠️ API method not available")
                                continue
                                
                            st.subheader(symbol.replace('.NS', ''))
                            
                            if price_data and 'ltp' in price_data and price_data['ltp'] > 0:
                                price = float(price_data['ltp'])
                                change = float(price_data.get('change', 0))
                                change_pct = float(price_data.get('change_percent', 0))
                                
                                # Color based on change
                                if change > 0:
                                    delta_color = "normal"
                                elif change < 0:
                                    delta_color = "inverse"
                                else:
                                    delta_color = "off"
                                
                                # Change text
                                if change != 0:
                                    change_text = f"{change:+.2f} ({change_pct:+.2f}%)"
                                else:
                                    change_text = "No change"
                                
                                st.metric(
                                    "Price",
                                    f"₹{price:,.2f}",
                                    delta=change_text if change != 0 else None,
                                    delta_color=delta_color
                                )
                                
                                st.caption("� Live data")
                                
                                # Show timestamp
                                timestamp = price_data.get('timestamp')
                                if timestamp:
                                    if isinstance(timestamp, str):
                                        st.caption(f"⏰ {timestamp}")
                                    elif isinstance(timestamp, datetime):
                                        time_str = timestamp.strftime('%H:%M:%S')
                                        st.caption(f"⏰ {time_str}")
                            else:
                                st.metric("Price", "₹ --")
                                if is_market_open:
                                    st.caption("⚠️ Waiting for data...")
                                else:
                                    st.caption("⚠️ No data available")
                                    
                        except Exception as e:
                            st.subheader(symbol.replace('.NS', ''))
                            st.error(f"Error: {str(e)}")
                            continue
            
            # Show debug info if WebSocket found
            if has_websocket:
                with st.expander("WebSocket Debug Info", expanded=False):
                    st.write(f"WebSocket symbols found: {list(websocket_data.keys())}")
                    st.write(f"Total symbols: {len(websocket_data)}")
                    for symbol, data in websocket_data.items():
                        st.write(f"{symbol}: {data}")
            
            # Add refresh control (user can click to update manually)
            col1, col2, col3 = st.columns([1, 1, 2])
            with col1:
                if st.button("🔄 Refresh Prices", help="Click to refresh live prices"):
                    st.rerun()
            with col2:
                st.write(f"Last update: {datetime.now().strftime('%H:%M:%S')}")
            with col3:
                if has_websocket:
                    st.success("🔴 WebSocket Active")
                else:
                    st.warning("📡 API Only")
                        
        except Exception as e:
            logger.error(f"Error in _fetch_and_display_live_prices: {str(e)}", exc_info=True)
            st.error(f"Error updating prices: {str(e)}")

    def render_trading_tab(self):
        """Render the trading tab with live price streaming"""
        try:
            st.header("Trading Dashboard")
            
            # Get symbols
            all_symbols = self._get_symbols()
            
            # Searchable multi-select for all stocks
            monitored_symbols = st.multiselect(
                "Select stocks to monitor (search and select)",
                options=all_symbols,
                default=all_symbols[:4],  # Default to first 4 symbols
                help="Search and select stocks to monitor live. All stocks are available for training."
            )
            
            if not monitored_symbols:
                st.info("No stocks selected for monitoring. Use the search box above to add.")
            else:
                # Call the fragment for live price streaming
                # This will auto-update every 2 seconds
                self._fetch_and_display_live_prices(monitored_symbols)

            # Trading mode: always live
            st.subheader("Trading Mode")
            st.success("LIVE Trading mode is ACTIVE. Real trades will be placed if you proceed.")

            # Trading controls
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Trading Controls")
                is_active = st.toggle("Enable Trading", value=False)
                if is_active:
                    st.success("Trading active")
                else:
                    st.warning("Trading paused")
            with col2:
                st.subheader("Risk Settings")
                trading_config = self.config.get('trading', {})
                max_position = st.number_input(
                    "Max Position Size",
                    min_value=0.0,
                    max_value=1.0,
                    value=float(trading_config.get('max_position_size', 0.1)),
                    step=0.01,
                    format="%.2f"
                )
            # --- Fully Autonomous Mode Toggle ---
            st.subheader("Autonomous AI Trading")
            autonomous_mode = not st.toggle(
                "Disable Fully Autonomous Mode (AI + RL + News)",
                value=False,
                help="AI is ON by default. Toggle to turn OFF autonomous trading."
            )
            if autonomous_mode:
                st.success("Fully Autonomous Trading is ENABLED. The AI will make and execute all trading decisions.")
                # Start autonomous trader if not already running
                if not self.autonomous_trader:
                    self.autonomous_trader = AutonomousTrader(
                        self.config,
                        self.data_collector,
                        self.trade_manager,
                        self.technical_analyzer,
                        self.ai_model
                    )
                if not self.autonomous_thread or not self.autonomous_thread.is_alive():
                    self.autonomous_trader.set_autonomous_mode(True)
                    self.autonomous_thread = threading.Thread(target=self.autonomous_trader.run, daemon=True)
                    self.autonomous_thread.start()
                # --- Transparency: Show latest signals, ensemble, and reasoning ---
                st.subheader("AI Model & Sentiment Signals (Explainability)")
                if hasattr(self.autonomous_trader, 'last_reasoning'):
                    for symbol, info in self.autonomous_trader.last_reasoning.items():
                        st.markdown(f"**{symbol}**")
                        st.json(info)
                else:
                    st.info("No trade reasoning available yet. Will appear after first autonomous trade.")
            else:
                st.info("Manual or semi-automatic mode. You control trade execution.")
                # Stop autonomous trader if running
                if self.autonomous_trader:
                    self.autonomous_trader.set_autonomous_mode(False)

            # Display positions if any
            if self.trade_manager:
                positions = self.trade_manager.get_positions()
                if positions:
                    st.subheader("Open Positions")
                    position_df = pd.DataFrame(positions)
                    st.dataframe(position_df, use_container_width=True)
                else:
                    st.info("No open positions")
        except Exception as e:
            logger.error(f"Error in trading tab: {str(e)}")
            st.error(f"[DEBUG] Error in trading tab: {str(e)}")
            import traceback
            st.text(traceback.format_exc())
    
    def render_analysis_tab(self):
        import streamlit as st
        st.info("[DEBUG] render_analysis_tab() called")
        try:
            st.header("Technical Analysis Dashboard")
            
            # Check for technical analyzer
            if not self.technical_analyzer:
                st.warning("Technical analysis module is not initialized")
                return
                
            # Get configured symbols
            trading_config = self.config.get('trading', {})
            default_symbols = ['RELIANCE.NS', 'TCS.NS', 'HDFCBANK.NS', 'INFY.NS', 'ICICIBANK.NS']
            configured_symbols = trading_config.get('symbols', default_symbols)
            
            # Symbol selection
            selected_symbol = st.selectbox("Select Stock for Analysis", configured_symbols, key="analysis_symbol")
            
            # Analysis Configuration
            st.subheader("Analysis Configuration")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                days = st.number_input("Analysis Period (Days)", min_value=1, value=30, max_value=365)
            
            with col2:
                interval = st.selectbox(
                    "Timeframe",
                    options=["1m", "5m", "15m", "30m", "1h", "1d"],
                    index=5,
                    help="Select the time interval for analysis",
                    key="analysis_interval"
                )
            
            with col3:
                selected_indicators = st.multiselect(
                    "Select Indicators",
                    options=["SMA", "EMA", "RSI", "MACD", "Bollinger Bands"],
                    default=["RSI", "MACD"],
                    help="Choose technical indicators to display"
                )
            
            try:
                with st.spinner("Loading technical analysis..."):
                    # Get historical data
                    df = self.data_collector.get_historical_data(
                        symbol=selected_symbol,
                        interval=interval,
                        days=days
                    )
                    
                    if df is None or len(df) == 0:
                        st.warning(f"No historical data available for {selected_symbol}")
                        return
                        
                    # Configure indicators
                    indicator_config = {
                        'trend_indicators': {},
                        'momentum_indicators': {},
                        'volatility_indicators': {},
                        'volume_indicators': {'volume_sma': 20}  # Always include volume
                    }
                    
                    # Map selected indicators to config
                    for indicator in selected_indicators:
                        if indicator == 'SMA':
                            indicator_config['trend_indicators']['sma'] = [10, 20, 50]
                        elif indicator == 'EMA':
                            indicator_config['trend_indicators']['ema'] = [9, 21]
                        elif indicator == 'RSI':
                            indicator_config['momentum_indicators']['rsi'] = {'period': 14}
                        elif indicator == 'MACD':
                            indicator_config['trend_indicators']['macd'] = {'fast': 12, 'slow': 26, 'signal': 9}
                        elif indicator == 'Bollinger Bands':
                            indicator_config['volatility_indicators']['bollinger_bands'] = {'period': 20, 'std_dev': 2}
                    
                    # Run technical analysis
                    analysis = self.technical_analyzer.analyze(df, indicator_config)
                    
                    if not analysis:
                        st.error("Technical analysis failed - no results returned")
                        return
                    
                    # Display results
                    self._display_technical_analysis(df, analysis, selected_indicators)
                    
            except Exception as e:
                logger.error(f"Error performing technical analysis: {str(e)}")
                st.error(f"Error performing technical analysis: {str(e)}")
                
        except Exception as e:
            logger.error(f"Error in analysis tab: {str(e)}")
            st.error(f"[DEBUG] Error in analysis tab: {str(e)}")
            import traceback
            st.text(traceback.format_exc())
    
    def render_settings_tab(self):
        import streamlit as st
        st.info("[DEBUG] render_settings_tab() called")
        try:
            st.header("System Settings")
            trading_config = self.config.get('trading', {})
            # --- Sanitize and cap values before passing to Streamlit ---
            raw_max_position = trading_config.get('max_position_size', 0.1)
            # If > 1, assume it's already percent; else convert decimal to percent
            if raw_max_position > 1.0:
                max_position_percent = float(raw_max_position)
            else:
                max_position_percent = float(raw_max_position) * 100.0
            max_position_percent = min(max(max_position_percent, 1.0), 100.0)

            stop_loss = float(trading_config.get('stop_loss', 2.0))
            stop_loss = min(max(stop_loss, 0.1), 10.0)
            take_profit = float(trading_config.get('take_profit', 5.0))
            take_profit = min(max(take_profit, 0.1), 20.0)
            max_trades = int(trading_config.get('max_trades', 3))
            max_trades = min(max(max_trades, 1), 10)

            # Trading Mode Selection
            mode = st.selectbox(
                "Trading Mode",
                options=["live"],
                index=0,
                help="Live trading only. Simulation mode is disabled.",
                key="settings_trading_mode"
            )

            # Risk Management Settings
            st.subheader("Risk Management")
            col1, col2 = st.columns(2)
            with col1:
                max_position = st.number_input(
                    "Max Position Size (%)",
                    min_value=1.0,
                    max_value=100.0,
                    value=max_position_percent,
                    step=1.0,
                    help="Maximum position size as a percentage of portfolio"
                )
                stop_loss = st.number_input(
                    "Default Stop Loss (%)",
                    min_value=0.1,
                    max_value=10.0,
                    value=stop_loss,
                    step=0.1,
                    help="Default stop loss percentage"
                )
            with col2:
                take_profit = st.number_input(
                    "Default Take Profit (%)",
                    min_value=0.1,
                    max_value=20.0,
                    value=take_profit,
                    step=0.1,
                    help="Default take profit percentage"
                )
                max_trades = st.number_input(
                    "Max Concurrent Trades",
                    min_value=1,
                    max_value=10,
                    value=max_trades,
                    help="Maximum number of concurrent trades"
                )

            # API Configuration
            st.subheader("API Configuration")
            api_settings = self.config.get('api', {})
            col1, col2 = st.columns(2)
            with col1:
                api_key = st.text_input(
                    "API Key",
                    value="●" * 10,
                    type="password"
                )
            with col2:
                api_secret = st.text_input(
                    "API Secret",
                    value="●" * 10,
                    type="password"
                )

            # Save Settings Button
            if st.button("Save Settings", key="save_settings"):
                try:
                    # Validate settings before saving
                    if max_position < 1.0 or max_position > 100.0:
                        st.error("Invalid position size. Must be between 1 and 100%")
                        return
                    if stop_loss < 0.1 or stop_loss > 10.0:
                        st.error("Invalid stop loss. Must be between 0.1 and 10%")
                        return
                    if take_profit < 0.1 or take_profit > 20.0:
                        st.error("Invalid take profit. Must be between 0.1 and 20%")
                        return
                    if max_trades < 1 or max_trades > 10:
                        st.error("Invalid max trades. Must be between 1 and 10")
                        return

                    # Update configuration
                    updated_config = {
                        'mode': mode,
                        'max_position_size': float(max_position) / 100.0,  # Store as decimal
                        'stop_loss': float(stop_loss),
                        'take_profit': float(take_profit),
                        'max_trades': int(max_trades)
                    }

                    # Update API credentials if changed
                    if api_key != "●" * 10:
                        updated_config['api_key'] = api_key
                    if api_secret != "●" * 10:
                        updated_config['api_secret'] = api_secret

                    # Update trading config
                    self.config['trading'].update(updated_config)

                    # Save to file
                    config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'config.yaml')
                    with open(config_path, 'w') as f:
                        yaml.dump(self.config, f, default_flow_style=False)

                    st.success("Settings saved successfully!")

                    # Notify components of config change
                    if self.trade_manager:
                        self.trade_manager.update_config(self.config['trading'])

                except Exception as e:
                    logger.error(f"Error saving settings: {e}")
                    st.error(f"Failed to save settings: {str(e)}")

        except Exception as e:
            logger.error(f"Error in settings tab: {e}")
            st.error(f"[DEBUG] Error in settings tab: {e}")
            import traceback
            st.text(traceback.format_exc())
    
    def render_broker_tab(self):
        """
        Show live broker portfolio, order book, real-time P&L, and trading balance from Angel One.
        Also allows manual order placement via the official API.
        """
        import streamlit as st
        st.info("[DEBUG] render_broker_tab() called")
        try:
            st.header("Broker Portfolio & Orders (Angel One)")
            # Live Holdings
            st.subheader("Live Holdings")
            try:
                broker_holdings = self.data_collector.get_broker_portfolio() if hasattr(self.data_collector, 'get_broker_portfolio') else []
            except Exception as e:
                st.error(f"Error fetching live holdings: {e}")
                broker_holdings = []
            if broker_holdings:
                st.dataframe(broker_holdings, use_container_width=True)
            else:
                st.info("No live holdings data available or API error.")
            # Live Trading Balance (Funds/Margins)
            st.subheader("Trading Balance (Funds/Margins)")
            broker_balance = self.data_collector.get_broker_balance() if hasattr(self.data_collector, 'get_broker_balance') else None
            if broker_balance:
                if broker_balance.get('error') or broker_balance.get('errorcode') or broker_balance.get('message'):
                    # Show error if any error-related key is present
                    st.error(f"Trading balance error: [{broker_balance.get('errorcode', 'UNKNOWN')}] {broker_balance.get('message', broker_balance.get('error', 'Unknown error'))}")
                elif any([broker_balance.get('available_cash'), broker_balance.get('net'), broker_balance.get('utilised_debits')]):
                    st.write({
                        'Available Cash': broker_balance.get('available_cash'),
                        'Net': broker_balance.get('net'),
                        'Utilised Debits': broker_balance.get('utilised_debits')
                    })
                else:
                    st.info("No trading balance data available.")
            else:
                st.info("No trading balance data available.")
            # Live Orders
            st.subheader("Order Book")
            broker_orders = self.data_collector.get_broker_orders() if hasattr(self.data_collector, 'get_broker_orders') else []
            if broker_orders:
                st.dataframe(broker_orders, use_container_width=True)
            else:
                st.info("No live order data available.")
            # Manual Order Placement
            st.subheader("Place Manual Order (Live)")
            with st.form("manual_order_form"):
                col1, col2, col3 = st.columns(3)
                with col1:
                    tradingsymbol = st.text_input("Trading Symbol", "RELIANCE-EQ")
                    symboltoken = st.text_input("Symbol Token", "2885")
                    exchange = st.selectbox("Exchange", ["NSE", "BSE"], index=0)
                with col2:
                    transactiontype = st.selectbox("Transaction Type", ["BUY", "SELL"], index=0)
                    ordertype = st.selectbox("Order Type", ["MARKET", "LIMIT"], index=0)
                    producttype = st.selectbox("Product Type", ["INTRADAY", "DELIVERY"], index=1)
                with col3:
                    price = st.number_input("Price", min_value=0.0, value=0.0, step=0.05, format="%.2f")
                    quantity = st.number_input("Quantity", min_value=1, value=1, step=1)
                    duration = st.selectbox("Duration", ["DAY", "IOC"], index=0)
                submitted = st.form_submit_button("Place Order")
            if submitted:
                order_params = {
                    "variety": "NORMAL",
                    "tradingsymbol": tradingsymbol,
                    "symboltoken": symboltoken,
                    "transactiontype": transactiontype,
                    "exchange": exchange,
                    "ordertype": ordertype,
                    "producttype": producttype,
                    "duration": duration,
                    "price": price,
                    "quantity": int(quantity)
                }
                result = self.data_collector.place_broker_order(order_params)
                if result.get("status"):
                    st.success(f"Order placed successfully! Order ID: {result.get('data', {}).get('orderid', 'N/A')}")
                else:
                    st.error(f"Order failed: {result.get('message', result)}")
            # Live P&L
            st.subheader("Live P&L Snapshot")
            if hasattr(self.trade_manager, 'get_portfolio_metrics'):
                metrics = self.trade_manager.get_portfolio_metrics()
                st.json(metrics)
            else:
                st.info("No live P&L data available.")
        except Exception as e:
            st.error(f"[DEBUG] Error in broker tab: {e}")
            import traceback
            st.text(traceback.format_exc())

    def render(self):
        """Main render method"""
        logger.info("=== ENTER DashboardUI.render() ===")
        print("=== ENTER DashboardUI.render() ===")
        try:
            st.title("Dashboard")
            st.info("[DEBUG] DashboardUI.render() started - if you see this, Streamlit is running.")
            # Sidebar
            with st.sidebar:
                st.header("Controls")
            # --- Persistent trade notifications at top right ---
            trade_log = None
            if self.trade_manager and hasattr(self.trade_manager, 'get_trade_log'):
                trade_log = self.trade_manager.get_trade_log()
            elif hasattr(self, 'last_sim_trade_log'):
                trade_log = self.last_sim_trade_log
            if trade_log is not None:
                self.render_trade_notifications_top_right(trade_log)
            # Main content
            tabs = st.tabs(["Trading", "Analysis", "Settings", "Broker", "Logs"])
            with tabs[0]:
                st.info("[DEBUG] Entering Trading Tab")
                logger.info("=== DashboardUI: render_trading_tab() ===")
                print("=== DashboardUI: render_trading_tab() ===")
                self.render_trading_tab()
                logger.info("=== DashboardUI: render_trading_tab() done ===")
                print("=== DashboardUI: render_trading_tab() done ===")
            with tabs[1]:
                st.info("[DEBUG] Entering Analysis Tab")
                logger.info("=== DashboardUI: render_analysis_tab() ===")
                print("=== DashboardUI: render_analysis_tab() ===")
                self.render_analysis_tab()
                logger.info("=== DashboardUI: render_analysis_tab() done ===")
                print("=== DashboardUI: render_analysis_tab() done ===")
            with tabs[2]:
                st.info("[DEBUG] Entering Settings Tab")
                logger.info("=== DashboardUI: render_settings_tab() ===")
                print("=== DashboardUI: render_settings_tab() ===")
                self.render_settings_tab()
                logger.info("=== DashboardUI: render_settings_tab() done ===")
                print("=== DashboardUI: render_settings_tab() done ===")
            with tabs[3]:
                st.info("[DEBUG] Entering Broker Tab")
                logger.info("=== DashboardUI: render_broker_tab() ===")
                print("=== DashboardUI: render_broker_tab() ===")
                self.render_broker_tab()
                logger.info("=== DashboardUI: render_broker_tab() done ===")
                print("=== DashboardUI: render_broker_tab() done ===")
            with tabs[4]:
                st.info("[DEBUG] Entering Logs Tab")
                logger.info("=== DashboardUI: render_logs_tab() ===")
                print("=== DashboardUI: render_logs_tab() ===")
                self.render_logs_tab()
                logger.info("=== DashboardUI: render_logs_tab() done ===")
                print("=== DashboardUI: render_logs_tab() done ===")
            logger.info("Dashboard render cycle complete")
        except Exception as e:
            logger.error(f"Error in dashboard render cycle: {e}")
            print(f"Error in dashboard render cycle: {e}")
            st.error(f"[FATAL] An error occurred while rendering the dashboard: {e}")
            import traceback
            st.text(traceback.format_exc())
        logger.info("=== EXIT DashboardUI.render() ===")
        print("=== EXIT DashboardUI.render() ===")
    
    def _display_technical_analysis(self, df, analysis, selected_indicators):
        """Display technical analysis results. Minimal stub to prevent errors."""
        import streamlit as st
        st.subheader("Technical Analysis Results")
        st.write("DataFrame:")
        st.dataframe(df)
        st.write("Analysis:")
        st.json(analysis)
        st.write("Indicators:")
        st.write(selected_indicators)

    def render_logs_tab(self):
        """Display recent log entries in the dashboard."""
        import streamlit as st
        import os
        from datetime import datetime
        st.header("Logs")
        # Compute log file path
        log_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), '..', 'logs')
        log_file = os.path.join(log_dir, datetime.now().strftime('%Y-%m-%d'), 'app.log')
        st.caption(f"Log file path: {log_file}")
        try:
            if not os.path.exists(log_file):
                st.info("Log file does not exist yet. No logs to display.")
                return
            with open(log_file, "r", encoding="utf-8", errors="ignore") as f:
                logs = f.read()
            if not logs.strip():
                st.info("Log file is empty.")
            else:
                st.text_area("Application Logs", logs, height=400)
        except Exception as e:
            st.error(f"Failed to load logs: {e}")
            import traceback
            st.text(traceback.format_exc())
