import streamlit as st
import plotly.graph_objects as go
from datetime import datetime
import pandas as pd
import yaml
import logging
import time
from typing import Dict, Any, Optional, List, Tuple
from functools import lru_cache
from datetime import datetime, timedelta
import threading
from collections import defaultdict
import os
from src.trading.paper_trading import PaperTradingEngine
from src.trading.strategy import EnhancedTradingStrategy
from src.autonomous_trader import AutonomousTrader

# Configure logging to prevent duplicates
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

class RateLimiter:
    """Rate limiter based on Angel One API limits"""
    
    LIMITS = {
        'getLtpData': {'per_second': 10, 'per_minute': 500, 'per_hour': 5000},
        'getCandleData': {'per_second': 3, 'per_minute': 180, 'per_hour': 5000},
        'default': {'per_second': 1, 'per_minute': None, 'per_hour': None}
    }
    
    def __init__(self):
        self.locks = defaultdict(threading.Lock)
        self.last_request_time = defaultdict(lambda: defaultdict(float))
        self.request_counts = defaultdict(lambda: defaultdict(int))
    
    def _reset_counts(self, endpoint: str, time_window: str):
        """Reset counts for a specific time window"""
        self.request_counts[endpoint][time_window] = 0
        
    def can_request(self, endpoint: str) -> Tuple[bool, float]:
        """Check if a request can be made and return wait time if needed"""
        limits = self.LIMITS.get(endpoint, self.LIMITS['default'])
        now = time.time()
        
        with self.locks[endpoint]:
            # Check second limit
            if limits['per_second']:
                second_count = self.request_counts[endpoint]['second']
                if second_count >= limits['per_second']:
                    last_time = self.last_request_time[endpoint]['second']
                    if now - last_time < 1.0:
                        return False, 1.0 - (now - last_time)
                    else:
                        self._reset_counts(endpoint, 'second')
            
            # Check minute limit
            if limits['per_minute']:
                minute_count = self.request_counts[endpoint]['minute']
                if minute_count >= limits['per_minute']:
                    last_time = self.last_request_time[endpoint]['minute']
                    if now - last_time < 60.0:
                        return False, 60.0 - (now - last_time)
                    else:
                        self._reset_counts(endpoint, 'minute')
            
            return True, 0.0
    
    def record_request(self, endpoint: str):
        """Record a successful request"""
        now = time.time()
        with self.locks[endpoint]:
            self.request_counts[endpoint]['second'] += 1
            self.request_counts[endpoint]['minute'] += 1
            self.last_request_time[endpoint]['second'] = now
            self.last_request_time[endpoint]['minute'] = now

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
        self._rate_limiter = RateLimiter()
        self._last_render = {}
        self._update_interval = 5  # seconds
        
        # Performance metrics
        self._perf_metrics = {
            'cpu_usage': [],
            'memory_usage': [],
            'api_latency': [],
            'update_times': [],
            'error_count': 0,
            'last_error': None,
            'successful_updates': 0,
            'failed_updates': 0
        }
        
        # Dashboard state
        self._state = {
            'is_trading_active': False,
            'last_update': None,
            'active_symbols': set(),
            'error_symbols': set(),
            'market_status': 'unknown'
        }
        
        # Autonomous trading
        self.autonomous_trader = None
        self.autonomous_thread = None
        
        logger.info("DashboardUI initialized")
    
    def _update_perf_metrics(self):
        """Update performance metrics"""
        try:
            import psutil
            
            # System metrics
            self._perf_metrics['cpu_usage'].append(psutil.cpu_percent())
            self._perf_metrics['memory_usage'].append(psutil.Process().memory_info().rss / 1024 / 1024)  # MB
            
            # Keep only last 100 measurements
            max_history = 100
            for metric in ['cpu_usage', 'memory_usage', 'api_latency', 'update_times']:
                if len(self._perf_metrics[metric]) > max_history:
                    self._perf_metrics[metric] = self._perf_metrics[metric][-max_history:]
                    
        except Exception as e:
            logger.error(f"Error updating performance metrics: {str(e)}")
    
    def _log_error(self, error: str, context: str = None):
        """Log an error and update metrics"""
        self._perf_metrics['error_count'] += 1
        self._perf_metrics['last_error'] = {
            'error': str(error),
            'context': context,
            'timestamp': datetime.now()
        }
        self._perf_metrics['failed_updates'] += 1
        logger.error(f"Dashboard error in {context}: {error}")
    
    def _log_success(self):
        """Log a successful update"""
        self._perf_metrics['successful_updates'] += 1
    
    def _get_symbols(self):
        """Get the list of symbols to monitor, respecting config and auto mode, and using model/ai_model if available."""
        # Prefer ai_model or technical_analyzer if they have get_symbols
        if hasattr(self.ai_model, 'get_symbols'):
            try:
                return self.ai_model.get_symbols(self.data_collector)
            except Exception as e:
                logger.error(f"Error getting symbols from ai_model: {e}")
        if hasattr(self.technical_analyzer, 'get_symbols'):
            try:
                return self.technical_analyzer.get_symbols(self.data_collector)
            except Exception as e:
                logger.error(f"Error getting symbols from technical_analyzer: {e}")
        # Fallback to data_collector
        if self.data_collector and hasattr(self.data_collector, 'get_symbols_from_config'):
            try:
                return self.data_collector.get_symbols_from_config()
            except Exception as e:
                logger.error(f"Error getting symbols from data_collector: {e}")
        # Fallback to config
        trading_config = self.config.get('trading', {})
        data_config = trading_config.get('data', {})
        if data_config.get('mode') == 'manual':
            return data_config.get('manual_symbols', [])
        elif data_config.get('mode') == 'auto':
            # If auto mode but collector not available, fallback to manual_symbols or default
            return data_config.get('manual_symbols', ['RELIANCE.NS', 'TCS.NS', 'HDFCBANK.NS', 'INFY.NS', 'ICICIBANK.NS'])
        return trading_config.get('symbols', ['RELIANCE.NS', 'TCS.NS', 'HDFCBANK.NS', 'INFY.NS', 'ICICIBANK.NS'])

    def _fetch_and_display_live_prices(self, symbols: List[str] = None):
        """Fetch and display live prices for symbols, always prefer WebSocket, fallback to REST API only if needed"""
        try:
            if not self.data_collector:
                return
            if symbols is None:
                symbols = self._get_symbols()
            cols = st.columns(len(symbols))
            for i, symbol in enumerate(symbols):
                with cols[i]:
                    price_data = None
                    last_update = None
                    used_websocket = False
                    # --- Always try WebSocket first ---
                    websocket = getattr(self.data_collector, 'websocket', None)
                    if websocket and hasattr(websocket, 'get_market_data'):
                        ws_data = websocket.get_market_data(symbol)
                        if ws_data and 'ltp' in ws_data:
                            price_data = ws_data
                            last_update = datetime.now()
                            used_websocket = True
                            self.market_data_cache.update(symbol, price_data)
                    # --- Fallback: use cache, then REST API ---
                    if not price_data:
                        # Try cache (could be from previous WebSocket or REST)
                        price_data = self.market_data_cache.get(symbol)
                        last_update = self.market_data_cache.get_last_update(symbol)
                    if not price_data:
                        # Fallback to REST API via DataCollector
                        price_data = self.data_collector.get_last_known_ltp(symbol)
                        if price_data:
                            self.market_data_cache.update(symbol, price_data)
                            last_update = datetime.now()
                    st.subheader(symbol.replace('.NS', ''))
                    if price_data and 'ltp' in price_data:
                        price = float(price_data['ltp'])
                        change = float(price_data.get('change', 0))
                        change_pct = float(price_data.get('change_percent', 0))
                        price_color = 'green' if change >= 0 else 'red'
                        change_text = f"{change:+.2f} ({change_pct:+.2f}%)"
                        st.metric(
                            "Price",
                            f"₹{price:,.2f}",
                            delta=change_text
                        )
                        if last_update:
                            time_diff = datetime.now() - last_update
                            if time_diff.seconds < 60:
                                st.caption(f"Updated {time_diff.seconds}s ago" + (" (WebSocket)" if used_websocket else " (REST API)"))
                            else:
                                st.caption(f"Updated {time_diff.seconds // 60}m ago" + (" (WebSocket)" if used_websocket else " (REST API)"))
                        elif used_websocket:
                            st.caption("Source: WebSocket")
                        else:
                            st.caption("Source: REST API (fallback)")
                    else:
                        st.metric("Price", "₹ --")
                        st.caption("Waiting for data...")
        except Exception as e:
            logger.error(f"Error displaying live prices: {str(e)}")
            st.error("Error updating prices. Check logs for details.")

    def render_trading_tab(self):
        """Render trading controls and positions, with simulation as default and live switch button."""
        try:
            st.header("Trading Dashboard")
            all_symbols = self._get_symbols()
            # --- Searchable multi-select for all stocks ---
            monitored_symbols = st.multiselect(
                "Select stocks to monitor (star to add/remove)",
                options=all_symbols,
                default=st.session_state.get('monitored_symbols', []),
                help="Search and select stocks to monitor live. All stocks are available for training."
            )
            st.session_state['monitored_symbols'] = monitored_symbols
            if not monitored_symbols:
                st.info("No stocks selected for monitoring. Use the search box above to add.")
            else:
                self._fetch_and_display_live_prices(monitored_symbols)

            # Trading mode: simulation by default, switch to live with button
            st.subheader("Trading Mode")
            mode = st.session_state.get('trading_mode', 'simulation')
            col1, col2 = st.columns([2, 1])
            with col1:
                st.info("Simulation mode is running by default. All trades are paper trades.")
            with col2:
                if st.button("Switch to LIVE Trading", key="switch_live"):
                    st.session_state['trading_mode'] = 'live'
                    mode = 'live'
            if mode == 'simulation':
                st.success("Paper Trading (Simulation) is ACTIVE.")
                # Run simulation automatically
                if hasattr(self, 'market_data_cache') and hasattr(self.market_data_cache, 'get_latest_data'):
                    data = self.market_data_cache.get_latest_data(all_symbols)
                else:
                    data = None
                if data is not None and not data.empty:
                    self.run_simulation_mode(data)
                else:
                    st.warning("No market data available for simulation.")
            else:
                st.warning("LIVE Trading mode is enabled. Real trades will be placed if you proceed.")
                # Here you would call your live trading logic

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
            autonomous_mode = st.toggle(
                "Enable Fully Autonomous Mode (AI + RL + News)",
                value=st.session_state.get('autonomous_mode', False),
                help="Let the AI make all trading decisions, including news sentiment via Gemini."
            )
            st.session_state['autonomous_mode'] = autonomous_mode
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
            st.error("Error in trading tab. Check logs for details.")
    
    def render_analysis_tab(self):
        """Render analysis tab with Technical Analysis"""
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
            st.error("Error updating analysis view. Check logs for details.")
    
    def render_settings_tab(self):
        """Render settings tab"""
        try:
            st.header("System Settings")
            
            trading_config = self.config.get('trading', {})
            
            # Trading Mode Selection
            mode = st.selectbox(
                "Trading Mode",
                options=["simulation", "live"],
                index=0 if trading_config.get('mode') == 'simulation' else 1,
                help="Select trading mode: Simulation or Live trading",
                key="settings_trading_mode"
            )
            
            # Risk Management Settings
            st.subheader("Risk Management")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Convert max_position from decimal to percentage
                current_max_position = float(trading_config.get('max_position_size', 0.1))
                max_position = st.number_input(
                    "Max Position Size (%)",
                    min_value=1.0,
                    max_value=100.0,
                    value=min(current_max_position * 100, 100.0),
                    step=1.0,
                    help="Maximum position size as a percentage of portfolio"
                )
                
                stop_loss = st.number_input(
                    "Default Stop Loss (%)",
                    min_value=0.1,
                    max_value=10.0,
                    value=float(trading_config.get('stop_loss', 2.0)),
                    step=0.1,
                    help="Default stop loss percentage"
                )
            
            with col2:
                take_profit = st.number_input(
                    "Default Take Profit (%)",
                    min_value=0.1,
                    max_value=20.0,
                    value=float(trading_config.get('take_profit', 5.0)),
                    step=0.1,
                    help="Default take profit percentage"
                )
                
                max_trades = st.number_input(
                    "Max Concurrent Trades",
                    min_value=1,
                    max_value=10,
                    value=int(trading_config.get('max_trades', 3)),
                    help="Maximum number of concurrent trades"
                )
            
            # API Configuration
            st.subheader("API Configuration")
            api_settings = self.config.get('api', {})
            
            # Save Settings Button
            if st.button("Save Settings", key="save_settings"):
                try:
                    # Validate settings before saving
                    if max_position <= 0 or max_position > 100:
                        st.error("Invalid position size. Must be between 1 and 100%")
                        return
                        
                    if 'stop_loss' in locals() and (stop_loss <= 0 or stop_loss > 20):
                        st.error("Invalid stop loss. Must be between 0.1 and 20%")
                        return
                        
                    # Update configuration
                    updated_config = {
                        'mode': mode,
                        'max_position_size': float(max_position) / 100.0,  # Convert to decimal
                    }
                    
                    if 'stop_loss' in locals():
                        updated_config['stop_loss'] = float(stop_loss)
                    if 'take_profit' in locals():
                        updated_config['take_profit'] = float(take_profit)
                    
                    # Update trading config
                    self.config['trading'].update(updated_config)
                    
                    # Save to file
                    config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'config.yaml')
                    with open(config_path, 'w') as f:
                        yaml.dump(self.config, f, default_flow_style=False)
                    
                    # Show success message
                    st.success("Settings saved successfully!")
                    
                    # Notify components of config change
                    if self.trade_manager:
                        self.trade_manager.update_config(self.config['trading'])
                    
                except Exception as e:
                    logger.error(f"Error saving settings: {e}")
                    st.error(f"Failed to save settings: {str(e)}")
            
            # Trading Mode Selection
            mode = st.selectbox(
                "Trading Mode",
                options=["simulation", "live"],
                index=0 if trading_config.get('mode') == 'simulation' else 1,
                help="Select trading mode: Simulation or Live trading"
            )
            
            # Risk Management
            st.subheader("Risk Management")
            col1, col2 = st.columns(2)
            with col1:
                max_position = st.number_input(
                    "Max Position Size (%)",
                    min_value=1.0,
                    max_value=100.0,
                    value=float(trading_config.get('max_position_size', 10)) * 100,
                    step=1.0
                ) / 100.0
            
            with col2:
                stop_loss = st.number_input(
                    "Default Stop Loss (%)",
                    min_value=0.1,
                    max_value=10.0,
                    value=float(trading_config.get('stop_loss', 2.0)),
                    step=0.1
                )
            
            # API Settings
            st.subheader("API Configuration")
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
            
            # Save Settings
            if st.button("Save Settings"):
                try:
                    # Update configuration
                    trading_config.update({
                        'mode': mode,
                        'max_position_size': max_position,
                        'stop_loss': stop_loss,
                    })
                    
                    # Update API credentials if changed
                    if api_key != "●" * 10:
                        trading_config['api_key'] = api_key
                    if api_secret != "●" * 10:
                        trading_config['api_secret'] = api_secret
                    
                    # Save to config file
                    self.config['trading'] = trading_config
                    st.success("Settings saved successfully!")
                    
                except Exception as e:
                    logger.error(f"Error saving settings: {e}")
                    st.error("Failed to save settings")
            
        except Exception as e:
            logger.error(f"Error in settings tab: {e}")
            st.error("Error updating settings")

    def _display_technical_analysis(self, df: pd.DataFrame, analysis: Dict, selected_indicators: List[str]):
        """Display technical analysis results using plotly
        
        Args:
            df: DataFrame with price data
            analysis: Dictionary with technical analysis results
            selected_indicators: List of selected indicators to display
        """
        try:
            if df is None or df.empty or not analysis:
                st.warning("No technical analysis data to display.")
                return
            
            # Create main price chart
            fig = go.Figure()
            
            # Add candlestick chart
            fig.add_trace(go.Candlestick(
                x=df['timestamp'],
                open=df['open'],
                high=df['high'],
                low=df['low'],
                close=df['close'],
                name='Price'
            ))
            
            # Add selected indicators
            for indicator in selected_indicators:
                if indicator == 'SMA':
                    for period in [10, 20, 50]:
                        sma = analysis['trend_indicators'].get(f'sma_{period}')
                        if sma is not None:
                            fig.add_trace(go.Scatter(
                                x=df['timestamp'],
                                y=sma,
                                name=f'SMA {period}',
                                line=dict(width=1)
                            ))
                
                elif indicator == 'EMA':
                    for period in [9, 21]:
                        ema = analysis['trend_indicators'].get(f'ema_{period}')
                        if ema is not None:
                            fig.add_trace(go.Scatter(
                                x=df['timestamp'],
                                y=ema,
                                name=f'EMA {period}',
                                line=dict(width=1)
                            ))
                
                elif indicator == 'MACD':
                    macd_data = analysis['trend_indicators'].get('macd')
                    if macd_data:
                        fig.add_trace(go.Scatter(
                            x=df['timestamp'],
                            y=macd_data['macd'],
                            name='MACD',
                            line=dict(width=1)
                        ))
                        fig.add_trace(go.Scatter(
                            x=df['timestamp'],
                            y=macd_data['signal'],
                            name='Signal',
                            line=dict(width=1)
                        ))
                
                elif indicator == 'RSI':
                    rsi = analysis['momentum_indicators'].get('rsi')
                    if rsi is not None:
                        fig.add_trace(go.Scatter(
                            x=df['timestamp'],
                            y=rsi,
                            name='RSI',
                            yaxis='y2',
                            line=dict(width=1)
                        ))
                
                elif indicator == 'Bollinger Bands':
                    bb_data = analysis['volatility_indicators'].get('bollinger_bands')
                    if bb_data:
                        for band, name in [('upper', 'Upper BB'), ('middle', 'Middle BB'), ('lower', 'Lower BB')]:
                            fig.add_trace(go.Scatter(
                                x=df['timestamp'],
                                y=bb_data[band],
                                name=name,
                                line=dict(width=1, dash='dot')
                            ))
            
            # Update layout
            fig.update_layout(
                title='Technical Analysis',
                yaxis_title='Price',
                template='plotly_dark',
                height=800,
                xaxis_rangeslider_visible=False
            )
            
            # Add secondary y-axis for indicators if needed
            if 'RSI' in selected_indicators:
                fig.update_layout(
                    yaxis2=dict(
                        title='RSI',
                        overlaying='y',
                        side='right',
                        range=[0, 100]
                    )
                )
            
            # Display the chart
            st.plotly_chart(fig, use_container_width=True)
            
            # Display indicator values for last data point
            with st.expander("Current Indicator Values"):
                cols = st.columns(len(selected_indicators))
                for i, indicator in enumerate(selected_indicators):
                    with cols[i]:
                        st.subheader(indicator)
                        try:
                            if indicator == 'RSI':
                                rsi_vals = analysis.get('momentum_indicators', {}).get('rsi', [])
                                if rsi_vals and not pd.isnull(rsi_vals[-1]):
                                    st.metric("RSI", f"{rsi_vals[-1]:.2f}")
                                else:
                                    st.warning("No valid RSI data.")
                            elif indicator == 'MACD':
                                macd_data = analysis.get('trend_indicators', {}).get('macd', {})
                                if macd_data and 'macd' in macd_data and len(macd_data['macd']) > 0:
                                    latest_macd = macd_data['macd'][-1]
                                    latest_signal = macd_data['signal'][-1]
                                    st.metric("MACD", f"{latest_macd:.2f}")
                                    st.metric("Signal", f"{latest_signal:.2f}")
                                else:
                                    st.warning("No valid MACD data.")
                            elif indicator == 'Bollinger Bands':
                                bb_data = analysis.get('volatility_indicators', {}).get('bollinger_bands', {})
                                if bb_data and all(isinstance(bb_data[k], (list, np.ndarray)) and len(bb_data[k]) > 0 for k in ['upper', 'middle', 'lower']):
                                    st.metric("Upper", f"{bb_data['upper'][-1]:.2f}")
                                    st.metric("Middle", f"{bb_data['middle'][-1]:.2f}")
                                    st.metric("Lower", f"{bb_data['lower'][-1]:.2f}")
                                else:
                                    st.warning("No valid Bollinger Bands data.")
                        except Exception as ind_e:
                            logger.error(f"Error displaying {indicator} value: {ind_e}")
                            st.warning(f"Error displaying {indicator} value.")
        except Exception as e:
            logger.error(f"Error displaying technical analysis: {str(e)}")
            st.error("Error displaying technical analysis. Check logs for details.")
            
    def render_sidebar(self):
        st.sidebar.header("Controls")
        market_status = self._state.get('market_status', 'unknown')
        if market_status == 'closed':
            st.sidebar.error("Market Closed")
            st.sidebar.caption("Displaying last known prices")
        elif market_status == 'open':
            st.sidebar.success("Market Open")
        else:
            st.sidebar.info("Market status unknown")
        # Add more sidebar controls as needed

    def render_trade_notifications_top_right(self, trade_log: pd.DataFrame):
        """Display persistent notification icon and rationale at the top right of all dashboard pages."""
        # Inject custom CSS for top-right notification area
        st.markdown(
            """
            <style>
            .trade-notification-area {
                position: fixed;
                top: 1.5rem;
                right: 2.5rem;
                width: 370px;
                z-index: 9999;
                background: rgba(34, 34, 34, 0.98);
                border-radius: 10px;
                box-shadow: 0 2px 8px rgba(0,0,0,0.15);
                padding: 1rem 1.2rem 0.5rem 1.2rem;
                color: #fff;
                font-size: 1rem;
                max-height: 60vh;
                overflow-y: auto;
            }
            .trade-notification-area h5 {
                margin-top: 0;
                margin-bottom: 0.5rem;
                font-size: 1.1rem;
            }
            .trade-notification-entry {
                border-bottom: 1px solid #444;
                margin-bottom: 0.5rem;
                padding-bottom: 0.5rem;
            }
            .trade-notification-entry:last-child {
                border-bottom: none;
            }
            </style>
            """,
            unsafe_allow_html=True
        )
        notif_html = [
            '<div class="trade-notification-area">',
            '<h5>🔔 Trade Notifications</h5>'
        ]
        if trade_log is None or trade_log.empty:
            notif_html.append('<div class="trade-notification-entry" style="text-align:center; color:#bbb;">No trades yet</div>')
        else:
            # Only show the last 5 trades for brevity
            recent_trades = trade_log.tail(5)
            for _, trade in recent_trades[::-1].iterrows():
                notif_html.append('<div class="trade-notification-entry">')
                notif_html.append(f"<b>{trade.get('action', '')}</b> | <b>{trade.get('symbol', '')}</b> @ ₹{trade.get('price', 0):,.2f}")
                if 'stop_loss' in trade and pd.notnull(trade['stop_loss']):
                    notif_html.append(f"<br><span style='color:#ff6666'>Stop Loss:</span> ₹{trade['stop_loss']:.2f}")
                if 'target' in trade and pd.notnull(trade['target']):
                    notif_html.append(f"<br><span style='color:#66ff66'>Target:</span> ₹{trade['target']:.2f}")
                rationale = trade.get('rationale', {})
                if isinstance(rationale, dict):
                    if rationale.get('reason', ''):
                        notif_html.append(f"<br><b>Reason:</b> {rationale.get('reason', '')}")
                    if rationale.get('technical', ''):
                        notif_html.append(f"<br><b>Technical:</b> {rationale.get('technical', '')}")
                    if rationale.get('news', ''):
                        notif_html.append(f"<br><b>Sentiment/News:</b> {rationale.get('news', '')}")
                elif rationale:
                    notif_html.append(f"<br><b>Rationale:</b> {rationale}")
                if 'pnl' in trade:
                    notif_html.append(f"<br><b>PnL:</b> ₹{trade['pnl']:.2f}")
                notif_html.append('</div>')
        notif_html.append('</div>')
        st.markdown(''.join(notif_html), unsafe_allow_html=True)

    def render(self):
        """Main render method"""
        try:
            st.title("Dashboard")

            # Performance metrics from last cycle
            if self._perf_metrics:
                logger.debug(f"Last cycle performance: {self._perf_metrics}")
            self._perf_metrics.clear()

            # Sidebar
            with st.sidebar:
                st.header("Controls")

            # --- Persistent trade notifications at top right ---
            # Try to get trade log from trade_manager or paper trading engine
            trade_log = None
            if self.trade_manager and hasattr(self.trade_manager, 'get_trade_log'):
                trade_log = self.trade_manager.get_trade_log()
            elif hasattr(self, 'last_sim_trade_log'):
                trade_log = self.last_sim_trade_log
            if trade_log is not None:
                self.render_trade_notifications_top_right(trade_log)

            # Main content
            tab1, tab2, tab3 = st.tabs(["Trading", "Analysis", "Settings"])

            with tab1:
                self.render_trading_tab()

            with tab2:
                self.render_analysis_tab()

            with tab3:
                self.render_settings_tab()
            logger.info("Dashboard render cycle complete")

        except Exception as e:
            logger.error(f"Error in dashboard render cycle: {e}")
            st.error("An error occurred while rendering the dashboard")

    def run(self):
        """Entry point for running the dashboard UI (for compatibility with main.py)."""
        self.render()

    def run_simulation_mode(self, data: pd.DataFrame):
        """Run paper trading simulation from the dashboard UI."""
        strategy = EnhancedTradingStrategy(self.config)
        engine = strategy.run_paper_trading(data, ai_model=self.ai_model)
        st.subheader("Paper Trading Results")
        st.write(f"Final Balance: ₹{engine.get_balance():,.2f}")
        st.write("Trade Log:")
        trade_log = engine.get_trade_log()
        # Store for persistent notification rendering
        self.last_sim_trade_log = trade_log
        st.dataframe(trade_log)
        st.line_chart(engine.get_equity_curve())
