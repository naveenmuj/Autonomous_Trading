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
    
    def _fetch_and_display_live_prices(self, symbols: List[str]):
        """Fetch and display live prices for symbols, fallback to last known price if needed"""
        try:
            if not self.data_collector:
                return
            cols = st.columns(len(symbols))
            for i, symbol in enumerate(symbols):
                with cols[i]:
                    price_data = self.market_data_cache.get(symbol)
                    last_update = self.market_data_cache.get_last_update(symbol)
                    if not price_data:
                        # Fallback: get last known price from DataCollector
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
                                st.caption(f"Updated {time_diff.seconds}s ago")
                            else:
                                st.caption(f"Updated {time_diff.seconds // 60}m ago")
                    else:
                        st.metric("Price", "₹ --")
                        st.caption("Waiting for data...")
        except Exception as e:
            logger.error(f"Error displaying live prices: {str(e)}")
            st.error("Error updating prices. Check logs for details.")
    
    def render_trading_tab(self):
        """Render trading controls and positions"""
        try:
            st.header("Trading Dashboard")
            
            # Display live prices first
            trading_config = self.config.get('trading', {})
            symbols = trading_config.get('symbols', ['RELIANCE.NS', 'TCS.NS', 'HDFCBANK.NS', 'INFY.NS', 'ICICIBANK.NS'])
            self._fetch_and_display_live_prices(symbols)
            
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
                max_position = st.number_input(
                    "Max Position Size",
                    min_value=0.0,
                    max_value=1.0,
                    value=float(trading_config.get('max_position_size', 0.1)),
                    step=0.01,
                    format="%.2f"
                )
            
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
            logger.error(f"Error in trading tab: {e}")
            st.error("Error updating trading view. Check logs for details.")
    
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
                        if indicator == 'RSI':
                            latest_rsi = analysis['momentum_indicators'].get('rsi', [])[-1]
                            st.metric("RSI", f"{latest_rsi:.2f}")
                        elif indicator == 'MACD':
                            macd_data = analysis['trend_indicators'].get('macd', {})
                            if macd_data:
                                latest_macd = macd_data['macd'][-1]
                                latest_signal = macd_data['signal'][-1]
                                st.metric("MACD", f"{latest_macd:.2f}")
                                st.metric("Signal", f"{latest_signal:.2f}")
                        elif indicator == 'Bollinger Bands':
                            bb_data = analysis['volatility_indicators'].get('bollinger_bands', {})
                            if bb_data:
                                st.metric("Upper", f"{bb_data['upper'][-1]:.2f}")
                                st.metric("Middle", f"{bb_data['middle'][-1]:.2f}")
                                st.metric("Lower", f"{bb_data['lower'][-1]:.2f}")
                                
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
        """Main entry point for running the dashboard"""
        try:
            self.render()
        except Exception as e:
            logger.error(f"Error running dashboard: {e}")
            st.error(f"An error occurred while running the dashboard: {str(e)}")
