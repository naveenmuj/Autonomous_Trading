import streamlit as st
import plotly.graph_objects as go
from datetime import datetime
import pandas as pd
import yaml
import logging
import time
from typing import Dict, Any, Optional, List

logger = logging.getLogger(__name__)

# Utility for INR formatting
def format_inr(value):
    try:
        return f"₹{value:,.2f}"
    except Exception:
        return f"₹{value}"

class DashboardUI:
    def __init__(self, config=None, data_collector=None, trade_manager=None, 
                 ai_trader=None, technical_analyzer=None, sentiment_analyzer=None):
        start_time = time.time()
        logger.info("Initializing DashboardUI...")
        
        try:
            self.config = config
            self.data_collector = data_collector
            self.trade_manager = trade_manager
            self.ai_trader = ai_trader
            self.technical_analyzer = technical_analyzer
            self.sentiment_analyzer = sentiment_analyzer
            self._perf_metrics = {}
            
            # Initialize UI components
            self._init_ui_components()
            
            init_time = time.time() - start_time
            logger.info(f"DashboardUI initialized in {init_time:.2f}s")
            
        except Exception as e:
            logger.error(f"Error initializing DashboardUI: {e}", exc_info=True)
            raise
    
    def _init_ui_components(self):
        """Initialize UI components and state"""
        try:
            # Initialize WebSocket related state
            if 'update_counter' not in st.session_state:
                st.session_state.update_counter = 0
            if 'last_prices' not in st.session_state:
                st.session_state.last_prices = {}
            if 'last_update' not in st.session_state:
                st.session_state.last_update = time.time()
            if 'refresh_interval' not in st.session_state:
                st.session_state.refresh_interval = 1.0  # Update interval in seconds
                
            # Initialize other UI state
            if 'watchlist' not in st.session_state:
                st.session_state.watchlist = []
            if 'trades' not in st.session_state:
                st.session_state.trades = []
            if 'positions' not in st.session_state:
                st.session_state.positions = []
                
            # Initialize WebSocket if needed
            if hasattr(self.data_collector, 'websocket') and not self.data_collector.websocket:
                self.data_collector._initialize_websocket()
                
        except Exception as e:
            logger.error(f"Error initializing UI components: {str(e)}")
            
    def _log_performance(self, component_name):
        """Context manager to track component render times"""
        class _PerfLogger:
            def __init__(self, component, metrics_dict):
                self.component = component
                self.metrics_dict = metrics_dict
                self.start_time = None
            
            def __enter__(self):
                self.start_time = time.time()
                logger.debug(f"Starting render of {self.component}")
                return self
            
            def __exit__(self, exc_type, exc_val, exc_tb):
                duration = time.time() - self.start_time
                self.metrics_dict[self.component] = duration
                logger.debug(f"Finished render of {self.component} in {duration:.2f}s")
                if exc_type:
                    logger.error(f"Error in {self.component}: {exc_val}")
                    return False
                return True
        
        return _PerfLogger(component_name, self._perf_metrics)
    
    def run(self):
        """Main dashboard rendering function"""
        logger.info("Starting dashboard render cycle")
        try:
            st.title("AI Trading Dashboard")
            
            # Performance metrics from last cycle
            if self._perf_metrics:
                logger.debug(f"Last cycle performance: {self._perf_metrics}")
            self._perf_metrics.clear()
            
            # Sidebar
            with self._log_performance("sidebar"):
                self.render_sidebar()
            
            # Main content
            with self._log_performance("tabs"):
                tab1, tab2, tab3 = st.tabs(["Trading", "Analysis", "Settings"])
            
            with tab1:
                with self._log_performance("trading_tab"):
                    self.render_trading_tab()
                
            with tab2:
                with self._log_performance("analysis_tab"):
                    self.render_analysis_tab()
                
            with tab3:
                with self._log_performance("settings_tab"):
                    self.render_settings_tab()
            
            logger.info("Dashboard render cycle complete")
        except Exception as e:
            logger.error(f"Error in dashboard render cycle: {e}", exc_info=True)
            st.error("An error occurred while rendering the dashboard. Check the logs for details.")

    def render_sidebar(self):
        """Render sidebar controls"""
        with st.sidebar:
            st.header("Controls")
            
            # Ensure session state is initialized
            if 'refresh_interval' not in st.session_state:
                st.session_state['refresh_interval'] = 1.0
            
            # Mode selection
            mode = st.selectbox(
                "Trading Mode",
                ["simulation", "live"],
                index=0 if self.config['trading']['mode'] == 'simulation' else 1
            )
            
            # WebSocket Status and Controls
            st.subheader("Connection Status")
            ws_status = "Connected" if (hasattr(self.data_collector, 'websocket') 
                                     and self.data_collector.websocket 
                                     and self.data_collector.websocket.is_connected) else "Disconnected"
            
            st.metric(
                "WebSocket Status",
                value=ws_status,
                delta="Live" if ws_status == "Connected" else "Offline",
                delta_color="normal" if ws_status == "Connected" else "off"
            )
            
            # Refresh Interval Control
            st.number_input(
                "Refresh Interval (seconds)",
                min_value=0.1,
                max_value=60.0,
                value=st.session_state.refresh_interval,
                step=0.1,
                key="refresh_interval_input"
            )
            
            if st.button("Reset WebSocket"):
                try:
                    if hasattr(self.data_collector, 'websocket'):
                        self.data_collector._initialize_websocket()
                        st.success("WebSocket connection reset")
                except Exception as e:
                    st.error(f"Error resetting WebSocket: {str(e)}")
                    
            # Action buttons
            if st.button("Fetch Data"):
                if self.data_collector:
                    with st.spinner("Fetching market data..."):
                        try:
                            watchlist = self.data_collector.get_watchlist()
                            if watchlist:
                                st.session_state.watchlist = watchlist
                                st.success("Market data fetched successfully!")
                        except Exception as e:
                            logger.error(f"Error fetching market data: {e}")
                            st.error(f"Error fetching data: {str(e)}")
                            
            # Trading controls in live mode
            if mode == "live":
                if st.button("Start Live Trading"):
                    logger.warning("Live trading button clicked but not implemented")
                    st.warning("Live trading interface not implemented yet")

    def render_trading_tab(self):
        """Render trading tab with enhanced portfolio monitoring"""
        try:
            # Portfolio Overview
            st.header("Portfolio Overview")
            col1, col2, col3, col4 = st.columns(4)
            
            # Get portfolio metrics
            portfolio_value = self.trade_manager.get_portfolio_value()
            daily_pnl = self.trade_manager.get_daily_pnl()
            metrics = self.trade_manager.get_portfolio_metrics()
            
            with col1:
                st.metric(
                    "Portfolio Value",
                    value=format_inr(portfolio_value),
                    delta=format_inr(daily_pnl)
                )
            
            with col2:
                st.metric(
                    "Win Rate",
                    value=f"{metrics['win_rate']:.1f}%",
                    delta=f"{metrics['daily_return']:.1f}%"
                )
                
            with col3:
                st.metric(
                    "Open Positions",
                    value=metrics['open_positions']
                )
                
            with col4:
                st.metric(
                    "Daily P&L",
                    value=format_inr(metrics['daily_pnl']),
                    delta=format_inr(daily_pnl)
                )

            # Trading Interface
            st.header("Trading Interface")
            
            col1, col2 = st.columns([2, 1])
            with col1:
                # Symbol Selection
                watchlist = self.data_collector.get_watchlist()
                symbol_list = [item['symbol'] for item in watchlist] if watchlist else []
                selected_symbol = st.selectbox("Select Symbol", symbol_list)
                
                # Initialize quote
                quote = None
                if selected_symbol:
                    # Get quote from WebSocket if available, fallback to REST API
                    quote = None
                    if (hasattr(self.data_collector, 'websocket') and 
                        self.data_collector.websocket and 
                        self.data_collector.websocket.is_connected):
                        token = self.data_collector.symbol_token_map.get(selected_symbol)
                        if token:
                            quote = self.data_collector.websocket.get_market_data(token)
                    
                    if not quote:
                        quote = self.data_collector.get_live_quote(selected_symbol)
                        
                    if quote:
                        self._update_price_display(selected_symbol, quote)
                    else:
                        st.warning("Unable to fetch current price")

            with col2:
                # Trading Form
                with st.form("trade_form"):
                    quantity = st.number_input("Quantity", min_value=1, value=1)
                    action = st.selectbox("Action", ["BUY", "SELL"])
                      # Get current price for stop loss/target calculations
                    current_price = quote.get('ltp', 0.0) if quote else 0.0
                    
                    # Risk Management Controls
                    st.write("Risk Management")
                    stop_loss = st.number_input(
                        "Stop Loss",
                        value=float(current_price * 0.98),
                        format="%.2f"
                    )
                    target = st.number_input(
                        "Target Price",
                        value=float(current_price * 1.02),
                        format="%.2f"
                    )
                    
                    if st.form_submit_button("Place Order"):
                        try:
                            trade = self.trade_manager.place_trade(
                                symbol=selected_symbol,
                                action=action,
                                quantity=quantity,
                                price=quote.get('ltp', 0),
                                stop_loss=stop_loss,
                                target=target
                            )
                            if trade:
                                st.success(f"{action} order placed successfully!")
                                st.json(trade)  # Show trade details
                            else:
                                st.error("Order placement failed")
                        except Exception as e:
                            st.error(f"Error placing order: {str(e)}")

            # Active Positions
            st.header("Active Positions")
            positions = self.trade_manager.positions
            if positions:
                position_data = []
                for symbol, pos in positions.items():
                    current_price = self.data_collector.get_ltp(symbol)
                    if current_price and pos['status'].upper() == 'OPEN':
                        pnl = (current_price - pos['entry_price']) * pos['quantity']
                        if pos.get('action') == 'SELL':  # For short positions
                            pnl = -pnl
                        
                        position_data.append({
                            'Symbol': symbol,
                            'Quantity': pos['quantity'],
                            'Entry Price': format_inr(pos['entry_price']),
                            'Current Price': format_inr(current_price),
                            'Stop Loss': format_inr(pos.get('stop_loss', 0)),
                            'Target': format_inr(pos.get('target', 0)),
                            'P&L': format_inr(pnl),
                            'Action': pos.get('action', 'BUY')
                        })
                
                if position_data:
                    st.dataframe(pd.DataFrame(position_data))
            else:
                st.info("No active positions")

            # Trade History
            st.header("Trade History")
            trades = self.trade_manager.get_trade_history()
            if trades:
                trade_df = pd.DataFrame(trades)
                trade_df['Entry Time'] = pd.to_datetime(trade_df['timestamp'])
                trade_df['Exit Time'] = pd.to_datetime(trade_df['exit_timestamp'])
                trade_df['P&L'] = trade_df['pnl'].apply(format_inr)
                
                # Show recent trades first
                trade_df = trade_df.sort_values('Entry Time', ascending=False)
                st.dataframe(trade_df)
            else:
                st.info("No trade history available")
            
        except Exception as e:
            logger.error(f"Error rendering trading tab: {str(e)}")
            st.error("Error loading trading data. Please try again.")

    def render_analysis_tab(self):
        """Render enhanced technical and market analysis"""
        try:
            st.header("Market Analysis")
              # Symbol Selection
            watchlist = self.data_collector.get_watchlist()
            symbol_list = [item['symbol'] for item in watchlist] if watchlist else []
            selected_symbol = st.selectbox("Select Symbol", symbol_list, key="analysis_symbol")
            
            if selected_symbol:
                # Fetch data
                historical_data = self.data_collector.get_historical_data(selected_symbol)
                live_quote = self.data_collector.get_live_quote(selected_symbol)
                
                if historical_data is not None and not historical_data.empty:
                    # Price Chart
                    st.subheader("Price Analysis")
                    fig = go.Figure(data=[
                        go.Candlestick(
                            x=historical_data.index,
                            open=historical_data['open'],
                            high=historical_data['high'],
                            low=historical_data['low'],
                            close=historical_data['close'],
                            name="OHLC"
                        )
                    ])
                    
                    # Add technical indicators if available
                    if 'SMA_20' in historical_data.columns:
                        fig.add_trace(go.Scatter(
                            x=historical_data.index,
                            y=historical_data['SMA_20'],
                            name="20-day SMA",
                            line=dict(color='orange')
                        ))
                    
                    if 'SMA_50' in historical_data.columns:
                        fig.add_trace(go.Scatter(
                            x=historical_data.index,
                            y=historical_data['SMA_50'],
                            name="50-day SMA",
                            line=dict(color='blue')
                        ))
                    
                    fig.update_layout(
                        title=f"{selected_symbol} Price Chart",
                        xaxis_title="Date",
                        yaxis_title="Price",
                        height=600
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Technical Indicators
                    st.subheader("Technical Indicators")
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        if 'RSI' in historical_data.columns:
                            current_rsi = historical_data['RSI'].iloc[-1]
                            st.metric(
                                "RSI (14)",
                                value=f"{current_rsi:.2f}",
                                delta="Overbought" if current_rsi > 70 else "Oversold" if current_rsi < 30 else "Neutral"
                            )
                            
                    with col2:
                        if 'MACD' in historical_data.columns:
                            current_macd = historical_data['MACD'].iloc[-1]
                            st.metric(
                                "MACD",
                                value=f"{current_macd:.2f}",
                                delta="Bullish" if current_macd > 0 else "Bearish"
                            )
                            
                    with col3:
                        if all(col in historical_data.columns for col in ['SMA_20', 'SMA_50']):
                            sma_20 = historical_data['SMA_20'].iloc[-1]
                            sma_50 = historical_data['SMA_50'].iloc[-1]
                            st.metric(
                                "Moving Averages",
                                value=f"20MA: {sma_20:.2f}",
                                delta=f"50MA: {sma_50:.2f}"
                            )
                    
                    # Volume Analysis
                    st.subheader("Volume Analysis")
                    volume_fig = go.Figure(data=[
                        go.Bar(
                            x=historical_data.index,
                            y=historical_data['volume'],
                            name="Volume"
                        )
                    ])
                    volume_fig.update_layout(
                        title="Trading Volume",
                        xaxis_title="Date",
                        yaxis_title="Volume",
                        height=300
                    )
                    st.plotly_chart(volume_fig, use_container_width=True)
                    
                    # Market Depth if available
                    if live_quote:
                        st.subheader("Market Depth")
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.write("Bid")
                            bid_data = live_quote.get('best_5_buy_data', [])
                            if bid_data:
                                st.dataframe(pd.DataFrame(bid_data))
                                
                        with col2:
                            st.write("Ask")
                            ask_data = live_quote.get('best_5_sell_data', [])
                            if ask_data:
                                st.dataframe(pd.DataFrame(ask_data))
                    
                else:
                    st.warning("No historical data available for analysis")
                    
        except Exception as e:
            logger.error(f"Error rendering analysis tab: {str(e)}")
            st.error("Error analyzing market data. Please try again.")

    def render_settings_tab(self):
        """Render settings tab with connection management"""
        st.header("System Settings")
        
        # Connection Status
        st.subheader("Connection Status")
        cols = st.columns(3)
        
        with cols[0]:
            # Angel One API Status
            api_status = "Connected" if self.data_collector and self.data_collector.angel_api else "Disconnected"
            st.metric(
                "Angel One API",
                value=api_status,
                delta="Online" if api_status == "Connected" else "Offline",
                delta_color="normal" if api_status == "Connected" else "off"
            )
            
        with cols[1]:
            # WebSocket Status
            ws_status = self.data_collector.websocket.get_connection_status() if hasattr(self.data_collector, 'websocket') else {}
            ws_connected = ws_status.get('connected', False)
            st.metric(
                "WebSocket Connection",
                value="Connected" if ws_connected else "Disconnected",
                delta=f"Attempts: {ws_status.get('reconnect_attempts', 0)}",
                delta_color="normal" if ws_connected else "off"
            )
            
        with cols[2]:
            # Trading Mode
            trading_mode = self.config['trading']['mode']
            st.metric(
                "Trading Mode",
                value=trading_mode.title(),
                delta="Live" if trading_mode == 'live' else "Simulation"
            )

        # Subscription Management
        st.subheader("Market Data Subscriptions")
        if hasattr(self.data_collector, 'websocket'):
            subscribed = self.data_collector.websocket.get_connection_status().get('subscribed_tokens', {})
            for exchange, tokens in subscribed.items():
                st.write(f"{exchange}: {len(tokens)} symbols subscribed")
                
            # Subscription Management
            with st.expander("Manage Subscriptions"):
                symbols = self.data_collector.get_watchlist()
                selected = st.multiselect("Select Symbols", symbols)
                
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("Subscribe"):
                        try:
                            self.data_collector.websocket.subscribe({
                                'NSE': [self.data_collector.get_token(s) for s in selected]
                            })
                            st.success("Subscription updated successfully")
                        except Exception as e:
                            st.error(f"Error updating subscriptions: {str(e)}")
                            
                with col2:
                    if st.button("Unsubscribe"):
                        try:
                            self.data_collector.websocket.unsubscribe({
                                'NSE': [self.data_collector.get_token(s) for s in selected]
                            })
                            st.success("Unsubscribed successfully")
                        except Exception as e:
                            st.error(f"Error unsubscribing: {str(e)}")

        # Risk Management Settings
        st.subheader("Risk Management")
        risk_config = self.config.get('trading', {}).get('risk', {})
        
        col1, col2 = st.columns(2)
        with col1:
            st.write("Position Limits")
            st.write(f"Max Position Size: {risk_config.get('max_position_size', 0.02)*100}%")
            st.write(f"Max Trades: {risk_config.get('max_trades', 5)}")
            
        with col2:
            st.write("Risk Limits")
            st.write(f"Stop Loss: {risk_config.get('stop_loss', 0.02)*100}%")
            st.write(f"Max Drawdown: {risk_config.get('max_drawdown', 0.15)*100}%")

        # Account Settings
        st.subheader("Account Settings")
        if st.button("Refresh Session"):
            try:
                self.data_collector._initialize_api()  # Re-authenticate
                st.success("Session refreshed successfully")
            except Exception as e:
                st.error(f"Error refreshing session: {str(e)}")

        # Save Settings
        if st.button("Save Settings"):
            try:
                self.trade_manager._save_account_value()
                st.success("Settings saved successfully")
            except Exception as e:
                st.error(f"Error saving settings: {str(e)}")
                
    def display_portfolio_metrics(self, portfolio_data):
        """Display portfolio metrics in a grid"""
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Value", f"₹{portfolio_data['total_value']:,.2f}", 
                     f"{portfolio_data['daily_return']}%")
            
        with col2:
            st.metric("Open Positions", portfolio_data['open_positions'])
            
        with col3:
            st.metric("Daily P/L", f"₹{portfolio_data['daily_pnl']:,.2f}")
            
        with col4:
            st.metric("Win Rate", f"{portfolio_data['win_rate']}%")
            
        logger.debug(f"Portfolio metrics displayed: {portfolio_data}")

    def plot_candlestick(self, data):
        """Plot candlestick chart with technical indicators"""
        logger.debug("Creating candlestick chart")
        
        try:
            fig = go.Figure(data=[go.Candlestick(
                x=data.index,
                open=data['Open'],
                high=data['High'],
                low=data['Low'],
                close=data['Close']
            )])
            
            # Add technical indicators if available
            if 'BB_upper' in data.columns:
                logger.debug("Adding Bollinger Bands to chart")
                fig.add_trace(go.Scatter(
                    x=data.index, y=data['BB_upper'],
                    line=dict(color='gray', dash='dot'),
                    name='Upper BB'
                ))
                fig.add_trace(go.Scatter(
                    x=data.index, y=data['BB_lower'],
                    line=dict(color='gray', dash='dot'),
                    name='Lower BB'
                ))
                
            fig.update_layout(
                title='Stock Price',
                yaxis_title='Price',
                xaxis_title='Date',
                template='plotly_dark' if self.config['ui']['theme'] == 'dark' else 'plotly_white'
            )
            
            st.plotly_chart(fig, use_container_width=True)
            logger.debug("Chart rendered successfully")
            
        except Exception as e:
            logger.error(f"Error creating chart: {e}")
            st.error("Error creating chart")
            
    def _format_currency(self, amount: float) -> str:
        """Format currency in Indian Rupee format with proper separators"""
        try:
            if pd.isna(amount):
                return "₹0.00"
            # Convert to Indian format (2 decimal places, comma separators)
            # Example: 1,00,000.00
            abs_amount = abs(amount)
            crore = abs_amount // 10000000
            remainder = abs_amount % 10000000
            lakh = remainder // 100000
            remainder = remainder % 100000
            thousand = remainder // 1000
            remainder = remainder % 1000
            
            parts = []
            if crore > 0:
                parts.append(f"{crore:,.0f}")
            if lakh > 0 or crore > 0:
                parts.append(f"{lakh:02.0f}")
            if thousand > 0 or lakh > 0 or crore > 0:
                parts.append(f"{thousand:03.0f}")
            parts.append(f"{remainder:03.2f}")
            
            formatted = ",".join(parts)
            return f"₹{'-' if amount < 0 else ''}{formatted}"
        except Exception as e:
            logger.error(f"Error formatting currency: {str(e)}")
            return "₹0.00"
            
    def _format_symbol(self, symbol: str) -> str:
        """Format stock symbol for NSE"""
        try:
            # Convert Yahoo Finance format to NSE format
            if '.NS' in symbol:
                return symbol.replace('.NS', '')
            # Convert Angel One format to NSE format
            if '-EQ' in symbol:
                return symbol.replace('-EQ', '')
            return symbol
        except Exception as e:
            logger.error(f"Error formatting symbol: {str(e)}")
            return symbol

    def _update_price_display(self, symbol: str, quote: Dict[str, Any]):
        """Update price display with WebSocket data"""
        try:
            if not quote:
                return
                
            # Get current price and change
            current_price = quote.get('ltp', 0.0)
            price_change = quote.get('price_change', 0.0)
            price_change_pct = quote.get('price_change_pct', 0.0)
            
            # Get last price from session state
            last_price = st.session_state.last_prices.get(symbol)
            
            # Update last price in session state
            st.session_state.last_prices[symbol] = current_price
            
            # Create price display container
            price_container = st.empty()
            with price_container:
                if last_price and current_price != last_price:
                    # Calculate delta for color coding
                    delta = current_price - last_price
                    delta_color = "green" if delta >= 0 else "red"
                    
                    # Display with color coding
                    st.markdown(
                        f"""
                        <div style='text-align: center'>
                            <h2 style='margin-bottom: 0;'>₹{current_price:,.2f}</h2>
                            <p style='color: {delta_color}; margin-top: 0;'>
                                ₹{price_change:+.2f} ({price_change_pct:+.2f}%)
                            </p>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                else:
                    # First display or no change
                    st.metric(
                        "Current Price",
                        value=f"₹{current_price:,.2f}",
                        delta=f"₹{price_change:+.2f} ({price_change_pct:+.2f}%)"
                    )
            
            # Check if it's time to refresh
            now = time.time()
            if 'last_update' not in st.session_state:
                st.session_state.last_update = now
            elif now - st.session_state.last_update >= st.session_state.refresh_interval:
                st.session_state.last_update = now
                st.session_state.update_counter = st.session_state.get('update_counter', 0) + 1
                time.sleep(0.1)  # Small delay to prevent too rapid updates
                try:
                    st.rerun()
                except Exception as e:
                    logger.error(f"Error during dashboard rerun: {str(e)}")
                    # Fallback to normal refresh
                    time.sleep(st.session_state.refresh_interval)
                
        except Exception as e:
            logger.error(f"Error updating price display: {str(e)}")
            st.error("Error updating price display")
