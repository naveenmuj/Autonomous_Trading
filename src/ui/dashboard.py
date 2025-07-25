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
        """Fetch and display live prices for symbols, with proper market status handling."""
        try:
            logger.info(f"[DEBUG] _fetch_and_display_live_prices called")
            
            if not self.data_collector:
                st.error("[DEBUG] self.data_collector is None!")
                return
                
            if not hasattr(self.data_collector, 'get_live_quote'):
                st.error("[DEBUG] get_live_quote method not found on data_collector!")
                return
            
            if symbols is None:
                symbols = self._get_symbols()
            
            # Check current market status
            current_time = datetime.now()
            market_open_time = current_time.replace(hour=9, minute=15, second=0, microsecond=0)
            market_close_time = current_time.replace(hour=15, minute=30, second=0, microsecond=0)
            is_market_open = market_open_time <= current_time <= market_close_time
            
            # Display market status
            if is_market_open:
                st.success("🟢 Market is OPEN - Live prices streaming")
            else:
                st.info("🔴 Market is CLOSED - Showing Last Traded Prices (LTP)")
            
            cols = st.columns(len(symbols))
            for i, symbol in enumerate(symbols):
                with cols[i]:
                    try:
                        # Always try to get the latest available price data
                        logger.info(f"[DEBUG] Calling get_live_quote for {symbol}")
                        price_data = self.data_collector.get_live_quote(symbol)
                        logger.info(f"[DEBUG] get_live_quote({symbol}) returned: {price_data}")
                        
                        if price_data and 'ltp' in price_data and price_data['ltp'] > 0:
                            st.subheader(symbol.replace('.NS', ''))
                            price = float(price_data['ltp'])
                            change = float(price_data.get('change', 0))
                            change_pct = float(price_data.get('change_percent', 0))
                            
                            # Determine price color based on change
                            if change > 0:
                                delta_color = "normal"  # Green for positive
                            elif change < 0:
                                delta_color = "inverse"  # Red for negative
                            else:
                                delta_color = "off"  # No color for no change
                            
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
                            
                            # Show data source and timestamp
                            data_source = price_data.get('source', 'Unknown')
                            timestamp = price_data.get('timestamp')
                            
                            if is_market_open:
                                if data_source == 'websocket':
                                    st.caption("📡 Live WebSocket data")
                                elif data_source == 'api':
                                    st.caption("🔄 Live API data")
                                else:
                                    st.caption("📊 Live data")
                            else:
                                if data_source == 'websocket':
                                    st.caption("💾 Cached WebSocket LTP")
                                elif data_source == 'api':
                                    st.caption("💾 Historical API LTP")
                                elif data_source == 'csv':
                                    st.caption("📁 CSV Fallback LTP")
                                else:
                                    st.caption("📈 Last Traded Price")
                            
                            # Show last update time if available
                            if timestamp:
                                if isinstance(timestamp, datetime):
                                    time_str = timestamp.strftime('%H:%M:%S')
                                    st.caption(f"⏰ Last updated: {time_str}")
                        
                        else:
                            # No price data available
                            st.subheader(symbol.replace('.NS', ''))
                            st.metric("Price", "₹ --")
                            if is_market_open:
                                st.caption("⚠️ Waiting for live data...")
                            else:
                                st.caption("⚠️ No LTP data available")
                            
                    except Exception as symbol_error:
                        st.subheader(symbol.replace('.NS', ''))
                        st.error(f"Error loading {symbol}: {str(symbol_error)}")
                        logger.error(f"Error processing {symbol}: {symbol_error}")
                        
        except Exception as e:
            logger.error(f"Error in _fetch_and_display_live_prices: {str(e)}", exc_info=True)
            st.error(f"Error updating prices: {str(e)}")

    def render_trading_tab(self):
        import streamlit as st
        st.info("[DEBUG] render_trading_tab() called")
        try:
            st.header("Trading Dashboard")
            all_symbols = self._get_symbols()
            st.info(f"[DEBUG] all_symbols: {all_symbols}")
            
            # TEMP: Direct test of get_live_quote method
            if self.data_collector and hasattr(self.data_collector, 'get_live_quote'):
                st.info("[DEBUG] Testing get_live_quote method directly...")
                test_symbol = 'RELIANCE.NS'
                try:
                    test_result = self.data_collector.get_live_quote(test_symbol)
                    st.info(f"[DEBUG] get_live_quote('{test_symbol}') returned: {test_result}")
                    st.info(f"[DEBUG] Result type: {type(test_result)}")
                    if test_result:
                        st.info(f"[DEBUG] Result keys: {list(test_result.keys()) if isinstance(test_result, dict) else 'Not a dict'}")
                except Exception as e:
                    st.error(f"[DEBUG] get_live_quote failed with error: {e}")
                    import traceback
                    st.text(traceback.format_exc())
            
            # DEBUG: Show recent logs
            with st.expander("Debug Logs", expanded=False):
                try:
                    import logging
                    # Check if we have the StreamlitLogHandler
                    streamlit_handlers = [h for h in logging.getLogger().handlers if hasattr(h, 'log_messages')]
                    if streamlit_handlers:
                        recent_logs = streamlit_handlers[0].log_messages[-20:]  # Show last 20 messages
                        for log_msg in recent_logs:
                            st.text(log_msg)
                    else:
                        st.text("No debug handler found")
                except Exception as e:
                    st.text(f"Error showing logs: {e}")
            
            # --- Searchable multi-select for all stocks ---
            monitored_symbols = st.multiselect(
                "Select stocks to monitor (star to add/remove)",
                options=all_symbols,
                default=all_symbols[:5],
                help="Search and select stocks to monitor live. All stocks are available for training."
            )
            st.info(f"[DEBUG] monitored_symbols: {monitored_symbols}")
            if not monitored_symbols:
                st.info("No stocks selected for monitoring. Use the search box above to add.")
            else:
                st.info(f"[DEBUG] About to call _fetch_and_display_live_prices with {monitored_symbols}")
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
