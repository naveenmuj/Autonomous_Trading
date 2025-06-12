import streamlit as st
import plotly.graph_objects as go
from datetime import datetime
import pandas as pd
import yaml
import logging
import time

logger = logging.getLogger(__name__)

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
        """Initialize UI components with timing"""
        start_time = time.time()
        
        # Initialize performance tracking
        self._component_times = {
            'sidebar': [],
            'trading_tab': [],
            'analysis_tab': [],
            'settings_tab': []
        }
        
        # Cache frequently used UI elements
        self._cached_elements = {}
        
        init_time = time.time() - start_time
        logger.debug(f"UI components initialized in {init_time:.2f}s")
        
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
            
            # Mode selection
            mode = st.selectbox(
                "Trading Mode",
                ["simulation", "live"],
                index=0 if self.config['trading']['mode'] == 'simulation' else 1
            )
            logger.debug(f"Trading mode selected: {mode}")
            
            # Stock selection
            stock = st.text_input("Stock Symbol", "RELIANCE")
            if stock:
                logger.debug(f"Stock symbol entered: {stock}")
            
            # Action buttons
            if st.button("Fetch Data"):
                if self.data_collector:
                    logger.info(f"Fetching data for stock: {stock}")
                    with st.spinner("Fetching data..."):
                        try:
                            data = self.data_collector.get_historical_data(stock)
                            st.session_state.current_data = data
                            logger.info(f"Data fetched successfully for {stock}")
                            st.success("Data fetched successfully!")
                        except Exception as e:
                            logger.error(f"Error fetching data for {stock}: {e}")
                            st.error(f"Error fetching data: {str(e)}")
                        
            if mode == "live" and st.button("Start Trading"):
                logger.warning("Live trading button clicked but not implemented")
                st.warning("Live trading interface not implemented yet")

    def render_trading_tab(self):
        """Render trading interface"""
        cols = st.columns([2, 1])
        
        with cols[0]:
            # Chart
            if 'current_data' in st.session_state:
                logger.debug("Rendering chart with current data")
                self.plot_candlestick(st.session_state.current_data)
            else:
                logger.debug("No data available for chart")
                st.info("Fetch data for a stock to see the chart")
                
        with cols[1]:
            # Trading stats
            st.subheader("Trading Statistics")
            placeholder_stats = {
                'total_value': 100000,
                'daily_return': 0.5,
                'open_positions': 2,
                'daily_pnl': 500,
                'win_rate': 65
            }
            logger.debug("Displaying portfolio metrics")
            self.display_portfolio_metrics(placeholder_stats)

    def render_analysis_tab(self):
        """Render analysis interface"""
        st.subheader("Technical Analysis")
        if 'current_data' in st.session_state:
            data = st.session_state.current_data
            logger.debug("Rendering technical analysis")
            
            cols = st.columns(2)
            with cols[0]:
                st.write("Technical Indicators")
                if 'RSI' in data.columns:
                    current_rsi = data['RSI'].iloc[-1]
                    logger.debug(f"Current RSI: {current_rsi:.2f}")
                    st.metric("RSI", f"{current_rsi:.2f}")
                if 'MACD' in data.columns:
                    current_macd = data['MACD'].iloc[-1]
                    logger.debug(f"Current MACD: {current_macd:.2f}")
                    st.metric("MACD", f"{current_macd:.2f}")
                    
            with cols[1]:
                st.write("Predictions")
                logger.debug("AI predictions not yet implemented")
                st.info("AI predictions coming soon...")
        else:
            logger.debug("No data available for analysis")
            st.info("Fetch data for a stock to see analysis")

    def render_settings_tab(self):
        """Render settings interface"""
        st.subheader("Risk Management")
        
        col1, col2 = st.columns(2)
        with col1:
            max_loss = st.number_input(
                "Max Loss per Trade (%)",
                min_value=0.1,
                max_value=10.0,
                value=float(self.config['risk_management']['max_loss_per_trade']),
                step=0.1
            )
            logger.debug(f"Max loss per trade set to: {max_loss}%")
            
        with col2:
            max_risk = st.number_input(
                "Max Portfolio Risk (%)",
                min_value=1.0,
                max_value=20.0,
                value=float(self.config['risk_management']['max_portfolio_risk']),
                step=0.5
            )
            logger.debug(f"Max portfolio risk set to: {max_risk}%")
            
    def display_portfolio_metrics(self, portfolio_data):
        """Display portfolio metrics in a grid"""
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Value", f"${portfolio_data['total_value']:,.2f}", 
                     f"{portfolio_data['daily_return']}%")
            
        with col2:
            st.metric("Open Positions", portfolio_data['open_positions'])
            
        with col3:
            st.metric("Daily P/L", f"${portfolio_data['daily_pnl']:,.2f}")
            
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
