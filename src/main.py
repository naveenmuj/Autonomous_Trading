import sys
import os
from pathlib import Path
import streamlit as st  # Add Streamlit import

# Add project root to path first
# Get the absolute path to the project root
project_root = str(Path(__file__).parent.parent.absolute())

# Add to Python path if not already there
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import warnings
import logging
from datetime import datetime
import traceback
import psutil

# Enhanced logging configuration
log_dir = os.path.join(project_root, 'logs', datetime.now().strftime('%Y-%m-%d'))
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, 'app.log')

# Custom formatter with thread info and memory usage
class DetailedFormatter(logging.Formatter):
    def format(self, record):
        process = psutil.Process(os.getpid())
        record.memory_usage = f"{process.memory_info().rss / 1024 / 1024:.1f}MB"
        return super().format(record)

formatter = DetailedFormatter(
    '%(asctime)s - [%(name)s] - %(levelname)s - [%(threadName)s] - '
    '[Mem: %(memory_usage)s] - %(message)s'
)

# Configure file handler with rotation
file_handler = logging.FileHandler(log_file)
file_handler.setFormatter(formatter)

# Configure console handler
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setFormatter(formatter)

# Root logger configuration
root_logger = logging.getLogger()
root_logger.setLevel(logging.INFO)
root_logger.addHandler(file_handler)
root_logger.addHandler(console_handler)

# Module logger
logger = logging.getLogger(__name__)

# Suppress warnings
warnings.filterwarnings('ignore')

# Initialize Streamlit
st.set_page_config(
    page_title="AI Trading System",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Log system info at startup
logger.info("=== Starting Trading System ===")
logger.info(f"Python version: {sys.version}")
logger.info(f"Working directory: {os.getcwd()}")
logger.info(f"Log directory: {log_dir}")

# Now import the modules
import yaml
from typing import Dict, Any, Optional
import pandas as pd
import numpy as np

from data.collector import DataCollector  # Using WebSocket-enabled collector
from ai.training_pipeline import TrainingPipeline
from trading.strategy import EnhancedTradingStrategy
from ui.dashboard import DashboardUI

class TradingSystem:
    def __init__(self, config_path: str):
        """Initialize the trading system"""
        self.config = self._load_config(config_path)
        self.collector = DataCollector(self.config)
        self.training_pipeline = TrainingPipeline(self.config)
        self.models = {}
        self.strategy = None
        self.dashboard = None
        
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            logger.info("Configuration loaded successfully")
            return config
        except Exception as e:
            logger.error(f"Error loading configuration: {str(e)}")
            raise
            
    def initialize(self):
        """Initialize system components"""
        try:
            logger.info("Initializing system components...")
            
            # Get initial market data
            data = self.collector.get_market_data()
            if data.empty:
                logger.warning("No live market data available (market may be closed)")
            
            # Train models with available data
            logger.info("Training new models...")
            try:
                training_data, market_data = self.training_pipeline.prepare_training_data()
                
                # Train models if we have data
                if not training_data.empty:
                    self.models = self.training_pipeline.train_models(training_data)
                    logger.info("Models trained successfully")
                else:
                    logger.warning("No training data available, using default models")
                    # Initialize with default model parameters
                    self.models = self.training_pipeline.get_default_models()
                
            except Exception as e:
                logger.error(f"Error during model training: {str(e)}")
                logger.warning("Using default models")
                self.models = self.training_pipeline.get_default_models()
            
            # Initialize trading strategy
            self.strategy = EnhancedTradingStrategy(
                self.config,
                self.models,
                self.collector
            )
            
            # Initialize dashboard
            self.dashboard = DashboardUI(
                self.config,
                self.collector,
                self.strategy
            )
            
            logger.info("System initialization complete")
            
        except Exception as e:
            logger.error(f"Error during system initialization: {str(e)}")
            raise
            
    def run(self):
        """Main application entry point"""
        try:
            logger.info("Application entry point")
            logger.info("Starting application...")
            
            # Page setup
            st.title("AI Trading Dashboard")
            
            # Initialize components if needed
            if not self.dashboard:
                logger.info("Setting up Streamlit page...")
                self.initialize()
            
            # Run dashboard
            self.dashboard.render()
            
        except Exception as e:
            logger.error(f"Application error: {str(e)}")
            st.error("An error occurred. Please check the logs for details.")
            raise

def main():
    try:
        logger.info("Starting application...")
        
        # Initialize and run the trading system
        config_path = os.path.join(os.path.dirname(__file__), '..', 'config.yaml')
        system = TradingSystem(config_path)
        system.run()
        
    except Exception as e:
        logger.error(f"Fatal error: {str(e)}")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings("ignore")

import streamlit as st
from ui.dashboard import DashboardUI
from data.collector import DataCollector, TimeoutError as TimeoutException
from trading.manager import TradeManager
from ai.models import AITrader, TechnicalAnalysisModel, SentimentAnalyzer
import yaml
import threading
import time

@st.cache_resource
def load_config():
    """Load configuration file (cached by Streamlit)"""
    logger.info("Loading configuration...")
    try:
        with open('config.yaml', 'r') as file:
            config = yaml.safe_load(file)
              # Validate required sections and handle plural/singular forms
        section_mappings = {
            'api': ['api', 'apis'],
            'trading': ['trading'],
            'data': ['data'],
            'models': ['models', 'model']
        }
        
        missing_sections = []
        for required, alternatives in section_mappings.items():
            if not any(alt in config for alt in alternatives):
                missing_sections.append(required)
        
        if missing_sections:
            raise ValueError(f"Missing required config sections: {missing_sections}")
            
        # Normalize config to use expected keys
        if 'apis' in config and 'api' not in config:
            config['api'] = config['apis']
        if 'model' in config and 'models' not in config:
            config['models'] = config['model']
            
        logger.info("Configuration loaded successfully")
        logger.debug(f"Config sections: {list(config.keys())}")
        
        # Log configuration details (excluding sensitive data)
        safe_config = config.copy()
        if 'api' in safe_config:
            safe_config['api'] = {k: '***' for k in safe_config['api'].keys()}
        logger.debug(f"Sanitized config: {safe_config}")
        
        return config
    except FileNotFoundError:
        logger.error("config.yaml not found in working directory")
        raise
    except yaml.YAMLError as e:
        logger.error(f"YAML parsing error in config.yaml: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error loading config: {e}")
        logger.error(traceback.format_exc())
        raise

@st.cache_resource
def initialize_components(config):
    """Initialize all system components with progress tracking"""
    logger.info("Initializing system components...")
    
    # Create a progress bar
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        # Initialize DataCollector (25%)
        status_text.text("Initializing Data Collector...")
        start_time = time.time()
        collector = DataCollector(config)
        logger.info(f"DataCollector initialized in {time.time() - start_time:.2f}s")
        progress_bar.progress(25)
        
        # Initialize TradeManager (50%)
        status_text.text("Initializing Trade Manager...")
        start_time = time.time()
        trade_manager = TradeManager(config, collector)
        logger.info(f"TradeManager initialized in {time.time() - start_time:.2f}s")
        progress_bar.progress(50)
        
        # Initialize AI components (75%)
        status_text.text("Initializing AI Models...")
        start_time = time.time()
        ai_trader = AITrader(config)
        technical_analyzer = TechnicalAnalysisModel(config)
        sentiment_analyzer = SentimentAnalyzer(config)
        logger.info(f"AI components initialized in {time.time() - start_time:.2f}s")
        
        # Verify AI model training
        if not ai_trader.is_trained:
            logger.info("AI model not trained, fetching data for training...")
            # Get training data
            symbol = config.get('trading', {}).get('default_symbol', 'RELIANCE.NS')
            training_data = collector.get_historical_data(symbol)
            if not training_data.empty:
                training_data = collector.add_technical_indicators(training_data)
                logger.info(f"Training AI model with {len(training_data)} data points...")
                try:
                    ai_trader.train(training_data)
                    logger.info("AI model training completed")
                except Exception as e:
                    logger.error(f"Error training AI model: {str(e)}")
            else:
                logger.warning("No training data available")
        
        progress_bar.progress(75)
        
        # Initialize Dashboard (100%)
        status_text.text("Initializing Dashboard...")
        start_time = time.time()
        dashboard = DashboardUI(
            config=config,
            data_collector=collector,
            trade_manager=trade_manager,
            ai_trader=ai_trader,
            technical_analyzer=technical_analyzer,
            sentiment_analyzer=sentiment_analyzer
        )
        logger.info(f"DashboardUI initialized in {time.time() - start_time:.2f}s")
        progress_bar.progress(100)
        
        # Clear temporary UI elements
        status_text.empty()
        progress_bar.empty()
        
        return dashboard
        
    except Exception as e:
        progress_bar.empty()
        status_text.empty()
        logger.error(f"Error initializing components: {e}", exc_info=True)
        raise

def monitor_trading(data_collector, trade_manager, ai_trader, technical_analyzer, 
                   sentiment_analyzer, config):
    """Monitor trading in a separate thread"""
    logger = logging.getLogger('trading_monitor')
    logger.info("Starting trading monitor thread...")
    
    try:
        while True:
            logger.debug("Trading monitor iteration starting...")
            # Add your trading logic here
            time.sleep(config['ui']['update_interval'] * 60)  # Sleep for update_interval minutes
            
    except Exception as e:
        logger.error(f"Error in trading monitor: {e}")
        raise

def main():
    logger.info("Starting application...")
    start_time = time.time()
    
    try:
        # Set up Streamlit page
        logger.info("Setting up Streamlit page...")
        st.set_page_config(
            page_title="AI Trading Dashboard",
            page_icon="📈",
            layout="wide"
        )
        
        # Create title and initialization message
        st.title("AI Trading Dashboard")
        init_message = st.empty()
        init_message.info("Initializing trading system... This may take a few moments.")
        
        # Add initialization progress tracking
        progress_text = st.empty()
        progress_bar = st.progress(0)
        
        try:
            # Load configuration (cached)
            progress_text.text("Loading configuration...")
            logger.info("Loading configuration...")
            config = load_config()
            progress_bar.progress(25)
            
            # Initialize components (cached)
            progress_text.text("Initializing components...")
            logger.info("Initializing components...")
            dashboard = initialize_components(config)
            progress_bar.progress(75)
            
            # Clear initialization UI
            init_message.empty()
            progress_text.empty()
            progress_bar.empty()
            
            # Run the dashboard
            progress_text.text("Starting dashboard...")
            dashboard.run()
            progress_bar.progress(100)
            progress_text.empty()
            progress_bar.empty()
            
            # Log total initialization time
            total_time = time.time() - start_time
            logger.info(f"Application initialized in {total_time:.2f}s")
            
        except TimeoutException as e:
            logger.error(f"Initialization timed out: {e}")
            st.error("""
            ⚠️ Initialization Timeout
            
            The system initialization took too long. This might be due to:
            1. Slow network connection
            2. API service issues
            3. Heavy system load
            
            Please try again in a few moments.
            """)
            
        except ValueError as e:
            logger.error(f"Configuration error: {e}")
            st.error(f"""
            ⚠️ Configuration Error: {str(e)}
            
            Please check your config.yaml file and ensure it contains all required sections:
            - api/apis: API keys and credentials
            - trading: Trading parameters
            - data: Data collection settings
            - models/model: AI model configurations
            """)
            
        except Exception as e:
            logger.error(f"Error in main: {e}", exc_info=True)
            st.error(f"""
            ⚠️ System Error
            
            An unexpected error occurred: {str(e)}
            
            Please check the logs at {os.path.join('logs', datetime.now().strftime('%Y-%m-%d'), 'app.log')} for more details.
            """)
            
    finally:
        # Clear any remaining progress indicators
        if 'progress_bar' in locals():
            progress_bar.empty()
        if 'progress_text' in locals():
            progress_text.empty()
        if 'init_message' in locals():
            init_message.empty()

if __name__ == "__main__":
    logger.info("Application entry point")
    main()
