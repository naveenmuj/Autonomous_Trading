import streamlit as st
# Must set page config as the first Streamlit command
st.set_page_config(
    page_title="Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/yourusername/trading-dashboard',
        'Report a bug': "https://github.com/yourusername/trading-dashboard/issues",
        'About': "# AI Trading Dashboard\nA sophisticated trading platform powered by AI."
    }
)

import sys
import os
import yaml
import warnings
import psutil
import logging.handlers
from pathlib import Path
from datetime import datetime
import traceback
import time
import pandas as pd
import numpy as np
from data.collector import DataCollector
from trading.manager import TradeManager
from ai.training_pipeline import TrainingPipeline
from trading.strategy import EnhancedTradingStrategy
from ui.dashboard import DashboardUI
from ai.models import AITrader, TechnicalAnalysisModel, SentimentAnalyzer

# Remove any existing handlers from the root logger
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)

# Configure logging with a single handler
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [%(name)s] - [%(levelname)s] - %(message)s',
    handlers=[
        logging.StreamHandler()  # Console handler
    ]
)

# Project paths setup
project_root = str(Path(__file__).parent.parent.absolute())
src_path = str(Path(__file__).parent.absolute())
for path in [project_root, src_path]:
    if path not in sys.path:
        sys.path.insert(0, path)

# Enhanced logging configuration
log_dir = os.path.join(project_root, 'logs', datetime.now().strftime('%Y-%m-%d'))
os.makedirs(log_dir, exist_ok=True)

# Separate log files for different components
app_log = os.path.join(log_dir, 'app.log')
trade_log = os.path.join(log_dir, 'trading.log')
model_log = os.path.join(log_dir, 'model.log')
debug_log = os.path.join(log_dir, 'debug.log')

class DetailedFormatter(logging.Formatter):
    def format(self, record):
        # Add memory usage info
        process = psutil.Process(os.getpid())
        record.memory_usage = f"{process.memory_info().rss / 1024 / 1024:.1f}MB"
        
        # Add timestamp with milliseconds
        record.created_fmt = datetime.fromtimestamp(record.created).strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
        
        return super().format(record)

# Create formatters
detailed_formatter = DetailedFormatter(
    '%(created_fmt)s - [%(name)s] - %(levelname)s - [%(threadName)s] - '
    '[Mem: %(memory_usage)s] - %(message)s'
)
simple_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

# Configure root logger
root_logger = logging.getLogger()
root_logger.setLevel(logging.DEBUG)  # Set to DEBUG to capture all levels

# App handler - Info and above
app_handler = logging.handlers.RotatingFileHandler(
    app_log, maxBytes=10*1024*1024, backupCount=5
)
app_handler.setLevel(logging.INFO)
app_handler.setFormatter(detailed_formatter)

# Trading handler - All levels
trade_handler = logging.handlers.RotatingFileHandler(
    trade_log, maxBytes=10*1024*1024, backupCount=5
)
trade_handler.setLevel(logging.DEBUG)
trade_handler.setFormatter(detailed_formatter)

# Model handler - All levels
model_handler = logging.handlers.RotatingFileHandler(
    model_log, maxBytes=10*1024*1024, backupCount=5
)
model_handler.setLevel(logging.DEBUG)
model_handler.setFormatter(detailed_formatter)

# Debug handler - Debug only
debug_handler = logging.handlers.RotatingFileHandler(
    debug_log, maxBytes=20*1024*1024, backupCount=10
)
debug_handler.setLevel(logging.DEBUG)
debug_handler.setFormatter(detailed_formatter)

# Console handler - Info and above
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(simple_formatter)

# Add all handlers to root logger
root_logger.addHandler(app_handler)
root_logger.addHandler(trade_handler)
root_logger.addHandler(model_handler)
root_logger.addHandler(debug_handler)
root_logger.addHandler(console_handler)

# Create module loggers
logger = logging.getLogger(__name__)
trading_logger = logging.getLogger('trading')
model_logger = logging.getLogger('model')
data_logger = logging.getLogger('data')

# Suppress warnings
warnings.filterwarnings('ignore')

# Log system info at startup
logger.info("=== System Information ===")
logger.info(f"Python version: {sys.version}")
logger.info(f"CPU cores: {psutil.cpu_count()}")
logger.info(f"Memory available: {psutil.virtual_memory().available / (1024*1024*1024):.1f}GB")
logger.info(f"Disk space available: {psutil.disk_usage('/').free / (1024*1024*1024):.1f}GB")
logger.info("=== System Check Complete ===")

@st.cache_resource
def load_config():
    """Load configuration file (cached by Streamlit)"""
    logger.info("Loading configuration...")
    try:
        with open('config.yaml', 'r') as file:
            config = yaml.safe_load(file)
        
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
        return config
        
    except Exception as e:
        logger.error(f"Error loading config: {e}")
        raise

@st.cache_resource
def initialize_components(config):
    """Initialize all system components"""
    logger.info("Initializing system components...")
    
    try:
        # Initialize DataCollector
        collector = DataCollector(config)
        
        # Initialize Trade Manager
        trade_manager = TradeManager(config, collector)
        
        # Initialize Models
        technical_analyzer = TechnicalAnalysisModel(config)
        ai_model = AITrader(config)
        
        # Initialize Dashboard
        dashboard = DashboardUI(
            config=config,
            data_collector=collector,
            trade_manager=trade_manager,
            technical_analyzer=technical_analyzer,
            ai_model=ai_model
        )
        
        logger.info("All components initialized successfully")
        return dashboard
        
    except Exception as e:
        logger.error(f"Error initializing components: {str(e)}")
        raise

def initialize_system():
    """Initialize system components with proper configuration"""
    try:
        # Load config
        config_path = os.path.join(project_root, 'config.yaml')
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Update symbol names to match Angel One format
        trading_config = config.get('trading', {})
        symbol_mapping = {
            'HDFC.NS': 'HDFCBANK.NS',  # Correct HDFC to HDFCBANK
            'ICICI.NS': 'ICICIBANK.NS',  # Correct ICICI to ICICIBANK
        }
        
        symbols = trading_config.get('symbols', [])
        corrected_symbols = [symbol_mapping.get(s, s) for s in symbols]
        trading_config['symbols'] = corrected_symbols
        config['trading'] = trading_config
        
        # Save corrected config
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
            
        return config
        
    except Exception as e:
        logger.error(f"Error initializing system: {e}")
        raise

def main():
    logger.info("Starting application...")
    
    try:
        # Create title and initialization message
        st.title("")
        init_message = st.empty()
        init_message.info("Initializing trading system... This may take a few moments.")
        
        # Add initialization progress tracking
        progress_text = st.empty()
        progress_bar = st.progress(0)
        
        try:
            # Load configuration (cached)
            progress_text.text("Loading configuration...")
            config = load_config()
            progress_bar.progress(25)
            
            # Initialize components (cached)
            progress_text.text("Initializing components...")
            dashboard = initialize_components(config)
            progress_bar.progress(75)
            
            # Clear initialization UI
            init_message.empty()
            progress_text.empty()
            progress_bar.empty()
            
            # Run the dashboard
            dashboard.run()
            
        except Exception as e:
            logger.error(f"Error in main: {e}", exc_info=True)
            st.error(f"""
            ⚠️ System Error
              An unexpected error occurred: {str(e)}
            
            Please check the logs in the {log_dir} directory for more details.
            """)
            
    finally:
        # Clear any remaining progress indicators
        if 'progress_bar' in locals(): progress_bar.empty()
        if 'progress_text' in locals(): progress_text.empty()
        if 'init_message' in locals(): init_message.empty()

if __name__ == "__main__":
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    warnings.filterwarnings("ignore")
    main()
