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

try:
    import sys
    print(f"[DEBUG] sys.version: {sys.version}")
except Exception as e:
    print(f"[DEBUG] sys import failed: {e}")

import sys
import os
import yaml
import warnings
import psutil
import logging.handlers
from pathlib import Path
from datetime import datetime, timedelta
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

# TensorFlow will be loaded lazily to avoid Streamlit conflicts
TF_AVAILABLE = False
load_model = None

def initialize_tensorflow():
    """Lazy load TensorFlow after Streamlit is fully initialized"""
    global TF_AVAILABLE, load_model
    try:
        from tensorflow.keras.models import load_model as tf_load_model
        load_model = tf_load_model
        TF_AVAILABLE = True
        print("[INFO] TensorFlow loaded successfully (lazy loading)")
        return True
    except ImportError as e:
        print(f"[WARNING] TensorFlow not available: {e}")
        TF_AVAILABLE = False
        def dummy_load_model(*args, **kwargs):
            print("[INFO] Dummy load_model called (TensorFlow unavailable)")
            return None
        load_model = dummy_load_model
        return False

import torch
from apscheduler.schedulers.background import BackgroundScheduler
import subprocess
import csv
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from pytz import timezone

# Make concurrent_log_handler optional
try:
    from concurrent_log_handler import ConcurrentRotatingFileHandler
    CONCURRENT_LOG_AVAILABLE = True
except ImportError:
    from logging.handlers import RotatingFileHandler as ConcurrentRotatingFileHandler
    CONCURRENT_LOG_AVAILABLE = False
    print("[WARNING] concurrent_log_handler not available, using standard RotatingFileHandler")

# Remove any existing handlers from the root logger
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)

# Configure logging with both console and Streamlit visibility
import streamlit as st

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [%(name)s] - [%(levelname)s] - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)  # Explicit stdout for Streamlit
    ],
    force=True  # Force reconfiguration
)

# Also add a custom handler for Streamlit
class StreamlitLogHandler(logging.Handler):
    def __init__(self):
        super().__init__()
        self.log_messages = []
    
    def emit(self, record):
        msg = self.format(record)
        self.log_messages.append(msg)
        # Keep only last 100 messages
        if len(self.log_messages) > 100:
            self.log_messages = self.log_messages[-100:]
        print(f"[STREAMLIT LOG] {msg}")

# Add the Streamlit handler
streamlit_handler = StreamlitLogHandler()
streamlit_handler.setLevel(logging.INFO)
logging.getLogger().addHandler(streamlit_handler)

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
app_handler = ConcurrentRotatingFileHandler(
    app_log, maxBytes=10*1024*1024, backupCount=5
)
app_handler.setLevel(logging.INFO)
app_handler.setFormatter(detailed_formatter)

# Trading handler - All levels
trade_handler = ConcurrentRotatingFileHandler(
    trade_log, maxBytes=10*1024*1024, backupCount=5
)
trade_handler.setLevel(logging.DEBUG)
trade_handler.setFormatter(detailed_formatter)

# Model handler - All levels
model_handler = ConcurrentRotatingFileHandler(
    model_log, maxBytes=10*1024*1024, backupCount=5
)
model_handler.setLevel(logging.DEBUG)
model_handler.setFormatter(detailed_formatter)

# Debug handler - Debug only
debug_handler = ConcurrentRotatingFileHandler(
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

def log_system_info():
    """Log system information once during initialization"""
    logger.info("=== System Information ===")
    logger.info(f"Python version: {sys.version}")
    logger.info(f"CPU cores: {psutil.cpu_count()}")
    logger.info(f"Memory available: {psutil.virtual_memory().available / (1024*1024*1024):.1f}GB")
    try:
        # Use Windows drive path instead of Unix '/'
        disk_usage = psutil.disk_usage('C:\\')
        logger.info(f"Disk space available: {disk_usage.free / (1024*1024*1024):.1f}GB")
    except Exception as e:
        logger.warning(f"Could not get disk usage: {e}")
    logger.info("=== System Check Complete ===")

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
            
        # Validate critical API configuration
        api_config = config.get('api', {}) or config.get('apis', {})
        angel_one_config = api_config.get('angel_one', {})
        
        required_angel_one_fields = ['api_key', 'client_id', 'mpin', 'totp_secret']
        missing_angel_fields = [field for field in required_angel_one_fields 
                               if not angel_one_config.get(field)]
        
        if missing_angel_fields:
            raise ValueError(f"Missing required Angel One API fields: {missing_angel_fields}")
            
        # Validate API key format (basic checks)
        api_key = angel_one_config.get('api_key', '')
        if api_key and len(api_key) < 6:
            raise ValueError("Angel One API key appears invalid (too short)")
            
        # Validate TOTP secret format
        totp_secret = angel_one_config.get('totp_secret', '')
        if totp_secret and len(totp_secret) < 16:
            raise ValueError("TOTP secret appears invalid (too short)")
        
        # Validate client_id format  
        client_id = angel_one_config.get('client_id', '')
        if client_id and len(client_id) < 8:
            raise ValueError("Angel One client_id appears invalid (too short)")
        
        # Validate trading configuration
        trading_config = config.get('trading', {})
        if 'risk_management' in trading_config:
            risk_config = trading_config['risk_management']
            max_position_size = risk_config.get('max_position_size_percent')
            if max_position_size is not None and (max_position_size <= 0 or max_position_size > 100):
                raise ValueError(f"Invalid max_position_size_percent: {max_position_size}. Must be between 0 and 100")
                
        logger.info("Configuration validation passed")
            
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

def load_pretrained_models(config):
    """Load pre-trained models for live trading. Retrain separately and reload as needed."""
    models = {}
    # LSTM Technical Analysis Model
    try:
        lstm_path = os.path.join(project_root, 'models', 'ta_lstm_model.h5')
        if os.path.exists(lstm_path):
            models['ta_lstm'] = load_model(lstm_path)
            logger.info(f"Loaded LSTM model from {lstm_path}")
        else:
            logger.warning(f"LSTM model not found at {lstm_path}")
    except Exception as e:
        logger.error(f"Failed to load LSTM model: {e}")
    # RL Model (Stable Baselines3 PPO)
    try:
        rl_path = os.path.join(project_root, 'models', 'trading_model_rl.zip')
        if os.path.exists(rl_path):
            from stable_baselines3 import PPO
            models['ppo'] = PPO.load(rl_path)
            logger.info(f"Loaded RL model from {rl_path}")
        else:
            logger.warning(f"RL model not found at {rl_path}")
    except Exception as e:
        logger.error(f"Failed to load RL model: {e}")
    # Transformer Model (PyTorch)
    try:
        transformer_path = os.path.join(project_root, 'models', 'transformer_model_best.pt')
        input_dim = None
        # Try to get input_dim from config or a metadata file
        try:
            input_dim = config.get('model', {}).get('transformer', {}).get('input_dim')
        except Exception:
            input_dim = None
        if input_dim is None:
            # Try to read from a metadata file if available
            meta_path = os.path.join(project_root, 'models', 'transformer_model_meta.yaml')
            if os.path.exists(meta_path):
                import yaml
                with open(meta_path, 'r') as f:
                    meta = yaml.safe_load(f)
                    input_dim = meta.get('input_dim', 20)
            else:
                input_dim = 20  # Fallback to 20 (matches training)
        if os.path.exists(transformer_path):
            from ai.train import TimeSeriesTransformer
            model = TimeSeriesTransformer(input_dim=input_dim, output_dim=2, d_model=128, nhead=8, num_layers=3, dropout=0.2)
            state_dict = torch.load(transformer_path, map_location=torch.device('cpu'))
            try:
                model.load_state_dict(state_dict)
                model.eval()
                models['transformer'] = model
                logger.info(f"Loaded Transformer model from {transformer_path} with input_dim={input_dim}")
            except RuntimeError as e:
                logger.error(f"Transformer model state_dict loading failed: {e}\nCheck that input_dim used for inference matches training (expected {input_dim}).")
        else:
            logger.warning(f"Transformer model not found at {transformer_path}")
    except Exception as e:
        logger.error(f"Failed to load Transformer model: {e}")
    return models

# @st.cache_resource
def initialize_components(config):
    """Initialize all system components, loading pre-trained models only."""
    import streamlit as st
    logger.info("=== ENTER initialize_components() ===")
    print("=== ENTER initialize_components() ===")
    try:
        # Show progress for each step
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        status_text.text("🔧 Initializing Data Collector...")
        progress_bar.progress(10)
        try:
            logger.info("About to create DataCollector instance...")
            collector = DataCollector(config)
            logger.info("DataCollector instance created successfully")
            
            # Initialize websocket connection for live prices
            logger.info("Initializing websocket connection for live price data...")
            try:
                collector.ensure_websocket_connected()
                logger.info("WebSocket connection initialized")
            except Exception as ws_e:
                logger.warning(f"WebSocket initialization failed: {ws_e}")
            
            logger.info("Step 1: DataCollector initialized")
        except Exception as e:
            logger.error(f"Error creating DataCollector: {e}")
            logger.exception("Full exception details:")
            raise
        
        # WebSocket is already initialized in DataCollector constructor
        status_text.text("🌐 Live data stream already connected...")
        progress_bar.progress(25)
        logger.info("Step 1.1: WebSocket already initialized in DataCollector")
            
        status_text.text("💼 Initializing Trade Manager...")
        progress_bar.progress(40)
        trade_manager = TradeManager(config, collector)
        logger.info("Step 2: TradeManager initialized")
        
        status_text.text("📈 Loading Technical Analysis Models...")
        progress_bar.progress(60)
        technical_analyzer = TechnicalAnalysisModel(config)
        logger.info("Step 3: TechnicalAnalysisModel initialized")
        
        status_text.text("🤖 Initializing AI Models...")
        progress_bar.progress(75)
        ai_model = AITrader(config)
        logger.info("Step 4: AITrader initialized")
        
        status_text.text("🧠 Loading Pre-trained Models...")
        progress_bar.progress(85)
        pretrained_models = load_pretrained_models(config)
        logger.info("Step 5: Pre-trained models loaded")
        if hasattr(ai_model, 'set_loaded_models'):
            ai_model.set_loaded_models(pretrained_models)
        if hasattr(technical_analyzer, 'set_loaded_models'):
            technical_analyzer.set_loaded_models(pretrained_models)
            
        status_text.text("🎨 Initializing Dashboard UI...")
        progress_bar.progress(95)
        logger.info("Step 6: Initializing DashboardUI...")
        print("=== initialize_components(): Instantiating DashboardUI ===")
        dashboard = DashboardUI(
            config=config,
            data_collector=collector,
            trade_manager=trade_manager,
            technical_analyzer=technical_analyzer,
            ai_model=ai_model
        )
        
        status_text.text("✅ All components initialized!")
        progress_bar.progress(100)
        logger.info("All components initialized successfully")
        print("=== EXIT initialize_components() ===")
        logger.info("=== EXIT initialize_components() ===")
        
        # Clear the progress indicators
        import time
        time.sleep(1)
        progress_bar.empty()
        status_text.empty()
        
        return dashboard
    except Exception as e:
        logger.error(f"Error initializing components: {str(e)}", exc_info=True)
        print(f"Error initializing components: {str(e)}")
        return None

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

def send_retrain_notification(subject, body):
    # Read SMTP config from config.yaml
    try:
        with open(os.path.join(project_root, 'config.yaml'), 'r') as f:
            config = yaml.safe_load(f)
        smtp_config = config.get('notifications', {}).get('smtp', {})
        SMTP_SERVER = smtp_config.get('server', 'smtp.example.com')
        SMTP_PORT = int(smtp_config.get('port', 587))
        SMTP_USER = smtp_config.get('user', 'your@email.com')
        SMTP_PASS = smtp_config.get('password', 'yourpassword')
        TO_EMAIL = smtp_config.get('to', 'your@email.com')
        FROM_EMAIL = SMTP_USER
    except Exception as e:
        logger.error(f"Failed to load SMTP config from config.yaml: {e}")
        return
    try:
        msg = MIMEMultipart()
        msg['From'] = FROM_EMAIL
        msg['To'] = TO_EMAIL
        msg['Subject'] = subject
        msg.attach(MIMEText(body, 'plain'))
        with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
            server.starttls()
            server.login(SMTP_USER, SMTP_PASS)
            server.sendmail(FROM_EMAIL, TO_EMAIL, msg.as_string())
        logger.info(f"Retrain notification sent to {TO_EMAIL}")
    except Exception as e:
        logger.error(f"Failed to send retrain notification: {e}")

def run_training_pipeline():
    # Create a unique scheduler log folder for this run
    run_start = datetime.now()
    run_dir = os.path.join(project_root, 'logs', run_start.strftime('%Y-%m-%d'), f'scheduler_{run_start.strftime("%H-%M-%S")}')
    os.makedirs(run_dir, exist_ok=True)
    summary_log_file = os.path.join(run_dir, 'scheduler_run.log')
    debug_log_file = os.path.join(run_dir, 'debug.log')
    with open(summary_log_file, 'a', encoding='utf-8') as f_summary, open(debug_log_file, 'a', encoding='utf-8') as f_debug:
        f_summary.write(f"=== SCHEDULED TRAINING STARTED at {run_start.strftime('%Y-%m-%d %H:%M:%S')} ===\n")
        try:
            train_models_path = os.path.join(project_root, 'src', 'ai', 'train_models.py')
            train_agent_path = os.path.join(project_root, 'src', 'ai', 'train_agent.py')
            f_summary.write(f"Running train_models.py: {train_models_path}\n")
            t1_start = datetime.now()
            result1 = subprocess.run([sys.executable, train_models_path], capture_output=True, text=True)
            t1_end = datetime.now()
            f_summary.write(f"train_models.py started at {t1_start.strftime('%H:%M:%S')}, ended at {t1_end.strftime('%H:%M:%S')}\n")
            f_summary.write(f"train_models.py status: {'SUCCESS' if result1.returncode == 0 else 'FAIL'}\n")
            f_debug.write(f"\n--- train_models.py STDOUT ---\n{result1.stdout}\n")
            if result1.stderr:
                f_debug.write(f"\n--- train_models.py STDERR ---\n{result1.stderr}\n")
            if result1.returncode != 0:
                f_summary.write(f"train_models.py exited with code {result1.returncode}\n")
                f_summary.write(f"=== SCHEDULED TRAINING FAILED at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ===\n")
                return
            f_summary.write(f"Running train_agent.py: {train_agent_path}\n")
            t2_start = datetime.now()
            result2 = subprocess.run([sys.executable, train_agent_path], capture_output=True, text=True)
            t2_end = datetime.now()
            f_summary.write(f"train_agent.py started at {t2_start.strftime('%H:%M:%S')}, ended at {t2_end.strftime('%H:%M:%S')}\n")
            f_summary.write(f"train_agent.py status: {'SUCCESS' if result2.returncode == 0 else 'FAIL'}\n")
            f_debug.write(f"\n--- train_agent.py STDOUT ---\n{result2.stdout}\n")
            if result2.stderr:
                f_debug.write(f"\n--- train_agent.py STDERR ---\n{result2.stderr}\n")
            if result2.returncode != 0:
                f_summary.write(f"train_agent.py exited with code {result2.returncode}\n")
                f_summary.write(f"=== SCHEDULED TRAINING FAILED at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ===\n")
                return
            f_summary.write(f"=== SCHEDULED TRAINING SUCCESS at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ===\n")
        except Exception as e:
            f_summary.write(f"Exception in scheduled training: {e}\n")
            f_summary.write(f"=== SCHEDULED TRAINING FAILED at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ===\n")
            f_debug.write(f"\n--- Exception in scheduled training ---\n{e}\n")
    # ...existing code...

def main():
    print("=== ENTER main() ===")
    logging.info("=== ENTER main() ===")
    
    # Initialize TensorFlow lazily after Streamlit is ready
    print("Initializing TensorFlow...")
    initialize_tensorflow()
    
    try:
        config = load_config()
        print("=== main(): load_config() returned ===")
        logging.info("=== main(): load_config() returned ===")

        dashboard = initialize_components(config)
        if dashboard is not None:
            dashboard.render()
        else:
            print("Dashboard failed to initialize. Check logs for details.")
            return
    except Exception as e:
        logging.exception("Exception in main(): %s", e)
        print(f"Exception during initialization: {e}")
        import traceback
        print(traceback.format_exc())
        return

# Initialize components using Streamlit session state for persistence
if 'dashboard' not in st.session_state:
    print("=== Initializing dashboard in session state ===")
    logger.info("=== Initializing dashboard in session state ===")
    
    # Log system information once
    log_system_info()
    
    # Show loading indicator
    with st.spinner("🚀 Initializing AI Trading System..."):
        st.info("📊 Loading configuration...")
        try:
            config = load_config()
            print("=== Session state init: load_config() returned ===")
            logger.info("=== Session state init: load_config() returned ===")
            
            st.info("🔌 Connecting to market data sources...")
            dashboard = initialize_components(config)
            if dashboard is not None:
                st.session_state.dashboard = dashboard
                print("=== Dashboard stored in session state ===")
                logger.info("=== Dashboard stored in session state ===")
                st.success("✅ AI Trading System initialized successfully!")
            else:
                st.error("❌ Dashboard failed to initialize. Check logs for details.")
                st.stop()
        except Exception as e:
            logger.exception("Exception during initialization: %s", e)
            st.error(f"❌ Exception during initialization: {e}")
            st.stop()

# Render the dashboard
if 'dashboard' in st.session_state:
    st.session_state.dashboard.render()

# NOTE: Model training is NOT performed at startup. Run ai/train.py or ai/training_pipeline.py separately to retrain models.
# Reload models by restarting the app or by implementing a reload mechanism if needed.

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings("ignore")
