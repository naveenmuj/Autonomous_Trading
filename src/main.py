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
from tensorflow.keras.models import load_model
import torch
from apscheduler.schedulers.background import BackgroundScheduler
import subprocess
import csv
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

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

@st.cache_resource
def initialize_components(config):
    """Initialize all system components, loading pre-trained models only."""
    logger.info("Initializing system components...")
    try:
        # Initialize DataCollector
        collector = DataCollector(config)
        
        # Initialize Trade Manager
        trade_manager = TradeManager(config, collector)
        
        # Initialize Models (load pre-trained only)
        technical_analyzer = TechnicalAnalysisModel(config)
        ai_model = AITrader(config)
        pretrained_models = load_pretrained_models(config)
        # Optionally, pass loaded models to AITrader/TechnicalAnalysisModel if needed
        if hasattr(ai_model, 'set_loaded_models'):
            ai_model.set_loaded_models(pretrained_models)
        if hasattr(technical_analyzer, 'set_loaded_models'):
            technical_analyzer.set_loaded_models(pretrained_models)
        
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

def retrain_model_job():
    logger.info("Scheduled retraining started...")
    csv_path = os.path.join(project_root, 'logs', 'retrain_history.csv')
    now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    try:
        # Run your training script as a subprocess
        result = subprocess.run(
            [sys.executable, os.path.join(project_root, "src", "ai", "train.py")],
            capture_output=True, text=True
        )
        success = result.returncode == 0
        log_row = {
            'timestamp': now,
            'status': 'success' if success else 'error',
            'stdout': result.stdout.strip()[:500],
            'stderr': result.stderr.strip()[:500]
        }
        # Write/append to CSV
        file_exists = os.path.isfile(csv_path)
        with open(csv_path, 'a', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=['timestamp', 'status', 'stdout', 'stderr'])
            if not file_exists:
                writer.writeheader()
            writer.writerow(log_row)
        if success:
            logger.info("Retraining completed successfully.")
            send_retrain_notification(
                subject="Model Retraining Success",
                body=f"Retraining completed successfully at {now}.\n\nOutput:\n{result.stdout.strip()[:500]}"
            )
        else:
            logger.error(f"Retraining failed: {result.stderr}")
            send_retrain_notification(
                subject="Model Retraining FAILED",
                body=f"Retraining failed at {now}.\n\nError:\n{result.stderr.strip()[:500]}"
            )
            raise RuntimeError(f"Retraining failed: {result.stderr}")
    except Exception as e:
        logger.error(f"Retraining job error: {e}")
        # Log error to CSV as well
        with open(csv_path, 'a', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=['timestamp', 'status', 'stdout', 'stderr'])
            if os.path.getsize(csv_path) == 0:
                writer.writeheader()
            writer.writerow({
                'timestamp': now,
                'status': 'error',
                'stdout': '',
                'stderr': str(e)
            })
        send_retrain_notification(
            subject="Model Retraining ERROR",
            body=f"Retraining job error at {now}.\n\nException:\n{str(e)}"
        )
        raise

def main():
    logger.info("Starting application...")
    scheduler = BackgroundScheduler()
    # Schedule retraining every day at 2am
    scheduler.add_job(retrain_model_job, 'cron', hour=2, minute=0)
    scheduler.start()
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

# NOTE: Model training is NOT performed at startup. Run ai/train.py or ai/training_pipeline.py separately to retrain models.
# Reload models by restarting the app or by implementing a reload mechanism if needed.
