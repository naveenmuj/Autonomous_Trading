import os
import sys
import yaml
import logging
import pandas as pd
import numpy as np
import time
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, Tuple
from src.data.collector import DataCollector
from src.ai.agent import TradingAgent
from src.ai.models import TechnicalAnalysisModel

# Setup logging to logs/<date>/train_agent.log and console
log_dir = os.path.join('logs', datetime.now().strftime('%Y-%m-%d'))
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, 'train_agent.log')
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

def prepare_training_data(data: pd.DataFrame, config: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
    """Prepare data for training"""
    try:
        logger.info("Calculating technical indicators for training data...")
        ta_model = TechnicalAnalysisModel(config)
        processed_data = ta_model.calculate_indicators(data)
        
        # Get required features from config and always include 'pattern' and technical indicators
        features = config.get('model', {}).get('env', {}).get('features', [])
        # Add 'pattern' and all technical indicator columns if not present
        tech_cols = [col for col in processed_data.columns if col not in ['target','returns','timestamp','symbol']]
        if 'pattern' not in features:
            features.append('pattern')
        for col in tech_cols:
            if col not in features:
                features.append(col)
        # Ensure all required features exist
        missing_features = [f for f in features if f not in processed_data.columns]
        if missing_features:
            logger.error(f"Missing required features: {missing_features}")
            raise ValueError(f"Missing required features: {missing_features}")
        # Split into train/eval
        train_size = int(len(processed_data) * 0.8)
        train_data = processed_data[:train_size]
        eval_data = processed_data[train_size:]
        logger.info(f"Prepared training and evaluation data: train={len(train_data)}, eval={len(eval_data)}")
        return train_data[features].values, eval_data[features].values
        
    except Exception as e:
        logger.error(f"Error preparing training data: {str(e)}")
        raise

def get_symbols_from_config(config: Dict[str, Any], data_collector: Optional[object] = None) -> list:
    logger.info("Selecting symbols for training based on config...")
    trading_symbols = config.get('trading', {}).get('symbols', {})
    # If trading.symbols is a list, return it directly
    if isinstance(trading_symbols, list):
        logger.info(f"Symbols from trading.symbols (list): {trading_symbols}")
        if trading_symbols:
            return trading_symbols
        # If empty, fallback
        symbols = config.get('trading', {}).get('manual_symbols', [])
        if symbols:
            logger.info(f"Symbols from trading.manual_symbols: {symbols}")
            return symbols
        symbols = config.get('data', {}).get('fallback_stocks', [])
        logger.info(f"Symbols from data.fallback_stocks: {symbols}")
        return symbols
    # If trading.symbols is a dict, handle auto/manual modes
    elif isinstance(trading_symbols, dict):
        mode = trading_symbols.get('mode', 'manual')
        logger.info(f"Symbol selection mode: {mode}")
        if mode == 'auto':
            top_n = trading_symbols.get('top_n_stocks', 50)
            if data_collector and hasattr(data_collector, 'get_top_n_stocks'):
                symbols = data_collector.get_top_n_stocks(top_n)
                logger.info(f"Symbols from get_top_n_stocks({top_n}): {symbols}")
                if symbols:
                    return symbols
            symbols = trading_symbols.get('manual_list', [])
            if symbols:
                logger.info(f"Symbols from trading.symbols.manual_list: {symbols}")
                return symbols
            symbols = config.get('trading', {}).get('manual_symbols', [])
            if symbols:
                logger.info(f"Symbols from trading.manual_symbols: {symbols}")
                return symbols
            symbols = config.get('data', {}).get('fallback_stocks', [])
            logger.info(f"Symbols from data.fallback_stocks: {symbols}")
            return symbols
        else:
            symbols = trading_symbols.get('manual_list', [])
            if symbols:
                logger.info(f"Symbols from trading.symbols.manual_list: {symbols}")
                return symbols
            symbols = config.get('trading', {}).get('manual_symbols', [])
            if symbols:
                logger.info(f"Symbols from trading.manual_symbols: {symbols}")
                return symbols
            symbols = config.get('data', {}).get('fallback_stocks', [])
            logger.info(f"Symbols from data.fallback_stocks: {symbols}")
            return symbols
    # If trading.symbols is neither dict nor list, fallback
    symbols = config.get('trading', {}).get('manual_symbols', [])
    if symbols:
        logger.info(f"Symbols from trading.manual_symbols: {symbols}")
        return symbols
    symbols = config.get('data', {}).get('fallback_stocks', [])
    logger.info(f"Symbols from data.fallback_stocks: {symbols}")
    return symbols

def train_agent(config_path: str = 'config.yaml') -> None:
    """Train the trading agent"""
    try:
        logger.info(f"Loading config from {config_path}")
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        logger.info("Initializing DataCollector...")
        collector = DataCollector(config)

        # Use new symbol selection logic
        symbols = get_symbols_from_config(config, data_collector=collector)
        logger.info(f"Symbols selected for training: {symbols}")
        if not symbols:
            logger.error("No symbols configured for training (checked auto/manual modes and fallbacks)")
            raise ValueError("No symbols configured for training (checked auto/manual modes and fallbacks)")

        all_data = []
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365)  # 1 year of data
        logger.info(f"Collecting historical data for {len(symbols)} symbols from {start_date.date()} to {end_date.date()}...")
        for symbol in symbols:
            logger.info(f"Fetching data for symbol: {symbol}")
            data = collector.get_historical_data(
                symbol=symbol,
                days=365
            )
            if data is not None:
                logger.info(f"Collected {len(data)} records for {symbol}")
                all_data.append(data)
            else:
                logger.warning(f"No data returned for {symbol}")
            time.sleep(1)  # Add 1 second delay to avoid API rate limit
        if not all_data:
            logger.error("No data collected for training")
            raise ValueError("No data collected for training")
        combined_data = pd.concat(all_data)
        combined_data.sort_index(inplace=True)
        logger.info(f"Total combined training records: {len(combined_data)}")
        train_data, eval_data = prepare_training_data(combined_data, config)
        logger.info("Initializing TradingAgent and starting training...")
        agent = TradingAgent(config)
        agent.train(train_data, eval_data)
        logger.info("Agent training completed successfully")
    except Exception as e:
        logger.error(f"Error in training pipeline: {str(e)}")
        raise

if __name__ == "__main__":
    # Logging is already set up at the top
    config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'config.yaml')
    train_agent(config_path)
