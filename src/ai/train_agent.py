import os
import sys
import yaml
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, Tuple
from ..data.collector import DataCollector
from .agent import TradingAgent
from .models import TechnicalAnalysisModel

logger = logging.getLogger(__name__)

def prepare_training_data(data: pd.DataFrame, config: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
    """Prepare data for training"""
    try:
        # Calculate technical indicators
        ta_model = TechnicalAnalysisModel(config)
        processed_data = ta_model.calculate_indicators(data)
        
        # Get required features from config
        features = config.get('model', {}).get('env', {}).get('features', [])
        
        # Ensure all required features exist
        missing_features = [f for f in features if f not in processed_data.columns]
        if missing_features:
            raise ValueError(f"Missing required features: {missing_features}")
            
        # Split into train/eval
        train_size = int(len(processed_data) * 0.8)
        train_data = processed_data[:train_size]
        eval_data = processed_data[train_size:]
        
        return train_data[features].values, eval_data[features].values
        
    except Exception as e:
        logger.error(f"Error preparing training data: {str(e)}")
        raise

def train_agent(config_path: str = 'config.yaml') -> None:
    """Train the trading agent"""
    try:
        # Load configuration
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        
        # Initialize data collector
        collector = DataCollector(config)
        
        # Get training data
        symbols = config.get('data', {}).get('symbols', [])
        if not symbols:
            raise ValueError("No symbols configured for training")
            
        all_data = []
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365)  # 1 year of data
        
        for symbol in symbols:
            data = collector.get_historical_data(
                symbol=symbol,
                start_date=start_date,
                end_date=end_date,
                interval='1d'
            )
            if data is not None:
                all_data.append(data)
        
        if not all_data:
            raise ValueError("No data collected for training")
            
        # Combine all data
        combined_data = pd.concat(all_data)
        combined_data.sort_index(inplace=True)
        
        # Prepare training data
        train_data, eval_data = prepare_training_data(combined_data, config)
        
        # Initialize and train agent
        agent = TradingAgent(config)
        agent.train(train_data, eval_data)
        
        logger.info("Agent training completed successfully")
        
    except Exception as e:
        logger.error(f"Error in training pipeline: {str(e)}")
        raise

if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('logs/training.log'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    # Get the absolute path to config.yaml
    config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'config.yaml')
    
    # Train the agent
    train_agent(config_path)
