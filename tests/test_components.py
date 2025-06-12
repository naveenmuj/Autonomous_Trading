#!/usr/bin/env python
"""Component Test Script

This script tests each component of the trading system individually to verify they work correctly.
"""

import os
import sys
import logging
import pandas as pd
import numpy as np
from datetime import datetime

# Add project root to python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.collector import DataCollector
from src.trading.manager import TradeManager
from src.ai.models import AITrader, TechnicalAnalysisModel, SentimentAnalyzer

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)

def test_data_collector(logger, config):
    """Test data collection functionality"""
    logger.info("Testing DataCollector...")
    
    try:
        collector = DataCollector(config)
        symbol = "RELIANCE"
        
        # Test historical data
        data = collector.get_historical_data(symbol)
        if isinstance(data, pd.DataFrame):
            logger.info(f"Successfully fetched historical data for {symbol}")
            logger.info(f"Data shape: {data.shape}")
            logger.info(f"Columns: {data.columns.tolist()}")
        else:
            logger.error(f"Failed to get proper DataFrame for {symbol}")
        
        return collector
        
    except Exception as e:
        logger.error(f"DataCollector test failed: {e}")
        raise

def test_trade_manager(logger, config, mock_data_collector):
    """Test trade management functionality"""
    logger.info("Testing TradeManager...")
    
    try:
        manager = TradeManager(config, mock_data_collector)
        
        # Test trade simulation
        trade = manager.place_trade(
            symbol="RELIANCE",
            action="BUY",
            quantity=1,
            price=2500.0,
            stop_loss=2450.0,
            target=2600.0
        )
        
        if trade:
            logger.info("Successfully placed test trade")
            logger.info(f"Trade details: {trade}")
            
        return manager
        
    except Exception as e:
        logger.error(f"TradeManager test failed: {e}")
        raise

def test_ai_models(logger, config):
    """Test AI models functionality"""
    logger.info("Testing AI models...")
    
    try:
        # Test AITrader
        trader = AITrader(config)
        
        # Test TechnicalAnalysisModel
        ta_model = TechnicalAnalysisModel()
        dummy_data = np.random.random((1, 60, 5))
        prediction = ta_model.predict(dummy_data)
        logger.info(f"Technical analysis prediction shape: {prediction.shape}")
        
        # Test SentimentAnalyzer
        sentiment = SentimentAnalyzer()
        test_news = ["Positive market outlook", "Negative earnings report"]
        score = sentiment.analyze_news(test_news)
        logger.info(f"Sentiment score for test news: {score}")
        
        return trader, ta_model, sentiment
        
    except Exception as e:
        logger.error(f"AI models test failed: {e}")
        raise

def main():
    logger = setup_logging()
    logger.info("Starting component tests...")
    
    try:
        # Load config
        import yaml
        with open('config.yaml', 'r') as file:
            config = yaml.safe_load(file)
            
        # Test each component
        collector = test_data_collector(logger, config)
        manager = test_trade_manager(logger, config, collector)
        trader, ta_model, sentiment = test_ai_models(logger, config)
        
        logger.info("All component tests completed successfully!")
        
    except Exception as e:
        logger.error(f"Component testing failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
