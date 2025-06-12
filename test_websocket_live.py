import os
import sys
import yaml
import time
import logging
import pytz
from datetime import datetime
from src.data.collector import DataCollector
from src.data.market_utils import is_market_open, format_time_until_market_open

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [%(name)s] - [%(levelname)s] - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    try:
        # Load configuration
        with open('config.yaml', 'r') as f:
            config = yaml.safe_load(f)
            
        # Override config to add SUZLON for testing
        if 'trading' not in config:
            config['trading'] = {}
        if 'data' not in config['trading']:
            config['trading']['data'] = {}
        config['trading']['data']['manual_symbols'] = ['SUZLON-EQ']
        
        # Check market hours
        if not is_market_open():
            wait_time = format_time_until_market_open()
            logger.warning(f"Market is currently closed. Next market opening in {wait_time}")
            logger.info("Running in simulation mode for testing...")
            
        # Initialize DataCollector
        collector = DataCollector(config)
        
        # Wait for WebSocket connection
        time.sleep(2)
        
        try:
            # Keep running and print updates
            while True:
                market_data = collector.get_market_data()
                current_time = datetime.now(pytz.timezone('Asia/Kolkata'))
                
                logger.info("\nMarket Status Update:")
                logger.info(f"Current Time (IST): {current_time.strftime('%Y-%m-%d %H:%M:%S')}")
                logger.info(f"Market Status: {'OPEN' if is_market_open() else 'CLOSED'}")
                
                if not market_data.empty:
                    logger.info("\nCurrent Market Data:")
                    logger.info(market_data[['symbol', 'ltp', 'volume', 'bid_price', 'ask_price']].to_string())
                else:
                    if is_market_open():
                        logger.warning("No market data received (Market is open)")
                    else:
                        next_open = format_time_until_market_open()
                        logger.info(f"No market data available - Market is closed. Opens in {next_open}")
                    
        except KeyboardInterrupt:
            logger.info("Stopping data collection...")
            collector.cleanup()
            
    except Exception as e:
        logger.error(f"Error in main: {str(e)}")
        raise

if __name__ == "__main__":
    main()
