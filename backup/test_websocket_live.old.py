import os
import sys
import yaml
import time
import logging
import pytz
import signal
from datetime import datetime
from src.data.collector import DataCollector
from src.data.market_utils import is_market_open, format_time_until_market_open

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [%(name)s] - [%(levelname)s] - %(message)s'
)
logger = logging.getLogger(__name__)

# Global flag for program control
running = True

def signal_handler(signum, frame):
    """Handle interrupt signals"""
    global running
    logger.info("\nReceived interrupt signal. Cleaning up...")
    running = False

def on_market_data(tick_data):
    """Callback function for market data updates"""
    current_time = datetime.now(pytz.timezone('Asia/Kolkata'))
    
    # Print market data update
    logger.info(f"\nMarket Data Update at {current_time.strftime('%Y-%m-%d %H:%M:%S')}:")
    logger.info(f"Symbol: {tick_data['symbol']}")
    logger.info(f"LTP: {tick_data['ltp']}")
    logger.info(f"Volume: {tick_data['volume']}")
    logger.info(f"Bid/Ask: {tick_data['bid_price']}/{tick_data['ask_price']}")

def main():
    global running
    collector = None
    
    try:
        # Set up signal handlers
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

        # Load configuration
        with open('config.yaml', 'r') as f:
            config = yaml.safe_load(f)
            
        # Ensure trading config sections exist
        if 'trading' not in config:
            config['trading'] = {}
        if 'data' not in config['trading']:
            config['trading']['data'] = {'mode': 'manual', 'manual_symbols': []}

        # Check market hours
        if not is_market_open():
            wait_time = format_time_until_market_open()
            logger.warning(f"Market is currently closed. Next market opening in {wait_time}")
            logger.info("Running in simulation mode for testing...")
            
        # Initialize DataCollector and set up real-time data handling
        collector = DataCollector(config)
        
        # Wait briefly for WebSocket connection to establish
        time.sleep(2)
        
        if not collector.websocket or not collector.websocket.is_connected:
            raise Exception("WebSocket connection failed to establish")
            
        # Register our callback with the websocket
        collector.websocket.add_tick_callback(on_market_data)
        
        # Initial market status
        if not is_market_open():
            next_open = format_time_until_market_open()
            logger.info(f"Market is closed. Opens in {next_open}")
        else:
            logger.info("Market is open. Waiting for real-time updates...")
        
        # Keep the main thread alive until interrupted
        while running:
            time.sleep(1)  # Sleep to prevent CPU usage
            
    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt. Shutting down...")
    except Exception as e:
        logger.error(f"Error in main: {str(e)}")
    finally:
        if collector:
            collector.cleanup()
        logger.info("Application shutdown complete")

if __name__ == "__main__":
    main()
