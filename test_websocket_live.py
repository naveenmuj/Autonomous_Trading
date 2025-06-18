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

def on_market_data(tick_data: dict):
    """Callback function for market data updates"""
    try:
        # Format the output similar to test_angel_one_websocket.py
        print("\033[H\033[2J", end="")  # Clear screen
        print("\n" + "═" * 50)
        print(f"  MARKET DATA UPDATE")
        print("═" * 50)
        print(f"  Time: {tick_data['timestamp'].strftime('%H:%M:%S')}")
        print(f"  Token: {tick_data['token']}")
        print(f"  LTP: ₹{tick_data['ltp']:,.2f}")
        if 'bid_price' in tick_data and 'ask_price' in tick_data:
            print(f"  Bid/Ask: ₹{tick_data['bid_price']:,.2f} / ₹{tick_data['ask_price']:,.2f}")
        if 'volume' in tick_data:
            print(f"  Volume: {tick_data['volume']:,}")
        if 'oi' in tick_data:
            print(f"  Open Interest: {tick_data['oi']:,}")
        print("═" * 50)
        print("\nPress Ctrl+C to exit")
        
    except Exception as e:
        logger.error(f"Error in market data callback: {str(e)}")

def main():
    """Main test function"""
    global running
    collector = None
    
    try:
        # Set up signal handlers
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

        # Load configuration
        with open('config.yaml', 'r') as f:
            config = yaml.safe_load(f)
            
        # Check market hours
        if not is_market_open():
            wait_time = format_time_until_market_open()
            logger.warning(f"Market is currently closed. Next market opening in {wait_time}")
            logger.info("Running in simulation mode for testing...")
            
        # Initialize DataCollector and set up real-time data handling
        collector = DataCollector(config)
        
        # Wait briefly for WebSocket connection to establish
        time.sleep(2)
        
        if not collector.websocket or not collector.websocket.connection_state != collector.websocket.CONNECTED:
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
