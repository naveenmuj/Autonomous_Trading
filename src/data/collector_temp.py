import os
import sys
import json
import logging
import threading
import time
import requests
from typing import Dict, List, Optional, Any, Union, Tuple
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd
import numpy as np
import pyotp
from SmartApi import SmartConnect
import yfinance as yf
import pandas_ta as ta
from functools import wraps
from .websocket import MarketDataWebSocket
import contextlib
import codecs

logger = logging.getLogger(__name__)

# Configure logger for UTF-8 encoding
handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
handler.stream.reconfigure(encoding='utf-8')  # Set UTF-8 encoding
logger.addHandler(handler)

# Session renewal interval (8 hours in seconds)
SESSION_RENEWAL_INTERVAL = 8 * 60 * 60

@contextlib.contextmanager
def utf8_stdout():
    """Context manager that temporarily sets stdout to use UTF-8 encoding"""
    old_stdout = sys.stdout
    try:
        sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer)
        yield
    finally:
        sys.stdout = old_stdout

def with_timeout(timeout_seconds: int):
    """Windows-compatible timeout decorator"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            import threading
            import queue

            result = queue.Queue()
            def worker():
                try:
                    r = func(*args, **kwargs)
                    result.put(('success', r))
                except Exception as e:
                    result.put(('error', e))

            thread = threading.Thread(target=worker)
            thread.daemon = True
            thread.start()
            thread.join(timeout_seconds)

            if thread.is_alive():
                thread.join(0)  # Non-blocking join
                raise TimeoutError(f"Function call timed out after {timeout_seconds} seconds")

            status, value = result.get()
            if status == 'error':
                raise value
            return value

        return wrapper
    return decorator

class DataCollector:
    """Data collection class for market data"""
    
    _instance = None  # Singleton instance
    INSTRUMENT_MASTER_URL = "https://margincalculator.angelbroking.com/OpenAPI_File/files/OpenAPIScripMaster.json"
    NSE_STOCKS_URL = "https://archives.nseindia.com/content/equities/EQUITY_L.csv"
    NIFTY_50_URL = "https://archives.nseindia.com/content/indices/ind_nifty50list.csv"
    NIFTY_100_URL = "https://archives.nseindia.com/content/indices/ind_nifty100list.csv"
    
    def __new__(cls, config: Optional[Dict] = None) -> 'DataCollector':
        if cls._instance is None:
            cls._instance = super(DataCollector, cls).__new__(cls)
        return cls._instance
        
    def __init__(self, config: Optional[Dict] = None):
        if not hasattr(self, 'initialized'):
            self.config = config if config else {}
            self.angel_api = None
            self.websocket = None
            self.token_mapping = {}
            self.exchange_manager = None
            self.live_data = {}  # Store real-time data
            self.initialized = True
            self.session_timer = None
            self.is_running = True
            
            # Initialize Angel One API
            self._initialize_angel_api()
            # Initialize token mapping for NSE symbols
            self._initialize_token_mapping()
            # Start WebSocket connection
            self._initialize_websocket()
