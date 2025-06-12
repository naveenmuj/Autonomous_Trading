import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import time

def main():
    # Test symbols to try
    test_symbols = [
        "RELIANCE.NS",    # Old format
        "TATAMOTORS.NS", # Alternative stock
        "TCS.NS",        # Another major stock
        "^NSEI"          # Nifty 50 index
    ]
    
    print("Testing yfinance version:", yf.__version__)
    
    for symbol in test_symbols:
        try:
            print(f"\nTesting symbol: {symbol}")
            
            # Get last 5 days of data
            end_date = datetime.now()
            start_date = end_date - timedelta(days=5)
            
            print(f"Fetching data from {start_date.date()} to {end_date.date()}")
            data = yf.download(symbol, start=start_date, end=end_date, progress=True)
            
            if data.empty:
                print(f"No data received for {symbol}")
                continue
            
            print(f"Successfully fetched data for {symbol}:")
            print(f"Shape of data: {data.shape}")
            print(f"Columns: {data.columns.tolist()}")
            print("\nLast 2 rows of data:")
            print(data.tail(2))
            return True
            
        except Exception as e:
            print(f"Error fetching {symbol}: {str(e)}")
            time.sleep(1)  # Wait a bit before trying next symbol
    
    print("\nFailed to fetch data from any test symbols")
    return False

if __name__ == "__main__":
    main()
