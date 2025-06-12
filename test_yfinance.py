import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import time

def test_yfinance():
    print("\nChecking yfinance installation...")
    print("yfinance version:", yf.__version__)
    
    # Test symbols to try
    test_symbols = [
        "RELIANCE.BO",    # BSE format
        "RELIANCE.NSE",   # Alternative NSE format
        "NSEI.BO",        # Alternative Nifty 50 format
        "^BSESN",         # BSE Sensex
        "INFY.BO",        # Infosys BSE
        "HDFCBANK.BO",    # HDFC Bank BSE
        "TATASTEEL.BO"    # Tata Steel BSE
    ]
    
    success = False
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
                
            # Print the results
            print(f"\nSuccessfully fetched data for {symbol}:")
            print("Shape of data:", data.shape)
            print("Columns:", data.columns.tolist())
            print("\nLast 2 rows of data:")
            print(data.tail(2))
            
            success = True
            break  # Found working symbol, can stop testing
            
        except Exception as e:
            print(f"Error fetching {symbol}: {str(e)}")
            time.sleep(1)  # Wait a bit before trying next symbol
            continue
    
    if not success:
        print("\nFailed to fetch data from any test symbols")
    
    return success

if __name__ == "__main__":
    test_yfinance()
            
        print("\nSuccessfully fetched data from yfinance:")
        print("\nShape of data:", data.shape)
        print("\nColumns:", data.columns.tolist())
        print("\nLast 2 rows of data:")
        print(data.tail(2))
        return True
        
    except Exception as e:
        print(f"Error testing yfinance: {str(e)}")
        return False

if __name__ == "__main__":
    test_yfinance()
