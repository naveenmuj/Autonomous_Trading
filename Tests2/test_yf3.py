import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import time
from typing import Dict, Optional, Tuple

def try_fetch_data(symbol: str, start_date: datetime, end_date: datetime) -> Tuple[Optional[pd.DataFrame], str]:
    """
    Try to fetch data for a given symbol, returns the data and the successful symbol format
    """
    try:
        data = yf.download(symbol, start=start_date, end=end_date, progress=True, auto_adjust=True)
        if not data.empty:
            return data, symbol
    except Exception as e:
        print(f"Error fetching {symbol}: {str(e)}")
    return None, ""

def get_exchange_symbols(base_symbol: str) -> list:
    """
    Generate different exchange symbol formats to try
    """
    return [
        f"{base_symbol}.BO",  # Try BSE first as it's working
        f"{base_symbol}.NS",  # Then NSE
        base_symbol           # Plain format (for indices)
    ]

def main():
    # Test symbols with their base names
    test_symbols = {
        "RELIANCE": "Reliance Industries",
        "TCS": "Tata Consultancy Services",
        "HDFCBANK": "HDFC Bank",
        "TATAMOTORS": "Tata Motors",
        "INFY": "Infosys"
    }
    
    # Add indices
    indices = {
        "^NSEI": "NIFTY 50",
        "^BSESN": "BSE SENSEX"
    }
    
    print("\nChecking yfinance installation...")
    print(f"yfinance version: {yf.__version__}")
    
    # Store successful symbols for future reference
    working_symbols = {}
    
    # Test regular stocks first
    print("\nTesting stock symbols...")
    end_date = datetime.now()
    start_date = end_date - timedelta(days=5)
    
    for base_symbol, company_name in test_symbols.items():
        print(f"\nTesting {company_name} ({base_symbol})...")
        
        # Try different exchange formats
        for symbol in get_exchange_symbols(base_symbol):
            try:
                print(f"Attempting symbol: {symbol}")
                data, success_symbol = try_fetch_data(symbol, start_date, end_date)
                
                if data is not None:
                    print(f"✓ Successfully fetched data from {symbol}")
                    print(f"Shape: {data.shape}")
                    print(f"Columns: {list(data.columns)}")
                    print("\nLast 2 rows:")
                    print(data.tail(2))
                    working_symbols[base_symbol] = symbol
                    break  # Found working symbol format
            except Exception as e:
                print(f"× Failed with {symbol}: {str(e)}")
            
            time.sleep(1)  # Brief pause between attempts
            
    # Test indices
    print("\nTesting market indices...")
    for index_symbol, index_name in indices.items():
        try:
            print(f"\nTesting {index_name} ({index_symbol})...")
            data, success_symbol = try_fetch_data(index_symbol, start_date, end_date)
            
            if data is not None:
                print(f"✓ Successfully fetched index data from {index_symbol}")
                print(f"Shape: {data.shape}")
                print("\nLast 2 rows:")
                print(data.tail(2))
                working_symbols[index_name] = index_symbol
        except Exception as e:
            print(f"× Failed with {index_symbol}: {str(e)}")
        
        time.sleep(1)  # Brief pause between attempts
    
    # Summary
    print("\n=== Summary of Working Symbols ===")
    if working_symbols:
        print("\nWorking symbols found:")
        for name, symbol in working_symbols.items():
            print(f"- {name}: {symbol}")
        return True
    else:
        print("\nNo working symbols found!")
        return False

if __name__ == "__main__":
    main()
