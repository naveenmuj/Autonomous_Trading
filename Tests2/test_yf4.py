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
        print(f"Attempting to fetch {symbol}...")
        data = yf.download(symbol, start=start_date, end=end_date, progress=True, auto_adjust=True)
        if not data.empty:
            return data, symbol
    except Exception as e:
        print(f"Error fetching {symbol}: {str(e)}")
    return None, ""

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
    print(f"yfinance version: {yf.__version__}\n")
    
    # Store working symbols for each exchange
    bse_working = {}
    nse_working = {}
    
    # Test regular stocks
    end_date = datetime.now()
    start_date = end_date - timedelta(days=5)
    
    print("=" * 50)
    print("Testing both BSE and NSE for each stock:")
    print("=" * 50)
    
    for base_symbol, company_name in test_symbols.items():
        print(f"\nTesting {company_name} ({base_symbol}):")
        print("-" * 40)
        
        # Try BSE
        bse_symbol = f"{base_symbol}.BO"
        data, _ = try_fetch_data(bse_symbol, start_date, end_date)
        if data is not None:
            print(f"✓ BSE ({bse_symbol}): Working")
            print(f"  Last close: {data['Close'].iloc[-1]}")
            bse_working[base_symbol] = bse_symbol
        else:
            print(f"× BSE ({bse_symbol}): Not working")
            
        # Try NSE
        nse_symbol = f"{base_symbol}.NS"
        data, _ = try_fetch_data(nse_symbol, start_date, end_date)
        if data is not None:
            print(f"✓ NSE ({nse_symbol}): Working")
            print(f"  Last close: {data['Close'].iloc[-1]}")
            nse_working[base_symbol] = nse_symbol
        else:
            print(f"× NSE ({nse_symbol}): Not working")
        
        time.sleep(1)  # Brief pause between stocks
    
    # Test indices
    print("\n" + "=" * 50)
    print("Testing Market Indices:")
    print("=" * 50)
    
    indices_working = {}
    for index_symbol, index_name in indices.items():
        print(f"\nTesting {index_name} ({index_symbol}):")
        data, _ = try_fetch_data(index_symbol, start_date, end_date)
        if data is not None:
            print(f"✓ {index_name}: Working")
            print(f"  Last close: {data['Close'].iloc[-1]}")
            indices_working[index_name] = index_symbol
        else:
            print(f"× {index_name}: Not working")
        
        time.sleep(1)
    
    # Summary
    print("\n" + "=" * 50)
    print("SUMMARY OF FINDINGS")
    print("=" * 50)
    
    print("\nBSE Symbols Working:")
    for symbol in bse_working.values():
        print(f"- {symbol}")
        
    print("\nNSE Symbols Working:")
    for symbol in nse_working.values():
        print(f"- {symbol}")
        
    print("\nIndices Working:")
    for symbol in indices_working.values():
        print(f"- {symbol}")
        
    # Recommendation
    print("\n" + "=" * 50)
    print("RECOMMENDATION")
    print("=" * 50)
    if len(bse_working) > len(nse_working):
        print("\nUse BSE (.BO) symbols as they are more reliable.")
    elif len(nse_working) > len(bse_working):
        print("\nUse NSE (.NS) symbols as they are more reliable.")
    else:
        if bse_working:
            print("\nBoth exchanges have same reliability. BSE symbols are currently working.")
        else:
            print("\nNeither exchange's symbols are working reliably.")

if __name__ == "__main__":
    main()
