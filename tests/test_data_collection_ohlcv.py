import pandas as pd
import pytest
from src.data.collector import DataCollector
import yaml

# Load config for DataCollector
with open('src/config.yaml', 'r') as f:
    config = yaml.safe_load(f)

def get_missing_business_days(df, start_date, end_date, date_col='date'):
    all_days = pd.bdate_range(start=start_date, end=end_date)
    df_dates = pd.to_datetime(df[date_col])
    missing = [d for d in all_days if d not in df_dates.values]
    return missing

def test_data_collection_ohlcv():
    """
    Test that DataCollector collects all required OHLCV data for a symbol and period,
    and that there are no missing business days or missing columns.
    """
    days = 365
    interval = 'ONE_DAY'
    # Patch config to match expected structure in DataCollector
    patched_config = dict(config)
    patched_config['apis'] = config['api']
    collector = DataCollector(patched_config)
    candidate_symbols = collector.get_symbols_from_config()
    mapped_symbols = [s for s in candidate_symbols if s in collector.symbol_token_map]
    print(f"Candidate symbols from config: {candidate_symbols}")
    print(f"Mapped symbols with valid token: {mapped_symbols}")
    tested = 0
    passed = False
    required_cols = {'timestamp', 'open', 'high', 'low', 'close', 'volume'}
    from datetime import datetime, timedelta
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    if not mapped_symbols:
        print("No mapped symbols with valid token found. Check your config, instrument file, and mapping logic.")
        assert False, "No mapped symbols with valid token found. Test cannot proceed."
    for symbol in mapped_symbols[:3]:
        print(f"\nTrying symbol: {symbol}")
        try:
            df = collector.get_historical_data(symbol, days, interval)
            if df.empty:
                print(f"No data returned for symbol {symbol}.")
                continue
            missing_cols = required_cols - set(df.columns)
            if missing_cols:
                print(f"Missing columns in data for {symbol}: {missing_cols}")
                continue
            missing = get_missing_business_days(df, start_date, end_date, date_col='timestamp')
            if missing:
                print(f"Missing business days for {symbol}: {missing}")
                continue
            for col in required_cols - {'timestamp'}:
                if not df[col].notnull().all():
                    print(f"Column {col} contains null values for {symbol}")
                    break
                if not (df[col] != '').all():
                    print(f"Column {col} contains empty values for {symbol}")
                    break
            else:
                print(f"Symbol {symbol} passed all checks.")
                passed = True
                break
        except Exception as e:
            print(f"Exception for symbol {symbol}: {e}")
        tested += 1
    assert passed, f"No valid OHLCV data found for any of the first {tested} mapped symbols. Check instrument file, config, and API/data availability."
