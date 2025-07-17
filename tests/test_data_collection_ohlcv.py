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
    symbol = 'RELIANCE.NS'  # You can parameterize this for more symbols
    days = 10
    interval = 'ONE_DAY'
    # Patch config to match expected structure in DataCollector
    patched_config = dict(config)
    patched_config['apis'] = config['api']
    collector = DataCollector(patched_config)
    print(f"Mapped symbols: {list(collector.symbol_token_map.keys())}")
    # Use RELIANCE.NS if mapped, else use the first available mapped symbol
    if symbol not in collector.symbol_token_map:
        print(f"Symbol {symbol} not mapped. Using first available mapped symbol.")
        if collector.symbol_token_map:
            symbol = list(collector.symbol_token_map.keys())[0]
        else:
            raise AssertionError("No symbols mapped to tokens. Check instrument file and config.")
    df = collector.get_historical_data(symbol, days, interval)
    # Check for empty DataFrame
    assert not df.empty, f"No data returned for symbol {symbol}. Check if the symbol is present in the config and instrument file."
    # Check for required columns (including timestamp)
    required_cols = {'timestamp', 'open', 'high', 'low', 'close', 'volume'}
    missing_cols = required_cols - set(df.columns)
    assert not missing_cols, f"Missing columns in data: {missing_cols}"
    # Use the last 'days' business days as the expected range
    from datetime import datetime, timedelta
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    # Check for missing business days using the 'timestamp' column
    missing = get_missing_business_days(df, start_date, end_date, date_col='timestamp')
    assert not missing, f"Missing business days: {missing}"
    # Check for empty columns (except timestamp)
    for col in required_cols - {'timestamp'}:
        assert df[col].notnull().all(), f"Column {col} contains null values"
        assert (df[col] != '').all(), f"Column {col} contains empty values"
