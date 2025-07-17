import pandas as pd
import pytest

def get_missing_business_days(df, start_date, end_date, date_col='date'):
    """
    Returns a list of missing business days between start_date and end_date
    that are not present in df[date_col].
    """
    all_days = pd.bdate_range(start=start_date, end=end_date)
    df_dates = pd.to_datetime(df[date_col])
    missing = [d for d in all_days if d not in df_dates.values]
    return missing

def test_missing_business_days():
    # Simulate collected data with missing business days
    start = '2025-03-19'
    end = '2025-03-31'
    # Simulate missing 2025-03-21 and 2025-03-25
    data = {
        'date': ['2025-03-19', '2025-03-20', '2025-03-24', '2025-03-26', '2025-03-27', '2025-03-28', '2025-03-31'],
        'open': [1,2,3,4,5,6,7],
        'close': [2,3,4,5,6,7,8]
    }
    df = pd.DataFrame(data)
    missing = get_missing_business_days(df, start, end)
    expected = [pd.Timestamp('2025-03-21'), pd.Timestamp('2025-03-25')]
    assert missing == expected, f"Missing days: {missing}"
