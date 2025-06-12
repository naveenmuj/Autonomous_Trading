import pytz
from datetime import datetime, time, timedelta

def is_market_open() -> bool:
    """Check if the market is currently open"""
    india_tz = pytz.timezone('Asia/Kolkata')
    current_time = datetime.now(india_tz)
    current_day = current_time.weekday()
    
    # Check if it's a weekend (5 is Saturday, 6 is Sunday)
    if current_day in [5, 6]:
        return False
        
    # Market hours 9:15 AM to 3:30 PM IST
    market_open = time(9, 15)
    market_close = time(15, 30)
    current_time = current_time.time()
    
    return market_open <= current_time <= market_close

def format_time_until_market_open() -> str:
    """Format time remaining until next market open"""
    india_tz = pytz.timezone('Asia/Kolkata')
    current_time = datetime.now(india_tz)
    current_day = current_time.weekday()
    
    # Create market open time for today
    market_open = datetime.combine(current_time.date(), time(9, 15))
    market_open = india_tz.localize(market_open)
    
    # If current time is past market close (3:30 PM), move to next day
    if current_time.time() >= time(15, 30):
        market_open += timedelta(days=1)
        current_day = (current_day + 1) % 7
    
    # If it's Friday after market close, move to Monday
    if current_day == 5:  # Friday
        market_open += timedelta(days=3)
    # If it's Saturday, move to Monday
    elif current_day == 6:  # Saturday
        market_open += timedelta(days=2)
    
    # Calculate time difference
    time_diff = market_open - current_time
    
    # Format the time difference
    days = time_diff.days
    hours, remainder = divmod(time_diff.seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    
    if days > 0:
        return f"{days}d {hours}h {minutes}m"
    elif hours > 0:
        return f"{hours}h {minutes}m"
    else:
        return f"{minutes}m {seconds}s"
