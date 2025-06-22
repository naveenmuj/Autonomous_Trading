import logging
import pandas as pd
from datetime import datetime
import os

logger = logging.getLogger(__name__)

class PaperTradingEngine:
    def __init__(self, initial_balance=100000, commission=0.0):
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.position = 0  # +1 for long, -1 for short, 0 for flat
        self.entry_price = None
        self.trade_log = []
        self.commission = commission
        self.equity_curve = [initial_balance]

    def reset(self):
        self.balance = self.initial_balance
        self.position = 0
        self.entry_price = None
        self.trade_log = []
        self.equity_curve = [self.initial_balance]

    def on_signal(self, signal, price, timestamp=None, symbol=None, stop_loss=None, target=None, rationale=None):
        timestamp = timestamp or datetime.now()
        trade_info = {
            'action': 'BUY' if signal == 1 and self.position == 0 else 'SELL' if signal == -1 and self.position == 1 else None,
            'price': price,
            'timestamp': timestamp,
            'symbol': symbol,
            'stop_loss': stop_loss,
            'target': target,
            'rationale': rationale
        }
        if signal == 1 and self.position == 0:
            # Enter long
            self.position = 1
            self.entry_price = price
            logger.info(f"PaperTrade: BUY at {price:.2f} on {timestamp} | Symbol: {symbol} | SL: {stop_loss} | Target: {target} | Reason: {rationale}")
            self.trade_log.append(trade_info)
            # Save to CSV after each trade
            self.save_trade_log_to_csv()
        elif signal == -1 and self.position == 1:
            # Exit long
            pnl = price - self.entry_price - self.commission
            self.balance += pnl
            logger.info(f"PaperTrade: SELL at {price:.2f} on {timestamp} | PnL: {pnl:.2f} | Symbol: {symbol} | SL: {stop_loss} | Target: {target} | Reason: {rationale}")
            trade_info['pnl'] = pnl
            self.trade_log.append(trade_info)
            self.position = 0
            self.entry_price = None
            # Save to CSV after each trade
            self.save_trade_log_to_csv()
        # Update equity curve
        equity = self.balance
        if self.position == 1 and self.entry_price is not None:
            equity += price - self.entry_price
        self.equity_curve.append(equity)

    def get_trade_log(self):
        return pd.DataFrame(self.trade_log)

    def get_equity_curve(self):
        return self.equity_curve

    def get_balance(self):
        return self.balance

    def get_position(self):
        return self.position

    def save_trade_log_to_csv(self, filename=None):
        """Save the trade log to a CSV file in the project root, with all relevant fields."""
        if not filename:
            filename = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'paper_trade_log.csv')
        # Prepare DataFrame with all required columns
        df = pd.DataFrame(self.trade_log)
        # Ensure all columns are present
        columns = ['action', 'symbol', 'price', 'timestamp', 'stop_loss', 'target', 'rationale', 'pnl']
        for col in columns:
            if col not in df.columns:
                df[col] = None
        # Convert rationale dict to string for CSV
        df['rationale'] = df['rationale'].apply(lambda x: str(x) if isinstance(x, dict) else (x if x is not None else ''))
        # Format timestamp as string
        df['timestamp'] = df['timestamp'].astype(str)
        df.to_csv(filename, columns=columns, index=False)
