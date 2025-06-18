import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from .strategy import EnhancedTradingStrategy

logger = logging.getLogger(__name__)

class Backtester:
    def __init__(self, strategy: EnhancedTradingStrategy, initial_capital: float = 100000):
        self.strategy = strategy
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.positions = {}
        self.trades = []
        self.trade_history = pd.DataFrame()
        
    def run(self, data: pd.DataFrame, start_date: Optional[str] = None, 
            end_date: Optional[str] = None) -> Dict[str, Any]:
        """Run backtest on historical data"""
        try:
            # Prepare data
            df = data.copy()
            df = df.sort_index()
            
            if start_date:
                df = df[df.index >= start_date]
            if end_date:
                df = df[df.index <= end_date]
            
            # Initialize tracking variables
            equity_curve = []
            daily_returns = []
            max_drawdown = 0
            peak_value = self.initial_capital
            
            # Run through each day
            for date in df.index:
                daily_data = df.loc[:date]  # Use data up to current date
                
                # Get trading signals
                signals = self.strategy.generate_signals(daily_data)
                
                # Execute trades based on signals
                self._execute_trades(signals.iloc[-1], df.loc[date])
                
                # Track performance
                current_value = self._calculate_portfolio_value(df.loc[date])
                equity_curve.append(current_value)
                
                # Calculate daily return
                daily_return = (current_value / equity_curve[-2] - 1) if len(equity_curve) > 1 else 0
                daily_returns.append(daily_return)
                
                # Update max drawdown
                peak_value = max(peak_value, current_value)
                drawdown = (peak_value - current_value) / peak_value
                max_drawdown = max(max_drawdown, drawdown)
            
            # Calculate performance metrics
            total_return = (equity_curve[-1] / self.initial_capital - 1) * 100
            sharpe_ratio = self._calculate_sharpe_ratio(daily_returns)
            win_rate = self._calculate_win_rate()
            
            # Prepare results
            results = {
                'total_return': total_return,
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': max_drawdown * 100,
                'win_rate': win_rate,
                'total_trades': len(self.trades),
                'equity_curve': equity_curve,
                'daily_returns': daily_returns,
                'trade_history': self.trade_history
            }
            
            logger.info(f"Backtest completed. Total return: {total_return:.2f}%, Sharpe ratio: {sharpe_ratio:.2f}")
            return results
            
        except Exception as e:
            logger.error(f"Error in backtest: {str(e)}")
            raise
            
    def _execute_trades(self, signal: pd.Series, current_data: pd.Series) -> None:
        """Execute trades based on signals"""
        try:
            symbol = current_data.name
            
            # Check if we should exit existing position
            if symbol in self.positions:
                position = self.positions[symbol]
                exit_price = current_data['close']
                
                # Check stop loss and take profit
                if self._should_exit_position(position, current_data):
                    pnl = (exit_price - position['entry_price']) * position['quantity']
                    self.current_capital += pnl
                    
                    # Record trade
                    self.trades.append({
                        'symbol': symbol,
                        'entry_price': position['entry_price'],
                        'exit_price': exit_price,
                        'quantity': position['quantity'],
                        'entry_time': position['entry_time'],
                        'exit_time': current_data.name,
                        'pnl': pnl,
                        'return': (exit_price / position['entry_price'] - 1) * 100
                    })
                    
                    del self.positions[symbol]
            
            # Check for new entry signals
            if signal['signal'] != 0 and symbol not in self.positions:
                # Calculate position size
                risk_amount = self.current_capital * self.strategy.config['risk']['position_size']
                stop_loss = self.strategy.calculate_stop_loss(current_data)
                quantity = self._calculate_position_size(
                    risk_amount, current_data['close'], stop_loss
                )
                
                if quantity > 0:
                    self.positions[symbol] = {
                        'entry_price': current_data['close'],
                        'quantity': quantity,
                        'stop_loss': stop_loss,
                        'entry_time': current_data.name
                    }
                    
        except Exception as e:
            logger.error(f"Error executing trades: {str(e)}")
            raise
            
    def _should_exit_position(self, position: Dict, current_data: pd.Series) -> bool:
        """Check if position should be exited"""
        current_price = current_data['close']
        
        # Check stop loss
        if current_price <= position['stop_loss']:
            return True
            
        # Check take profit
        take_profit = position['entry_price'] * (1 + self.strategy.config['profit_target'])
        if current_price >= take_profit:
            return True
            
        return False
        
    def _calculate_position_size(self, risk_amount: float, 
                               entry_price: float, 
                               stop_loss: float) -> int:
        """Calculate position size based on risk management rules"""
        risk_per_share = entry_price - stop_loss
        if risk_per_share <= 0:
            return 0
            
        quantity = int(risk_amount / risk_per_share)
        return max(0, quantity)
        
    def _calculate_portfolio_value(self, current_data: pd.Series) -> float:
        """Calculate current portfolio value"""
        portfolio_value = self.current_capital
        
        for symbol, position in self.positions.items():
            current_price = current_data['close']
            position_value = position['quantity'] * current_price
            portfolio_value += position_value
            
        return portfolio_value
        
    def _calculate_sharpe_ratio(self, returns: List[float], 
                              risk_free_rate: float = 0.02) -> float:
        """Calculate Sharpe ratio"""
        if not returns:
            return 0
            
        returns = np.array(returns)
        excess_returns = returns - (risk_free_rate / 252)  # Daily risk-free rate
        if len(excess_returns) < 2:
            return 0
            
        return np.sqrt(252) * np.mean(excess_returns) / np.std(excess_returns)
        
    def _calculate_win_rate(self) -> float:
        """Calculate win rate of trades"""
        if not self.trades:
            return 0
            
        winning_trades = sum(1 for trade in self.trades if trade['pnl'] > 0)
        return winning_trades / len(self.trades)
