import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from .strategy import EnhancedTradingStrategy
from .manager import TradeManager

logger = logging.getLogger(__name__)

class Backtester:
    def __init__(self, strategy: EnhancedTradingStrategy, initial_capital: float = 100000, config=None, data_collector=None):
        self.strategy = strategy
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.positions = {}
        self.trades = []
        self.trade_history = pd.DataFrame()
        # Use TradeManager for risk logic
        self.trade_manager = TradeManager(config, data_collector) if config else None
        
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
            # Generate signals for the whole period first
            signals = self.strategy.generate_signals(df)
            # Log signal distribution before trade execution
            if 'signal' in signals.columns:
                logger.info(f"Signal value counts before trade execution: {signals['signal'].value_counts().to_dict()}")
                if signals['signal'].abs().sum() == 0:
                    logger.error("No non-zero signals found. Aborting backtest. Check signal generation logic and thresholds.")
                    return {
                        'total_return': 0,
                        'sharpe_ratio': 0,
                        'max_drawdown': 0,
                        'win_rate': 0,
                        'total_trades': 0,
                        'equity_curve': [],
                        'daily_returns': [],
                        'trade_history': pd.DataFrame(),
                        'error': 'No non-zero signals found. Backtest aborted.'
                    }
            else:
                logger.warning("No 'signal' column found before trade execution!")
                return {
                    'total_return': 0,
                    'sharpe_ratio': 0,
                    'max_drawdown': 0,
                    'win_rate': 0,
                    'total_trades': 0,
                    'equity_curve': [],
                    'daily_returns': [],
                    'trade_history': pd.DataFrame(),
                    'error': "No 'signal' column found. Backtest aborted."
                }
            # Initialize tracking variables
            equity_curve = []
            daily_returns = []
            max_drawdown = 0
            peak_value = self.initial_capital
            # Run through each day
            for date in df.index:
                try:
                    # Use precomputed signals
                    signal_row = signals.loc[date] if date in signals.index else pd.Series({'signal': 0})
                    # Determine the symbol for this row
                    symbol = df.loc[date]['symbol'] if 'symbol' in df.columns else None
                    # Execute trades based on signals
                    self._execute_trades(signal_row, df.loc[date])
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
                except Exception as e:
                    logger.error(f"Error in backtest loop: {str(e)}")
            # Calculate performance metrics
            total_return = (equity_curve[-1] / self.initial_capital - 1) * 100 if equity_curve else 0
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
            logger.info(f"Total trades executed: {len(self.trades)}")
            if len(self.trades) == 0:
                logger.warning("No trades executed in backtest. Check signal generation and thresholds.")
            else:
                logger.info(f"Sample trades: {self.trades[:3]}")
            return results
        except Exception as e:
            logger.error(f"Error running backtest: {str(e)}")
            raise
            
    def _execute_trades(self, signal: pd.Series, current_data: pd.Series) -> None:
        """Execute trades based on signals using TradeManager for risk logic"""
        try:
            symbol = current_data['symbol'] if 'symbol' in current_data else current_data.name
            logger.debug(f"_execute_trades: symbol={symbol}, signal={signal.get('signal', None)}, in_position={symbol in self.positions}")
            if symbol in self.positions:
                position = self.positions[symbol]
                price = current_data['close']
                logger.debug(f"Checking exit for {symbol}: action={position['action']}, price={price}, stop_loss={position['stop_loss']}, target={position['target']}")
                # Check for exit (stop-loss/target)
                if position['action'] == 'BUY' and (price <= position['stop_loss'] or price >= position['target']):
                    logger.info(f"Exiting BUY position for {symbol} at price {price}")
                    self.trades.append({**position, 'exit_price': price, 'exit_time': current_data.name})
                    del self.positions[symbol]
                elif position['action'] == 'SELL' and (price >= position['stop_loss'] or price <= position['target']):
                    logger.info(f"Exiting SELL position for {symbol} at price {price}")
                    self.trades.append({**position, 'exit_price': price, 'exit_time': current_data.name})
                    del self.positions[symbol]
            # Entry logic
            if signal.get('signal', 0) != 0 and symbol not in self.positions:
                action = 'BUY' if signal['signal'] == 1 else 'SELL'
                price = current_data['close']
                logger.debug(f"Attempting entry for {symbol}: action={action}, price={price}")
                # Use TradeManager for position sizing and risk
                if self.trade_manager:
                    order = self.trade_manager.place_order(symbol, signal['signal'])
                    if order:
                        logger.info(f"Trade executed: {order}")
                        self.positions[symbol] = {
                            'symbol': symbol,
                            'action': action,
                            'quantity': order['quantity'],
                            'entry_price': price,
                            'stop_loss': order['stop_loss'],
                            'target': order['profit_target'],
                            'status': 'open',
                            'entry_time': current_data.name
                        }
                    else:
                        logger.warning(f"TradeManager.place_order returned None for {symbol} {action}. Possible risk rejection or zero quantity.")
                else:
                    logger.warning("No TradeManager instance available. Skipping trade execution.")
        except Exception as e:
            logger.error(f"Error executing trades in backtest: {str(e)}")
            raise
            
    def _should_exit_position(self, position: Dict, current_data: pd.Series) -> bool:
        """Check if position should be exited"""
        current_price = current_data['close']
        
        # Check stop loss
        if current_price <= position['stop_loss']:
            return True
            
        # Check take profit
        take_profit = position['entry_price'] * (1 + self.strategy.profit_target)
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
