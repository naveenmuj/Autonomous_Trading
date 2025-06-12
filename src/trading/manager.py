import yaml
from datetime import datetime
import numpy as np
import logging
from typing import Dict, List, Optional, Tuple, Any
import pandas as pd
from SmartApi import SmartConnect  # Using the official import style
from src.data.collector import DataCollector

logger = logging.getLogger(__name__)

class RiskManager:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.risk_config = config.get('risk_management', {})
        if not self.risk_config:
            self.risk_config = config.get('trading', {}).get('risk', {})
            
        # Required parameters with defaults
        self.max_position_size = self.risk_config.get('max_position_size', 0.02)
        self.max_trades = self.risk_config.get('max_trades', 5)
        self.stop_loss = self.risk_config.get('stop_loss', 0.02)
        self.max_sector_exposure = self.risk_config.get('max_sector_exposure', 0.25)
        self.max_drawdown = self.risk_config.get('max_drawdown', 0.15)
        self.correlation_threshold = self.risk_config.get('correlation_threshold', 0.7)
        self.max_correlated_trades = self.risk_config.get('max_correlated_trades', 2)
        self.kelly_fraction = self.risk_config.get('kelly_fraction', 0.5)
        self.max_leverage = self.risk_config.get('max_leverage', 1.0)
        
        logger.info(f"RiskManager initialized with max position size: {self.max_position_size}")

class TradeManager:
    def __init__(self, config: Dict[str, Any], data_collector: Optional[DataCollector] = None):
        self.config = config
        self.data_collector = data_collector or DataCollector(config)
        self.positions = {}
        self.trade_history = []
        self.account_value = config.get('initial_account_value', 100000)
        self.logger = logging.getLogger(__name__)
        self.risk_manager = RiskManager(config)
        logger.info(f"TradeManager initialized with initial capital: {self.account_value}")

    def place_trade(self, symbol: str, action: str, quantity: int, price: float, stop_loss: float, target: float) -> Dict[str, Any]:
        """Place a new trade"""
        try:
            # Validate inputs
            if not all([symbol, action, quantity, price, stop_loss, target]):
                raise ValueError("All trade parameters must be provided")
                
            if action not in ['BUY', 'SELL']:
                raise ValueError("Action must be either 'BUY' or 'SELL'")
                
            if quantity <= 0:
                raise ValueError("Quantity must be positive")
                
            # Check risk management rules
            trade = {
                'symbol': symbol,
                'action': action,
                'quantity': quantity,
                'entry_price': price,
                'stop_loss': stop_loss,
                'target': target,
                'timestamp': datetime.now(),
                'status': 'open',
                'type': self.config['trading']['mode']
            }
            
            is_valid, reason = self.risk_manager.validate_trade(trade, self.account_value)
            if not is_valid:
                logger.warning(f"Trade rejected: {reason}")
                return None
                
            if self.config['trading']['mode'] == 'simulation':
                return self._simulate_trade(symbol, action, quantity, price)
            else:
                return self._execute_live_trade(trade)
                
        except Exception as e:
            logger.error(f"Error placing trade: {str(e)}")
            raise

    def _simulate_trade(self, symbol: str, action: str, quantity: int, price: float,
                       stop_loss: Optional[float] = None, profit_target: Optional[float] = None) -> Dict[str, Any]:
        """Simulate a trade execution."""
        try:
            timestamp = datetime.now()
            trade = {
                'symbol': symbol,
                'action': action,
                'quantity': quantity,
                'price': price,
                'timestamp': timestamp,
                'stop_loss': stop_loss,
                'profit_target': profit_target
            }
            
            self.logger.info(f"Executing trade: {trade}")
            self.trade_history.append(trade)
            
            if action == 'BUY':
                self.account_value -= quantity * price
            else:  # SELL
                self.account_value += quantity * price
            
            self._update_position(trade)
            return trade
        except Exception as e:
            self.logger.error(f"Error simulating trade: {str(e)}")
            raise

    def _execute_live_trade(self, trade: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a live trade"""
        try:
            # Place the order
            order = self.place_order(
                symbol=trade['symbol'],
                order_type=trade['action'],
                quantity=trade['quantity'],
                price=trade['entry_price']
            )
            
            if order and order['status'] == 'COMPLETE':
                self._update_position(trade)
                self.trade_history.append(trade)
                logger.info(f"Live trade executed: {trade['symbol']} {trade['action']} {trade['quantity']}")
                return trade
            else:
                logger.error(f"Order placement failed: {order.get('message', 'Unknown error')}")
                return None
                
        except Exception as e:
            logger.error(f"Error executing live trade: {str(e)}")
            raise

    def calculate_position_size(self, capital: float, price: float) -> int:
        """Calculate position size based on risk management rules."""
        try:
            max_position_size = self.config['trading']['risk']['max_position_size']
            max_risk_amount = capital * max_position_size
            
            # Calculate quantity based on price and max risk
            quantity = int(max_risk_amount / price)
            
            return max(1, quantity)  # Ensure at least 1 unit
            
        except Exception as e:
            self.logger.error(f"Error calculating position size: {str(e)}")
            return 1

    def _update_position(self, trade: Dict[str, Any]):
        """Update position based on trade."""
        try:
            symbol = trade['symbol']
            action = trade['action']
            quantity = trade['quantity']
            price = trade['price']
            
            if symbol not in self.positions:
                self.positions[symbol] = {
                    'quantity': 0,
                    'avg_price': 0,
                    'stop_loss': trade.get('stop_loss'),
                    'profit_target': trade.get('profit_target')
                }
            
            position = self.positions[symbol]
            
            if action == 'BUY':
                new_quantity = position['quantity'] + quantity
                new_cost = (position['quantity'] * position['avg_price']) + (quantity * price)
                position['quantity'] = new_quantity
                position['avg_price'] = new_cost / new_quantity
            else:  # SELL
                if position['quantity'] < quantity:
                    raise ValueError(f"Insufficient position for {symbol}")
                position['quantity'] -= quantity
                if position['quantity'] == 0:
                    del self.positions[symbol]
            
            self.logger.info(f"Updated position for {symbol}: {self.positions[symbol]}")
        except Exception as e:
            self.logger.error(f"Error updating position: {str(e)}")
            raise

    def place_order(self, symbol: str, order_type: str, quantity: int, price: float = None) -> Dict[str, Any]:
        """Place an order with the broker"""
        try:
            if self.config['trading']['mode'] == 'simulation':
                order = {
                    'symbol': symbol,
                    'type': order_type,
                    'quantity': quantity,
                    'price': price or 0,
                    'status': 'COMPLETE',
                    'timestamp': datetime.now(),
                    'order_id': f"SIM_{len(self.trade_history)}",
                }
                return order
            else:
                # Implement actual broker order placement here
                # This is a placeholder for the actual implementation
                raise NotImplementedError("Live trading not implemented")
                
        except Exception as e:
            logger.error(f"Error placing order: {str(e)}")
            raise

    def update_positions(self, current_prices: Dict[str, float]) -> None:
        """Update all open positions"""
        try:
            # Create a list of positions to update to avoid modifying dict during iteration
            positions_to_update = [(symbol, position.copy()) for symbol, position in self.positions.items() if position['status'] == 'open']
            
            for symbol, position in positions_to_update:
                current_price = current_prices.get(symbol)
                if current_price is None:
                    continue
                    
                # Check stop loss and target
                if position['action'] == 'BUY':
                    if current_price <= position['stop_loss']:
                        self._close_position(symbol, current_price, 'stop_loss')
                    elif current_price >= position['target']:
                        self._close_position(symbol, current_price, 'target')
                else:  # SELL position
                    if current_price >= position['stop_loss']:
                        self._close_position(symbol, current_price, 'stop_loss')
                    elif current_price <= position['target']:
                        self._close_position(symbol, current_price, 'target')
                        
        except Exception as e:
            logger.error(f"Error updating positions: {str(e)}")
            raise

    def _close_position(self, symbol: str, price: float, reason: str) -> None:
        """Close an open position"""
        try:
            position = self.positions[symbol]
            position['exit_price'] = price
            position['exit_timestamp'] = datetime.now()
            position['status'] = 'closed'
            position['close_reason'] = reason
            
            if position['action'] == 'BUY':
                pnl = (position['exit_price'] - position['entry_price']) * position['quantity']
            else:
                pnl = (position['entry_price'] - position['exit_price']) * position['quantity']
                
            position['pnl'] = pnl
            self.trade_history.append(position)
            del self.positions[symbol]
            
            logger.info(f"Position closed: {symbol} at {price} ({reason})")
            
        except Exception as e:
            logger.error(f"Error closing position: {str(e)}")
            raise

    def get_portfolio_metrics(self) -> Dict[str, float]:
        """Calculate portfolio metrics"""
        try:
            total_value = self.account_value
            open_positions = len(self.positions)
            daily_pnl = 0
            wins = 0
            total_trades = len(self.trade_history)
            
            # Calculate P&L for open positions
            for position in self.positions.values():
                if position['status'] == 'open':
                    current_price = self.data_collector.get_latest_price(position['symbol'])
                    if current_price:
                        if position['action'] == 'BUY':
                            pnl = (current_price - position['entry_price']) * position['quantity']
                        else:
                            pnl = (position['entry_price'] - current_price) * position['quantity']
                        total_value += pnl
            
            # Calculate metrics for closed trades
            for trade in self.trade_history:
                if trade['status'] == 'closed':
                    if trade['pnl'] > 0:
                        wins += 1
                    daily_pnl += trade['pnl']
            
            win_rate = (wins / total_trades * 100) if total_trades > 0 else 0
            
            return {
                'total_value': total_value,
                'open_positions': open_positions,
                'daily_pnl': daily_pnl,
                'win_rate': win_rate,
                'daily_return': (daily_pnl / total_value * 100) if total_value > 0 else 0
            }
            
        except Exception as e:
            logger.error(f"Error calculating portfolio metrics: {str(e)}")
            raise

    def update_position(self, symbol: str, quantity: int, price: float) -> None:
        """Update position for a symbol"""
        try:
            if symbol in self.positions:
                position = self.positions[symbol]
                position['quantity'] = quantity
                position['entry_price'] = price
                position['last_update'] = datetime.now()
            else:
                self.positions[symbol] = {
                    'symbol': symbol,
                    'quantity': quantity,
                    'entry_price': price,
                    'entry_time': datetime.now(),
                    'last_update': datetime.now(),
                    'status': 'open'
                }
            logger.info(f"Updated position for {symbol}: {quantity} @ {price}")
        except Exception as e:
            logger.error(f"Error updating position: {str(e)}")
            raise

    def has_position(self, symbol: str) -> bool:
        """Check if there is a position for the symbol."""
        return symbol in self.positions and self.positions[symbol]['quantity'] != 0

    def get_position(self, symbol: str) -> Dict[str, Any]:
        """Get position details for a symbol."""
        return self.positions.get(symbol, {
            'quantity': 0,
            'average_price': 0,
            'direction': None,
            'status': 'CLOSED'
        })

    def close_position(self, symbol: str) -> None:
        """Close position for a symbol."""
        try:
            if symbol in self.positions:
                self.positions[symbol]['status'] = 'CLOSED'
                self.positions[symbol]['quantity'] = 0
                self.logger.info(f"Closed position for {symbol}")
            
        except Exception as e:
            self.logger.error(f"Error closing position: {str(e)}")
            raise

    def get_portfolio_value(self) -> float:
        """Calculate total portfolio value."""
        try:
            total_value = self.account_value
            
            for symbol, position in self.positions.items():
                if position['status'] == 'OPEN':
                    current_price = self.data_collector.get_latest_price(symbol)
                    if current_price:
                        position_value = position['quantity'] * current_price
                        if position['action'] == 'BUY':
                            total_value += position_value
                        else:  # SHORT
                            total_value -= position_value
            
            return total_value
            
        except Exception as e:
            self.logger.error(f"Error calculating portfolio value: {str(e)}")
            return self.account_value

    def log_trade(self, trade: Dict[str, Any]) -> None:
        """Log a completed trade."""
        try:
            self.trade_history.append(trade)
            self.logger.info(f"Logged trade: {trade}")
            
        except Exception as e:
            self.logger.error(f"Error logging trade: {str(e)}")

    def get_trade_history(self) -> List[Dict[str, Any]]:
        """Get trade history."""
        return self.trade_history

    def execute_trade(self, symbol: str, action: str, quantity: int, price: float,
                     stop_loss: Optional[float] = None, profit_target: Optional[float] = None) -> Dict[str, Any]:
        """Execute a trade with risk management."""
        try:
            # Check risk limits
            if not self.risk_manager.check_risk_limits(symbol, action, quantity, price):
                raise ValueError("Trade exceeds risk limits")
            
            # Execute trade
            trade = self._simulate_trade(symbol, action, quantity, price, stop_loss, profit_target)
            
            # Update risk metrics
            self.risk_manager.update_risk_metrics(trade)
            
            return trade
        except Exception as e:
            self.logger.error(f"Error executing trade: {str(e)}")
            raise
