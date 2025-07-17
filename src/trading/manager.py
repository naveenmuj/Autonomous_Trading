import yaml
from datetime import datetime
import numpy as np
import logging
from typing import Dict, List, Optional, Tuple, Any
import pandas as pd
from SmartApi import SmartConnect  # Using the official import style
from src.data.collector import DataCollector
import os

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
        self.logger = logging.getLogger(__name__)
        self.risk_manager = RiskManager(config)
        self._initialize_account_value()  # Initialize account value based on mode

    def place_trade(self, symbol: str, action: str, quantity: int, price: float, stop_loss: float, target: float) -> Dict[str, Any]:
        """Place a new trade"""
        try:
            # Validate inputs
            if not all([symbol, action, quantity, price, stop_loss, target]):
                logger.warning(f"Trade skipped: Missing parameters symbol={symbol}, action={action}, quantity={quantity}, price={price}, stop_loss={stop_loss}, target={target}")
                raise ValueError("All trade parameters must be provided")
            if action not in ['BUY', 'SELL']:
                logger.warning(f"Trade skipped: Invalid action {action} for symbol {symbol}")
                raise ValueError("Action must be either 'BUY' or 'SELL'")
            if quantity <= 0:
                logger.warning(f"Trade skipped: Zero or negative quantity for {symbol}")
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
                logger.warning(f"Trade rejected by risk manager for {symbol}: {reason}")
                return None
            if self.config['trading']['mode'] == 'simulation':
                logger.info(f"Simulating trade for {symbol}: {trade}")
                return self._simulate_trade(symbol, action, quantity, price)
            else:
                logger.info(f"Executing live trade for {symbol}: {trade}")
                return self._execute_live_trade(trade)
        except Exception as e:
            logger.error(f"Error placing trade for {symbol}: {str(e)}")
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
            # Place the order using the new place_order logic
            order = self.place_order(
                symbol=trade['symbol'],
                order_type=trade['action'],
                quantity=trade['quantity'],
                price=trade['entry_price']
            )
            if order and (order.get('status') == 'COMPLETE' or order.get('status') == True):
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
        """Calculate position size based on risk management rules, with logging."""
        try:
            max_position_size = self.config['trading']['risk']['max_position_size']
            max_risk_amount = capital * max_position_size
            quantity = int(max_risk_amount / price)
            logger.info(f"Position sizing for capital={capital}, price={price}: max_risk_amount={max_risk_amount}, quantity={quantity}")
            if quantity <= 0:
                logger.warning(f"Position sizing resulted in zero quantity for price={price}, capital={capital}")
            return max(1, quantity)  # Ensure at least 1 unit
        except Exception as e:
            self.logger.error(f"Error calculating position size: {str(e)}")
            return 1

    def _update_position(self, trade: Dict[str, Any]):
        """Update position based on trade, with logging."""
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
                logger.info(f"Position updated (BUY) for {symbol}: {position}")
            else:  # SELL
                if position['quantity'] < quantity:
                    logger.warning(f"Trade skipped: Insufficient position to sell {quantity} of {symbol}. Current: {position['quantity']}")
                    raise ValueError(f"Insufficient position for {symbol}")
                position['quantity'] -= quantity
                if position['quantity'] == 0:
                    del self.positions[symbol]
                logger.info(f"Position updated (SELL) for {symbol}: {position if symbol in self.positions else 'CLOSED'}")
        except Exception as e:
            self.logger.error(f"Error updating position for {symbol}: {str(e)}")
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
                # Live trading: use DataCollector.place_broker_order
                if not self.data_collector or not hasattr(self.data_collector, 'place_broker_order'):
                    raise NotImplementedError("Live trading not available: DataCollector.place_broker_order missing.")
                # Prepare order params as per Angel One API
                # You may want to map symbol to tradingsymbol and symboltoken here
                tradingsymbol = symbol.replace('.NS', '-EQ') if symbol.endswith('.NS') else symbol
                symboltoken = self.data_collector.symbol_token_map.get(symbol) or self.data_collector.symbol_token_map.get(tradingsymbol)
                if not symboltoken:
                    raise ValueError(f"No symbol token found for {symbol}")
                order_params = {
                    "variety": "NORMAL",
                    "tradingsymbol": tradingsymbol,
                    "symboltoken": str(symboltoken),
                    "transactiontype": order_type,
                    "exchange": "NSE",
                    "ordertype": "MARKET" if price is None else "LIMIT",
                    "producttype": "DELIVERY",
                    "price": 0 if price is None else float(price),
                    "quantity": int(quantity)
                }
                response = self.data_collector.place_broker_order(order_params)
                return response
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
            
            # Use live broker data if available
            if hasattr(self, 'sync_with_broker'):
                self.sync_with_broker()
            if hasattr(self, 'sync_orders_with_broker'):
                self.sync_orders_with_broker()
            
            # Calculate P&L for open positions
            for position in self.positions.values():
                if position.get('status') == 'open':
                    current_price = self.data_collector.get_latest_price(position['symbol'])
                    if current_price:
                        if position.get('action', 'BUY') == 'BUY':
                            pnl = (current_price - position['avg_price']) * position['quantity']
                        else:
                            pnl = (position['avg_price'] - current_price) * position['quantity']
                        total_value += pnl
            
            # Calculate metrics for closed trades
            for trade in self.trade_history:
                if trade.get('status') == 'closed':
                    if trade.get('pnl', 0) > 0:
                        wins += 1
                    daily_pnl += trade.get('pnl', 0)
            
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
            # In live mode, get value from Angel One API
            if self.config['trading']['mode'] == 'live':
                return self._get_angel_portfolio_balance()

            # In simulation mode, calculate from account value and positions
            total_value = self.account_value
            for symbol, position in self.positions.items():
                if position['status'].upper() == 'OPEN':
                    current_price = self.data_collector.get_ltp(symbol)
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

    def get_daily_pnl(self) -> float:
        """Calculate daily profit and loss"""
        try:
            total_pnl = 0.0
            today = datetime.now().date()
            
            # Calculate realized P&L from today's closed trades
            for trade in self.trade_history:
                if trade.get('exit_timestamp') and trade['exit_timestamp'].date() == today:
                    if trade['action'] == 'BUY':
                        pnl = (trade['exit_price'] - trade['entry_price']) * trade['quantity']
                    else:  # SELL
                        pnl = (trade['entry_price'] - trade['exit_price']) * trade['quantity']
                    total_pnl += pnl
              # Calculate unrealized P&L for open positions opened today
            for symbol, position in self.positions.items():
                if position['status'] == 'open' and position.get('timestamp', datetime.now()).date() == today:
                    current_price = self.data_collector.get_ltp(symbol)
                    if current_price:
                        if position['action'] == 'BUY':
                            pnl = (current_price - position['entry_price']) * position['quantity']
                        else:  # SELL
                            pnl = (position['entry_price'] - current_price) * position['quantity']
                        total_pnl += pnl
            
            return round(total_pnl, 2)
            
        except Exception as e:
            self.logger.error(f"Error calculating daily PnL: {str(e)}")
            return 0.0

    def _get_angel_portfolio_balance(self) -> float:
        """Get portfolio balance from Angel One API in live mode using robust DataCollector method."""
        try:
            if not self.data_collector or not hasattr(self.data_collector, 'get_broker_balance'):
                logger.warning("Angel One DataCollector.get_broker_balance not available")
                return 0.0
            logger.info("Fetching broker balance using DataCollector.get_broker_balance()")
            balance = self.data_collector.get_broker_balance()
            logger.info(f"Fetched broker balance: {balance}")
            if balance:
                available_cash = float(balance.get('available_cash') or 0)
                net = float(balance.get('net') or 0)
                logger.info(f"Available Cash: {available_cash}, Net: {net}")
                return net
            else:
                logger.error("No balance data returned from Angel One RMS endpoint.")
                return 0.0
        except Exception as e:
            logger.error(f"Error getting Angel One portfolio balance: {str(e)}")
            return 0.0

    def _save_account_value(self) -> None:
        """Save current account value to config file in simulation mode"""
        try:
            if self.config['trading']['mode'] == 'simulation':
                self.config['initial_account_value'] = self.account_value
                config_file = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'config.yaml')
                with open(config_file, 'w') as f:
                    yaml.dump(self.config, f)
                logger.info(f"Saved account value {self.account_value} to config")
        except Exception as e:
            logger.error(f"Error saving account value: {str(e)}")

    def _initialize_account_value(self) -> None:
        """Initialize account value based on trading mode"""
        try:
            if self.config['trading']['mode'] == 'live':
                # Get real portfolio value from Angel One
                self.account_value = self._get_angel_portfolio_balance()
                logger.info(f"Initialized live account value: {self.account_value}")
            else:
                # Use value from config for simulation mode
                self.account_value = self.config.get('initial_account_value', 100000)
                logger.info(f"Initialized simulation account value: {self.account_value}")
        except Exception as e:
            logger.error(f"Error initializing account value: {str(e)}")
            self.account_value = self.config.get('initial_account_value', 100000)

    def get_positions(self) -> List[Dict]:
        """Get current positions with enhanced error handling"""
        try:
            if hasattr(self, '_positions'):
                return [
                    {
                        'symbol': pos['symbol'],
                        'quantity': pos['quantity'],
                        'entry_price': pos['entry_price'],
                        'current_price': pos['current_price'],
                        'pnl': pos['pnl'],
                        'status': pos['status']
                    }
                    for pos in self._positions.values()
                ]
            return []
        except Exception as e:
            logger.error(f"Error getting positions: {e}")
            return []

    def place_order(self, symbol: str, signal: int, sentiment=None, reasoning=None):
        """
        Place an order with full risk management and explainability.
        signal: 1=buy, -1=sell
        """
        try:
            action = 'BUY' if signal == 1 else 'SELL'
            price = self.data_collector.get_latest_price(symbol)
            if not price:
                logger.warning(f"No price available for {symbol}, skipping order.")
                return None
            # Position sizing
            capital = self.account_value
            quantity = self.calculate_position_size(capital, price)
            # Stop-loss and target
            stop_loss_pct = self.risk_manager.stop_loss
            profit_target_pct = self.config.get('trading', {}).get('strategy', {}).get('profit_target', 0.03)
            stop_loss = price * (1 - stop_loss_pct) if action == 'BUY' else price * (1 + stop_loss_pct)
            target = price * (1 + profit_target_pct) if action == 'BUY' else price * (1 - profit_target_pct)
            # Log reasoning for explainability
            logger.info(f"Placing order: {symbol} | {action} | Qty: {quantity} | Price: {price} | SL: {stop_loss} | Target: {target} | Sentiment: {sentiment} | Reasoning: {reasoning}")
            # Enforce risk checks
            trade = {
                'symbol': symbol,
                'action': action,
                'quantity': quantity,
                'entry_price': price,
                'stop_loss': stop_loss,
                'target': target,
                'timestamp': datetime.now(),
                'status': 'open',
                'type': self.config['trading']['mode'],
                'sentiment': sentiment,
                'reasoning': reasoning
            }
            is_valid, reason = self.risk_manager.validate_trade(trade, self.account_value)
            if not is_valid:
                logger.warning(f"Trade rejected: {reason}")
                return None
            # Simulate or execute trade
            if self.config['trading']['mode'] == 'simulation':
                return self._simulate_trade(symbol, action, quantity, price, stop_loss, target)
            else:
                return self._execute_live_trade(trade)
        except Exception as e:
            logger.error(f"Error placing order: {str(e)}")
            return None

    def sync_with_broker(self):
        """
        Fetch live holdings from Angel One and update self.positions using robust DataCollector.get_broker_portfolio().
        """
        try:
            if not hasattr(self.data_collector, 'get_broker_portfolio'):
                logger.error("DataCollector missing get_broker_portfolio method.")
                return
            broker_positions = self.data_collector.get_broker_portfolio()
            if not broker_positions:
                logger.warning("No holdings data returned from Angel One (getHolding). Portfolio may be empty or API error.")
                return
            self.update_positions_from_broker(broker_positions)
            logger.info("Angel One portfolio sync complete.")
        except Exception as e:
            logger.error(f"Error in sync_with_broker: {e}")

    def update_positions_from_broker(self, broker_positions):
        """
        Update self.positions from Angel One holdings data.
        broker_positions: list of dicts from Angel One API.
        """
        try:
            self.positions.clear()
            for pos in broker_positions:
                symbol = pos.get('tradingsymbol') or pos.get('symbol')
                qty = pos.get('quantity') or pos.get('qty')
                avg_price = pos.get('averageprice') or pos.get('avg_price')
                if symbol and qty:
                    self.positions[symbol] = {
                        'quantity': qty,
                        'avg_price': avg_price,
                        'stop_loss': None,
                        'profit_target': None,
                        'status': 'open',
                        'from_broker': True
                    }
            logger.info(f"Updated positions from broker: {list(self.positions.keys())}")
        except Exception as e:
            logger.error(f"Error updating positions from broker: {e}")

    def sync_orders_with_broker(self):
        """
        Fetch live orders from Angel One and update self.trade_history.
        """
        try:
            api_key = self.config.get('angel_one', {}).get('api_key')
            client_id = self.config.get('angel_one', {}).get('client_id')
            password = self.config.get('angel_one', {}).get('password')
            totp_secret = self.config.get('angel_one', {}).get('totp_secret')
            if not all([api_key, client_id, password, totp_secret]):
                logger.warning("Angel One API credentials missing in config. Cannot sync orders.")
                return
            smart_api = SmartConnect(api_key)
            try:
                smart_api.generateSession(client_id, password, smart_api.get_totp(totp_secret))
            except Exception as e:
                logger.error(f"Angel One login failed: {e}")
                return
            try:
                orders = smart_api.orderBook()
                if not orders or 'data' not in orders or not orders['data']:
                    logger.info("Angel One API returned no order data (empty order book or API response).")
                    return
                self.update_orders_from_broker(orders['data'])
                logger.info("Angel One order sync complete.")
            except Exception as e:
                logger.error(f"Error fetching orders from Angel One: {e}")
        except Exception as e:
            logger.error(f"Error in sync_orders_with_broker: {e}")

    def update_orders_from_broker(self, broker_orders):
        """
        Update self.trade_history from Angel One order data.
        broker_orders: list of dicts from Angel One API.
        """
        try:
            self.trade_history.clear()
            for order in broker_orders:
                symbol = order.get('tradingsymbol') or order.get('symbol')
                qty = order.get('quantity') or order.get('qty')
                price = order.get('averageprice') or order.get('price')
                status = order.get('status')
                action = order.get('transactiontype') or order.get('action')
                if symbol and qty:
                    self.trade_history.append({
                        'symbol': symbol,
                        'action': action,
                        'quantity': qty,
                        'price': price,
                        'status': status,
                        'order_id': order.get('orderid'),
                        'timestamp': order.get('orderentrytime')
                    })
            logger.info(f"Updated trade history from broker: {len(self.trade_history)} orders")
        except Exception as e:
            logger.error(f"Error updating orders from broker: {e}")

    def get_portfolio_metrics(self) -> Dict[str, float]:
        """
        Calculate portfolio metrics, using live broker data if available.
        """
        try:
            total_value = self.account_value
            open_positions = len(self.positions)
            daily_pnl = 0
            wins = 0
            total_trades = len(self.trade_history)
            # Use live broker data if available
            if hasattr(self, 'sync_with_broker'):
                self.sync_with_broker()
            if hasattr(self, 'sync_orders_with_broker'):
                self.sync_orders_with_broker()
            # Calculate P&L for open positions
            for position in self.positions.values():
                if position.get('status') == 'open':
                    current_price = self.data_collector.get_latest_price(position['symbol'])
                    if current_price:
                        if position.get('action', 'BUY') == 'BUY':
                            pnl = (current_price - position['avg_price']) * position['quantity']
                        else:
                            pnl = (position['avg_price'] - current_price) * position['quantity']
                        total_value += pnl
            # Calculate metrics for closed trades
            for trade in self.trade_history:
                if trade.get('status') == 'closed':
                    if trade.get('pnl', 0) > 0:
                        wins += 1
                    daily_pnl += trade.get('pnl', 0)
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
