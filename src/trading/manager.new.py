import yaml
import os
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
        self.logger = logging.getLogger(__name__)
        self.risk_manager = RiskManager(config)
        self._initialize_account_value()  # Initialize account value based on mode

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

    def get_daily_pnl(self) -> float:
        """Calculate daily profit and loss"""
        try:
            # For live mode, get PnL from Angel One portfolio
            if self.config['trading']['mode'] == 'live':
                response = self.data_collector.angel_api.portfolio()
                if response and response.get('status') and response.get('data'):
                    return float(response['data'].get('daypl', 0))

            # For simulation mode, calculate PnL
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

    # ... (rest of the TradeManager methods remain unchanged)
