import pytest
import pandas as pd
import numpy as np
import os
import sys
from unittest.mock import MagicMock, patch
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)

# Add project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.ai.models import AITrader, TechnicalAnalysisModel, SentimentAnalyzer
from src.data.collector import DataCollector
from src.trading.manager import TradeManager
from src.trading.strategy import EnhancedTradingStrategy

@pytest.fixture
def mock_config():
    return {
        'trading': {
            'risk': {
                'max_position_size': 0.02,  # 2% of capital
                'stop_loss': 0.02,          # 2% stop loss
                'max_trades': 5,            # Max concurrent trades
            },
            'strategy': {
                'lookback_period': 20,
                'rsi_period': 14,
                'rsi_overbought': 70,
                'rsi_oversold': 30,
                'profit_target': 0.03       # 3% profit target
            }
        },
        'risk_management': {
            'max_loss_per_trade': 0.02,     # 2% max loss per trade
            'max_portfolio_risk': 0.10,      # 10% max portfolio risk
            'max_total_risk': 0.06,         # 6% max total risk
            'position_sizing': 'fixed',      # fixed or kelly
            'max_correlated_trades': 2,      # max trades in correlated instruments
            'max_sector_exposure': 0.20,     # 20% max sector exposure
            'stop_loss_type': 'fixed',      # fixed or atr-based
            'profit_target_mult': 2.0,      # Risk:reward ratio target
            'trailing_stop': True,          # Enable trailing stops
            'max_drawdown': 0.15,          # 15% max drawdown
            'risk_free_rate': 0.03,        # 3% risk-free rate
            'kelly_fraction': 0.5,         # Half-Kelly criterion
            'correlation_threshold': 0.7    # Correlation threshold for pairs
        }
    }

@pytest.fixture
def strategy(mock_config):
    models = {
        'technical': TechnicalAnalysisModel(mock_config),
        'ai': AITrader(mock_config),
        'sentiment': SentimentAnalyzer(mock_config)
    }
    return EnhancedTradingStrategy(mock_config, models)

@pytest.fixture
def manager(mock_config):
    return TradeManager(mock_config)

@pytest.mark.timeout(5)
class TestTrading:
    def test_trade_simulation(self, config, mock_angel_api):
        """Test trade simulation functionality"""
        logger.info("Testing trade simulation")
        trade_manager = TradeManager(config=config, data_collector=DataCollector(config))
        
        # Test trade placement
        trade = trade_manager._simulate_trade(
            symbol='RELIANCE',
            action='BUY',
            quantity=100,
            price=2500.0,
            stop_loss=2450.0,
            target=2600.0
        )
        
        # Verify trade details
        assert trade['symbol'] == 'RELIANCE'
        assert trade['action'] == 'BUY'
        assert trade['quantity'] == 100
        assert trade['status'] == 'open'
        logger.info("Trade simulation test passed")

    def test_risk_management(self, config):
        """Test risk management rules"""
        logger.info("Testing risk management")
        trade_manager = TradeManager(config=config)
        
        # Test position sizing
        position = trade_manager._calculate_position_size(
            capital=100000.0,  # 1L capital
            risk_per_trade=2.0,  # 2% risk per trade
            entry_price=2500.0,
            stop_loss=2450.0
        )
        assert position > 0
        assert isinstance(position, int)  # Should return whole number of shares
        logger.info("Risk management test passed")

    def test_trade_execution(self, config, mock_angel_api):
        """Test trade execution workflow"""
        logger.info("Testing trade execution")
        trade_manager = TradeManager(config=config, data_collector=DataCollector(config))
        
        # Test order placement
        order = trade_manager.place_order(
            symbol='RELIANCE',
            order_type='MARKET',
            quantity=100,
            price=2500.0  # Optional for market orders
        )
        assert order is not None
        assert order['status'] == 'COMPLETE'  # Should be COMPLETE in simulation mode
        assert order['symbol'] == 'RELIANCE'
        assert order['quantity'] == 100
        assert 'order_id' in order
        logger.info("Trade execution test passed")

    def test_strategy_initialization(self, strategy):
        """Test trading strategy initialization"""
        assert strategy is not None
        assert strategy.config == mock_config

    def test_signal_generation(self, strategy):
        """Test trading signal generation"""
        data = pd.DataFrame({
            'open': np.random.randn(100) + 100,
            'high': np.random.randn(100) + 102,
            'low': np.random.randn(100) + 98,
            'close': np.random.randn(100) + 101,
            'volume': np.random.randint(1000, 2000, 100),
            'rsi': np.random.randn(100) * 10 + 50,
            'sma_50': np.random.randn(100) + 100
        })
        
        signals = strategy.generate_signals(data)
        assert isinstance(signals, pd.DataFrame)
        assert 'signal' in signals.columns
        assert all(signals['signal'].isin([-1, 0, 1]))  # Sell, Hold, Buy

    def test_risk_management(self, manager):
        """Test risk management rules"""
        capital = 100000
        price = 100
        
        # Test position sizing
        position_size = manager.calculate_position_size(capital, price)
        max_risk = capital * manager.config['trading']['risk']['max_position_size']
        assert position_size * price <= max_risk

    def test_order_placement(self, manager):
        """Test order placement logic"""
        with patch('src.trading.manager.SmartConnect') as mock_smart_connect:
            mock_api = MagicMock()
            mock_smart_connect.return_value = mock_api
            mock_api.placeOrder.return_value = {
                'status': True,
                'data': {'orderId': '123'}
            }
            
            result = manager.place_order(
                symbol='RELIANCE',
                quantity=10,
                order_type='MARKET',
                transaction_type='BUY'
            )
            
            assert result['status']
            assert result['data']['orderId'] == '123'

    def test_portfolio_management(self, manager):
        """Test portfolio management"""
        # Add test positions
        manager.update_position('RELIANCE', 10, 2000)
        manager.update_position('TCS', 5, 3500)
        
        # Test portfolio value calculation
        portfolio_value = manager.get_portfolio_value()
        expected_value = 10 * 2000 + 5 * 3500
        assert portfolio_value == expected_value

    def test_stop_loss_handling(self, strategy):
        """Test stop loss handling"""
        entry_price = 100
        current_price = 95
        stop_loss_pct = strategy.config['trading']['risk']['stop_loss']
        
        should_exit = strategy.check_stop_loss(
            entry_price=entry_price,
            current_price=current_price,
            stop_loss_pct=stop_loss_pct
        )
        assert should_exit

    def test_profit_target_handling(self, strategy):
        """Test profit target handling"""
        entry_price = 100
        current_price = 110
        target_pct = strategy.config['trading']['strategy']['profit_target']
        
        should_exit = strategy.check_profit_target(
            entry_price=entry_price,
            current_price=current_price,
            target_pct=target_pct
        )
        assert should_exit

    def test_position_tracking(self, manager):
        """Test position tracking"""
        # Add position
        manager.update_position('RELIANCE', 10, 2000)
        assert manager.has_position('RELIANCE')
        
        # Update position
        manager.update_position('RELIANCE', 5, 2100)
        position = manager.get_position('RELIANCE')
        assert position['quantity'] == 5
        
        # Close position
        manager.close_position('RELIANCE')
        assert not manager.has_position('RELIANCE')

    def test_trade_logging(self, manager):
        """Test trade logging"""
        trade = {
            'symbol': 'RELIANCE',
            'entry_price': 2000,
            'exit_price': 2100,
            'quantity': 10,
            'entry_time': datetime.now(),
            'exit_time': datetime.now() + timedelta(days=1),
            'pnl': 1000
        }
        
        manager.log_trade(trade)
        trade_history = manager.get_trade_history()
        assert len(trade_history) > 0
        assert trade_history[-1]['symbol'] == trade['symbol']

    @pytest.mark.integration
    def test_end_to_end_trading(self, strategy, manager):
        """Test end-to-end trading workflow"""
        # Generate sample data
        data = pd.DataFrame({
            'open': np.random.randn(100) + 100,
            'high': np.random.randn(100) + 102,
            'low': np.random.randn(100) + 98,
            'close': np.random.randn(100) + 101,
            'volume': np.random.randint(1000, 2000, 100),
            'rsi': np.random.randn(100) * 10 + 50,
            'sma_50': np.random.randn(100) + 100
        })
        
        # Get trading signals
        signals = strategy.generate_signals(data)
        
        # Simulate trading
        for idx, row in signals.iterrows():
            if row['signal'] == 1:  # Buy signal
                price = data.loc[idx, 'close']
                quantity = manager.calculate_position_size(100000, price)
                
                with patch.object(manager, 'place_order') as mock_order:
                    mock_order.return_value = {'status': True, 'data': {'orderId': '123'}}
                    manager.execute_trade('RELIANCE', quantity, 'BUY', price)
                    
                    assert mock_order.called
                    assert manager.has_position('RELIANCE')
