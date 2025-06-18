import os
import sys
import pytest
import logging
import pandas as pd
import numpy as np
from unittest.mock import MagicMock, patch
from datetime import datetime, timedelta
from typing import Dict, Any

# Import our mocks
from tests.mocks.smartapi_mock import SmartConnect
from tests.mocks.mock_smartwebsocket import MockSmartWebSocketV2

# Create mock modules
mock_smartapi = MagicMock()
mock_smartapi.SmartConnect = SmartConnect
sys.modules['smartapi'] = mock_smartapi

mock_websocket = MagicMock()
mock_websocket.SmartWebSocketV2 = MockSmartWebSocketV2
sys.modules['SmartApi.smartWebSocketV2'] = mock_websocket

# Add project root and src to Python path
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
src_dir = os.path.join(root_dir, 'src')
sys.path.insert(0, root_dir)
sys.path.insert(0, src_dir)

# Set testing environment
os.environ['TESTING'] = '1'

# Import after setting TESTING environment
from src.ai.models import AITrader, TechnicalAnalysisModel, SentimentAnalyzer
from src.data.collector import DataCollector
from src.trading.manager import TradeManager, RiskManager
from src.trading.strategy import EnhancedTradingStrategy

# Reset singleton instances between tests
@pytest.fixture(autouse=True)
def cleanup_singletons():
    yield
    # Clear singleton instances after each test
    if hasattr(DataCollector, '_instances'):
        DataCollector._instances.clear()

# Default test configuration
TEST_CONFIG = {
    'initial_account_value': 100000,
    'trading': {
        'risk': {
            'max_position_size': 0.02,
            'max_trades': 5,
            'stop_loss': 0.02,
            'max_sector_exposure': 0.25  # Maximum exposure per sector (25%)
        },
        'strategy': {
            'lookback_period': 20,
            'profit_target': 0.03,
            'rsi_overbought': 70,
            'rsi_oversold': 30,
            'macd_fast': 12,
            'macd_slow': 26,
            'macd_signal': 9
        }
    },
    'risk_management': {
        'max_drawdown': 0.15,
        'correlation_threshold': 0.7,
        'max_correlated_trades': 2,
        'kelly_fraction': 0.5,
        'position_sizing': {
            'method': 'fixed_fraction',
            'fraction': 0.02
        },
        'max_sector_exposure': 0.25,  # Maximum exposure per sector (25%)
        'max_leverage': 1.0  # No leverage
    },
    'features': {
        'technical_indicators': {
            'trend': ['sma', 'ema'],
            'momentum': ['rsi', 'macd'],
            'volatility': ['bbands', 'atr'],
            'volume': ['obv', 'adl'],
            'sma_periods': [20, 50, 200],
            'ema_periods': [12, 26],
            'rsi_period': 14,
            'macd_fast': 12,
            'macd_slow': 26,
            'macd_signal': 9,
            'bb_period': 20,
            'bb_std': 2,
            'atr_period': 14
        }
    },
    'credentials': {
        'angel_one': {
            'api_key': 'test_api_key',
            'client_id': 'test_client_id',
            'pin': 'test_pin',
            'totp_key': 'test_totp_key'
        }
    }
}

@pytest.fixture
def mock_config() -> Dict[str, Any]:
    """Provide test configuration."""
    return {
        'angel_one': {
            'api_key': 'test_key',
            'client_id': 'test_client',
            'pin': '1234'
        },
        'technical_indicators': {
            'trend': ['SMA', 'EMA'],
            'momentum': ['RSI', 'MACD'],
            'volatility': ['BB', 'ATR'],
            'SMA_periods': [20, 50, 200],
            'EMA_periods': [9, 21]
        },
        'risk_management': {
            'max_position_size': 0.02,
            'max_trades': 5,
            'stop_loss': 0.02,
            'max_drawdown': 0.15,
            'max_sector_exposure': 0.20,
            'correlation_threshold': 0.7,
            'kelly_fraction': 0.5,
            'max_correlated_trades': 2
        },
        'trading': {
            'symbols': ['RELIANCE.NS', 'INFY.NS', 'TCS.NS'],
            'interval': '1d',
            'strategy': {
                'lookback_period': 20,
                'rsi_oversold': 30,
                'rsi_overbought': 70,
                'profit_target': 0.03
            }
        },
        'model': {
            'input_dim': 20,
            'batch_size': 32,
            'epochs': 100,
            'learning_rate': 0.001,
            'validation_split': 0.2
        },
        'data': {
            'timeout': 30,
            'max_retries': 3
        }
    }

@pytest.fixture
def sample_market_data():
    dates = pd.date_range(start='2020-01-01', periods=100, freq='D')
    np.random.seed(42)
    data = pd.DataFrame({
        'open': np.random.normal(100, 2, 100).cumsum(),
        'high': np.random.normal(101, 2, 100).cumsum(),
        'low': np.random.normal(99, 2, 100).cumsum(),
        'close': np.random.normal(100, 2, 100).cumsum(),
        'volume': np.random.randint(1000, 2000, 100),
        'rsi': np.random.uniform(0, 100, 100),
        'sma_50': np.random.normal(100, 1, 100).cumsum()
    }, index=dates)
    data['symbol'] = 'TEST'
    return data

@pytest.fixture
def mock_angel_api():
    """Mock Angel One API."""
    with patch('src.data.collector.SmartConnect') as mock:
        api = MagicMock()
        api.generateSession.return_value = {'status': True}
        api.getProfile.return_value = {'status': True, 'data': {'name': 'Test User'}}
        api.getNseSymbols.return_value = [
            {'name': 'RELIANCE', 'token': '2885'},
            {'name': 'TCS', 'token': '11536'},
            {'name': 'HDFCBANK', 'token': '1333'},
            {'name': 'INFY', 'token': '1594'},
            {'name': 'ICICIBANK', 'token': '4963'}
        ]
        mock.return_value = api
        yield api

@pytest.fixture
def mock_yfinance():
    """Mock yfinance for testing."""
    with patch('yfinance.download') as mock:
        def mock_download(*args, **kwargs):
            dates = pd.date_range(start='2024-01-01', end='2024-01-10', freq='D')
            data = {
                'Open': np.random.uniform(100, 200, len(dates)),
                'High': np.random.uniform(200, 300, len(dates)),
                'Low': np.random.uniform(50, 100, len(dates)),
                'Close': np.random.uniform(100, 200, len(dates)),
                'Volume': np.random.uniform(1000, 5000, len(dates))
            }
            return pd.DataFrame(data, index=dates)
        mock.side_effect = mock_download
        yield mock

@pytest.fixture
def mock_news_api():
    """Mock News API for testing."""
    with patch('newsapi.NewsApiClient') as mock:
        api = MagicMock()
        api.get_everything.return_value = {
            'status': 'ok',
            'articles': [
                {
                    'title': 'Test News 1',
                    'description': 'Test Description 1',
                    'publishedAt': '2024-01-01T00:00:00Z'
                },
                {
                    'title': 'Test News 2',
                    'description': 'Test Description 2',
                    'publishedAt': '2024-01-02T00:00:00Z'
                }
            ]
        }
        mock.return_value = api
        yield api

@pytest.fixture
def mock_risk_manager():
    """Mock RiskManager for testing."""
    with patch('src.trading.manager.RiskManager') as mock:
        manager = MagicMock()
        manager.check_risk_limits.return_value = True
        manager.update_risk_metrics.return_value = None
        mock.return_value = manager
        yield manager

@pytest.fixture
def mock_data_collector():
    collector = MagicMock(spec=DataCollector)
    collector.get_historical_data.return_value = pd.DataFrame({
        'open': np.random.normal(100, 2, 100).cumsum(),
        'high': np.random.normal(101, 2, 100).cumsum(),
        'low': np.random.normal(99, 2, 100).cumsum(),
        'close': np.random.normal(100, 2, 100).cumsum(),
        'volume': np.random.randint(1000, 2000, 100),
        'rsi': np.random.uniform(0, 100, 100),
        'macd': np.random.normal(0, 1, 100),
        'macd_signal': np.random.normal(0, 1, 100),
        'macd_hist': np.random.normal(0, 1, 100),
        'bb_upper': np.random.normal(102, 1, 100).cumsum(),
        'bb_middle': np.random.normal(100, 1, 100).cumsum(),
        'bb_lower': np.random.normal(98, 1, 100).cumsum(),
        'sma_50': np.random.normal(100, 1, 100).cumsum(),
        'atr': np.random.uniform(1, 3, 100),
        'obv': np.random.normal(1000000, 100000, 100).cumsum(),
        'adx': np.random.uniform(0, 100, 100),
        'symbol': ['TEST'] * 100
    }, index=pd.date_range(start='2020-01-01', periods=100, freq='D'))
    return collector

@pytest.fixture
def dummy_news():
    return [
        {
            'title': 'Stock Market Sees Positive Growth',
            'description': 'Markets rise on positive economic data'
        },
        {
            'title': 'Company XYZ Reports Strong Earnings',
            'description': 'Profits up 20% year over year'
        }
    ]

@pytest.fixture
def mock_trading_strategy():
    """Mock EnhancedTradingStrategy for testing."""
    with patch('src.trading.strategy.EnhancedTradingStrategy') as mock:
        strategy = MagicMock()
        strategy.generate_signals.return_value = pd.DataFrame({
            'symbol': ['RELIANCE'],
            'signal': [1],
            'confidence': [0.8],
            'timestamp': [datetime.now()]
        })
        mock.return_value = strategy
        yield strategy

@pytest.fixture
def mock_trade_manager():
    """Mock TradeManager for testing."""
    with patch('src.trading.manager.TradeManager') as mock:
        manager = MagicMock()
        manager.execute_trade.return_value = {
            'symbol': 'RELIANCE',
            'action': 'BUY',
            'quantity': 100,
            'price': 150.0,
            'timestamp': datetime.now()
        }
        mock.return_value = manager
        yield manager

@pytest.fixture
def config():
    """Function-scoped config fixture"""
    return {
        'trading': {
            'mode': 'simulation'
        },
        'risk_management': {
            'max_loss_per_trade': 2,
            'max_portfolio_risk': 5
        },
        'apis': {
            'angle_one': {
                'api_key': 'test_key',
                'client_id': 'test_client',
                'mpin': 'test_pin',
                'totp_secret': 'test_totp'
            }
        },
        'data': {
            'update_interval': 5,
            'historical_days': 30  # Reduced for testing
        },
        'models': {  # Changed to models to match validation
            'training': {
                'epochs': 2,  # Reduced for testing
                'batch_size': 32,
                'validation_split': 0.2
            }
        }
    }

@pytest.fixture
def logger():
    """Function-scoped logger fixture"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)

@pytest.fixture
def sample_trade_data():
    """Function-scoped sample data fixture"""
    return pd.DataFrame({
        'Open': [100.0] * 10,  # Reduced size
        'High': [102.0] * 10,
        'Low': [99.0] * 10,
        'Close': [101.0] * 10,
        'Volume': [1000] * 10
    }, index=pd.date_range(start='2025-06-08', periods=10, freq='1min'))

@pytest.fixture
def mock_data_collector(mock_angel_api):
    config = {
        'apis': {
            'angle_one': {
                'api_key': 'test_key',
                'client_id': 'test_client',
                'mpin': 'test_pin',
                'totp_secret': 'test_totp'
            }
        },
        'data': {
            'update_interval': 5
        }
    }
    collector = DataCollector(config)
    return collector

@pytest.fixture
def trade_config():
    return {
        'trading': {
            'mode': 'simulation'
        },
        'risk_management': {
            'max_loss_per_trade': 2,
            'max_portfolio_risk': 5
        }
    }

@pytest.fixture
def trade_manager(trade_config, mock_data_collector):
    return TradeManager(config=trade_config, data_collector=mock_data_collector)

@pytest.fixture
def trading_strategy(mock_config):
    return EnhancedTradingStrategy(config=mock_config)

@pytest.fixture
def ai_trader(mock_config):
    return AITrader(config=mock_config)

@pytest.fixture
def technical_model(mock_config):
    return TechnicalAnalysisModel(config=mock_config)

@pytest.fixture
def sentiment_analyzer(mock_config):
    return SentimentAnalyzer(config=mock_config)

@pytest.fixture
def mock_market_data():
    dates = pd.date_range(start='2020-01-01', periods=100)
    data = pd.DataFrame(index=dates)
    data['open'] = np.random.normal(100, 1, 100)
    data['high'] = data['open'] + abs(np.random.normal(0, 1, 100))
    data['low'] = data['open'] - abs(np.random.normal(0, 1, 100))
    data['close'] = np.random.normal(100, 1, 100)
    data['volume'] = np.random.randint(1000, 2000, 100)
    data['symbol'] = 'RELIANCE.NS'
    
    # Add technical indicators
    data['rsi'] = np.random.normal(50, 10, 100)
    data['sma_50'] = data['close'].rolling(window=50).mean()
    
    return data

@pytest.fixture
def test_news():
    return [
        {
            'title': 'Test News 1',
            'description': 'Positive market outlook',
            'sentiment': 0.8
        },
        {
            'title': 'Test News 2',
            'description': 'Negative market outlook',
            'sentiment': -0.3
        }
    ]

@pytest.fixture
def mock_smart_connect():
    with patch('src.data.collector.SmartConnect') as mock:
        mock_instance = MagicMock()
        mock.return_value = mock_instance
        mock_instance.generateSession.return_value = {'status': True, 'message': 'SUCCESS', 'data': {'jwtToken': 'dummy_token'}}
        mock_instance.getProfile.return_value = {'status': True, 'message': 'SUCCESS', 'data': {'name': 'Test User'}}
        yield mock_instance
