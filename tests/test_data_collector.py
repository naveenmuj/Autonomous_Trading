import pytest
import pandas as pd
import numpy as np
import os
import sys
import time
from unittest.mock import patch, MagicMock, ANY
import logging
from datetime import datetime, timedelta
import json
import requests

# Setup logging
logger = logging.getLogger(__name__)

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import after path setup
from src.data.collector import DataCollector

@pytest.fixture(autouse=True)
def cleanup():
    """Cleanup fixture that runs before and after each test"""
    # Before test
    DataCollector._instance = None
    
    yield
    
    # After test
    if DataCollector._instance:
        DataCollector._instance.cleanup()
        DataCollector._instance = None

@pytest.fixture
def mock_angel_response():
    """Mock successful Angel One API response"""
    return {
        'status': True,
        'message': 'Success',
        'data': {
            'feedToken': 'test_feed_token',
            'refreshToken': 'test_refresh_token',
            'jwtToken': 'test_jwt_token'
        }
    }

@pytest.fixture
def mock_instruments_response():
    """Mock instrument master file response"""
    return [
        {
            'token': '123',
            'symbol': 'RELIANCE',
            'exch_seg': 'NSE',
            'instrumenttype': 'EQ'
        },
        {
            'token': '456',
            'symbol': 'NIFTY',
            'exch_seg': 'NSE',
            'instrumenttype': 'INDICES'
        }
    ]

@pytest.fixture
def mock_candle_data():
    """Mock candle data response"""
    dates = pd.date_range(start='2025-06-09', periods=30, freq='1min')
    return {
        'status': True,
        'data': [
            [str(d), 100.0, 101.0, 99.0, 100.5, 1000] for d in dates
        ]
    }

@pytest.fixture
def data_config():
    """Test configuration"""
    return {
        'apis': {
            'angle_one': {
                'api_key': 'test_key',
                'client_id': 'test_client',
                'mpin': '1234',
                'totp_secret': 'test_secret'
            }
        },
        'timeout': {
            'api_call': 5,
            'cleanup': 1
        },
        'retry': {
            'max_attempts': 2,
            'delay': 1
        }
    }

@pytest.fixture
def mock_config():
    return {
        'apis': {
            'angle_one': {
                'api_key': 'test_key',
                'client_id': 'test_client',
                'mpin': '1234',
                'totp_secret': 'test_secret'
            },
            'newsapi': {
                'api_key': 'test_news_key'
            }
        },
        'model': {
            'features': {
                'technical_indicators': {
                    'trend': True,
                    'momentum': True,
                    'volatility': True,
                    'volume': True
                }
            }
        },
        'data': {
            'indices': {
                'nifty50': '^NSEI',
                'india_vix': '^INDIAVIX'
            },
            'fallback_stocks': ['RELIANCE', 'TCS']
        }
    }

@pytest.fixture
def collector(mock_config):
    with patch('src.data.collector.SmartConnect') as mock_smart_connect:
        # Mock Angel One API responses
        mock_api = MagicMock()
        mock_smart_connect.return_value = mock_api
        mock_api.generateSession.return_value = {
            'status': True,
            'data': {
                'feedToken': 'test_feed',
                'refreshToken': 'test_refresh'
            }
        }
        mock_api.getProfile.return_value = {
            'status': True,
            'data': {'name': 'Test User'}
        }
        
        collector = DataCollector(mock_config)
        yield collector

@pytest.mark.timeout(30)
@pytest.mark.usefixtures('mock_smart_connect')
class TestDataCollector:
    """Test cases for DataCollector"""
    
    def test_singleton_pattern(self, data_config):
        """Test singleton pattern implementation"""
        collector1 = DataCollector(data_config)
        collector2 = DataCollector(data_config)
        assert collector1 is collector2
        assert collector1._initialized
        logger.info("Singleton test passed")
        
    @patch('SmartApi.smartConnect.SmartConnect')
    def test_angel_one_authentication(self, mock_smart_connect, data_config, mock_angel_response):
        """Test Angel One authentication"""
        # Setup mock
        mock_instance = mock_smart_connect.return_value
        mock_instance.generateSession.return_value = mock_angel_response
        mock_instance.getProfile.return_value = {
            'status': True,
            'data': {'name': 'Test User'}
        }
        
        collector = DataCollector(data_config)
        assert collector.angel_api is not None
        mock_instance.generateSession.assert_called_once()
        mock_instance.getProfile.assert_called_once()
        logger.info("Angel One authentication test passed")
        
    @patch('requests.get')
    @patch('SmartApi.smartConnect.SmartConnect')
    def test_token_mapping_initialization(self, mock_smart_connect, mock_get, 
                                       data_config, mock_angel_response, mock_instruments_response):
        """Test token mapping initialization"""
        # Setup mocks
        mock_instance = mock_smart_connect.return_value
        mock_instance.generateSession.return_value = mock_angel_response
        mock_instance.getProfile.return_value = {'status': True, 'data': {'name': 'Test User'}}
        
        mock_response = MagicMock()
        mock_response.ok = True
        mock_response.json.return_value = mock_instruments_response
        mock_get.return_value = mock_response
        
        collector = DataCollector(data_config)
        assert hasattr(collector, '_token_mapping')
        assert 'RELIANCE.NS' in collector._token_mapping
        assert '^NIFTY' in collector._token_mapping
        logger.info("Token mapping test passed")
        
    @patch('SmartApi.smartConnect.SmartConnect')
    def test_historical_data_angel_one(self, mock_smart_connect, data_config, 
                                     mock_angel_response, mock_candle_data):
        """Test historical data retrieval from Angel One"""
        # Setup mock
        mock_instance = mock_smart_connect.return_value
        mock_instance.generateSession.return_value = mock_angel_response
        mock_instance.getProfile.return_value = {'status': True, 'data': {'name': 'Test User'}}
        mock_instance.getCandleData.return_value = mock_candle_data
        
        collector = DataCollector(data_config)
        data = collector.get_historical_data('RELIANCE.NS')
        
        assert isinstance(data, pd.DataFrame)
        assert not data.empty
        assert all(col.lower() in data.columns for col in ['open', 'high', 'low', 'close', 'volume'])
        logger.info("Angel One historical data test passed")
        
    @patch('yfinance.download')
    def test_historical_data_yfinance_fallback(self, mock_yf_download, data_config):
        """Test yfinance fallback for historical data"""
        # Setup mock data
        mock_data = pd.DataFrame({
            'Open': [100.0] * 30,
            'High': [101.0] * 30,
            'Low': [99.0] * 30,
            'Close': [100.5] * 30,
            'Volume': [1000] * 30
        }, index=pd.date_range(start='2025-06-09', periods=30))
        mock_yf_download.return_value = mock_data
        
        collector = DataCollector(data_config)
        data = collector.get_historical_data('RELIANCE.NS')
        
        assert isinstance(data, pd.DataFrame)
        assert not data.empty
        mock_yf_download.assert_called_once()
        logger.info("yfinance fallback test passed")
        
    def test_technical_indicators(self, data_config):
        """Test technical indicator calculation"""
        sample_data = pd.DataFrame({
            'open': [100.0] * 30,
            'high': [101.0] * 30,
            'low': [99.0] * 30,
            'close': [100.5] * 30,
            'volume': [1000] * 30
        }, index=pd.date_range(start='2025-06-09', periods=30))
        
        collector = DataCollector(data_config)
        data = collector._process_market_data(sample_data, 'RELIANCE.NS')
        
        # Check for key indicators
        for period in collector.config['data']['periods']:
            assert f'sma_{period}' in data.columns
            assert f'ema_{period}' in data.columns
        
        assert 'rsi' in data.columns
        assert not data.isnull().any().any()
        logger.info("Technical indicators test passed")
        
    @patch('requests.get')
    def test_nse_stocks_listing(self, mock_get, data_config):
        """Test NSE stocks listing retrieval"""
        # Mock response for NSE website
        mock_response = MagicMock()
        mock_response.ok = True
        mock_response.content = '''
            <table id="equityStockTable">
                <tr><th>Symbol</th></tr>
                <tr><td>RELIANCE</td></tr>
                <tr><td>TCS</td></tr>
            </table>
        '''.encode()
        mock_get.return_value = mock_response
        
        collector = DataCollector(data_config)
        stocks = collector.get_nse_stocks()
        
        assert isinstance(stocks, list)
        assert len(stocks) > 0
        assert 'RELIANCE' in stocks
        logger.info("NSE stocks listing test passed")
        
    def test_error_handling_and_fallback(self, data_config):
        """Test error handling and fallback mechanism"""
        collector = DataCollector(data_config)
        
        # Test with invalid symbol
        with pytest.raises(Exception):
            collector.get_historical_data('INVALID_SYMBOL_123')
        
        # Test fallback to test data when both Angel One and yfinance fail
        data = collector.get_historical_data('NIFTYFMCG.NS')
        assert isinstance(data, pd.DataFrame)
        assert not data.empty
        logger.info("Error handling and fallback test passed")
            
    def test_cleanup(self, data_config):
        """Test cleanup functionality"""
        collector = DataCollector(data_config)
        collector.cleanup()
        assert collector.angel_api is None
        assert not collector._initialized
        logger.info("Cleanup test passed")
    
    def test_singleton_pattern(self, mock_config):
        """Test that DataCollector implements singleton pattern"""
        collector1 = DataCollector(mock_config)
        collector2 = DataCollector(mock_config)
        assert id(collector1) == id(collector2)

    @patch('requests.get')
    def test_token_mapping_initialization(self, mock_get, collector):
        """Test token mapping initialization from instrument master"""
        # Mock instrument master response
        mock_get.return_value.ok = True
        mock_get.return_value.json.return_value = [
            {'exch_seg': 'NSE', 'symbol': 'NIFTY', 'token': '26000', 'instrumenttype': 'INDICES'},
            {'exch_seg': 'NSE', 'symbol': 'RELIANCE', 'token': '123', 'instrumenttype': 'EQ'}
        ]
        
        collector._init_token_mapping(collector.angel_api)
        
        # Verify mappings
        assert collector._token_mapping['RELIANCE'] == '123'
        assert collector._token_mapping['^NIFTY'] == '26000'

    @patch('yfinance.download')
    def test_historical_data_fetch(self, mock_yf, collector):
        """Test historical data fetching with fallback"""
        # Mock yfinance data
        mock_data = pd.DataFrame({
            'Open': [100, 101],
            'High': [102, 103],
            'Low': [98, 99],
            'Close': [101, 102],
            'Volume': [1000, 1100]
        }, index=pd.date_range('2025-01-01', '2025-01-02'))
        mock_yf.return_value = mock_data
        
        data = collector.get_historical_data('RELIANCE.NS')
        assert isinstance(data, pd.DataFrame)
        assert not data.empty
        assert 'close' in data.columns.str.lower()

    def test_technical_indicators(self, collector):
        """Test technical indicator calculation"""
        data = pd.DataFrame({
            'open': np.random.randn(100) + 100,
            'high': np.random.randn(100) + 102,
            'low': np.random.randn(100) + 98,
            'close': np.random.randn(100) + 101,
            'volume': np.random.randint(1000, 2000, 100)
        })
        
        processed = collector._process_market_data(data, 'TEST')
        assert 'rsi' in processed.columns
        assert 'sma_50' in processed.columns
        assert 'ema_20' in processed.columns

    @patch('requests.get')
    def test_nse_stocks_fetch(self, mock_get, collector):
        """Test NSE stock list fetching"""
        # Mock instrument master
        mock_get.return_value.ok = True
        mock_get.return_value.json.return_value = [
            {'exch_seg': 'NSE', 'symbol': 'RELIANCE', 'token': '123', 'instrumenttype': 'EQ'},
            {'exch_seg': 'NSE', 'symbol': 'TCS', 'token': '456', 'instrumenttype': 'EQ'}
        ]
        
        stocks = collector.get_nse_stocks()
        assert isinstance(stocks, list)
        assert 'RELIANCE' in stocks
        assert 'TCS' in stocks

    def test_market_data_collection(self, collector):
        """Test market-wide data collection"""
        with patch.object(collector, 'get_historical_data') as mock_get_data:
            mock_get_data.return_value = pd.DataFrame({
                'open': [100],
                'high': [101],
                'low': [99],
                'close': [100.5],
                'volume': [1000]
            })
            
            market_data = collector.get_market_data()
            assert 'nifty50' in market_data
            assert 'india_vix' in market_data

    def test_error_handling_and_retries(self, collector):
        """Test error handling and retry mechanism"""
        with patch('yfinance.download') as mock_yf:
            # Make first call fail, second succeed
            mock_yf.side_effect = [
                Exception("API Error"),
                pd.DataFrame({'Close': [100, 101]})
            ]
            
            data = collector.get_historical_data('RELIANCE.NS')
            assert isinstance(data, pd.DataFrame)
            assert not data.empty

    @pytest.mark.timeout(5)
    def test_timeout_handling(self, collector):
        """Test timeout handling"""
        with patch('yfinance.download') as mock_yf:
            mock_yf.side_effect = TimeoutError("Request timed out")
            
            with pytest.raises(Exception):
                collector.get_historical_data('RELIANCE.NS')

    def test_angel_one_session_renewal(self, collector):
        """Test Angel One session renewal"""
        with patch.object(collector, '_renew_session') as mock_renew:
            mock_renew.return_value = True
            
            # Simulate session expiry and renewal
            with patch.object(collector.angel_api, 'getCandleData') as mock_get_data:
                mock_get_data.side_effect = [
                    Exception("Session Expired"),
                    {'status': True, 'data': []}
                ]
                
                collector._get_angel_one_data('RELIANCE.NS', 
                                            datetime.now() - timedelta(days=10),
                                            datetime.now(),
                                            '1d')
                
                assert mock_renew.called

    def test_data_validation(self, collector):
        """Test data validation and cleaning"""
        test_data = collector._generate_test_data(
            start_date=datetime.now() - timedelta(days=10),
            end_date=datetime.now(),
            interval='1d'
        )
        
        assert isinstance(test_data, pd.DataFrame)
        assert len(test_data) > 0
        assert all(col in test_data.columns 
                  for col in ['open', 'high', 'low', 'close', 'volume'])
