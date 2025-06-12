import os
import sys
import json
import pytest
import logging
from unittest.mock import Mock, patch
from datetime import datetime
from src.data.websocket import MarketDataWebSocket
from src.data.collector_new import DataCollector

logger = logging.getLogger(__name__)

@pytest.fixture
def mock_config():
    return {
        'apis': {
            'angel_one': {
                'api_key': 'test_api_key',
                'client_id': 'test_client',
                'mpin': 'test_pin',
                'totp_secret': 'test_totp'
            }
        },
        'data': {
            'websocket': {
                'max_retries': 3,
                'retry_delay': 1,
                'retry_multiplier': 2,
                'retry_duration': 5,
                'mode': 'QUOTE',
                'auto_reconnect': True
            }
        }
    }

@pytest.fixture
def mock_angel_api():
    with patch('src.data.collector_new.SmartConnect') as mock_api:
        mock_instance = Mock()
        mock_instance.auth_token = 'test_auth_token'
        mock_instance.feed_token = 'test_feed_token'
        mock_instance.generateSession.return_value = {
            'status': True,
            'data': {'refreshToken': 'test_refresh'}
        }
        mock_instance.getProfile.return_value = {
            'status': True,
            'data': {'name': 'Test User'}
        }
        mock_api.return_value = mock_instance
        yield mock_instance

def test_websocket_initialization():
    """Test WebSocket initialization with valid credentials"""
    ws = MarketDataWebSocket(
        auth_token='test_auth',
        api_key='test_api',
        client_code='test_client',
        feed_token='test_feed'
    )
    assert ws is not None
    assert ws.auth_token == 'test_auth'
    assert ws.api_key == 'test_api'
    assert ws.client_code == 'test_client'
    assert ws.feed_token == 'test_feed'

@patch('src.data.websocket.SmartWebSocketV2')
def test_websocket_connection(mock_websocket):
    """Test WebSocket connection and subscription"""
    # Setup mock
    mock_instance = Mock()
    mock_websocket.return_value = mock_instance
    
    # Create WebSocket instance
    ws = MarketDataWebSocket(
        auth_token='test_auth',
        api_key='test_api',
        client_code='test_client',
        feed_token='test_feed'
    )
    
    # Test connection
    ws.connect()
    mock_websocket.assert_called_once()
    
    # Test subscription
    tokens = ['123', '456']
    ws.subscribe(tokens)
    mock_instance.subscribe.assert_called_once()
    
    # Verify callbacks are set
    assert mock_instance.on_open is not None
    assert mock_instance.on_error is not None
    assert mock_instance.on_close is not None
    assert mock_instance.on_message is not None

def test_datacollector_websocket_integration(mock_config, mock_angel_api):
    """Test DataCollector integration with WebSocket"""
    collector = DataCollector(mock_config)
    assert collector.websocket is not None
    assert collector.angel_api is not None
    
    # Test live quote retrieval
    mock_tick = {
        'token': '12345',
        'ltp': 100.50,
        'volume': 1000,
        'bid_price': 100.40,
        'ask_price': 100.60,
        'timestamp': datetime.now()
    }
    collector.websocket.live_feed['12345'] = mock_tick
    
    quote = collector.get_live_quote('RELIANCE.NS')
    assert quote is not None
    assert 'symbol' in quote
    assert quote['symbol'] == 'RELIANCE.NS'

def test_websocket_error_handling():
    """Test WebSocket error handling and reconnection"""
    ws = MarketDataWebSocket(
        auth_token='test_auth',
        api_key='test_api',
        client_code='test_client',
        feed_token='test_feed'
    )
    
    # Test error callback
    ws._on_error("Test error")
    assert not ws.is_connected
    
    # Test close callback
    ws._on_close()
    assert not ws.is_connected
    
    # Test successful connection
    ws._on_open()
    assert ws.is_connected

def test_websocket_message_handling():
    """Test WebSocket message processing"""
    ws = MarketDataWebSocket(
        auth_token='test_auth',
        api_key='test_api',
        client_code='test_client',
        feed_token='test_feed'
    )
    
    # Mock callback
    mock_callback = Mock()
    ws.add_tick_callback(mock_callback)
    
    # Test message handling
    test_message = {
        'token': '12345',
        'last_traded_price': 10050,  # In paise
        'volume_trade_for_the_day': 1000,
        'best_5_buy_data': [{'price': 10040}],
        'best_5_sell_data': [{'price': 10060}],
        'exchange_timestamp': int(datetime.now().timestamp() * 1000)
    }
    
    ws._on_message(test_message)
    mock_callback.assert_called_once()
    
    # Verify processed data
    processed_tick = ws.get_live_feed('12345')
    assert processed_tick is not None
    assert processed_tick['ltp'] == 100.50  # Converted to rupees
    assert processed_tick['volume'] == 1000
    assert processed_tick['bid_price'] == 100.40
    assert processed_tick['ask_price'] == 100.60
