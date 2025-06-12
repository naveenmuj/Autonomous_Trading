import os
import sys
import json
import pytest
import logging
import threading
import time
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime
from src.data.websocket import MarketDataWebSocket

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
                'mode': 2,  # QUOTE mode
                'max_retries': 3,
                'retry_delay': 1
            }
        }
    }

@pytest.fixture
def mock_websocket():
    websocket = MarketDataWebSocket(
        auth_token='test_auth_token',
        api_key='test_api_key',
        client_code='test_client',
        feed_token='test_feed_token',
        config=mock_config()
    )
    return websocket

def test_websocket_initialization(mock_websocket):
    """Test WebSocket initialization"""
    assert mock_websocket.auth_token == 'test_auth_token'
    assert mock_websocket.api_key == 'test_api_key'
    assert mock_websocket.client_code == 'test_client'
    assert mock_websocket.feed_token == 'test_feed_token'
    assert not mock_websocket.is_connected
    assert isinstance(mock_websocket.subscribed_tokens, dict)
    assert isinstance(mock_websocket.live_feed, dict)

@patch('src.data.websocket.SmartWebSocketV2')
def test_websocket_connect(mock_smartwebsocket, mock_websocket):
    """Test WebSocket connection"""
    # Setup mock
    mock_ws = MagicMock()
    mock_smartwebsocket.return_value = mock_ws
    
    # Connect
    mock_websocket.connect()
    
    # Verify
    assert mock_smartwebsocket.called
    assert mock_ws.on_open == mock_websocket._on_open
    assert mock_ws.on_message == mock_websocket._on_message
    assert mock_ws.on_error == mock_websocket._on_error
    assert mock_ws.on_close == mock_websocket._on_close

def test_websocket_callbacks(mock_websocket):
    """Test WebSocket callbacks"""
    # Test on_open
    mock_websocket._on_open()
    assert mock_websocket.is_connected
    assert mock_websocket.reconnect_count == 0
    
    # Test on_error
    mock_websocket._on_error('ws', 'test error')
    assert not mock_websocket.is_connected
    
    # Test on_close
    mock_websocket._on_close('ws', 1000, 'normal closure')
    assert not mock_websocket.is_connected

def test_websocket_message_handling(mock_websocket):
    """Test WebSocket message handling"""
    # Setup test data
    test_message = {
        'token': '123456',
        'last_traded_price': 10000,  # 100.00 after division
        'volume_trade_for_the_day': 1000,
        'best_5_buy_data': [{'price': 9900}],  # 99.00 after division
        'best_5_sell_data': [{'price': 10100}],  # 101.00 after division
        'open_interest': 500,
        'exchange_timestamp': int(time.time() * 1000)
    }
    
    # Setup mock callback
    mock_callback = Mock()
    mock_websocket.add_tick_callback(mock_callback)
    
    # Process message
    mock_websocket._on_message('ws', test_message)
    
    # Verify live feed update
    assert '123456' in mock_websocket.live_feed
    tick_data = mock_websocket.live_feed['123456']
    assert tick_data['ltp'] == 100.00
    assert tick_data['volume'] == 1000
    assert tick_data['bid_price'] == 99.00
    assert tick_data['ask_price'] == 101.00
    assert tick_data['oi'] == 500
    
    # Verify callback was called
    assert mock_callback.called
    
def test_websocket_subscription(mock_websocket):
    """Test WebSocket subscription"""
    with patch.object(mock_websocket, 'ws') as mock_ws:
        # Setup
        mock_websocket.is_connected = True
        test_tokens = ['123456', '789012']
        
        # Subscribe
        mock_websocket.subscribe(test_tokens)
        
        # Verify
        assert mock_ws.subscribe.called
        call_args = mock_ws.subscribe.call_args[1]
        assert 'token_list' in call_args
        assert len(call_args['token_list']) == 1
        assert call_args['token_list'][0]['tokens'] == test_tokens
        
        # Verify token storage
        for token in test_tokens:
            assert token in mock_websocket.subscribed_tokens

def test_websocket_unsubscription(mock_websocket):
    """Test WebSocket unsubscription"""
    with patch.object(mock_websocket, 'ws') as mock_ws:
        # Setup
        mock_websocket.is_connected = True
        test_tokens = ['123456', '789012']
        for token in test_tokens:
            mock_websocket.subscribed_tokens[token] = token
            mock_websocket.live_feed[token] = {'some': 'data'}
        
        # Unsubscribe
        mock_websocket.unsubscribe(test_tokens)
        
        # Verify
        assert mock_ws.unsubscribe.called
        call_args = mock_ws.unsubscribe.call_args[1]
        assert 'token_list' in call_args
        
        # Verify token removal
        for token in test_tokens:
            assert token not in mock_websocket.subscribed_tokens
            assert token not in mock_websocket.live_feed

def test_websocket_cleanup(mock_websocket):
    """Test WebSocket cleanup"""
    # Setup
    mock_websocket.ws = Mock()
    mock_websocket.is_connected = True
    mock_websocket.subscribed_tokens = {'123': 'TEST'}
    mock_websocket.live_feed = {'123': {'data': 'test'}}
    
    # Close
    mock_websocket.close()
    
    # Verify
    assert mock_websocket.ws is None
    assert not mock_websocket.is_connected
    assert len(mock_websocket.subscribed_tokens) == 0
    assert len(mock_websocket.live_feed) == 0
