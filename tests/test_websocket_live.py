import pytest
import time
import threading
from unittest.mock import MagicMock, patch
from datetime import datetime
import pytz
from src.data.websocket import MarketDataWebSocket
from .mocks.mock_smartwebsocket import MockSmartWebSocketV2

# Mock SmartWebSocketV2
@pytest.fixture(autouse=True)
def mock_smartwebsocket():
    with patch('src.data.websocket.SmartWebSocketV2', MockSmartWebSocketV2):
        yield

@pytest.fixture
def mock_websocket(mocker):
    """Create a mock WebSocket connection with patched SmartWebSocketV2"""
    mocker.patch('src.data.websocket.SmartWebSocketV2', new=MockSmartWebSocketV2)
    ws = MarketDataWebSocket(
        auth_token="test_token",
        api_key="test_api_key",
        client_code="test_client",
        feed_token="test_feed_token"
    )
    yield ws
    if ws:
        ws.close()

def test_websocket_initialization(mock_websocket):
    """Test WebSocket initialization"""
    assert mock_websocket.auth_token == "test_token"
    assert mock_websocket.api_key == "test_api_key"
    assert mock_websocket.client_code == "test_client"
    assert not mock_websocket.is_connected
    assert not mock_websocket.stopping

def test_websocket_connection(mock_websocket):
    """Test WebSocket connection process"""
    # Mock SmartWebSocketV2 methods
    mock_ws = MagicMock()
    mock_websocket.ws = mock_ws
    
    # Simulate connection
    mock_websocket._on_open(mock_ws)
    assert mock_websocket.is_connected
    assert mock_websocket.last_heartbeat is not None

def test_websocket_message_handling(mock_websocket):
    """Test WebSocket message handling"""
    test_message = {
        'token': '123',
        'last_traded_price': 10000,  # 100.00 after division
        'volume_trade_for_the_day': 1000,
        'best_5_buy_data': [{'price': 9900}],  # 99.00 after division
        'best_5_sell_data': [{'price': 10100}],  # 101.00 after division
        'open_interest': 500,
        'exchange_timestamp': int(time.time() * 1000)
    }
    
    # Add a test callback
    received_data = []
    def test_callback(data):
        received_data.append(data)
    
    mock_websocket.add_tick_callback(test_callback)
    mock_websocket.subscribed_tokens['123'] = 'TEST'
    mock_websocket.is_connected = True
    
    # Process test message
    mock_websocket._on_message(None, test_message)
    
    # Verify callback was called with correct data
    assert len(received_data) == 1
    data = received_data[0]
    assert data['symbol'] == 'TEST'
    assert data['ltp'] == 100.00
    assert data['volume'] == 1000
    assert data['bid_price'] == 99.00
    assert data['ask_price'] == 101.00

def test_websocket_cleanup(mock_websocket):
    """Test WebSocket cleanup process"""
    # Set up initial state
    mock_ws = MagicMock()
    mock_websocket.ws = mock_ws
    mock_websocket.is_connected = True
    mock_websocket.subscribed_tokens = {'123': 'TEST'}
    
    # Close connection
    mock_websocket.close()
    
    # Verify cleanup
    assert mock_websocket.stopping
    assert not mock_websocket.is_connected
    assert not mock_websocket.subscribed_tokens
    assert not mock_websocket.live_feed
    mock_ws.close.assert_called_once()

def test_websocket_reconnection(mock_websocket):
    """Test WebSocket reconnection on error"""
    # Set up initial state
    mock_websocket.is_connected = True
    mock_websocket.stopping = False
    
    # Simulate error
    mock_websocket._on_error(None, Exception("Test error"))
    
    # Verify reconnection was initiated
    assert not mock_websocket.is_connected
    
def test_websocket_heartbeat(mock_websocket):
    """Test WebSocket heartbeat monitoring"""
    mock_ws = MagicMock()
    mock_ws._ws_app = MagicMock()
    mock_websocket.ws = mock_ws
    mock_websocket.is_connected = True
    
    # Start heartbeat monitor
    mock_websocket._start_heartbeat_monitor()
    
    # Let it run briefly
    time.sleep(0.1)
    
    # Verify heartbeat was sent
    mock_ws._ws_app.send.assert_called_with('ping')

def test_websocket_heartbeat_timeout(mock_websocket):
    """Test WebSocket heartbeat timeout handling"""
    mock_websocket.is_connected = True
    mock_websocket.ws._ws_app.recv = MagicMock(return_value=None)  # Simulate no pong response
    
    # Start heartbeat monitor
    mock_websocket._start_heartbeat_monitor()
    time.sleep(0.2)  # Wait for heartbeat cycle
    
    # Should trigger reconnection due to timeout
    assert not mock_websocket.is_connected

def test_websocket_error_handling(mock_websocket):
    """Test WebSocket error handling"""
    mock_websocket.ws.on_error(None, Exception("Test error"))
    time.sleep(0.1)  # Let async reconnection start
    assert not mock_websocket.is_connected
    assert mock_websocket.current_reconnect_attempt > 0

def test_websocket_subscription(mock_websocket):
    """Test WebSocket subscription handling"""
    mock_ws = MagicMock()
    mock_websocket.ws = mock_ws
    mock_websocket.is_connected = True
    
    # Subscribe to test token
    mock_websocket.subscribe({'NSE': ['123']})
    
    # Verify subscription
    mock_ws.subscribe.assert_called_once()
    assert '123' in mock_websocket.subscribed_tokens

def test_websocket_unsubscription(mock_websocket):
    """Test WebSocket unsubscription handling"""
    mock_ws = MagicMock()
    mock_websocket.ws = mock_ws
    mock_websocket.is_connected = True
    mock_websocket.subscribed_tokens = {'123': 'TEST'}
    
    # Unsubscribe from test token
    mock_websocket.unsubscribe({'NSE': ['123']})
    
    # Verify unsubscription
    mock_ws.unsubscribe.assert_called_once()
    assert '123' not in mock_websocket.subscribed_tokens
