from unittest.mock import MagicMock

class MockSmartWebSocketV2:
    def __init__(self, auth_token=None, api_key=None, client_code=None, feed_token=None, **kwargs):
        self.auth_token = auth_token
        self.api_key = api_key
        self.client_code = client_code
        self.feed_token = feed_token
        self.on_open = None
        self.on_data = None
        self.on_error = None
        self.on_close = None
        self._ws_app = MagicMock()
        self._ws_app.send = MagicMock(return_value=None)
        self._ws_app.recv = MagicMock(return_value="pong")
        
    def connect(self):
        """Simulate successful connection"""
        if self.on_open:
            self.on_open(self)
        
    def close(self):
        """Simulate graceful closure"""
        if self.on_close:
            self.on_close(self, 1000, "Normal closure")
            
    def subscribe(self, correlation_id=None, mode=None, token_list=None, **kwargs):
        """Simulate successful subscription"""
        if self.on_data:
            self.on_data(self, {
                'type': 'success',
                'message': 'subscribed',
                'correlation_id': correlation_id
            })
        return True
        
    def unsubscribe(self, correlation_id=None, mode=None, token_list=None, **kwargs):
        """Simulate successful unsubscription"""
        if self.on_data:
            self.on_data(self, {
                'type': 'success',
                'message': 'unsubscribed',
                'correlation_id': correlation_id
            })
        return True
