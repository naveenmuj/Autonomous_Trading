def get_live_quote(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get live quote for a symbol ONLY from WebSocket data - no API fallbacks"""
        logger.info(f"[WEBSOCKET_ONLY] get_live_quote called for {symbol}")
        try:
            # Check if market is open for context
            current_time = datetime.now()
            market_open_time = current_time.replace(hour=9, minute=15, second=0, microsecond=0)
            market_close_time = current_time.replace(hour=15, minute=30, second=0, microsecond=0)
            is_market_open = market_open_time <= current_time <= market_close_time
            
            logger.info(f"[WEBSOCKET_ONLY] Market open status: {is_market_open}")
            
            # Ensure WebSocket is connected
            if not hasattr(self, 'websocket') or not self.websocket:
                logger.info(f"[WEBSOCKET_ONLY] WebSocket not available, attempting to initialize...")
                try:
                    self.ensure_websocket_connected()
                except Exception as ws_error:
                    logger.error(f"[WEBSOCKET_ONLY] WebSocket initialization failed: {ws_error}")
                    return {
                        'symbol': symbol,
                        'ltp': 0,
                        'source': 'no_websocket_data',
                        'market_status': 'open' if is_market_open else 'closed',
                        'error': 'WebSocket data not available'
                    }
            
            # ONLY use WebSocket data - check our internal live quotes cache
            if hasattr(self, '_live_quotes') and hasattr(self, '_live_quotes_lock'):
                with self._live_quotes_lock:
                    live_data = self._live_quotes.get(symbol)
                    logger.info(f"[WEBSOCKET_ONLY] WebSocket live_quotes for {symbol}: {live_data}")
                    
                    if live_data and live_data.get('ltp', 0) > 0:
                        logger.info(f"[WEBSOCKET_ONLY] Using WebSocket data for {symbol}: ₹{live_data.get('ltp')}")
                        
                        ltp = float(live_data.get('ltp', 0))
                        close_price = float(live_data.get('close', 0))
                        change = ltp - close_price if close_price > 0 else 0
                        change_percent = (change / close_price * 100) if close_price > 0 else 0
                        
                        return {
                            'symbol': symbol,
                            'ltp': ltp,
                            'change': change,
                            'change_percent': change_percent,
                            'volume': int(live_data.get('volume', 0)),
                            'timestamp': live_data.get('timestamp', datetime.now()),
                            'source': 'websocket_live' if is_market_open else 'websocket_ltp',
                            'market_status': 'open' if is_market_open else 'closed'
                        }
            
            # NO FALLBACKS - if no WebSocket data, return no data
            logger.warning(f"[WEBSOCKET_ONLY] No WebSocket data available for {symbol}")
            return {
                'symbol': symbol,
                'ltp': 0,
                'source': 'no_websocket_data',
                'market_status': 'open' if is_market_open else 'closed',
                'error': 'WebSocket data not available'
            }
            
        except Exception as e:
            logger.error(f"Error in get_live_quote for {symbol}: {str(e)}")
            return {
                'symbol': symbol,
                'ltp': 0,
                'source': 'error',
                'market_status': 'error',
                'error': str(e)
            }
