# Backup of old/duplicate get_historical_data implementation from collector.py
# This version is deprecated and should not be used. See main collector.py for the correct implementation.

def get_historical_data(self, symbol, days=10, interval='ONE_DAY'):
    from datetime import datetime, timedelta
    # --- Rate limit constants (official) ---
    HIST_API_LIMIT_SEC = 3
    HIST_API_LIMIT_MIN = 180
    HIST_API_LIMIT_HR = 5000

    def _rate_limit_wait():
        now = time.time()
        with self._hist_api_lock:
            # Clean up old timestamps
            self._hist_api_calls_sec = [t for t in self._hist_api_calls_sec if t > now - 1]
            self._hist_api_calls_min = [t for t in self._hist_api_calls_min if t > now - 60]
            self._hist_api_calls_hr = [t for t in self._hist_api_calls_hr if t > now - 3600]
            # If any limit exceeded, calculate wait time
            wait_times = []
            if len(self._hist_api_calls_sec) >= HIST_API_LIMIT_SEC:
                wait_times.append(1 - (now - self._hist_api_calls_sec[0]))
            if len(self._hist_api_calls_min) >= HIST_API_LIMIT_MIN:
                wait_times.append(60 - (now - self._hist_api_calls_min[0]))
            if len(self._hist_api_calls_hr) >= HIST_API_LIMIT_HR:
                wait_times.append(3600 - (now - self._hist_api_calls_hr[0]))
            if wait_times:
                wait_time = max(wait_times)
                logger.warning(f"[RATE LIMIT] Waiting {wait_time:.2f}s to comply with Angel One historical data API limits.")
                time.sleep(max(wait_time, 0.01))
            # Register this call
            self._hist_api_calls_sec.append(now)
            self._hist_api_calls_min.append(now)
            self._hist_api_calls_hr.append(now)

    """
    Fetch historical OHLCV data for a symbol, with candlestick pattern detection.
    This method should be called by all model/data consumers.
    """
    try:
        if self.angel_api is None:
            logger.error("Angel One API not initialized.")
            return pd.DataFrame()
        end_date = datetime.now()
        min_date = datetime(2000, 1, 1)
        if days < 1:
            logger.warning(f"Requested days < 1, adjusting to 1.")
            days = 1
        start_date = end_date - timedelta(days=days)
        if start_date < min_date:
            logger.warning(f"Requested start_date {start_date.date()} before 2000-01-01, adjusting to 2000-01-01.")
            start_date = min_date
        if end_date > datetime.now():
            logger.warning(f"Requested end_date {end_date.date()} is in the future, adjusting to today.")
            end_date = datetime.now()
        if interval == 'ONE_DAY':
            start_str = start_date.strftime('%Y-%m-%d') + ' 09:15'
            end_str = end_date.strftime('%Y-%m-%d') + ' 15:30'
        else:
            start_str = start_date.strftime('%Y-%m-%d %H:%M')
            end_str = end_date.strftime('%Y-%m-%d %H:%M')
        logger.info(f"Requesting historical data for {symbol} from {start_str} to {end_str} (interval: {interval})")
        token = self.symbol_token_map.get(symbol)
        if token is None:
            token = self.symbol_token_map.get(symbol.replace('.NS', ''))
        if token is None:
            token = self.symbol_token_map.get(symbol.replace('.NS', '-EQ'))
        if token is None:
            logger.error(f"No token found for symbol {symbol} (tried direct, base, and -EQ)")
            return pd.DataFrame()
        params = {
            "exchange": "NSE",
            "symboltoken": token,
            "interval": interval,
            "fromdate": start_str,
            "todate": end_str,
        }
        max_retries = 5
        retry_wait = self.config.get('api_rate_limit_wait', 2)
        for attempt in range(max_retries):
            _rate_limit_wait()
            response = self.angel_api.getCandleData(params)
            if response and 'data' in response:
                candles = response['data']
                df = pd.DataFrame(candles, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                # --- DIAGNOSTIC: OHLCV completeness check ---
                expected_rows = self._calculate_expected_rows(start_date, end_date, interval)
                actual_rows = len(df)
                missing_timestamps = []
                if 'timestamp' in df.columns:
                    pass
                logger.info(f"[DIAGNOSTIC] {symbol}: Expected rows={expected_rows}, Actual rows={actual_rows}, Missing timestamps={len(missing_timestamps)}")
                if missing_timestamps:
                    pass
                if actual_rows < expected_rows:
                    pass
                for col in ['open', 'high', 'low', 'close', 'volume']:
                    pass
                # Verification log: check required columns
                required_cols = ['open', 'high', 'low', 'close', 'volume']
                missing_cols = [col for col in required_cols if col not in df.columns]
                if missing_cols:
                    pass
                if df.isnull().any().any():
                    pass
                if len(df) < 60:
                    pass
                df = self.detect_candlestick_patterns(df)
                if 'pattern' not in df.columns:
                    pass
                else:
                    pass

            else:
                pass
        logger.error(f"Failed to fetch historical data for {symbol} after {max_retries} retries.")
        return pd.DataFrame()
    except Exception as e:
        logger.error(f"Error fetching historical data for {symbol}: {e}")
        return pd.DataFrame()
