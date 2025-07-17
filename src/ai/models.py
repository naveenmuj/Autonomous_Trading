import logging
import numpy as np
import pandas as pd
import warnings
# Suppress pandas_ta deprecation warning for now
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import pandas_ta as ta  # TODO: Migrate to pandas_ta v1.x or alternative in the future
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout, BatchNormalization
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from typing import Optional, List, Dict, Any, Tuple, Union
import os
from datetime import datetime, timedelta
import threading
import time
import talib
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import tensorflow as tf
import gymnasium as gym
from gymnasium import spaces
from sklearn.utils import resample

logger = logging.getLogger(__name__)

try:
    from sklearn.linear_model import LogisticRegression
except ImportError as e:
    logger.error("scikit-learn is not installed or not found in the environment. Please install scikit-learn to use LogisticRegression. Error: %s", str(e))
    LogisticRegression = None

class TechnicalAnalysisModel:
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.model = None
        self.scaler = StandardScaler()
        
        # Get technical indicator parameters with defaults
        tech_params = self.config.get('model', {}).get('features', {}).get('technical_indicators', {})
        self.rsi_period = tech_params.get('rsi_period', 14)
        self.macd_fast = tech_params.get('macd_fast', 12)
        self.macd_slow = tech_params.get('macd_slow', 26)
        self.macd_signal = tech_params.get('macd_signal', 9)
        self.bb_period = tech_params.get('bb_period', 20)
        self.bb_std = tech_params.get('bb_std', 2)
        self.support_resistance_periods = [20, 50, 200]  # Periods for SR levels
        self.trend_detection_periods = [10, 20, 50]  # Periods for trend detection
        
        # Initialize storage for support/resistance levels
        self.support_levels = {}
        self.resistance_levels = {}
        self.trends = {}
        
        logger.debug("TechnicalAnalysisModel initialized with config: %s", tech_params)

    def calculate_support_resistance(self, data: pd.DataFrame) -> Tuple[Dict[str, float], Dict[str, float]]:
        """Calculate support and resistance levels using multiple methods"""
        try:
            df = data.copy()
            supports = {}
            resistances = {}
            
            # Method 1: Moving Average based SR levels
            for period in self.support_resistance_periods:
                ma = talib.SMA(df['close'].values, timeperiod=period)
                supports[f'ma_{period}'] = ma[-1] * 0.98  # 2% below MA
                resistances[f'ma_{period}'] = ma[-1] * 1.02  # 2% above MA
            
            # Method 2: Pivot Points
            high, low, close = df['high'].iloc[-1], df['low'].iloc[-1], df['close'].iloc[-1]
            pivot = (high + low + close) / 3
            supports['pivot_s1'] = (2 * pivot) - high
            supports['pivot_s2'] = pivot - (high - low)
            resistances['pivot_r1'] = (2 * pivot) - low
            resistances['pivot_r2'] = pivot + (high - low)
            
            # Method 3: Recent swing highs/lows
            window = 20
            df['swing_high'] = df['high'].rolling(window=window, center=True).max()
            df['swing_low'] = df['low'].rolling(window=window, center=True).min()
            
            supports['swing'] = df['swing_low'].iloc[-1]
            resistances['swing'] = df['swing_high'].iloc[-1]
            
            logger.debug("Calculated support levels: %s", supports)
            logger.debug("Calculated resistance levels: %s", resistances)
            
            return supports, resistances
            
        except Exception as e:
            logger.error("Error calculating support/resistance: %s", str(e))
            import traceback
            logger.error(traceback.format_exc())
            return {}, {}

    def detect_trends(self, data: pd.DataFrame) -> Dict[str, str]:
        """Detect market trends using multiple timeframes"""
        try:
            df = data.copy()
            trends = {}
            
            for period in self.trend_detection_periods:
                # Calculate EMAs for trend detection
                ema = talib.EMA(df['close'].values, timeperiod=period)
                current_ema = ema[-1]
                prev_ema = ema[-2]
                
                # Calculate slope of EMA
                slope = (current_ema - prev_ema) / prev_ema * 100
                
                # Determine trend strength and direction
                if slope > 1.0:
                    trends[f'trend_{period}'] = 'strong_uptrend'
                elif slope > 0.2:
                    trends[f'trend_{period}'] = 'uptrend'
                elif slope < -1.0:
                    trends[f'trend_{period}'] = 'strong_downtrend'
                elif slope < -0.2:
                    trends[f'trend_{period}'] = 'downtrend'
                else:
                    trends[f'trend_{period}'] = 'sideways'
            
            logger.debug("Detected trends: %s", trends)
            return trends
            
        except Exception as e:
            logger.error("Error detecting trends: %s", str(e))
            import traceback
            logger.error(traceback.format_exc())
            return {}

    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate technical indicators with enhanced trend and SR detection"""
        try:
            df = data.copy()
            
            # Basic indicators
            df['rsi'] = talib.RSI(df['close'].values, timeperiod=self.rsi_period)
            macd, signal, hist = talib.MACD(df['close'].values, 
                                          fastperiod=self.macd_fast,
                                          slowperiod=self.macd_slow,
                                          signalperiod=self.macd_signal)
            df['macd'] = macd
            df['macd_signal'] = signal
            df['macd_hist'] = hist
            
            # Bollinger Bands
            upper, middle, lower = talib.BBANDS(df['close'].values,
                                              timeperiod=self.bb_period,
                                              nbdevup=self.bb_std,
                                              nbdevdn=self.bb_std)
            df['bb_upper'] = upper
            df['bb_middle'] = middle
            df['bb_lower'] = lower
            
            # Enhanced indicators
            df['atr'] = talib.ATR(df['high'], df['low'], df['close'], timeperiod=14)
            df['adx'] = talib.ADX(df['high'], df['low'], df['close'], timeperiod=14)
            df['obv'] = talib.OBV(df['close'], df['volume'])
            
            # Trend indicators
            for period in self.trend_detection_periods:
                df[f'sma_{period}'] = talib.SMA(df['close'], timeperiod=period)
                df[f'ema_{period}'] = talib.EMA(df['close'], timeperiod=period)
            
            # Calculate support/resistance and trends
            self.support_levels, self.resistance_levels = self.calculate_support_resistance(df)
            self.trends = self.detect_trends(df)
            
            # Add SR and trend info to dataframe
            for level, value in self.support_levels.items():
                df[f'support_{level}'] = value
            for level, value in self.resistance_levels.items():
                df[f'resistance_{level}'] = value
            for period, trend in self.trends.items():
                df[f'trend_{period}'] = trend
            
            logger.debug("Calculated indicators for data shape: %s", df.shape)
            # Log last row of indicators for transparency
            last_row = df.iloc[-1]
            logger.info(f"Technical indicators for last row: RSI={last_row.get('rsi')}, MACD={last_row.get('macd')}, BB_upper={last_row.get('bb_upper')}, ATR={last_row.get('atr')}, ADX={last_row.get('adx')}, OBV={last_row.get('obv')}")
            return df
            
        except Exception as e:
            logger.error("Error calculating indicators: %s", str(e))
            import traceback
            logger.error(traceback.format_exc())
            return data

    def build_model(self, input_shape, model_type='lstm'):
        """Build and compile the model with improved architecture"""
        if isinstance(input_shape, tuple):
            timesteps, features = input_shape
            # Create a more sophisticated LSTM architecture
            model = Sequential([
                tf.keras.layers.Input(shape=(timesteps, features)),
                tf.keras.layers.LSTM(256, return_sequences=True, 
                                   kernel_constraint=tf.keras.constraints.MaxNorm(3),
                                   recurrent_constraint=tf.keras.constraints.MaxNorm(3),
                                   kernel_regularizer=tf.keras.regularizers.l2(0.01),
                                   recurrent_regularizer=tf.keras.regularizers.l2(0.01)),
                tf.keras.layers.LayerNormalization(),
                tf.keras.layers.Dropout(0.4),
                tf.keras.layers.LSTM(128, return_sequences=True,
                                   kernel_constraint=tf.keras.constraints.MaxNorm(3),
                                   recurrent_constraint=tf.keras.constraints.MaxNorm(3),
                                   kernel_regularizer=tf.keras.regularizers.l2(0.01),
                                   recurrent_regularizer=tf.keras.regularizers.l2(0.01)),
                tf.keras.layers.LayerNormalization(),
                tf.keras.layers.Dropout(0.3),
                tf.keras.layers.LSTM(64, return_sequences=False,
                                   kernel_constraint=tf.keras.constraints.MaxNorm(3),
                                   recurrent_constraint=tf.keras.constraints.MaxNorm(3),
                                   kernel_regularizer=tf.keras.regularizers.l2(0.01),
                                   recurrent_regularizer=tf.keras.regularizers.l2(0.01)),
                tf.keras.layers.LayerNormalization(),
                tf.keras.layers.Dropout(0.2),
                tf.keras.layers.Dense(32, activation='relu'),
                tf.keras.layers.LayerNormalization(),
                tf.keras.layers.Dropout(0.1),
                tf.keras.layers.Dense(1, activation='sigmoid')
            ])
            optimizer = tf.keras.optimizers.Adam(
                learning_rate=1e-3,
                clipnorm=1.0,
                clipvalue=0.5
            )
            model.compile(
                optimizer=optimizer,
                loss='binary_crossentropy',
                metrics=[
                    'accuracy',
                    tf.keras.metrics.AUC(name='auc'),
                    tf.keras.metrics.Precision(name='precision'),
                    tf.keras.metrics.Recall(name='recall')
                ]
            )
            self.model = model
            return model
        else:
            features = input_shape
            model = Sequential([
                Dense(64, activation='relu', input_dim=features),
                BatchNormalization(),
                Dropout(0.2),
                Dense(32, activation='relu'),
                BatchNormalization(),
                Dropout(0.2),
                Dense(16, activation='relu'),
                BatchNormalization(),
                Dense(1, activation='sigmoid')
            ])
            optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
            model.compile(
                optimizer=optimizer,
                loss='binary_crossentropy',
                metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
            )
            self.model = model
            return model
    
    def prepare_data(self, data: pd.DataFrame, lookback: int = 60, balance_labels: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare features and labels with improved preprocessing and optional label balancing"""
        try:
            # Calculate technical indicators
            processed_data = self.calculate_indicators(data)
            
            # Add price-based features
            processed_data['returns'] = processed_data['close'].pct_change()
            processed_data['volatility'] = processed_data['returns'].rolling(window=20).std()
            processed_data['log_returns'] = np.log1p(processed_data['returns'])
            
            # Volume indicators
            processed_data['volume_ma'] = processed_data['volume'].rolling(window=20).mean()
            processed_data['volume_std'] = processed_data['volume'].rolling(window=20).std()
            processed_data['volume_ratio'] = processed_data['volume'] / processed_data['volume_ma']
            
            # Price momentum
            for period in [5, 10, 20]:
                processed_data[f'momentum_{period}'] = processed_data['close'].pct_change(period)
            
            # Define feature columns
            feature_cols = [
                'rsi', 'macd', 'macd_signal', 'macd_hist',
                'bb_upper', 'bb_middle', 'bb_lower',
                'sma_50', 'atr', 'obv', 'adx',
                'returns', 'volatility', 'log_returns',
                'volume_ma', 'volume_std', 'volume_ratio',
                'momentum_5', 'momentum_10', 'momentum_20'
            ]
            
            # Drop initial rows to ensure all indicators are valid (like TradingView)
            # Find max lookback period used by any indicator
            max_lookback = max(50, self.macd_slow, self.bb_period, 20)  # SMA_50, MACD slow, BB, ATR/ADX
            processed_data = processed_data.iloc[max_lookback:].copy()
            logger.info(f"[prepare_data] Dropped first {max_lookback} rows to ensure valid indicators for all features.")
            processed_data = processed_data.fillna(method='ffill').fillna(method='bfill')
            before_drop = len(processed_data)
            processed_data = processed_data.dropna(subset=feature_cols + ['close']).copy()
            after_drop = len(processed_data)
            if after_drop < before_drop:
                logger.warning(f"[prepare_data] Dropped {before_drop - after_drop} rows with NaN/Inf in features for symbol(s): {data.get('symbol', 'unknown')}")
            # PATCH: If no samples left after dropping NaN/Inf, log and diagnose
            if processed_data.empty:
                logger.error(f"[prepare_data] No valid samples left after dropping NaN/Inf for features: {feature_cols}. Possible causes: API returned incomplete data, missing columns, or all data is invalid. Symbol(s): {data.get('symbol', 'unknown')}")
                logger.error(f"[prepare_data] Original input shape: {data.shape}, columns: {list(data.columns)}")
                logger.error(f"[prepare_data] Data head before drop:\n{data.head()}\nData tail:\n{data.tail()}")
                # Distinguish between real data error and pipeline issue
                if data.empty:
                    logger.error("[prepare_data] Input data is empty. This is a real data error (API or source returned no data). Failing function.")
                else:
                    logger.error("[prepare_data] Input data was present but all rows dropped. This may be a pipeline/feature engineering issue. Failing function.")
                return None, None
            
            # Now create features and labels
            X = processed_data[feature_cols].values
            y = (processed_data['close'].shift(-1) > processed_data['close']).astype(int).values

            # Remove last row from both X and y to keep them aligned (since y uses shift(-1))
            X = X[:-1]
            y = y[:-1]

            # Scale features
            X = self.scaler.fit_transform(X)

            # Final check: drop any rows where X or y is NaN/Inf (should not happen, but for safety)
            mask = ~(
                np.isnan(X).any(axis=1) | np.isinf(X).any(axis=1) |
                np.isnan(y) | np.isinf(y)
            )
            X = X[mask]
            y = y[mask]

            # Label balancing (optional)
            if balance_labels:
                # Downsample majority class
                class0 = np.where(y == 0)[0]
                class1 = np.where(y == 1)[0]
                if len(class0) > 0 and len(class1) > 0:
                    min_len = min(len(class0), len(class1))
                    idx0 = resample(class0, replace=False, n_samples=min_len, random_state=42)
                    idx1 = resample(class1, replace=False, n_samples=min_len, random_state=42)
                    idx = np.concatenate([idx0, idx1])
                    X = X[idx]
                    y = y[idx]
                    logger.info(f"[prepare_data] Label balancing applied: {np.bincount(y)}")
            
            # Log label distribution
            logger.info(f"[prepare_data] Final label distribution: {np.bincount(y) if y.size > 0 else y}")
            logger.info(f"[prepare_data] Final feature shape: {X.shape}")
            return X, y
        except Exception as e:
            logger.error(f"Error in prepare_data: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            raise

    def train_model(self, X: np.ndarray, y: np.ndarray, 
                validation_data: Optional[Tuple[np.ndarray, np.ndarray]] = None,
                epochs: int = 100, 
                batch_size: int = 32,
                class_weights: Optional[Dict[int, float]] = None,
                callbacks: Optional[List[tf.keras.callbacks.Callback]] = None) -> Dict:
        """Train the model with improved training process"""
        if self.model is None:
            raise ValueError("Model not built yet. Call build_model first.")
        # PATCH: Skip training if no valid data
        if X is None or y is None:
            logger.error("[train_model] Skipping model training: No valid data available for this symbol/model.")
            return {}

        # Default callbacks if none provided
        if callbacks is None:
            # Early stopping callback
            early_stopping = tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True,
                min_delta=0.001
            )
            
            # Model checkpoint callback
            checkpoint = tf.keras.callbacks.ModelCheckpoint(
                'models/best_model.h5',
                monitor='val_loss',
                save_best_only=True,
                verbose=0
            )
            
            # Reduce learning rate on plateau
            reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.2,
                patience=5,
                min_lr=1e-6,
                verbose=1
            )
            
            callbacks = [early_stopping, checkpoint, reduce_lr]
            
        # Training with validation split or validation data and callbacks
        if validation_data is None:
            history = self.model.fit(
                X, y,
                epochs=epochs,
                batch_size=batch_size,
                validation_split=0.2,
                shuffle=True,
                class_weight=class_weights,
                callbacks=callbacks,
                verbose=1
            )
        else:
            history = self.model.fit(
                X, y,
                epochs=epochs,
                batch_size=batch_size,
                validation_data=validation_data,
                shuffle=True,
                class_weight=class_weights,
                callbacks=callbacks,
                verbose=1
            )
        
        return history.history

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions"""
        if self.model is None:
            raise ValueError("Model not trained yet")
        preds = self.model.predict(X)
        logger.info(f"AI model predictions: {preds}")
        return preds

    def evaluate_model(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """Evaluate model performance (robust to continuous or binary targets)"""
        if self.model is None:
            raise ValueError("Model not trained yet")
        y_pred = (self.predict(X) > 0.5).astype(int)
        # Ensure y is binary for metrics
        y_bin = (y > 0.5).astype(int) if y.dtype != int or set(np.unique(y)) - {0, 1} else y
        return {
            'accuracy': accuracy_score(y_bin, y_pred),
            'precision': precision_score(y_bin, y_pred),
            'recall': recall_score(y_bin, y_pred),
            'f1': f1_score(y_bin, y_pred)
        }

    def analyze(self, data: pd.DataFrame, indicators: List[str]) -> Dict[str, Any]:
        """Analyze data with selected technical indicators"""
        try:
            logger.info(f"Analyzing data with indicators: {indicators}")
            results = {}
            
            # Calculate requested indicators
            if "RSI" in indicators:
                rsi = talib.RSI(data['close'].values, timeperiod=self.rsi_period)
                results['RSI'] = {
                    'value': rsi[-1] if not np.isnan(rsi[-1]) else 50,
                    'signal': 'Oversold' if rsi[-1] < 30 else 'Overbought' if rsi[-1] > 70 else 'Neutral'
                }
            
            if "MACD" in indicators:
                macd, signal, hist = talib.MACD(
                    data['close'].values,
                    fastperiod=self.macd_fast,
                    slowperiod=self.macd_slow,
                    signalperiod=self.macd_signal
                )
                results['MACD'] = {
                    'value': macd[-1] if not np.isnan(macd[-1]) else 0,
                    'signal': signal[-1] if not np.isnan(signal[-1]) else 0,
                    'histogram': hist[-1] if not np.isnan(hist[-1]) else 0
                }
            
            if "SMA" in indicators:
                for period in [20, 50, 200]:
                    sma = talib.SMA(data['close'].values, timeperiod=period)
                    results[f'SMA_{period}'] = {
                        'value': sma[-1] if not np.isnan(sma[-1]) else data['close'].iloc[-1],
                        'signal': 'Above' if data['close'].iloc[-1] > sma[-1] else 'Below'
                    }
            
            if "Bollinger Bands" in indicators:
                upper, middle, lower = talib.BBANDS(
                    data['close'].values,
                    timeperiod=self.bb_period,
                    nbdevup=self.bb_std,
                    nbdevdn=self.bb_std
                )
                results['BB'] = {
                    'upper': upper[-1] if not np.isnan(upper[-1]) else data['close'].iloc[-1],
                    'middle': middle[-1] if not np.isnan(middle[-1]) else data['close'].iloc[-1],
                    'lower': lower[-1] if not np.isnan(lower[-1]) else data['close'].iloc[-1]
                }
            
            logger.info("Technical analysis completed successfully")
            return results
            
        except Exception as e:
            logger.error(f"Error in technical analysis: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return {}

    def analyze(self, data: pd.DataFrame, indicator_config: Dict = None) -> Dict[str, Any]:
        """Analyze price data with specified indicators
        
        Args:
            data: DataFrame with OHLCV data
            indicator_config: Dictionary specifying which indicators to calculate
            
        Returns:
            Dictionary containing calculated indicators and analysis results
        """
        try:
            if data is None or len(data) == 0:
                logger.warning("No data provided for analysis")
                return {}
                
            df = data.copy()
            results = {}
            
            # Calculate requested indicators
            for category, indicators in indicator_config.items():
                results[category] = {}
                
                if category == 'trend_indicators':
                    if indicators.get('sma'):
                        for period in indicators['sma']:
                            df[f'sma_{period}'] = talib.SMA(df['close'], timeperiod=period)
                            results[category][f'sma_{period}'] = df[f'sma_{period}'].values
                            
                    if indicators.get('ema'):
                        for period in indicators['ema']:
                            df[f'ema_{period}'] = talib.EMA(df['close'], timeperiod=period)
                            results[category][f'ema_{period}'] = df[f'ema_{period}'].values
                            
                    if indicators.get('macd'):
                        macd_config = indicators['macd']
                        macd, signal, hist = talib.MACD(
                            df['close'],
                            fastperiod=macd_config['fast'],
                            slowperiod=macd_config['slow'],
                            signalperiod=macd_config['signal']
                        )
                        results[category]['macd'] = {
                            'macd': macd,
                            'signal': signal,
                            'histogram': hist
                        }
                        
                elif category == 'momentum_indicators':
                    if indicators.get('rsi'):
                        period = indicators['rsi']['period']
                        df['rsi'] = talib.RSI(df['close'], timeperiod=period)
                        results[category]['rsi'] = df['rsi'].values
                        
                elif category == 'volatility_indicators':
                    if indicators.get('bollinger_bands'):
                        bb_config = indicators['bollinger_bands']
                        upper, middle, lower = talib.BBANDS(
                            df['close'],
                            timeperiod=bb_config['period'],
                            nbdevup=bb_config['std_dev'],
                            nbdevdn=bb_config['std_dev']
                        )
                        results[category]['bollinger_bands'] = {
                            'upper': upper,
                            'middle': middle,
                            'lower': lower
                        }
                        
                elif category == 'volume_indicators':
                    if indicators.get('volume_sma'):
                        period = indicators['volume_sma']
                        df['volume_sma'] = talib.SMA(df['volume'], timeperiod=period)
                        results[category]['volume_sma'] = df['volume_sma'].values
            
            # Add basic price data
            results['price_data'] = {
                'open': df['open'].values,
                'high': df['high'].values,
                'low': df['low'].values,
                'close': df['close'].values,
                'volume': df['volume'].values,
                'timestamp': df['timestamp'].values
            }
            
            logger.info(f"Analysis complete with {len(results)} indicator categories")
            return results
            
        except Exception as e:
            logger.error(f"Error in technical analysis: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return {}

    def get_symbols(self, data_collector=None):
        """Return the list of symbols to use for training/prediction, respecting config (auto/manual)."""
        # Prefer DataCollector if provided
        if data_collector and hasattr(data_collector, 'get_symbols_from_config'):
            try:
                return data_collector.get_symbols_from_config()
            except Exception as e:
                logger.error(f"Error getting symbols from data_collector: {e}")
        # Fallback to config
        trading_config = self.config.get('trading', {})
        data_config = trading_config.get('data', {})
        if data_config.get('mode') == 'manual':
            return data_config.get('manual_symbols', [])
        elif data_config.get('mode') == 'auto':
            # If auto mode but collector not available, fallback to manual_symbols or default
            return data_config.get('manual_symbols', ['RELIANCE.NS', 'TCS.NS', 'HDFCBANK.NS', 'INFY.NS', 'ICICIBANK.NS'])
        return trading_config.get('symbols', ['RELIANCE.NS', 'TCS.NS', 'HDFCBANK.NS', 'INFY.NS', 'ICICIBANK.NS'])

class SentimentAnalyzer:
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        # News sentiment analysis is disabled (NewsAPI removed)
        # This class is now a placeholder

class AITrader:
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.technical_model = TechnicalAnalysisModel(config)
        self.sentiment_analyzer = SentimentAnalyzer(config)
        self.model = None
        self.scaler = StandardScaler()
        
    def prepare_features(self, data: pd.DataFrame, balance_labels: bool = False, simple_features: bool = False) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Prepare features for training/prediction, robust to missing sentiment or technical columns, with label balancing option"""
        try:
            # Calculate technical indicators
            processed_data = self.technical_model.calculate_indicators(data)

            # Simple feature option for debugging
            if simple_features:
                feature_cols = ['rsi', 'macd', 'macd_signal', 'macd_hist', 'sma_50', 'atr', 'adx']
            else:
                feature_cols = ['rsi', 'macd', 'macd_signal', 'macd_hist',
                              'bb_upper', 'bb_middle', 'bb_lower',
                              'sma_50', 'atr', 'obv', 'adx']

            # Check for sentiment columns
            sentiment_cols = [col for col in processed_data.columns if 'sentiment' in col]
            if sentiment_cols:
                feature_cols.extend(sentiment_cols)
            else:
                logger.warning("No sentiment data columns found in input data")

            # Check if all required feature columns exist
            missing_cols = [col for col in feature_cols if col not in processed_data.columns]
            if missing_cols:
                logger.error(f"[prepare_features] Missing required feature columns: {missing_cols}. Skipping this symbol. Columns present: {list(processed_data.columns)}")
                logger.error(f"[prepare_features] Data head:\n{processed_data.head()}\nData tail:\n{processed_data.tail()}")
                return None, None

            before_drop = len(processed_data)
            # Drop rows with any NaN in feature columns before scaling
            processed_data = processed_data.dropna(subset=feature_cols)
            after_drop = len(processed_data)
            if after_drop < before_drop:
                logger.warning(f"[prepare_features] Dropped {before_drop - after_drop} rows with NaN in features for symbol(s): {data.get('symbol', 'unknown')}")

            # Log if any NaN remain (should not happen)
            if processed_data[feature_cols].isnull().any().any():
                nan_cols = processed_data[feature_cols].columns[processed_data[feature_cols].isnull().any()].tolist()
                logger.error(f"[prepare_features] NaN still present in columns after dropna: {nan_cols}")
                logger.error(f"[prepare_features] First few rows with NaN: {processed_data[feature_cols][processed_data[feature_cols].isnull().any(axis=1)].head()}")
                logger.error(f"[prepare_features] Data head before drop:\n{data.head()}\nData tail:\n{data.tail()}")
                # Distinguish between real data error and pipeline issue
                if data.empty:
                    logger.error("[prepare_features] Input data is empty. This is a real data error (API or source returned no data).")
                else:
                    logger.error("[prepare_features] Input data was present but all rows dropped. This may be a pipeline/feature engineering issue.")
                return None, None

            # Calculate labels (y) based on future price movement
            processed_data['future_close'] = processed_data['close'].shift(-1)
            processed_data['label'] = (processed_data['future_close'] > processed_data['close']).astype(int)
            # Log label distribution for debugging
            label_counts = processed_data['label'].value_counts().to_dict()
            logger.info(f"[AI MODEL] Label distribution: {label_counts}")
            
            # Drop rows with NaN in 'close' or 'label' after shifting
            processed_data = processed_data.dropna(subset=['close', 'label'])

            # Now create features and labels
            X = processed_data[feature_cols].values
            y = processed_data['label'].values

            # Scale features
            X = self.scaler.fit_transform(X)

            # Final check: drop any rows where X or y is NaN/Inf (should not happen, but for safety)
            mask = ~(
                np.isnan(X).any(axis=1) | np.isinf(X).any(axis=1) |
                np.isnan(y) | np.isinf(y)
            )
            X = X[mask]
            y = y[mask]
            # Label balancing (optional)
            if balance_labels:
                class0 = np.where(y == 0)[0]
                class1 = np.where(y == 1)[0]
                if len(class0) > 0 and len(class1) > 0:
                    min_len = min(len(class0), len(class1))
                    idx0 = resample(class0, replace=False, n_samples=min_len, random_state=42)
                    idx1 = resample(class1, replace=False, n_samples=min_len, random_state=42)
                    idx = np.concatenate([idx0, idx1])
                    X = X[idx]
                    y = y[idx]
                    logger.info(f"[prepare_features] Label balancing applied: {np.bincount(y)}")
            logger.info(f"[prepare_features] Final label distribution: {np.bincount(y) if y.size > 0 else y}")
            logger.info(f"[prepare_features] Final feature shape: {X.shape}")
            return X, y
        except Exception as e:
            logger.error(f"Error in prepare_features: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return None, None
        # Final unconditional return to guarantee a tuple (should never be hit, but for safety)
        logger.error("prepare_features: Reached end of function without returning (should not happen). Returning (None, None)")
        return None, None

    def train(self, X, y, X_val=None, y_val=None, use_simple_model=False):
        """Train the AI trading model, robust to missing features, with option for simple model"""
        try:
            logger.info(f"Training labels distribution: {np.bincount(y) if hasattr(np, 'bincount') and y.dtype==int else y}")
            logger.info(f"Feature NaN count: {np.isnan(X).sum()}, Inf count: {np.isinf(X).sum()}")
            if np.isnan(X).any() or np.isinf(X).any() or np.isnan(y).any() or np.isinf(y).any():
                logger.error("NaN or Inf detected in training data. Aborting training.")
                raise ValueError("NaN or Inf in training data")
            if len(np.unique(y)) < 2:
                logger.error("Not enough label variety for training. Aborting.")
                raise ValueError("Not enough label variety for training.")
            if use_simple_model:
                logger.info("Using LogisticRegression for debugging.")
                self.model = LogisticRegression(max_iter=200)
                self.model.fit(X, y)
                if hasattr(self.model, 'coef_'):
                    logger.info(f"Feature importances (coef): {self.model.coef_}")
                return {"status": "ok", "model": "logistic_regression"}
            # Build model if not exists
            if self.model is None:
                if len(X.shape) == 3:
                    input_shape = (X.shape[1], X.shape[2])
                else:
                    input_shape = X.shape[1]
                self.model = self.technical_model.build_model(input_shape)
            history = self.model.fit(
                X, y,
                epochs=100,
                batch_size=32,
                validation_data=(X_val, y_val) if X_val is not None and y_val is not None else None,
                verbose=1
            )
            train_metrics = self.model.evaluate(X, y, verbose=0)
            val_metrics = self.model.evaluate(X_val, y_val, verbose=0) if X_val is not None and y_val is not None else None
            def safe_float(val):
                try:
                    return float(val)
                except Exception:
                    return float('nan')
            train_loss = safe_float(train_metrics[0]) if isinstance(train_metrics, (list, tuple)) and len(train_metrics) > 0 else float('nan')
            train_acc = safe_float(train_metrics[1]) if isinstance(train_metrics, (list, tuple)) and len(train_metrics) > 1 else float('nan')
            if val_metrics is not None and isinstance(val_metrics, (list, tuple)) and len(val_metrics) > 1:
                val_loss = safe_float(val_metrics[0])
                val_acc = safe_float(val_metrics[1])
            elif isinstance(val_metrics, dict):
                val_loss = safe_float(val_metrics.get('loss', float('nan')))
                val_acc = safe_float(val_metrics.get('accuracy', float('nan')))
            else:
                val_loss = float('nan')
                val_acc = float('nan')
            logger.info(f"Training completed: loss={train_loss:.4f}, accuracy={train_acc:.4f}, val_loss={val_loss:.4f}, val_accuracy={val_acc:.4f}")
            final_metrics = self.evaluate_model(X, y)
            accuracy = final_metrics['accuracy']
            f1 = final_metrics['f1']
            logger.info(f"Final training accuracy: {accuracy:.4f}, F1: {f1:.4f}")
            if accuracy < 0.7 or f1 < 0.7:
                logger.warning("Model metrics below threshold. Consider tuning or checking data.")
            logger.info("Forcing at least one buy and one sell signal in test data for debugging.")
            return history.history
        except Exception as e:
            logger.error(f"Error in model training: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return {"status": "error", "reason": str(e)}
        
    def predict(self, market_data: pd.DataFrame) -> np.ndarray:
        """Make trading predictions"""
        if self.model is None:
            raise ValueError("Model not trained yet")
            
        X, _ = self.prepare_features(market_data)
        return self.model.predict(X)
        
    def select_action(self, state: np.ndarray) -> int:
        """Select trading action based on current state"""
        if len(state.shape) == 1:
            state = state.reshape(1, -1)
        
        prediction = self.predict(pd.DataFrame(state))
        action = 1 if prediction[0] > 0.5 else 0  # 1 for buy, 0 for sell/hold
        logger.info(f"AI model select_action: prediction={prediction[0]}, action={action}")
        return action
        
    def calculate_reward(self, action: int, entry_price: float, exit_price: float, position_type: str = 'long') -> float:
        """Calculate reward for reinforcement learning"""
        if position_type == 'long':
            return (exit_price - entry_price) / entry_price if action == 1 else 0
        else:  # short position
            return (entry_price - exit_price) / entry_price if action == 1 else 0

    def evaluate_model(self, X, y):
        """Evaluate the trained AI trading model on given data and return metrics."""
        if self.model is None:
            logger.error("Model not trained yet. Cannot evaluate.")
            return {"accuracy": 0, "precision": 0, "recall": 0, "f1": 0}
        y_pred_prob = self.model.predict(X)
        y_pred = (y_pred_prob > 0.5).astype(int).flatten()
        accuracy = accuracy_score(y, y_pred)
        precision = precision_score(y, y_pred, zero_division=0)
        recall = recall_score(y, y_pred, zero_division=0)
        f1 = f1_score(y, y_pred, zero_division=0)
        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1
        }

class TradingEnvironment(gym.Env):
    """Custom Trading Environment for Reinforcement Learning"""
    
    metadata = {'render.modes': ['human']}
    
    def __init__(self, df: pd.DataFrame, config: Dict[str, Any]):
        super(TradingEnvironment, self).__init__()
        
        self.config = config
        env_config = config.get('env', {})
        
        # Get columns from config or use defaults
        self.numerical_cols = env_config.get('features', [
            'open', 'high', 'low', 'close', 'volume', 'SMA_20', 'Daily_Return'
        ])
        self.df = df[self.numerical_cols].copy()
        
        # Initialize scalers
        self.scalers = {}
        for col in self.df.columns:
            self.scalers[col] = StandardScaler()
            self.df[col] = self.scalers[col].fit_transform(self.df[col].values.reshape(-1, 1))
        
        # Get trading parameters from config
        self.initial_balance = env_config.get('initial_balance', 100000.0)
        self.buy_cost_pct = env_config.get('buy_cost_pct', 0.001)  # 0.1% trading cost
        self.sell_cost_pct = env_config.get('sell_cost_pct', 0.001)
        self.max_position = env_config.get('max_position', 0.9)  # Maximum amount to invest
        
        # Initialize trading state
        self.current_step = 0
        self.balance = self.initial_balance
        self.shares_held = 0
        self.cost_basis = 0
        self.total_shares_sold = 0
        self.total_sales_value = 0
        self.returns = []
        
        # Define action and observation spaces
        self.action_space = spaces.Discrete(3)  # Buy (1), Sell (2), Hold (0)
        self.observation_space = spaces.Box(
            low=-np.inf, 
            high=np.inf, 
            shape=(len(self.numerical_cols),),
            dtype=np.float32
        )
        
        self.reset()
    
    def reset(self, seed=None):
        """Reset the environment"""
        super().reset(seed=seed)
        self.current_step = 0
        self.balance = self.initial_balance
        self.shares_held = 0
        self.cost_basis = 0
        self.total_shares_sold = 0
        self.total_sales_value = 0
        self.returns = []
        
        return self._next_observation(), {}
        
    def _next_observation(self):
        """Get the next observation"""
        obs = self.df.iloc[self.current_step].values.astype(np.float32)
        return obs
        
    def step(self, action):
        """Execute one time step within the environment"""
        self._take_action(action)
        self.current_step += 1
        
        reward = self._calculate_reward(action)
        done = self.current_step >= len(self.df) - 1
        
        obs = self._next_observation()
        info = self._get_info()
        
        return obs, reward, done, False, info
        
    def _take_action(self, action):
        """Execute the trading action"""
        current_price = self.scalers['close'].inverse_transform(
            self.df.iloc[self.current_step]['close'].reshape(-1, 1)
        )[0][0]
        
        if action == 1:  # Buy
            # Calculate max shares we can buy
            max_shares = int((self.balance * self.max_position) / (
                current_price * (1 + self.buy_cost_pct)
            ))
            
            # Buy shares if we can afford at least one
            if max_shares > 0:
                shares_bought = max_shares
                purchase_cost = shares_bought * current_price * (1 + self.buy_cost_pct)
                self.balance -= purchase_cost
                self.shares_held += shares_bought
                self.cost_basis = current_price
                
        elif action == 2:  # Sell
            if self.shares_held > 0:
                # Calculate sale proceeds with transaction cost
                sale_proceeds = self.shares_held * current_price * (1 - self.sell_cost_pct)
                self.balance += sale_proceeds
                self.total_shares_sold += self.shares_held
                self.total_sales_value += sale_proceeds
                self.shares_held = 0
                self.cost_basis = 0

    def _calculate_reward(self, action):
        """Calculate reward for the action taken"""
        reward_config = self.config.get('env', {}).get('rewards', {})
        
        # Get the original (unscaled) close price for reward calculation
        current_price = self.scalers['close'].inverse_transform(
            self.df.iloc[self.current_step]['close'].reshape(-1, 1)
        )[0][0]
        
        if self.current_step > 0:
            prev_price = self.scalers['close'].inverse_transform(
                self.df.iloc[self.current_step - 1]['close'].reshape(-1, 1)
            )[0][0]
            price_change = (current_price - prev_price) / prev_price
            # Clip price change based on config
            price_change = np.clip(
                price_change,
                reward_config.get('min_price_change', -0.1),
                reward_config.get('max_price_change', 0.1)
            )
        else:
            price_change = 0
            
        # Calculate position value change
        position_value = self.shares_held * current_price + self.balance
        value_change = (position_value - self.initial_balance) / self.initial_balance
        value_change = np.clip(
            value_change,
            reward_config.get('min_value_change', -1.0),
            reward_config.get('max_value_change', 1.0)
        )
        
        # Base reward on returns with configurable weights
        reward = 0
        weights = reward_config.get('action_weights', {
            'buy': {'gain': 1.0, 'loss': 0.5},
            'sell': {'gain': 1.0, 'loss': 0.5},
            'hold': {'multiplier': 0.1}
        })
        
        if action == 1:  # Buy
            reward = price_change if price_change > 0 else -abs(price_change) * weights['buy']['loss']
        elif action == 2:  # Sell
            reward = -price_change if price_change < 0 else -abs(price_change) * weights['sell']['loss']
        else:  # Hold
            reward = price_change * weights['hold']['multiplier']
        
        # Add scaled position value change to reward
        self.returns.append(value_change)
        reward = np.clip(
            reward + value_change * reward_config.get('value_change_weight', 0.5),
            reward_config.get('min_reward', -1.0),
            reward_config.get('max_reward', 1.0)
        )
        
        return float(reward)
    
    def _get_info(self):
        """Get current trading information"""
        return {
            'balance': self.balance,
            'shares_held': self.shares_held,
            'total_shares_sold': self.total_shares_sold,
            'total_sales_value': self.total_sales_value,
            'avg_return': np.mean(self.returns) if self.returns else 0
        }
        
    def render(self, mode='human'):
        """Render the environment"""
        profit = self.balance - self.initial_balance
        logger.info(f'Step: {self.current_step}')
        logger.info(f'Balance: {self.balance:.2f}')
        logger.info(f'Shares held: {self.shares_held}')
        logger.info(f'Total profit: {profit:.2f}')
        logger.info(f'Avg return: {np.mean(self.returns):.4f}')
