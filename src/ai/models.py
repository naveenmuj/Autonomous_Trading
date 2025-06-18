import logging
import numpy as np
import pandas as pd
import pandas_ta as ta
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

logger = logging.getLogger(__name__)

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
            return df
            
        except Exception as e:
            logger.error("Error calculating indicators: %s", str(e))
            raise

    def build_model(self, input_shape, model_type='lstm'):
        """Build and compile the model with improved architecture"""
        if model_type == 'lstm':
            timesteps, features = input_shape
            
            # Create a more sophisticated LSTM architecture
            model = Sequential([
                # Input layer
                tf.keras.layers.Input(shape=(timesteps, features)),
                
                # First LSTM layer with larger units and gradient clipping
                tf.keras.layers.LSTM(256, return_sequences=True, 
                                   kernel_constraint=tf.keras.constraints.MaxNorm(3),
                                   recurrent_constraint=tf.keras.constraints.MaxNorm(3),
                                   kernel_regularizer=tf.keras.regularizers.l2(0.01),
                                   recurrent_regularizer=tf.keras.regularizers.l2(0.01)),
                tf.keras.layers.LayerNormalization(),
                tf.keras.layers.Dropout(0.4),
                
                # Second LSTM layer
                tf.keras.layers.LSTM(128, return_sequences=True,
                                   kernel_constraint=tf.keras.constraints.MaxNorm(3),
                                   recurrent_constraint=tf.keras.constraints.MaxNorm(3),
                                   kernel_regularizer=tf.keras.regularizers.l2(0.01),
                                   recurrent_regularizer=tf.keras.regularizers.l2(0.01)),
                tf.keras.layers.LayerNormalization(),
                tf.keras.layers.Dropout(0.3),
                
                # Third LSTM layer
                tf.keras.layers.LSTM(64, return_sequences=False,
                                   kernel_constraint=tf.keras.constraints.MaxNorm(3),
                                   recurrent_constraint=tf.keras.constraints.MaxNorm(3),
                                   kernel_regularizer=tf.keras.regularizers.l2(0.01),
                                   recurrent_regularizer=tf.keras.regularizers.l2(0.01)),
                tf.keras.layers.LayerNormalization(),
                tf.keras.layers.Dropout(0.2),
                
                # Dense layers with residual connections
                tf.keras.layers.Dense(32, activation='relu'),
                tf.keras.layers.LayerNormalization(),
                tf.keras.layers.Dropout(0.1),
                
                # Output layer
                tf.keras.layers.Dense(1, activation='sigmoid')
            ])
            
            # Create optimizer with learning rate scheduling and gradient clipping
            optimizer = tf.keras.optimizers.Adam(
                learning_rate=1e-3,
                clipnorm=1.0,
                clipvalue=0.5
            )
            
            # Compile the model with appropriate metrics
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
            # Standard dense model
            model = Sequential([
                Dense(64, activation='relu', input_dim=input_shape),
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
    
    def prepare_data(self, data: pd.DataFrame, lookback: int = 60) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare features and labels with improved preprocessing"""
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
            
            # Handle missing values
            processed_data = processed_data.fillna(method='ffill').fillna(method='bfill')
            
            # Create features and labels
            features = processed_data[feature_cols].values
            
            # Create target variable with multiple horizons
            horizons = [1, 5, 10]  # Predict returns for different horizons
            labels = []
            for horizon in horizons:
                future_returns = processed_data['returns'].shift(-horizon)
                labels.append((future_returns > 0).astype(int).values[:-max(horizons)])
            labels = np.column_stack(labels)
            labels = np.mean(labels, axis=1)  # Aggregate predictions across horizons
            
            # Scale features with robust scaler to handle outliers
            self.scaler = StandardScaler()
            scaled_features = self.scaler.fit_transform(features)
            
            if lookback > 0:
                # Create sequences for LSTM with overlapping windows
                X, y = [], []
                for i in range(lookback, len(scaled_features) - max(horizons)):
                    X.append(scaled_features[i - lookback:i])
                    y.append(labels[i])
                X = np.array(X)
                y = np.array(y)
            else:
                # For non-LSTM models
                X = scaled_features[:-max(horizons)]
                y = labels
            
            return X, y
            
        except Exception as e:
            logger.error(f"Error in prepare_data: {str(e)}")
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
        return self.model.predict(X)

    def evaluate_model(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """Evaluate model performance"""
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        y_pred = (self.predict(X) > 0.5).astype(int)
        
        return {
            'accuracy': accuracy_score(y, y_pred),
            'precision': precision_score(y, y_pred),
            'recall': recall_score(y, y_pred),
            'f1': f1_score(y, y_pred)
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
            return {}

class SentimentAnalyzer:
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        
    def analyze_news(self, news_data: List[Dict[str, str]]) -> float:
        """Analyze sentiment of news articles"""
        try:
            # Simple sentiment analysis using positive/negative word counting
            # In a real implementation, you'd use a proper NLP model
            positive_words = set(['up', 'rise', 'gain', 'positive', 'growth', 'profit'])
            negative_words = set(['down', 'fall', 'loss', 'negative', 'decline', 'drop'])
            
            total_score = 0
            for article in news_data:
                text = article.get('title', '') + ' ' + article.get('description', '')
                text = text.lower()
                
                pos_count = sum(1 for word in positive_words if word in text)
                neg_count = sum(1 for word in negative_words if word in text)
                
                if pos_count + neg_count > 0:
                    score = (pos_count - neg_count) / (pos_count + neg_count)
                    total_score += score
            
            return total_score / len(news_data) if news_data else 0
            
        except Exception as e:
            logger.error(f"Error analyzing sentiment: {str(e)}")
            return 0

class AITrader:
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.technical_model = TechnicalAnalysisModel(config)
        self.sentiment_analyzer = SentimentAnalyzer(config)
        self.model = None
        self.scaler = StandardScaler()
        
    def prepare_features(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare features for training/prediction"""
        try:
            # Calculate technical indicators
            processed_data = self.technical_model.calculate_indicators(data)
            
            # Define feature columns
            feature_cols = ['rsi', 'macd', 'macd_signal', 'macd_hist',
                          'bb_upper', 'bb_middle', 'bb_lower',
                          'sma_50', 'atr', 'obv', 'adx']
            
            # Check for sentiment columns
            sentiment_cols = [col for col in processed_data.columns if 'sentiment' in col]
            if sentiment_cols:
                feature_cols.extend(sentiment_cols)
            else:
                logger.warning("No sentiment data columns found in input data")
            
            try:
                X = processed_data[feature_cols].values
                y = (processed_data['close'].shift(-1) > processed_data['close']).astype(int).values[:-1]
                X = X[:-1]  # Remove last row to match y dimensions
                
                # Scale features
                X = self.scaler.fit_transform(X)
                
                return X, y
            except KeyError as e:
                logger.error(f"Error preparing features: {str(e)}")
                raise
                
        except Exception as e:
            logger.error(f"Error in prepare_features: {str(e)}")
            raise
            
    def train(self, market_data: pd.DataFrame, epochs: int = 100, batch_size: int = 32) -> Dict:
        """Train the AI trading model"""
        try:
            # Prepare features
            X, y = self.prepare_features(market_data)
            
            # Build model if not exists
            if self.model is None:
                self.model = self.technical_model.build_model(X.shape[1])
            
            # Train model
            history = self.model.fit(
                X, y,
                epochs=epochs,
                batch_size=batch_size,
                validation_split=0.2,
                verbose=0
            )
            return history.history
            
        except Exception as e:
            logger.error(f"Error in model training: {str(e)}")
            raise
            
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
        return 1 if prediction[0] > 0.5 else 0  # 1 for buy, 0 for sell/hold
        
    def calculate_reward(self, action: int, entry_price: float, exit_price: float, position_type: str = 'long') -> float:
        """Calculate reward for reinforcement learning"""
        if position_type == 'long':
            return (exit_price - entry_price) / entry_price if action == 1 else 0
        else:  # short position
            return (entry_price - exit_price) / entry_price if action == 1 else 0

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
