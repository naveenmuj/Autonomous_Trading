import sys
import os

# Add site-packages to path
python_path = os.path.dirname(os.__file__)
site_packages = os.path.join(python_path, 'site-packages')
if site_packages not in sys.path:
    sys.path.append(site_packages)

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional
from datetime import datetime, timedelta

# Configure TensorFlow logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TF logging
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Force CPU usage

logger = logging.getLogger(__name__)

def get_tf():
    """Lazy load TensorFlow to handle import issues gracefully"""
    try:
        import tensorflow as tf
        tf.get_logger().setLevel('ERROR')
        return tf
    except ImportError as e:
        logger.error(f"TensorFlow initialization failed: {e}")
        return None

def get_sklearn():
    """Lazy load scikit-learn components"""
    try:
        from sklearn.preprocessing import StandardScaler
        from sklearn.model_selection import TimeSeriesSplit
        return StandardScaler, TimeSeriesSplit
    except ImportError as e:
        logger.error(f"Scikit-learn import failed: {e}")
        return None, None

def get_talib():
    """Return pandas_ta as replacement for TA-Lib"""
    try:
        import pandas_ta as ta
        return ta
    except ImportError as e:
        logger.error(f"pandas_ta import failed: {e}")
        return None

class TrainingPipeline:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.X = None
        self.y = None
        self.model = None
        
        # Initialize dependencies
        self.tf = get_tf()
        StandardScaler, _ = get_sklearn()
        self.scaler = StandardScaler() if StandardScaler else None
        self.talib = get_talib()
        
        if self.tf is None:
            logger.warning("TensorFlow not available. Model training will be disabled.")
        if self.scaler is None:
            logger.warning("Scikit-learn not available. Data preprocessing will be limited.")
        if self.talib is None:
            logger.warning("TA-Lib not available. Technical analysis features will be limited.")

    def preprocess_data(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Preprocess data for model training"""
        try:
            if self.scaler is None:
                raise ImportError("Scikit-learn StandardScaler not available")
                
            # Calculate features first
            features = self.engineer_features(data)
            
            # Remove NaN values
            features = features.dropna()
            
            # Split into X (features) and y (labels)
            feature_cols = [col for col in features.columns if col not in ['target', 'returns']]
            X = self.scaler.fit_transform(features[feature_cols])
            y = features['target'].values if 'target' in features.columns else None
            
            # Store for hyperparameter tuning
            self.X = X
            self.y = y
            
            return X, y
        except Exception as e:
            logger.error(f"Error preprocessing data: {e}")
            raise

    def engineer_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Engineer technical indicators and other features"""
        try:
            df = data.copy()
            
            if self.talib:
                # Technical indicators using pandas_ta
                df['sma_10'] = self.talib.sma(df['close'], length=10)
                df['sma_20'] = self.talib.sma(df['close'], length=20)
                df['sma_50'] = self.talib.sma(df['close'], length=50)
                df['rsi'] = self.talib.rsi(df['close'], length=14)
                
                macd_data = self.talib.macd(df['close'], fast=12, slow=26, signal=9)
                df['macd'] = macd_data['MACD_12_26_9']
                df['macd_signal'] = macd_data['MACDs_12_26_9']
                
                bb_data = self.talib.bbands(df['close'], length=20, std=2)
                df['bb_upper'] = bb_data['BBU_20_2.0']
                df['bb_middle'] = bb_data['BBM_20_2.0']
                df['bb_lower'] = bb_data['BBL_20_2.0']
                
                df['atr'] = self.talib.atr(df['high'], df['low'], df['close'], length=14)
                df['volume_sma'] = self.talib.sma(df['volume'], length=20)
            else:
                # Fallback to basic calculations if TA-Lib is not available
                df['returns'] = df['close'].pct_change()
                df['volume_ma'] = df['volume'].rolling(window=20).mean()
            
            return df
        except Exception as e:
            logger.error(f"Error engineering features: {e}")
            raise

    def build_model(self, input_dim: int) -> Optional[Any]:
        """Build the neural network model"""
        if self.tf is None:
            logger.error("TensorFlow not available. Cannot build model.")
            return None

        try:
            model = self.tf.keras.Sequential([
                self.tf.keras.layers.Dense(64, activation='relu', input_shape=(input_dim,)),
                self.tf.keras.layers.Dropout(0.2),
                self.tf.keras.layers.Dense(32, activation='relu'),
                self.tf.keras.layers.Dropout(0.1),
                self.tf.keras.layers.Dense(1)
            ])
            
            model.compile(
                optimizer='adam',
                loss='mse',
                metrics=['mae']
            )
            
            return model
        except Exception as e:
            logger.error(f"Error building model: {e}")
            return None

    def train(self, X: np.ndarray, y: np.ndarray) -> Optional[Any]:
        """Train the model with error handling"""
        if self.tf is None:
            logger.error("TensorFlow not available. Cannot train model.")
            return None

        try:
            if X is None or y is None:
                raise ValueError("Training data not properly initialized")

            self.model = self.build_model(X.shape[1])
            if self.model is None:
                raise ValueError("Model initialization failed")

            history = self.model.fit(
                X, y,
                epochs=self.config.get('model', {}).get('epochs', 50),
                batch_size=self.config.get('model', {}).get('batch_size', 32),
                validation_split=0.2,
                verbose=1
            )
            
            return history
        except Exception as e:
            logger.error(f"Error training model: {e}")
            return None

    def save_model(self, path: str) -> bool:
        """Save the trained model"""
        if self.model is None:
            logger.error("No model to save")
            return False

        try:
            self.model.save(path)
            logger.info(f"Model saved successfully to {path}")
            return True
        except Exception as e:
            logger.error(f"Error saving model: {e}")
            return False

    def load_model(self, path: str) -> bool:
        """Load a trained model"""
        if self.tf is None:
            logger.error("TensorFlow not available. Cannot load model.")
            return False

        try:
            self.model = self.tf.keras.models.load_model(path)
            logger.info(f"Model loaded successfully from {path}")
            return True
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return False

    def prepare_training_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Prepare data for model training"""
        try:
            logger.info("Preparing training data...")
            
            from data.collector import DataCollector
            collector = DataCollector(self.config)
            
            # Set skip_indices flag for DataCollector
            self.config['skip_indices'] = True
            symbols = collector.get_symbols_from_config()
            self.config['skip_indices'] = False  # Reset after use
            if not symbols:
                logger.warning("No symbols discovered for training")
                symbols = ['RELIANCE-EQ', 'TCS-EQ', 'HDFCBANK-EQ']  # Default symbols
            
            # Fetch historical data
            all_data = []
            for symbol in symbols:
                try:
                    # Handle symbol format conversion
                    if '-EQ' in symbol:
                        nse_symbol = symbol.replace('-EQ', '.NS')
                    elif '.NS' in symbol:
                        nse_symbol = symbol
                    else:
                        nse_symbol = f"{symbol}.NS"
                    
                    # Get 1 year of historical data
                    end_date = datetime.now()
                    start_date = end_date - timedelta(days=365)
                    
                    data = collector.get_historical_data(
                        symbol=nse_symbol,
                        start_date=start_date,
                        end_date=end_date,
                        interval='1d'
                    )
                    
                    if not data.empty:
                        # Ensure numeric data types
                        numeric_cols = ['open', 'high', 'low', 'close', 'volume']
                        data[numeric_cols] = data[numeric_cols].astype(float)
                        
                        # Store original symbol for reference
                        data['symbol'] = symbol
                        all_data.append(data)
                        logger.info(f"Successfully fetched data for {symbol}")
                    else:
                        logger.warning(f"No historical data available for {symbol}")
                except Exception as e:
                    logger.error(f"Error fetching data for {symbol}: {str(e)}")
                    continue
            
            if not all_data:
                raise ValueError("No training data could be prepared")
            
            # Combine all data
            training_data = pd.concat(all_data)
            
            # Get current market data for live trading
            market_data = collector.get_market_data()
            if market_data.empty:
                logger.warning("No live market data available (market may be closed)")
                # Use most recent historical data as current market data
                market_data = training_data.groupby('symbol').last()
            
            logger.info(f"Prepared training data with {len(training_data)} records")
            return training_data, market_data
        except Exception as e:
            logger.error(f"Error preparing training data: {str(e)}")
            raise

    def get_default_models(self) -> Dict[str, Any]:
        """Get default models when training data is not available"""
        try:
            if self.tf is None:
                raise ImportError("TensorFlow not available")

            # Create minimal models with default parameters
            models = {
                'price_prediction': self.tf.keras.Sequential([
                    self.tf.keras.layers.Dense(64, activation='relu', input_shape=(10,)),
                    self.tf.keras.layers.Dropout(0.2),
                    self.tf.keras.layers.Dense(32, activation='relu'),
                    self.tf.keras.layers.Dense(1)
                ]),
                'trend_prediction': self.tf.keras.Sequential([
                    self.tf.keras.layers.Dense(64, activation='relu', input_shape=(10,)),
                    self.tf.keras.layers.Dropout(0.2),
                    self.tf.keras.layers.Dense(32, activation='relu'),
                    self.tf.keras.layers.Dense(3, activation='softmax')
                ])
            }
            
            # Compile models with default optimizers
            models['price_prediction'].compile(
                optimizer='adam',
                loss='mse',
                metrics=['mae']
            )
            
            models['trend_prediction'].compile(
                optimizer='adam',
                loss='categorical_crossentropy',
                metrics=['accuracy']
            )
            
            logger.info("Created default models")
            return models
        except Exception as e:
            logger.error(f"Error creating default models: {str(e)}")
            raise

    def train_models(self, data: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """Train and return the machine learning models"""
        try:
            if self.tf is None:
                raise ImportError("TensorFlow not available")

            if data is not None:
                X, y = self.preprocess_data(data)
            elif self.X is not None and self.y is not None:
                X, y = self.X, self.y
            else:
                logger.warning("No data available for training")
                return self.get_default_models()

            # Train models
            models = {}
            
            # LSTM model for sequence prediction
            models['lstm'] = self._train_lstm_model(X, y)
            
            # Feed-forward neural network for classification
            models['ffnn'] = self._train_ffnn_model(X, y)
            
            logger.info("Successfully trained all models")
            return models
        except Exception as e:
            logger.error(f"Error training models: {str(e)}")
            logger.warning("Using default models instead")
            return self.get_default_models()

    def _train_lstm_model(self, X: np.ndarray, y: np.ndarray) -> Any:
        """Train LSTM model for sequence prediction"""
        try:
            if self.tf is None:
                raise ImportError("TensorFlow not available")

            # Reshape data for LSTM [samples, timesteps, features]
            seq_length = self.config.get('ai', {}).get('lstm_seq_length', 10)
            X_lstm = self._prepare_sequences(X, seq_length)
            y_lstm = y[seq_length:]
            
            # Build LSTM model
            model = self.tf.keras.Sequential([
                self.tf.keras.layers.LSTM(64, input_shape=(seq_length, X.shape[1])),
                self.tf.keras.layers.Dense(32, activation='relu'),
                self.tf.keras.layers.Dense(1, activation='sigmoid')
            ])
            
            model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
            
            # Train model
            model.fit(X_lstm, y_lstm, epochs=10, batch_size=32, validation_split=0.2, verbose=0)
            
            return model
        except Exception as e:
            logger.error(f"Error training LSTM model: {str(e)}")
            return None

    def _train_ffnn_model(self, X: np.ndarray, y: np.ndarray) -> Any:
        """Train feed-forward neural network for classification"""
        try:
            if self.tf is None:
                raise ImportError("TensorFlow not available")

            # Build FFNN model
            model = self.tf.keras.Sequential([
                self.tf.keras.layers.Dense(64, activation='relu', input_shape=(X.shape[1],)),
                self.tf.keras.layers.Dropout(0.2),
                self.tf.keras.layers.Dense(32, activation='relu'),
                self.tf.keras.layers.Dense(1, activation='sigmoid')
            ])
            
            model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
            
            # Train model
            model.fit(X, y, epochs=10, batch_size=32, validation_split=0.2, verbose=0)
            
            return model
        except Exception as e:
            logger.error(f"Error training FFNN model: {str(e)}")
            return None

    def _prepare_sequences(self, data: np.ndarray, seq_length: int) -> np.ndarray:
        """Prepare sequences for LSTM model"""
        sequences = []
        for i in range(len(data) - seq_length):
            sequences.append(data[i:(i + seq_length)])
        return np.array(sequences)