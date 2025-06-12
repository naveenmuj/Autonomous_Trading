import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from datetime import datetime, timedelta
import optuna
import talib

logger = logging.getLogger(__name__)

class TrainingPipeline:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.X = None
        self.y = None
        self.model = None
        self.scaler = StandardScaler()

    def preprocess_data(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Preprocess data for model training"""
        try:
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
            logger.error(f"Error preprocessing data: {str(e)}")
            raise

    def engineer_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Engineer technical indicators and other features"""
        try:
            df = data.copy()
            
            # Ensure OHLCV columns exist
            required_cols = ['open', 'high', 'low', 'close', 'volume']
            if not all(col in df.columns for col in required_cols):
                raise ValueError(f"Missing required columns. Need: {required_cols}")

            # Technical Indicators
            df['sma_10'] = talib.SMA(df['close'], timeperiod=10)
            df['sma_20'] = talib.SMA(df['close'], timeperiod=20)
            df['sma_50'] = talib.SMA(df['close'], timeperiod=50)
            
            df['rsi'] = talib.RSI(df['close'], timeperiod=14)
            
            macd, macd_signal, _ = talib.MACD(df['close'])
            df['macd'] = macd
            df['macd_signal'] = macd_signal
            
            upper, middle, lower = talib.BBANDS(df['close'])
            df['bb_upper'] = upper
            df['bb_middle'] = middle
            df['bb_lower'] = lower
            
            # Volatility and Volume
            df['atr'] = talib.ATR(df['high'], df['low'], df['close'])
            df['volume_sma'] = talib.SMA(df['volume'], timeperiod=20)
            
            # Price based features
            df['returns'] = df['close'].pct_change()
            df['log_returns'] = np.log1p(df['returns'])
            
            # Target variable (next day return)
            df['target'] = df['returns'].shift(-1)
            
            return df
            
        except Exception as e:
            logger.error(f"Error engineering features: {str(e)}")
            raise

    def build_model(self, input_dim: int) -> tf.keras.Model:
        """Build and compile the neural network model"""
        try:
            model = tf.keras.Sequential([
                tf.keras.layers.Input(shape=(input_dim,)),
                tf.keras.layers.Dense(64, activation='relu'),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Dropout(0.2),
                tf.keras.layers.Dense(32, activation='relu'),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Dropout(0.1),
                tf.keras.layers.Dense(1, activation='tanh')
            ])
            
            model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                loss='mse',
                metrics=['mae']
            )
            
            return model
            
        except Exception as e:
            logger.error(f"Error building model: {str(e)}")
            raise

    def train(self, X: np.ndarray, y: np.ndarray) -> tf.keras.Model:
        """Train the model with given data"""
        try:
            self.model = self.build_model(X.shape[1])
            
            early_stopping = tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True
            )
            
            history = self.model.fit(
                X, y,
                epochs=self.config.get('model', {}).get('epochs', 100),
                batch_size=self.config.get('model', {}).get('batch_size', 32),
                validation_split=0.2,
                callbacks=[early_stopping],
                verbose=1
            )
            
            return self.model
            
        except Exception as e:
            logger.error(f"Error training model: {str(e)}")
            raise

    def hyperparameter_tune(self, trials: int = 100) -> Dict[str, Any]:
        """Perform hyperparameter tuning using Optuna"""
        try:
            def objective(trial):
                # Define hyperparameters to tune
                lr = trial.suggest_loguniform('learning_rate', 1e-5, 1e-2)
                units_1 = trial.suggest_int('units_1', 32, 128)
                units_2 = trial.suggest_int('units_2', 16, 64)
                dropout_1 = trial.suggest_uniform('dropout_1', 0.1, 0.5)
                dropout_2 = trial.suggest_uniform('dropout_2', 0.1, 0.3)
                
                # Build model with trial parameters
                model = tf.keras.Sequential([
                    tf.keras.layers.Input(shape=(self.X.shape[1],)),
                    tf.keras.layers.Dense(units_1, activation='relu'),
                    tf.keras.layers.BatchNormalization(),
                    tf.keras.layers.Dropout(dropout_1),
                    tf.keras.layers.Dense(units_2, activation='relu'),
                    tf.keras.layers.BatchNormalization(),
                    tf.keras.layers.Dropout(dropout_2),
                    tf.keras.layers.Dense(1, activation='tanh')
                ])
                
                model.compile(
                    optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
                    loss='mse',
                    metrics=['mae']
                )
                
                # Train and evaluate
                history = model.fit(
                    self.X, self.y,
                    epochs=50,
                    batch_size=32,
                    validation_split=0.2,
                    verbose=0
                )
                
                return min(history.history['val_loss'])
            
            # Create and run study
            study = optuna.create_study(direction='minimize')
            study.optimize(objective, n_trials=trials)
            
            return study.best_params
            
        except Exception as e:
            logger.error(f"Error during hyperparameter tuning: {str(e)}")
            raise

    def save_model(self, path: str):
        """Save the trained model"""
        try:
            if self.model is None:
                raise ValueError("No model to save")
            self.model.save(path)
            logger.info(f"Model saved to {path}")
        except Exception as e:
            logger.error(f"Error saving model: {str(e)}")
            raise

    def load_model(self, path: str):
        """Load a trained model"""
        try:
            self.model = tf.keras.models.load_model(path)
            logger.info(f"Model loaded from {path}")
            return self.model
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise

    def prepare_training_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Prepare data for model training"""
        try:
            logger.info("Preparing training data...")
            
            from data.collector import DataCollector
            collector = DataCollector(self.config)
            
            # Get configured symbols
            symbols = self.config.get('trading', {}).get('data', {}).get('manual_symbols', [])
            if not symbols:
                logger.warning("No symbols configured for training")
                symbols = ['RELIANCE-EQ', 'TCS-EQ', 'HDFCBANK-EQ']  # Default symbols
            
            # Fetch historical data
            all_data = []
            for symbol in symbols:
                try:
                    # Convert Angel One symbol format to NSE format
                    nse_symbol = symbol.replace('-EQ', '.NS')
                    
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
                        data['symbol'] = symbol
                        all_data.append(data)
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
            # Create minimal models with default parameters
            models = {
                'price_prediction': tf.keras.Sequential([
                    tf.keras.layers.Dense(64, activation='relu', input_shape=(10,)),
                    tf.keras.layers.Dropout(0.2),
                    tf.keras.layers.Dense(32, activation='relu'),
                    tf.keras.layers.Dense(1)
                ]),
                'trend_prediction': tf.keras.Sequential([
                    tf.keras.layers.Dense(64, activation='relu', input_shape=(10,)),
                    tf.keras.layers.Dropout(0.2),
                    tf.keras.layers.Dense(32, activation='relu'),
                    tf.keras.layers.Dense(3, activation='softmax')
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
