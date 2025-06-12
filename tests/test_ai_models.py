import pytest
import pandas as pd
import numpy as np
import os
import sys
from unittest.mock import patch, MagicMock
import logging

# Setup logging for tests
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Mock TensorFlow before importing models
mock_tf = MagicMock()
sys.modules['tensorflow'] = mock_tf
sys.modules['tensorflow.keras'] = MagicMock()
sys.modules['tensorflow.keras.models'] = MagicMock()

from src.ai.models import AITrader, TechnicalAnalysisModel, SentimentAnalyzer
from src.ai.training_pipeline import ModelTrainingPipeline

@pytest.fixture
def ai_config():
    return {
        'model': {  # Changed from 'models' to 'model'
            'training': {
                'epochs': 2,
                'batch_size': 32,
                'validation_split': 0.2,
                'patience': 5
            },
            'features': {
                'technical_indicators': {
                    'rsi_period': 14,
                    'macd_fast': 12,
                    'macd_slow': 26,
                    'macd_signal': 9,
                    'bb_period': 20,
                    'bb_std': 2
                },
                'use_sentiment': True,
                'columns': ['RSI', 'MACD', 'MACD_Signal', 'MACD_Hist', 'BB_upper', 'BB_middle', 'BB_lower']
            },
            'target_column': 'target'
        },
        'data': {
            'timeframe': '1d',
            'symbols': ['AAPL', 'MSFT'],
            'start_date': '2024-01-01',
            'end_date': '2024-12-31'
        }
    }

@pytest.fixture
def sample_data():
    dates = pd.date_range(start='2024-01-01', end='2024-01-10', freq='D')
    return pd.DataFrame({
        'date': dates,
        'open': np.random.uniform(100, 200, size=len(dates)),
        'high': np.random.uniform(100, 200, size=len(dates)),
        'low': np.random.uniform(100, 200, size=len(dates)),
        'close': np.random.uniform(100, 200, size=len(dates)),
        'volume': np.random.uniform(1000000, 5000000, size=len(dates)),
        'news_headlines': ['Sample headline ' + str(i) for i in range(len(dates))],
        'target': np.random.randint(0, 2, size=len(dates))
    })

@pytest.fixture
def mock_config():
    return {
        'model': {
            'technical': {
                'layers': [64, 32],
                'dropout': 0.2,
                'learning_rate': 0.001,
                'sequence_length': 10,
                'batch_size': 32,
                'epochs': 2
            },
            'reinforcement': {
                'state_size': 10,
                'action_size': 3,
                'gamma': 0.95,
                'epsilon': 0.1,
                'reward_scale': 1.0,
                'memory_size': 1000
            },
            'training': {
                'batch_size': 32,
                'epochs': 2,
                'validation_split': 0.2,
                'early_stopping_patience': 5,
                'up_threshold': 0.01,
                'down_threshold': -0.01
            }
        },
        'data': {
            'timeframes': ['1d', '1h'],
            'historical_days': 252,
            'technical_indicators': {
                'rsi_period': 14,
                'macd_fast': 12,
                'macd_slow': 26,
                'macd_signal': 9,
                'bb_period': 20,
                'bb_std': 2
            }
        }
    }

@pytest.fixture
def ai_trader(mock_config):
    return AITrader(mock_config)

@pytest.fixture
def ta_model(mock_config):
    return TechnicalAnalysisModel(mock_config)

@pytest.fixture
def pipeline(mock_config):
    return ModelTrainingPipeline(mock_config)

class TestAIModels:
    @pytest.mark.timeout(30)
    def test_ai_trader_initialization(self, ai_config):
        """Test AITrader initialization without models"""
        logger.info("Starting AI trader initialization test")
        trader = AITrader(ai_config)
        assert trader is not None
        assert trader.config == ai_config
        logger.info("AI trader initialization test passed")

    @pytest.mark.timeout(30)
    def test_technical_analysis_model(self, sample_market_data):
        """Test TechnicalAnalysisModel initialization and predictions"""
        logger.info("Starting technical analysis model test")
        model = TechnicalAnalysisModel()
        assert model is not None
        
        # Test basic technical indicators
        indicators = model.calculate_indicators(sample_market_data)
        assert isinstance(indicators, pd.DataFrame)
        assert 'RSI' in indicators.columns
        assert 'MACD' in indicators.columns
        logger.info("Technical analysis model test passed")

    @pytest.mark.timeout(30)
    def test_sentiment_analyzer(self):
        """Test SentimentAnalyzer"""
        logger.info("Starting sentiment analyzer test")
        analyzer = SentimentAnalyzer()
        assert analyzer is not None
        
        # Test sentiment analysis with dummy news
        dummy_news = ["Company XYZ reports positive earnings"]
        sentiment = analyzer.analyze_news(dummy_news)
        assert isinstance(sentiment, float)
        assert -1 <= sentiment <= 1
        logger.info("Sentiment analyzer test passed")

    @pytest.mark.timeout(30)
    def test_ai_trader_prediction(self, ai_config, sample_market_data):
        """Test AI trader model training and prediction"""
        logger.info("Starting AI trader prediction test")
        try:
            trader = AITrader(ai_config)
            assert trader is not None
            
            # Train model
            trader.train(sample_market_data)
            
            # Test prediction
            state = trader._prepare_features(sample_market_data)
            prediction = trader.predict(state[0])  # Test with single state
            assert isinstance(prediction, int)
            assert prediction in [0, 1, 2]  # Hold, Buy, Sell
            
            logger.info("AI trader prediction test passed")
        except Exception as e:
            logger.error(f"AI trader prediction test failed: {str(e)}")
            raise

    def test_model_initialization(self, ai_trader, ta_model, pipeline):
        """Test model initialization"""
        assert ai_trader is not None
        assert ta_model is not None
        assert pipeline is not None
        
    def test_technical_model_architecture(self, ta_model):
        """Test technical analysis model architecture"""
        model = ta_model.build_model()
        assert model is not None
        # Skip TensorFlow-specific assertions in test environment
        logger.info("Technical model architecture test passed")

    def test_data_preprocessing(self, pipeline):
        """Test data preprocessing"""
        dates = pd.date_range('2020-01-01', periods=100, freq='D')
        data = pd.DataFrame({
            'open': np.random.randn(100) + 100,
            'high': np.random.randn(100) + 102,
            'low': np.random.randn(100) + 98,
            'close': np.random.randn(100) + 101,
            'volume': np.random.randint(1000, 2000, 100),
            'rsi': np.random.randn(100) * 10 + 50,
            'sma_50': np.random.randn(100) + 100
        }, index=dates)
        
        X, y = pipeline.preprocess_data(data)
        assert isinstance(X, np.ndarray)
        assert isinstance(y, np.ndarray)
        assert X.shape[1] == len(data.columns)

    def test_reinforcement_learning(self, ai_trader):
        """Test reinforcement learning components"""
        state = np.random.randn(10)
        action = ai_trader.select_action(state)
        assert isinstance(action, int)
        assert action in [0, 1, 2]  # Buy, Sell, Hold

    def test_reward_calculation(self, ai_trader):
        """Test reward calculation"""
        entry_price = 100
        exit_price = 110
        action = 0  # Buy
        reward = ai_trader.calculate_reward(action, entry_price, exit_price)
        expected_reward = (exit_price - entry_price) / entry_price
        assert reward == pytest.approx(expected_reward)

    def test_model_training(self, ta_model):
        """Test model training"""
        X = np.random.randn(100, 10)
        y = np.random.randint(0, 3, 100)
        y = mock_tf.keras.utils.to_categorical(y, 3)
        
        with patch('tensorflow.keras.Sequential') as mock_sequential:
            model = mock_sequential.return_value
            history = ta_model.train_model(model, X, y, epochs=2)
            assert model.fit.called
            assert model.fit.call_args[0][0].shape == X.shape
            assert model.fit.call_args[0][1].shape == y.shape

    def test_model_prediction(self, ta_model):
        """Test model prediction"""
        X_test = np.random.randn(10, 10)
        
        with patch('tensorflow.keras.Sequential') as mock_sequential:
            model = mock_sequential.return_value
            model.predict.return_value = np.random.rand(10, 3)
            
            predictions = ta_model.predict(model, X_test)
            assert len(predictions) == 10
            assert all(isinstance(p, int) for p in predictions)
            assert all(0 <= p <= 2 for p in predictions)

    def test_model_evaluation(self, ta_model):
        """Test model evaluation"""
        X_test = np.random.randn(100, 10)
        y_test = np.random.randint(0, 3, 100)
        y_test = mock_tf.keras.utils.to_categorical(y_test, 3)
        
        with patch('tensorflow.keras.Sequential') as mock_sequential:
            model = mock_sequential.return_value
            metrics = ta_model.evaluate_model(model, X_test, y_test)
            assert model.evaluate.called
            assert isinstance(metrics, dict)
            assert 'loss' in metrics
            assert 'accuracy' in metrics

    def test_feature_engineering(self, pipeline):
        """Test feature engineering"""
        dates = pd.date_range('2020-01-01', periods=100, freq='D')
        close = 100 + np.random.randn(100).cumsum()
        data = pd.DataFrame({
            'open': close * (1 + np.random.randn(100) * 0.02),
            'high': close * (1 + abs(np.random.randn(100) * 0.03)),
            'low': close * (1 - abs(np.random.randn(100) * 0.03)),
            'close': close,
            'volume': np.abs(np.random.randint(1000, 10000, 100)),
            'daily_return': np.random.randn(100) * 0.02
        }, index=dates)
        
        features = pipeline.engineer_features(data)
        assert isinstance(features, pd.DataFrame)
        assert features.shape[1] > data.shape[1]

    def test_hyperparameter_tuning(self, pipeline):
        """Test hyperparameter tuning"""
        param_grid = {
            'learning_rate': [0.001, 0.01],
            'batch_size': [32, 64]
        }
        
        with patch('sklearn.model_selection.GridSearchCV') as mock_grid:
            mock_grid.return_value.best_params_ = {
                'learning_rate': 0.001,
                'batch_size': 32
            }
            
            best_params = pipeline.tune_hyperparameters(param_grid)
            assert isinstance(best_params, dict)
            assert 'learning_rate' in best_params
            assert 'batch_size' in best_params

    @pytest.mark.integration
    def test_end_to_end_training(self, ta_model, pipeline):
        """Test end-to-end model training pipeline"""
        # Generate sample data
        data = pd.DataFrame({
            'open': np.random.randn(1000) + 100,
            'high': np.random.randn(1000) + 102,
            'low': np.random.randn(1000) + 98,
            'close': np.random.randn(1000) + 101,
            'volume': np.random.randint(1000, 2000, 1000)
        })
        
        # Preprocess
        features = pipeline.engineer_features(data)
        X, y = pipeline.preprocess_data(features)
        
        # Train
        with patch('tensorflow.keras.Sequential') as mock_sequential:
            model = ta_model.build_model()
            history = ta_model.train_model(model, X, y, epochs=2)
            assert mock_sequential.called
            
            # Predict
            predictions = ta_model.predict(model, X[:10])
            assert len(predictions) == 10
