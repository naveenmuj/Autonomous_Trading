import os
import sys
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from src.data.collector import DataCollector
from src.ai.models import TechnicalAnalysisModel, AITrader
from src.trading.analysis import TechnicalAnalyzer
from src.trading.strategy import EnhancedTradingStrategy
from src.trading.backtester import Backtester
import yaml

logger = logging.getLogger(__name__)

class ModelTrainer:
    def __init__(self, config_path: str = 'config.yaml'):
        """Initialize model trainer with configuration"""
        try:
            with open(config_path, 'r') as file:
                self.config = yaml.safe_load(file)
            
            self.data_collector = DataCollector(self.config)
            self.tech_model = TechnicalAnalysisModel(self.config)
            self.ai_trader = AITrader(self.config)
            self.tech_analyzer = TechnicalAnalyzer()
            self.strategy = EnhancedTradingStrategy(self.config)
            self.backtester = Backtester(self.strategy)
            
            # Create models directory if not exists
            os.makedirs('models', exist_ok=True)
            
            logger.info("ModelTrainer initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing ModelTrainer: {str(e)}")
            raise

    def collect_training_data(self) -> pd.DataFrame:
        """Collect and prepare training data"""
        try:
            # Get configured symbols
            symbols = self.config.get('trading', {}).get('data', {}).get('manual_symbols', [])
            if not symbols:
                logger.warning("No symbols configured, using defaults")
                symbols = ['RELIANCE.NS', 'TCS.NS', 'INFY.NS']
            
            logger.info(f"Collecting data for symbols: {symbols}")
            
            # Collect historical data
            end_date = datetime.now()
            start_date = end_date - timedelta(days=365)  # 1 year of data
            
            all_data = []
            for symbol in symbols:
                try:
                    df = self.data_collector.get_historical_data(
                        symbol=symbol,
                        days=365
                    )
                    
                    if not df.empty:
                        # Add technical indicators
                        df = self.tech_model.calculate_indicators(df)
                        
                        # Add trend line analysis
                        trend_lines = self.tech_analyzer.detect_trend_lines(df)
                        breakouts = self.tech_analyzer.analyze_breakouts(df, trend_lines)
                        
                        # Store analysis results
                        df['trend_strength'] = self.strategy._calculate_trend_strength(df)
                        df['breakout_strength'] = self.strategy._calculate_breakout_strength(df)
                        
                        all_data.append(df)
                        logger.info(f"Collected and processed {len(df)} records for {symbol}")
                    else:
                        logger.warning(f"No data available for {symbol}")
                
                except Exception as e:
                    logger.error(f"Error processing {symbol}: {str(e)}")
            
            if not all_data:
                raise ValueError("No data collected for training")
            
            # Combine all data
            combined_data = pd.concat(all_data)
            logger.info(f"Total training records: {len(combined_data)}")
            
            return combined_data
            
        except Exception as e:
            logger.error(f"Error collecting training data: {str(e)}")
            raise

    def train_models(self, data: pd.DataFrame) -> Dict[str, float]:
        """Train all models with collected data"""
        try:
            logger.info("Starting model training")
            metrics = {}
            
            # Split data into train/validation/test sets
            train_size = int(len(data) * 0.7)
            val_size = int(len(data) * 0.15)
            
            train_data = data[:train_size]
            val_data = data[train_size:train_size + val_size]
            test_data = data[train_size + val_size:]
            
            # Train Technical Analysis Model
            logger.info("Training Technical Analysis Model...")
            X_train, y_train = self.tech_model.prepare_data(train_data)
            X_val, y_val = self.tech_model.prepare_data(val_data)
            
            history = self.tech_model.train_model(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=self.config.get('model', {}).get('training', {}).get('epochs', 100),
                batch_size=32
            )
            
            # Evaluate on test set
            X_test, y_test = self.tech_model.prepare_data(test_data)
            test_metrics = self.tech_model.evaluate_model(X_test, y_test)
            
            metrics.update({
                'tech_model_accuracy': test_metrics['accuracy'],
                'tech_model_precision': test_metrics['precision'],
                'tech_model_recall': test_metrics['recall'],
                'tech_model_f1': test_metrics['f1']
            })
            
            logger.info(f"Technical Model Metrics: {test_metrics}")
            
            # Train AI Trader
            logger.info("Training AI Trader Model...")
            self.ai_trader.train(train_data)
            
            # Backtest strategy
            logger.info("Running backtest...")
            backtest_results = self.backtester.run(test_data)
            
            metrics.update({
                'backtest_return': backtest_results['total_return'],
                'backtest_sharpe': backtest_results['sharpe_ratio'],
                'backtest_max_drawdown': backtest_results['max_drawdown'],
                'backtest_win_rate': backtest_results['win_rate']
            })
            
            logger.info(f"Backtest Results: {backtest_results}")
            
            # Save models
            self.save_models()
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error training models: {str(e)}")
            raise

    def save_models(self):
        """Save trained models"""
        try:
            models_dir = 'models'
            
            # Save Technical Analysis Model
            self.tech_model.model.save(os.path.join(models_dir, 'technical_model.h5'))
            
            # Save AI Trader Model
            self.ai_trader.model.save(os.path.join(models_dir, 'ai_trader_model.h5'))
            
            logger.info("Models saved successfully")
            
        except Exception as e:
            logger.error(f"Error saving models: {str(e)}")
            raise

    def load_models(self):
        """Load saved models"""
        try:
            models_dir = 'models'
            
            # Load Technical Analysis Model
            self.tech_model.model = tf.keras.models.load_model(
                os.path.join(models_dir, 'technical_model.h5')
            )
            
            # Load AI Trader Model
            self.ai_trader.model = tf.keras.models.load_model(
                os.path.join(models_dir, 'ai_trader_model.h5')
            )
            
            logger.info("Models loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading models: {str(e)}")
            raise

def main():
    """Main training script"""
    try:
        # Set up logging
        log_dir = os.path.join('logs', datetime.now().strftime('%Y-%m-%d'))
        os.makedirs(log_dir, exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(os.path.join(log_dir, 'training.log')),
                logging.StreamHandler(sys.stdout)
            ]
        )
        
        logger.info("Starting model training pipeline")
        
        # Initialize trainer
        trainer = ModelTrainer()
        
        # Collect and prepare data
        data = trainer.collect_training_data()
        
        # Train models
        metrics = trainer.train_models(data)
        
        logger.info("Training completed successfully")
        logger.info(f"Final Metrics: {metrics}")
        
    except Exception as e:
        logger.error(f"Training pipeline failed: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
