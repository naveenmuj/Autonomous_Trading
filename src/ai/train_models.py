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
        """Collect and prepare training data, skipping indices like NIFTY50/BANKNIFTY"""
        try:
            # Set skip_indices flag for DataCollector
            self.data_collector.config['skip_indices'] = True
            symbols = self.data_collector.get_symbols_from_config()
            self.data_collector.config['skip_indices'] = False  # Reset after use
            if not symbols:
                logger.warning("No symbols discovered, using defaults")
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
                        
                        # Add sentiment integration (robust, always present)
                        df = self.ai_trader.add_sentiment_to_market_data(symbol, df)
                        
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
            # --- FIX: Always match model type to data shape ---
            if len(X_train.shape) == 3:
                self.tech_model.build_model(X_train.shape[1:], model_type='lstm')
            else:
                self.tech_model.build_model(X_train.shape[1], model_type='dense')
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
            X_ai, y_ai = self.ai_trader.prepare_features(train_data)
            if X_ai is None or y_ai is None or len(X_ai) == 0 or len(y_ai) == 0:
                logger.error("AI Trader features or labels are empty. Skipping AI Trader training.")
                metrics['ai_trader_status'] = 'skipped'
            else:
                ai_train_result = self.ai_trader.train(X_ai, y_ai)
                if isinstance(ai_train_result, dict) and ai_train_result.get("status") == "error":
                    logger.error(f"AI Trader training failed: {ai_train_result.get('reason')}")
                    metrics['ai_trader_status'] = 'error'
                elif isinstance(ai_train_result, dict) and ai_train_result.get("status") == "skipped":
                    logger.warning(f"AI Trader training skipped: {ai_train_result.get('reason')}")
                    metrics['ai_trader_status'] = 'skipped'
                else:
                    metrics['ai_trader_status'] = 'success'

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

            # --- Robo trading readiness check ---
            min_acc = self.config.get('model', {}).get('min_accuracy', 0.7)
            min_f1 = self.config.get('model', {}).get('min_f1', 0.7)
            min_win = self.config.get('model', {}).get('min_win_rate', 0.55)
            ready = True
            if metrics['tech_model_accuracy'] < min_acc:
                logger.warning(f"Model accuracy {metrics['tech_model_accuracy']:.2f} below threshold {min_acc}")
                ready = False
            if metrics['tech_model_f1'] < min_f1:
                logger.warning(f"Model F1 {metrics['tech_model_f1']:.2f} below threshold {min_f1}")
                ready = False
            if metrics['backtest_win_rate'] < min_win:
                logger.warning(f"Backtest win rate {metrics['backtest_win_rate']:.2f} below threshold {min_win}")
                ready = False
            if ready:
                logger.info("MODEL IS READY FOR ROBO TRADING: All metrics above thresholds.")
            else:
                logger.error("MODEL IS NOT READY FOR ROBO TRADING: One or more metrics below threshold. Live trading will be blocked.")
            metrics['robo_trading_ready'] = ready
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
            import tensorflow as tf  # Fix for tf not defined
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
        
        # After training and backtest, check metrics and trades
        logger.info(f"Backtest metrics: accuracy={metrics['accuracy']:.4f}, F1={metrics['f1']:.4f}, win_rate={metrics['win_rate']:.4f}, trades={metrics['num_trades']}")
        if metrics['accuracy'] < 0.7 or metrics['f1'] < 0.7 or metrics['win_rate'] < 0.55 or metrics['num_trades'] == 0:
            logger.error("Pipeline not ready for AI trading: metrics or trades below threshold.")
        else:
            logger.info("Pipeline is READY FOR AI TRADING!")
        
    except Exception as e:
        logger.error(f"Training pipeline failed: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
