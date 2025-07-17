import os
import sys
import logging
from datetime import datetime
import time

# Set up logging at the very top, before any loggers are created or used
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

print("DEBUG: train_models.py started")
import os
print("DEBUG: Current working directory:", os.getcwd())

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
from src.ai import train_gemini_swing, train_sota_swing
import yaml
import smtplib

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
            # PATCH: Always pass config and data_collector to Backtester for TradeManager
            self.backtester = Backtester(self.strategy, config=self.config, data_collector=self.data_collector)
            
            # Create models directory if not exists
            os.makedirs('models', exist_ok=True)
            
            # SOTA data file path
            self.sota_data_path = 'data/swing_training_data.csv'
            
            logger.info("ModelTrainer initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing ModelTrainer: {str(e)}")
            raise
    def collect_training_data(self) -> pd.DataFrame:
        """Collect and prepare training data, always including pattern and technical features."""
        try:
            self.data_collector.config['skip_indices'] = True
            symbols = self.data_collector.get_symbols_from_config()
            self.data_collector.config['skip_indices'] = False
            if not symbols:
                logger.warning("No symbols discovered, using defaults")
                symbols = ['RELIANCE.NS', 'TCS.NS', 'INFY.NS']
            logger.info(f"Collecting data for symbols: {symbols}")
            # Collect historical data with pattern and indicators
            all_data = []
            for sym in symbols:
                df = self.data_collector.get_historical_data(sym, days=120)
                # Verification log: check required columns
                required_cols = ['open', 'high', 'low', 'close', 'volume']
                missing_cols = [col for col in required_cols if col not in df.columns]
                if not df.empty and not missing_cols:
                    # Add technical indicators if not present
                    if 'sma_10' not in df.columns:
                        try:
                            df = self.tech_model.calculate_indicators(df)
                        except Exception as e:
                            logger.warning(f"Could not calculate indicators for {sym}: {e}")
                    # 'pattern' column is already added by DataCollector.get_historical_data
                    # Ensure 'pattern' column exists in training data
                    if 'pattern' not in df.columns:
                        logger.warning("'pattern' column missing in training data. Adding default value.")
                        df['pattern'] = 'None'
                    df['symbol'] = sym
                    all_data.append(df)
                else:
                    logger.error(f"Missing or empty required columns {missing_cols} in collected data for {sym}. Failing data collection for this symbol.")
            if not all_data:
                logger.warning("No data collected for training.")
                return pd.DataFrame()
            data = pd.concat(all_data, axis=0)
            # Ensure 'pattern' column is present in the final DataFrame
            if 'pattern' not in data.columns:
                logger.warning("'pattern' column missing in final training data. Adding default value for all rows.")
                data['pattern'] = 'None'
            # Always include 'pattern' and indicator columns
            feature_cols = [col for col in data.columns if col not in ['target','returns','timestamp','symbol']]
            if 'pattern' not in feature_cols:
                feature_cols.append('pattern')
            logger.info(f"Training features used: {feature_cols}")
            return data
        except Exception as e:
            logger.error(f"Error in collect_training_data: {e}")
            return pd.DataFrame()
            start_date = end_date - timedelta(days=365)  # 1 year of data
            
            all_data = []
            for symbol in symbols:
                try:
                    df = self.data_collector.get_historical_data(
                        symbol=symbol,
                        days=365
                    )
                    
                    # Add 1 second delay to avoid API rate limit
                    time.sleep(1)
                    
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
            
            # Ensure 'pattern' column is present before saving to CSV
            if 'pattern' not in combined_data.columns:
                logger.warning("'pattern' column missing before saving to CSV. Adding default value for all rows.")
                combined_data['pattern'] = 'None'
            # Save to standard file for SOTA pipeline
            combined_data.to_csv(self.sota_data_path, index=False)
            logger.info(f"[AUTOMATION] SOTA training data prepared and saved to {self.sota_data_path}")
            
            return combined_data
            
        except Exception as e:
            logger.error(f"Error collecting training data: {str(e)}")
            raise

    def train_models(self, data: pd.DataFrame) -> Dict[str, float]:
        """Train all models with collected data"""
        try:
            logger.info("Starting model training")
            metrics = {}

            # Ensure 'pattern' column is present in data
            if 'pattern' not in data.columns:
                logger.warning("'pattern' column missing in input data. Adding default value.")
                data['pattern'] = 'None'

            # Split data into train/validation/test sets
            train_size = int(len(data) * 0.7)
            val_size = int(len(data) * 0.15)

            train_data = data[:train_size].copy()
            val_data = data[train_size:train_size + val_size].copy()
            test_data = data[train_size + val_size:].copy()

            # Ensure 'pattern' column is present in all splits
            for split_name, split_df in zip(['train', 'val', 'test'], [train_data, val_data, test_data]):
                if 'pattern' not in split_df.columns:
                    logger.warning(f"'pattern' column missing in {split_name}_data. Adding default value.")
                    split_df['pattern'] = 'None'

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

def load_automation_config():
    config_path = os.path.abspath('config.yaml')
    print(f"[DEBUG] Loading config from: {config_path}")
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    print(f"[DEBUG] FULL CONFIG: {config}")
    print(f"[DEBUG] ROOT KEYS: {list(config.keys()) if config else 'None'}")
    print(f"[DEBUG] automation config loaded: {config.get('automation', {})}")
    return config.get('automation', {})

def log_action(action):
    logger.info(f"[AUTOMATION] {action}")

def send_error_notification(subject, message):
    config = load_automation_config().get('error_notification', {})
    if not config.get('enabled', False):
        return
    # Example: email notification (expand for slack/telegram)
    try:
        recipients = config.get('recipients', [])
        if not recipients:
            return
        # Dummy SMTP logic for demonstration
        log_action(f"Would send error notification: {subject} to {recipients}")
    except Exception as e:
        logger.error(f"Failed to send error notification: {e}")

def check_retraining_schedule():
    retrain_cfg = load_automation_config().get('retraining', {})
    if not retrain_cfg.get('enabled', False):
        log_action("Retraining disabled by config.")
        return False
    # Frequency logic (daily/weekly/monthly)
    freq = retrain_cfg.get('frequency', 'daily')
    last_trained = retrain_cfg.get('last_trained')
    log_action(f"DEBUG: last_trained value: {last_trained} (type: {type(last_trained)})")
    now = datetime.now()
    # Treat None, 'null', or missing as not set
    if not last_trained or last_trained == 'null':
        log_action("Retraining is due (no last_trained set).")
        return True
    try:
        last_trained_dt = datetime.fromisoformat(last_trained)
        if freq == 'daily' and (now - last_trained_dt).days < 1:
            log_action("Retraining not due yet (daily schedule).")
            return False
        if freq == 'weekly' and (now - last_trained_dt).days < 7:
            log_action("Retraining not due yet (weekly schedule).")
            return False
        if freq == 'monthly' and (now - last_trained_dt).days < 28:
            log_action("Retraining not due yet (monthly schedule).")
            return False
    except Exception as e:
        log_action(f"DEBUG: Error parsing last_trained: {e}. Forcing retraining.")
        return True
    log_action("Retraining is due.")
    return True

def update_last_trained():
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    config['automation']['retraining']['last_trained'] = datetime.now().isoformat()
    with open('config.yaml', 'w') as f:
        yaml.safe_dump(config, f)
    log_action("Updated last_trained timestamp in config.")

def run_data_ingestion():
    cfg = load_automation_config().get('data_ingestion', {})
    if not cfg.get('enabled', False):
        log_action("Data ingestion disabled by config.")
        return
    log_action(f"Running data ingestion (schedule: {cfg.get('schedule')})")
    # Place your data ingestion logic here

def run_model_evaluation():
    cfg = load_automation_config().get('model_evaluation', {})
    if not cfg.get('enabled', False):
        log_action("Model evaluation disabled by config.")
        return
    log_action(f"Running model evaluation (threshold: {cfg.get('performance_threshold')})")
    # Place your model evaluation logic here

def run_hyperparameter_tuning():
    cfg = load_automation_config().get('hyperparameter_tuning', {})
    if not cfg.get('enabled', False):
        log_action("Hyperparameter tuning disabled by config.")
        return
    log_action(f"Running hyperparameter tuning (method: {cfg.get('method')})")
    # Place your hyperparameter tuning logic here

def run_feature_engineering():
    cfg = load_automation_config().get('feature_engineering', {})
    if not cfg.get('enabled', False):
        log_action("Feature engineering disabled by config.")
        return
    log_action(f"Running feature engineering (LLM suggest: {cfg.get('llm_suggest')})")
    # Place your feature engineering logic here

def run_journaling():
    cfg = load_automation_config().get('journaling', {})
    if not cfg.get('enabled', False):
        log_action("Journaling disabled by config.")
        return
    log_action(f"Running journaling (auto_review: {cfg.get('auto_review')})")
    # Place your journaling logic here

def run_api_key_rotation():
    cfg = load_automation_config().get('api_key_rotation', {})
    if not cfg.get('enabled', False):
        log_action("API key rotation disabled by config.")
        return
    log_action(f"Checking API key expiry (alert {cfg.get('alert_days_before_expiry')} days before)")
    # Place your API key rotation logic here

def run_deployment():
    cfg = load_automation_config().get('deployment', {})
    if not cfg.get('enabled', False):
        log_action("Deployment disabled by config.")
        return
    log_action(f"Running deployment (auto_promote_best: {cfg.get('auto_promote_best')})")
    # Place your deployment logic here

def run_documentation():
    cfg = load_automation_config().get('documentation', {})
    if not cfg.get('enabled', False):
        log_action("Documentation update disabled by config.")
        return
    log_action(f"Running documentation update (auto_update: {cfg.get('auto_update')})")
    # Place your documentation update logic here

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

        # --- Manual override: always run if executed directly from terminal ---
        is_manual = False
        if hasattr(sys, 'argv') and len(sys.argv) > 0 and sys.argv[0].endswith('train_models.py'):
            is_manual = True

        # Automation pipeline
        if not is_manual and not check_retraining_schedule():
            return
        run_data_ingestion()
        run_model_evaluation()
        run_hyperparameter_tuning()
        run_feature_engineering()
        run_journaling()
        run_api_key_rotation()
        run_deployment()
        run_documentation()
        
        # --- Gemini training flag logic ---
        import yaml
        with open('config.yaml', 'r') as f:
            config = yaml.safe_load(f)
        gemini_cfg = config.get('apis', {}).get('gemini', {}) or config.get('gemini', {})
        enable_gemini = gemini_cfg.get('enable_training', True)
        if enable_gemini:
            logger.info("Gemini training flag is true. Running Gemini swing training.")
            train_gemini_swing.main()
        else:
            logger.info("Gemini training flag is false. Skipping Gemini swing training as per config.")
        
        # Initialize trainer
        trainer = ModelTrainer()
        # Collect and prepare data for SOTA pipeline
        trainer.collect_training_data()
        # Run SOTA swing trading pipeline (it will pick up the latest data file)
        train_sota_swing.main()
        
        # Collect and prepare data
        data = trainer.collect_training_data()
        
        # Train models
        metrics = trainer.train_models(data)
        
        logger.info("Training completed successfully")
        logger.info(f"Final Metrics: {metrics}")
        
        # After training and backtest, check metrics and trades
        # Use correct keys for metrics
        accuracy = metrics.get('tech_model_accuracy', 0)
        f1 = metrics.get('tech_model_f1', 0)
        win_rate = metrics.get('backtest_win_rate', 0)
        num_trades = metrics.get('backtest_total_trades', metrics.get('num_trades', 0))
        logger.info(f"Backtest metrics: accuracy={accuracy:.4f}, F1={f1:.4f}, win_rate={win_rate:.4f}, trades={num_trades}")
        if accuracy < 0.7 or f1 < 0.7 or win_rate < 0.55 or num_trades == 0:
            logger.error("Pipeline not ready for AI trading: metrics or trades below threshold.")
        else:
            logger.info("Pipeline is READY FOR AI TRADING!")
        
        update_last_trained()
        
    except Exception as e:
        logger.error(f"Training pipeline failed: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
