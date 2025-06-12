import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from src.data.collector import DataCollector
import tensorflow as tf
import ta

logger = logging.getLogger(__name__)

class EnhancedTradingStrategy:
    def __init__(self, config: Dict[str, Any], models: Optional[Dict[str, Any]] = None):
        self.config = config
        
        # Load strategy parameters
        self.strategy_config = config.get('trading', {}).get('strategy', {})
        
        # Load risk parameters
        self.risk_config = config.get('trading', {}).get('risk', {})
        
        # Technical parameters
        self.lookback_period = self.strategy_config.get('lookback_period', 20)
        self.profit_target = self.strategy_config.get('profit_target', 0.03)
        self.stop_loss = self.risk_config.get('stop_loss', 0.02)
        
        # Initialize models
        self.models = models or {}
        
        # Initialize performance metrics
        self._metrics = {
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'total_profit': 0.0,
            'max_drawdown': 0.0
        }
        
        logger.info("Enhanced Trading Strategy initialized with config parameters")

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate trading signals based on technical analysis and AI predictions"""
        try:
            signals = pd.DataFrame(index=data.index)
            
            for symbol in data['symbol'].unique():
                symbol_data = data[data['symbol'] == symbol].copy()
                
                # Technical Analysis Signals
                signals = self._generate_technical_signals(symbol_data, signals, symbol)
                
                # AI Model Predictions
                if self.models and symbol in self.models:
                    signals = self._generate_ai_signals(symbol_data, signals, symbol)
                
                # Combine Signals
                signals[f'{symbol}_final'] = self._combine_signals(
                    technical_signals=signals[f'{symbol}_technical'] if f'{symbol}_technical' in signals.columns else None,
                    ai_signals=signals[f'{symbol}_ai'] if f'{symbol}_ai' in signals.columns else None
                )
            
            return signals
            
        except Exception as e:
            logger.error(f"Error generating signals: {str(e)}")
            raise

    def _generate_technical_signals(self, data: pd.DataFrame, signals: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """Generate signals based on technical indicators"""
        try:
            # RSI Signals
            if 'rsi' in data.columns:
                oversold = self.strategy_config.get('indicators', {}).get('rsi', {}).get('oversold', 30)
                overbought = self.strategy_config.get('indicators', {}).get('rsi', {}).get('overbought', 70)
                
                signals[f'{symbol}_rsi'] = np.where(
                    data['rsi'] < oversold, 1,  # Oversold - Buy
                    np.where(data['rsi'] > overbought, -1, 0)  # Overbought - Sell
                )
            
            # MACD Signals
            if all(x in data.columns for x in ['macd', 'macd_signal']):
                signals[f'{symbol}_macd'] = np.where(
                    data['macd'] > data['macd_signal'], 1, -1
                )
            
            # Bollinger Bands Signals
            if all(x in data.columns for x in ['bb_upper', 'bb_middle', 'bb_lower']):
                signals[f'{symbol}_bb'] = np.where(
                    data['close'] < data['bb_lower'], 1,  # Price below lower band - Buy
                    np.where(data['close'] > data['bb_upper'], -1, 0)  # Price above upper band - Sell
                )
            
            # Combined Technical Signal
            tech_columns = [col for col in signals.columns if col.startswith(f'{symbol}_') and col != f'{symbol}_technical']
            if tech_columns:
                signals[f'{symbol}_technical'] = signals[tech_columns].mean(axis=1)
            
            return signals
            
        except Exception as e:
            logger.error(f"Error generating technical signals: {str(e)}")
            raise

    def _generate_ai_signals(self, data: pd.DataFrame, signals: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """Generate signals based on AI model predictions"""
        try:
            model = self.models[symbol]
            features = self._prepare_features(data)
            
            predictions = model.predict(features)
            threshold = self.strategy_config.get('confidence_threshold', 0.6)
            
            signals[f'{symbol}_ai'] = np.where(
                predictions > threshold, 1,
                np.where(predictions < -threshold, -1, 0)
            )
            
            return signals
            
        except Exception as e:
            logger.error(f"Error generating AI signals: {str(e)}")
            raise

    def _combine_signals(self, technical_signals: Optional[pd.Series] = None, 
                        ai_signals: Optional[pd.Series] = None) -> pd.Series:
        """Combine technical and AI signals with configurable weights"""
        try:
            weights = self.strategy_config.get('signal_weights', {})
            tech_weight = weights.get('technical', 0.5)
            ai_weight = weights.get('ai', 0.5)
            
            if technical_signals is not None and ai_signals is not None:
                return (technical_signals * tech_weight + ai_signals * ai_weight).round()
            elif technical_signals is not None:
                return technical_signals.round()
            elif ai_signals is not None:
                return ai_signals.round()
            else:
                return pd.Series(0, index=technical_signals.index if technical_signals is not None else ai_signals.index)
                
        except Exception as e:
            logger.error(f"Error combining signals: {str(e)}")
            raise

    def _prepare_features(self, data: pd.DataFrame) -> np.ndarray:
        """Prepare features for AI model prediction"""
        try:
            feature_cols = self.strategy_config.get('model', {}).get('features', [])
            if not feature_cols:
                feature_cols = ['rsi', 'macd', 'bb_upper', 'bb_lower', 'volume']
            
            # Ensure all required features exist
            missing_cols = [col for col in feature_cols if col not in data.columns]
            if missing_cols:
                raise ValueError(f"Missing required features: {missing_cols}")
            
            return data[feature_cols].values
            
        except Exception as e:
            logger.error(f"Error preparing features: {str(e)}")
            raise

    def validate_signals(self, signals: pd.DataFrame, data: pd.DataFrame) -> pd.DataFrame:
        """Validate and filter trading signals based on risk management rules"""
        try:
            validated = signals.copy()
            
            for symbol in data['symbol'].unique():
                # Risk checks
                symbol_data = data[data['symbol'] == symbol]
                
                # Position size check
                position_size = self._calculate_position_size(symbol_data['close'].iloc[-1])
                if position_size == 0:
                    validated[f'{symbol}_final'] = 0
                    continue
                
                # Volatility check
                if self._is_volatility_high(symbol_data):
                    validated[f'{symbol}_final'] *= 0.5  # Reduce position size for high volatility
                
                # Trading hours check
                if not self._is_valid_trading_time():
                    validated[f'{symbol}_final'] = 0
            
            return validated
            
        except Exception as e:
            logger.error(f"Error validating signals: {str(e)}")
            raise

    def _calculate_position_size(self, current_price: float) -> float:
        """Calculate position size based on risk management rules"""
        try:
            account_size = self.risk_config.get('account_size', 100000)
            risk_per_trade = self.risk_config.get('risk_per_trade', 0.01)
            
            max_position = account_size * risk_per_trade
            return min(max_position / current_price, max_position)
            
        except Exception as e:
            logger.error(f"Error calculating position size: {str(e)}")
            raise

    def _is_volatility_high(self, data: pd.DataFrame, window: int = 20) -> bool:
        """Check if current volatility is above threshold"""
        try:
            returns = data['close'].pct_change()
            current_vol = returns.rolling(window=window).std().iloc[-1]
            threshold = self.risk_config.get('volatility_threshold', 0.02)
            
            return current_vol > threshold
            
        except Exception as e:
            logger.error(f"Error checking volatility: {str(e)}")
            raise

    def _is_valid_trading_time(self) -> bool:
        """Check if current time is within valid trading hours"""
        try:
            now = datetime.now()
            
            # Default trading hours (IST)
            market_open = datetime.strptime('09:15', '%H:%M').time()
            market_close = datetime.strptime('15:30', '%H:%M').time()
            
            return market_open <= now.time() <= market_close
            
        except Exception as e:
            logger.error(f"Error checking trading time: {str(e)}")
            raise

    def update_metrics(self, trade_result: Dict[str, Any]):
        """Update strategy performance metrics"""
        try:
            self._metrics['total_trades'] += 1
            
            if trade_result.get('profit', 0) > 0:
                self._metrics['winning_trades'] += 1
            else:
                self._metrics['losing_trades'] += 1
                
            self._metrics['total_profit'] += trade_result.get('profit', 0)
            
            # Update max drawdown
            current_drawdown = trade_result.get('drawdown', 0)
            self._metrics['max_drawdown'] = min(
                self._metrics['max_drawdown'],
                current_drawdown
            )
            
        except Exception as e:
            logger.error(f"Error updating metrics: {str(e)}")
            raise

    def get_metrics(self) -> Dict[str, Any]:
        """Get current strategy performance metrics"""
        try:
            metrics = self._metrics.copy()
            
            # Calculate derived metrics
            total_trades = metrics['total_trades']
            if total_trades > 0:
                metrics['win_rate'] = metrics['winning_trades'] / total_trades
                metrics['profit_factor'] = abs(metrics['total_profit']) / abs(metrics['max_drawdown']) if metrics['max_drawdown'] != 0 else float('inf')
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error getting metrics: {str(e)}")
            raise
