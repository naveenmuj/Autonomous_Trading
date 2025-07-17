import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from src.data.collector import DataCollector
import ta
from .analysis import TechnicalAnalyzer

# Lazy import of tensorflow to prevent initialization issues
def get_tf():
    try:
        import tensorflow as tf
        return tf
    except ImportError as e:
        logger.warning(f"TensorFlow import failed: {e}. AI predictions will be disabled.")
        return None

logger = logging.getLogger(__name__)

class EnhancedTradingStrategy:
    def __init__(self, config: Dict[str, Any], models: Optional[Dict[str, Any]] = None, collector: Optional['DataCollector'] = None):
        self.config = config
        self.collector = collector
        self.tech_analyzer = TechnicalAnalyzer()
        self.trend_lines = {'support': [], 'resistance': []}
        self.last_analysis_time = None
        self.analysis_interval = 60  # Minutes between trend line reanalysis
        
        # Load strategy parameters
        self.strategy_config = config.get('trading', {}).get('strategy', {})
        # --- LOWER THRESHOLDS FOR DEBUGGING ---
        # Lower signal and confidence thresholds to ensure more trades for debugging
        self.strategy_config['signal_threshold'] = 0.1
        self.strategy_config['confidence_threshold'] = 0.5
        
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
            if 'symbol' not in data.columns:
                logger.error("Input data for generate_signals is missing 'symbol' column. Returning empty signals.")
                return pd.DataFrame(index=data.index)
            signals = pd.DataFrame(index=data.index)
            primary_symbol = data['symbol'].iloc[0] if not data['symbol'].empty else None
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
            # Set main 'signal' column for the primary symbol
            if primary_symbol and f'{primary_symbol}_final' in signals.columns:
                signals['signal'] = signals[f'{primary_symbol}_final']
            else:
                logger.warning("No final signal column found for primary symbol. All signals set to 0.")
                signals['signal'] = 0
            # Log signal distribution
            logger.info(f"Signal distribution: {signals['signal'].value_counts().to_dict()}")
            # If all signals are zero, log and halt (no forced debug signals)
            if signals['signal'].abs().sum() == 0:
                logger.error("All signals are zero. No trades will be executed. Check indicator thresholds and data quality.")
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
                rsi_signal = np.where(
                    data['rsi'] < oversold, 1,  # Oversold - Buy
                    np.where(data['rsi'] > overbought, -1, 0)  # Overbought - Sell
                )
                signals[f'{symbol}_rsi'] = pd.Series(rsi_signal, index=data.index).reindex(signals.index)
            # MACD Signals
            if all(x in data.columns for x in ['macd', 'macd_signal']):
                macd_signal = np.where(
                    data['macd'] > data['macd_signal'], 1, -1
                )
                signals[f'{symbol}_macd'] = pd.Series(macd_signal, index=data.index).reindex(signals.index)
            # Bollinger Bands Signals
            if all(x in data.columns for x in ['bb_upper', 'bb_middle', 'bb_lower']):
                bb_signal = np.where(
                    data['close'] < data['bb_lower'], 1,  # Price below lower band - Buy
                    np.where(data['close'] > data['bb_upper'], -1, 0)  # Price above upper band - Sell
                )
                signals[f'{symbol}_bb'] = pd.Series(bb_signal, index=data.index).reindex(signals.index)
            # Combined Technical Signal (majority vote, not mean)
            tech_columns = [col for col in signals.columns if col.startswith(f'{symbol}_') and col not in [f'{symbol}_technical', f'{symbol}_ai', f'{symbol}_final']]
            if tech_columns:
                tech_signals = signals[tech_columns].reindex(signals.index).fillna(0)
                # Majority vote: if sum > 0, buy; < 0, sell; else hold
                tech_sum = tech_signals.sum(axis=1)
                tech_majority = tech_sum.apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))
                signals[f'{symbol}_technical'] = tech_majority
                logger.info(f"{symbol} technical signals (majority): {tech_majority.value_counts().to_dict()}")
            return signals
        except Exception as e:
            logger.error(f"Error generating technical signals: {str(e)}")
            raise
    def _generate_ai_signals(self, data: pd.DataFrame, signals: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """Generate signals based on AI model predictions"""
        try:
            if symbol not in self.models:
                logger.warning(f"No model found for symbol {symbol}")
                signals[f'{symbol}_ai'] = pd.Series(0, index=data.index).reindex(signals.index)
                return signals
            tf = get_tf()
            if tf is None:
                logger.warning("TensorFlow not available. Skipping AI predictions.")
                signals[f'{symbol}_ai'] = pd.Series(0, index=data.index).reindex(signals.index)
                return signals
            model = self.models[symbol]
            features = self._prepare_features(data)
            # Check if the model is a TensorFlow model and trained
            if hasattr(model, 'predict'):
                # Check for a 'trained' attribute or similar, or try/except predict
                try:
                    predictions = model.predict(features)
                except Exception as e:
                    logger.error(f"Model for {symbol} is not trained or failed to predict: {e}")
                    signals[f'{symbol}_ai'] = pd.Series(0, index=data.index).reindex(signals.index)
                    return signals
                threshold = self.strategy_config.get('confidence_threshold', 0.6)
                ai_signal = np.where(
                    predictions > threshold, 1,
                    np.where(predictions < -threshold, -1, 0)
                )
                signals[f'{symbol}_ai'] = pd.Series(ai_signal.flatten(), index=data.index).reindex(signals.index)
                logger.info(f"{symbol} AI predictions: {predictions[-5:].flatten()} | AI signals: {signals[f'{symbol}_ai'].iloc[-5:].values}")
            else:
                logger.warning(f"Model for {symbol} is not a valid TensorFlow model")
                signals[f'{symbol}_ai'] = pd.Series(0, index=data.index).reindex(signals.index)
            return signals
        except Exception as e:
            logger.error(f"Error generating AI signals: {str(e)}")
            signals[f'{symbol}_ai'] = pd.Series(0, index=data.index).reindex(signals.index)
            return signals

    def _combine_signals(self, technical_signals: Optional[pd.Series] = None, 
                        ai_signals: Optional[pd.Series] = None) -> pd.Series:
        """Combine technical and AI signals with configurable weights and validation"""
        try:
            weights = self.strategy_config.get('signal_weights', {})
            tech_weight = weights.get('technical', 0.5)
            ai_weight = weights.get('ai', 0.5)
            # Initialize with neutral signal
            if technical_signals is None and ai_signals is None:
                return pd.Series(0, index=technical_signals.index if technical_signals is not None else ai_signals.index)
            # Normalize weights
            total_weight = (tech_weight if technical_signals is not None else 0) + \
                         (ai_weight if ai_signals is not None else 0)
            if total_weight == 0:
                return pd.Series(0, index=technical_signals.index if technical_signals is not None else ai_signals.index)
            tech_weight = tech_weight / total_weight if technical_signals is not None else 0
            ai_weight = ai_weight / total_weight if ai_signals is not None else 0
            # Combine signals
            base_index = technical_signals.index if technical_signals is not None else ai_signals.index
            combined = pd.Series(0, index=base_index)
            if technical_signals is not None:
                combined = combined.add(technical_signals.reindex(base_index).fillna(0) * tech_weight, fill_value=0)
            if ai_signals is not None:
                combined = combined.add(ai_signals.reindex(base_index).fillna(0) * ai_weight, fill_value=0)
            return combined
        except Exception as e:
            logger.error(f"Error combining signals: {str(e)}")
            raise

    def _prepare_features(self, data: pd.DataFrame) -> np.ndarray:
        """Prepare features for AI model prediction with fallback calculations"""
        try:
            # Default technical features to calculate if missing
            default_features = {
                'rsi': lambda df: ta.momentum.RSIIndicator(df['close']).rsi(),
                'macd': lambda df: ta.trend.MACD(df['close']).macd(),
                'macd_signal': lambda df: ta.trend.MACD(df['close']).macd_signal(),
                'bb_upper': lambda df: ta.volatility.BollingerBands(df['close']).bollinger_hband(),
                'bb_middle': lambda df: ta.volatility.BollingerBands(df['close']).bollinger_mavg(),
                'bb_lower': lambda df: ta.volatility.BollingerBands(df['close']).bollinger_lband(),
                'volume_sma': lambda df: df['volume'].rolling(window=20, min_periods=1).mean()  # 20-period SMA
            }
            
            # Get configured features or use defaults
            feature_cols = self.strategy_config.get('model', {}).get('features', list(default_features.keys()))
            
            # Create working copy of data
            df = data.copy()
            
            # Calculate missing features
            for feature in feature_cols:
                if feature not in df.columns and feature in default_features:
                    logger.info(f"Calculating missing feature: {feature}")
                    try:
                        df[feature] = default_features[feature](df)
                    except Exception as e:
                        logger.warning(f"Could not calculate {feature}: {str(e)}")
                        df[feature] = 0  # Use safe default
            
            # Handle any remaining missing features with safe defaults
            for feature in feature_cols:
                if feature not in df.columns:
                    logger.warning(f"Feature {feature} not available and cannot be calculated")
                    df[feature] = 0
            
            # Ensure numeric types and handle NaN values
            df[feature_cols] = df[feature_cols].astype(float).fillna(0)
            
            return df[feature_cols].values
            
        except Exception as e:
            logger.error(f"Error preparing features: {str(e)}")
            # Return safe default features if error occurs
            return np.zeros((len(data), len(feature_cols)))

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

    def get_daily_pnl(self) -> float:
        """Calculate daily profit and loss"""
        try:
            if not self.collector or not hasattr(self.collector, 'positions'):
                return 0.0
            
            total_pnl = 0.0
            today = datetime.now().date()
            
            for position in self.collector.positions.values():
                # Only consider today's trades
                if position['entry_date'].date() == today:
                    if position['exit_price']:
                        pnl = (position['exit_price'] - position['entry_price']) * position['quantity']
                        if position['side'] == 'SELL':
                            pnl = -pnl
                        total_pnl += pnl
                        
            return round(total_pnl, 2)
        except Exception as e:
            logger.error(f"Error calculating daily PnL: {str(e)}")
            return 0.0

    def get_portfolio_value(self) -> float:
        """Calculate current portfolio value"""
        try:
            if not self.collector or not hasattr(self.collector, 'positions'):
                return 0.0
            
            total_value = 0.0
            for position in self.collector.positions.values():
                if not position['exit_price']:  # Only consider open positions
                    current_price = self.collector.get_last_price(position['symbol'])
                    if current_price:
                        position_value = current_price * position['quantity']
                        total_value += position_value
                        
            return round(total_value, 2)
        except Exception as e:
            logger.error(f"Error calculating portfolio value: {str(e)}")
            return 0.0

    def get_open_positions(self) -> Dict[str, Any]:
        """Get current open positions"""
        try:
            if not self.collector or not hasattr(self.collector, 'positions'):
                return {}
            
            open_positions = {}
            for symbol, position in self.collector.positions.items():
                if not position['exit_price']:  # Position is still open
                    current_price = self.collector.get_last_price(symbol)
                    if current_price:
                        pnl = (current_price - position['entry_price']) * position['quantity']
                        if position['side'] == 'SELL':
                            pnl = -pnl
                            
                        open_positions[symbol] = {
                            'quantity': position['quantity'],
                            'entry_price': position['entry_price'],
                            'current_price': current_price,
                            'side': position['side'],
                            'pnl': round(pnl, 2),
                            'pnl_percentage': round((pnl / (position['entry_price'] * position['quantity'])) * 100, 2)
                        }
            
            return open_positions
        except Exception as e:
            logger.error(f"Error getting open positions: {str(e)}")
            return {}

    def analyze_market(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Perform comprehensive market analysis"""
        try:
            current_time = datetime.now()
            
            # Update trend lines periodically
            if (self.last_analysis_time is None or 
                (current_time - self.last_analysis_time).total_seconds() / 60 >= self.analysis_interval):
                self.trend_lines = self.tech_analyzer.detect_trend_lines(data)
                self.last_analysis_time = current_time
            
            # Analyze potential breakouts
            breakouts = self.tech_analyzer.analyze_breakouts(data, self.trend_lines)
            
            # Get trading signals with enhanced analysis
            signals = self._generate_signals(data, breakouts)
            
            # Calculate risk levels
            risk_levels = self._calculate_risk_levels(data, signals)
            
            return {
                'signals': signals,
                'trend_lines': self.trend_lines,
                'breakouts': breakouts,
                'risk_levels': risk_levels
            }
            
        except Exception as e:
            logger.error(f"Error in market analysis: {str(e)}")
            return {
                'signals': pd.DataFrame(),
                'trend_lines': {'support': [], 'resistance': []},
                'breakouts': {'support': [], 'resistance': []},
                'risk_levels': {}
            }

    def _generate_signals(self, data: pd.DataFrame, 
                         breakouts: Dict[str, List[Dict]]) -> pd.DataFrame:
        """Generate trading signals with trend line analysis"""
        try:
            df = data.copy()
            
            # Calculate basic signals from indicators
            signals = pd.DataFrame(index=df.index)
            signals['signal'] = 0  # 0: Hold, 1: Buy, -1: Sell
            
            # Analyze trend line breakouts
            for breakout in breakouts['resistance']:
                if breakout['strength'] > 0.02:  # 2% breakout
                    idx = df.index[-1]
                    signals.loc[idx, 'signal'] = 1  # Buy on resistance breakout
                    
            for breakout in breakouts['support']:
                if breakout['strength'] > 0.02:
                    idx = df.index[-1]
                    signals.loc[idx, 'signal'] = -1  # Sell on support breakdown
            
            # Add confidence scores
            signals['confidence'] = self._calculate_confidence(df, signals)
            
            return signals
            
        except Exception as e:
            logger.error(f"Error generating signals: {str(e)}")
            return pd.DataFrame()

    def _calculate_confidence(self, data: pd.DataFrame, 
                            signals: pd.DataFrame) -> pd.Series:
        """Calculate confidence scores for signals"""
        try:
            confidence = pd.Series(index=signals.index, data=0.5)
            
            for idx in signals.index:
                if signals.loc[idx, 'signal'] != 0:
                    # Base confidence on multiple factors
                    trend_strength = self._calculate_trend_strength(data.loc[:idx])
                    breakout_strength = self._calculate_breakout_strength(data.loc[:idx])
                    volume_confirm = self._check_volume_confirmation(data.loc[:idx])
                    
                    # Combine factors for final confidence score
                    confidence.loc[idx] = (trend_strength + breakout_strength + volume_confirm) / 3
            
            return confidence
            
        except Exception as e:
            logger.error(f"Error calculating confidence: {str(e)}")
            return pd.Series()

    def _calculate_trend_strength(self, data: pd.DataFrame) -> float:
        """Calculate overall trend strength"""
        try:
            # Use ADX for trend strength
            if 'adx' in data.columns:
                return min(data['adx'].iloc[-1] / 100, 1.0)
            return 0.5
            
        except Exception as e:
            logger.error(f"Error calculating trend strength: {str(e)}")
            return 0.5

    def _calculate_breakout_strength(self, data: pd.DataFrame) -> float:
        """Calculate breakout strength"""
        try:
            if len(self.trend_lines['support']) == 0 and len(self.trend_lines['resistance']) == 0:
                return 0.5
                
            # Find nearest trend line and calculate strength
            current_price = data['close'].iloc[-1]
            nearest_line = None
            min_distance = float('inf')
            
            for line_type in ['support', 'resistance']:
                for line in self.trend_lines[line_type]:
                    distance = abs(current_price - line['current_value'])
                    if distance < min_distance:
                        min_distance = distance
                        nearest_line = line
            
            if nearest_line:
                return nearest_line['strength']
            return 0.5
            
        except Exception as e:
            logger.error(f"Error calculating breakout strength: {str(e)}")
            return 0.5

    def _check_volume_confirmation(self, data: pd.DataFrame) -> float:
        """Check volume confirmation for breakouts"""
        try:
            if 'volume' not in data.columns:
                return 0.5
                
            # Compare recent volume to average
            recent_vol = data['volume'].iloc[-1]
            avg_vol = data['volume'].rolling(window=20).mean().iloc[-1]
            
            return min(recent_vol / avg_vol, 1.0)
            
        except Exception as e:
            logger.error(f"Error checking volume confirmation: {str(e)}")
            return 0.5

    def _calculate_risk_levels(self, data: pd.DataFrame, 
                             signals: pd.DataFrame) -> Dict[str, float]:
        """Calculate risk levels for position sizing"""
        try:
            current_price = data['close'].iloc[-1]
            
            # Find nearest support and resistance
            supports = [line['current_value'] for line in self.trend_lines['support']]
            resistances = [line['current_value'] for line in self.trend_lines['resistance']]
            
            if supports:
                nearest_support = max([s for s in supports if s < current_price], default=current_price * 0.95)
            else:
                nearest_support = current_price * 0.95
                
            if resistances:
                nearest_resistance = min([r for r in resistances if r > current_price], default=current_price * 1.05)
            else:
                nearest_resistance = current_price * 1.05
            
            return {
                'stop_loss': nearest_support,
                'take_profit': nearest_resistance,
                'risk_reward_ratio': (nearest_resistance - current_price) / (current_price - nearest_support)
            }
            
        except Exception as e:
            logger.error(f"Error calculating risk levels: {str(e)}")
            return {
                'stop_loss': 0,
                'take_profit': 0,
                'risk_reward_ratio': 0
            }

    def calculate_stop_loss(self, entry_price: float, data: Optional[pd.DataFrame] = None, method: str = 'fixed', **kwargs) -> float:
        """Calculate stop loss price. Supports 'fixed' (percentage) and 'atr' (Average True Range) methods."""
        try:
            if method == 'atr' and data is not None and 'atr' in data.columns:
                atr_mult = self.risk_config.get('atr_multiplier', 1.5)
                atr = data['atr'].iloc[-1]
                stop_loss = entry_price - atr_mult * atr
                logger.info(f"ATR-based stop loss: entry={entry_price}, atr={atr}, mult={atr_mult}, stop_loss={stop_loss}")
                return stop_loss
            # Default: fixed percentage
            pct = self.risk_config.get('stop_loss', 0.02)
            stop_loss = entry_price * (1 - pct)
            logger.info(f"Fixed stop loss: entry={entry_price}, pct={pct}, stop_loss={stop_loss}")
            return stop_loss
        except Exception as e:
            logger.error(f"Error in calculate_stop_loss: {e}")
            return entry_price * 0.98  # fallback 2% stop

    def run_paper_trading(self, data: pd.DataFrame, ai_model=None):
        """Run paper trading simulation using this strategy and (optionally) an AI model."""
        from src.trading.paper_trading import PaperTradingEngine
        engine = PaperTradingEngine(initial_balance=100000)
        engine.reset()
        signals = self.generate_signals(data)
        for idx, row in data.iterrows():
            price = row['close']
            # Use AI model if provided, else use strategy signal
            if ai_model is not None:
                X, _ = ai_model.prepare_features(pd.DataFrame([row]))
                if X is not None:
                    pred = ai_model.model.predict(X)
                    signal = 1 if pred[0] > 0.5 else -1 if pred[0] < 0.5 else 0
                else:
                    signal = 0
            else:
                signal = signals.loc[idx, 'signal'] if 'signal' in signals.columns else 0
            engine.on_signal(signal, price, row['timestamp'] if 'timestamp' in row else None)
            logger.info(f"PaperTrade | Time: {row.get('timestamp', idx)} | Price: {price:.2f} | Signal: {signal} | Position: {engine.get_position()} | Balance: {engine.get_balance():.2f}")
        logger.info(f"Final Balance: {engine.get_balance():.2f}")
        logger.info("Trade Log:")
        logger.info(engine.get_trade_log())
        return engine
