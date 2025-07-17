import time
import logging
from src.ai.news_sentiment import NewsSentimentAnalyzer
from src.trading.strategy import EnhancedTradingStrategy
from src.ai.agent import TradingAgent

logger = logging.getLogger("autonomous_trader")

class AutonomousTrader:
    def __init__(self, config, data_collector, trade_manager, technical_analyzer, ai_model=None):
        self.config = config
        self.data_collector = data_collector
        self.trade_manager = trade_manager
        self.technical_analyzer = technical_analyzer
        self.ai_model = ai_model
        self.sentiment_analyzer = NewsSentimentAnalyzer(config)
        # Fix: Always pass a dict of models to EnhancedTradingStrategy
        symbols = []
        if hasattr(data_collector, 'get_symbols_from_config'):
            try:
                symbols = data_collector.get_symbols_from_config()
            except Exception:
                symbols = []
        if not symbols:
            symbols = ['DEFAULT']
        models = {symbol: ai_model for symbol in symbols} if ai_model is not None else {}
        self.strategy = EnhancedTradingStrategy(config, models=models, collector=data_collector)
        self.rl_agent = TradingAgent(config)
        self.autonomous_mode = False
        self.last_reasoning = {}  # For dashboard explainability

    def set_autonomous_mode(self, enabled: bool):
        self.autonomous_mode = enabled
        logger.info(f"Autonomous mode set to {enabled}")

    def combine_signals(self, technical_signal, ml_signal, rl_signal, sentiment):
        """
        Combine all model signals (technical, ML, RL) and sentiment into a robust ensemble decision.
        Returns the final trade signal and a reasoning dict for explainability.
        """
        # Configurable weights
        weights = self.config.get('ensemble', {
            'technical': 0.3,
            'ml': 0.3,
            'rl': 0.2,
            'sentiment': 0.2
        })
        # Fallback if any signal is None
        signals = {
            'technical': technical_signal if technical_signal is not None else 0,
            'ml': ml_signal if ml_signal is not None else 0,
            'rl': rl_signal if rl_signal is not None else 0,
            'sentiment': sentiment.get('score', 0) if sentiment else 0
        }
        # Weighted sum
        ensemble_score = (
            weights['technical'] * signals['technical'] +
            weights['ml'] * signals['ml'] +
            weights['rl'] * signals['rl'] +
            weights['sentiment'] * signals['sentiment']
        )
        # Threshold for action
        threshold = self.config.get('ensemble', {}).get('threshold', 0.5)
        if ensemble_score > threshold:
            final_signal = 1
        elif ensemble_score < -threshold:
            final_signal = -1
        else:
            final_signal = 0
        # Reasoning for explainability
        reasoning = {
            'signals': signals,
            'weights': weights,
            'ensemble_score': ensemble_score,
            'threshold': threshold,
            'final_signal': final_signal
        }
        return final_signal, reasoning

    def check_portfolio_constraints(self, symbol, signal):
        """
        Check sector exposure, correlation, and Kelly criterion before allowing a trade.
        Returns (allowed: bool, details: dict)
        """
        details = {}
        allowed = True
        # Sector exposure
        sector = self.data_collector.get_sector(symbol) if hasattr(self.data_collector, 'get_sector') else None
        if sector and hasattr(self.trade_manager, 'positions'):
            sector_positions = [s for s in self.trade_manager.positions if self.data_collector.get_sector(s) == sector]
            max_sector = self.config.get('trading', {}).get('risk', {}).get('max_sector_exposure', 0.25)
            sector_count = len(sector_positions)
            total_count = len(self.trade_manager.positions)
            sector_ratio = sector_count / total_count if total_count > 0 else 0
            details['sector'] = {'sector': sector, 'sector_ratio': sector_ratio, 'max_sector': max_sector}
            if sector_ratio > max_sector:
                allowed = False
                details['sector_blocked'] = True
                logger.debug(f"{symbol} blocked by sector exposure: {details['sector']}")
        # Correlation
        if hasattr(self.data_collector, 'get_correlation'):
            corr = self.data_collector.get_correlation(symbol, list(self.trade_manager.positions.keys()))
            corr_thresh = self.config.get('trading', {}).get('risk', {}).get('correlation_threshold', 0.7)
            details['correlation'] = {'correlation': corr, 'threshold': corr_thresh}
            if corr is not None and abs(corr) > corr_thresh:
                allowed = False
                details['correlation_blocked'] = True
                logger.debug(f"{symbol} blocked by correlation: {details['correlation']}")
        # Kelly criterion
        if hasattr(self.trade_manager.risk_manager, 'kelly_fraction'):
            kelly = self.trade_manager.risk_manager.kelly_fraction
            details['kelly_fraction'] = kelly
            logger.debug(f"{symbol} Kelly fraction: {kelly}")
        details['allowed'] = allowed
        return allowed, details

    def sync_portfolio_from_broker(self):
        """
        Fetch live portfolio/holdings from Angel One and update trade_manager.positions.
        Assumes trade_manager has a method to update positions from broker data.
        """
        try:
            if hasattr(self.trade_manager, 'sync_with_broker'):
                self.trade_manager.sync_with_broker()
                logger.info("Portfolio synchronized with Angel One broker.")
            elif hasattr(self.data_collector, 'get_broker_portfolio'):
                broker_positions = self.data_collector.get_broker_portfolio()
                if broker_positions and hasattr(self.trade_manager, 'update_positions_from_broker'):
                    self.trade_manager.update_positions_from_broker(broker_positions)
                    logger.info("Portfolio updated from broker data.")
                else:
                    logger.warning("No broker portfolio data or update method not implemented.")
            else:
                logger.warning("No broker sync method available.")
        except Exception as e:
            logger.error(f"Error syncing portfolio from broker: {e}")

    def sync_broker_state(self):
        """
        Sync both broker portfolio (holdings) and trading balance before trading decisions using robust DataCollector methods.
        """
        try:
            logger.info("Syncing broker portfolio and balance before trading decision...")
            if hasattr(self.trade_manager, 'sync_with_broker'):
                self.trade_manager.sync_with_broker()
                logger.info("Portfolio synchronized with Angel One broker.")
            if hasattr(self.data_collector, 'get_broker_balance'):
                balance = self.data_collector.get_broker_balance()
                logger.info(f"Broker trading balance: {balance}")
        except Exception as e:
            logger.error(f"Error syncing broker state: {e}")

    def run(self):
        logger.info("Starting Autonomous Trading Loop...")
        sync_interval = self.config.get('autonomous', {}).get('portfolio_sync_interval', 300)
        last_sync = 0
        while self.autonomous_mode:
            try:
                now = time.time()
                if now - last_sync > sync_interval:
                    self.sync_portfolio_from_broker()
                    last_sync = now
                symbols = self.data_collector.get_symbols_from_config()
                logger.debug(f"Fetched symbols from config: {symbols}")
                for symbol in symbols:
                    # 1. Get latest market data
                    df = self.data_collector.get_historical_data(symbol, interval='1d', days=30)
                    if df is None or df.empty:
                        logger.debug(f"No data for symbol {symbol}, skipping.")
                        continue
                    logger.debug(f"Fetched historical data for {symbol}, shape: {df.shape}")
                    # Technical/ML/RL signals
                    signals = self.strategy.generate_signals(df)
                    logger.debug(f"Signals DataFrame columns for {symbol}: {signals.columns}")
                    technical_signal = signals[f'{symbol}_technical'].iloc[-1] if f'{symbol}_technical' in signals else 0
                    ml_signal = signals[f'{symbol}_ai'].iloc[-1] if f'{symbol}_ai' in signals else 0
                    logger.info(f"{symbol} | Technical signal: {technical_signal} | ML signal: {ml_signal}")
                    rl_signal = 0
                    if hasattr(self.rl_agent, 'get_signal'):
                        rl_signal = self.rl_agent.get_signal(df)
                        logger.info(f"{symbol} | RL agent signal: {rl_signal}")
                    else:
                        logger.debug(f"RL agent does not implement get_signal for {symbol}")
                    # 3. News sentiment (Gemini)
                    sentiment = self.sentiment_analyzer.analyze_symbol(symbol)
                    logger.info(f"{symbol} | News sentiment: {sentiment}")
                    # Combine all signals
                    final_decision, reasoning = self.combine_signals(technical_signal, ml_signal, rl_signal, sentiment)
                    # Advanced portfolio checks
                    allowed, portfolio_details = self.check_portfolio_constraints(symbol, final_decision)
                    reasoning['portfolio_checks'] = portfolio_details
                    logger.info(f"{symbol} | Signals: {reasoning['signals']} | Ensemble Score: {reasoning['ensemble_score']:.2f} | Final: {final_decision} | Portfolio OK: {allowed} | Sentiment: {sentiment}")
                    logger.debug(f"{symbol} | Reasoning: {reasoning}")
                    # Save reasoning for dashboard
                    self.last_reasoning[symbol] = reasoning
                    # 5. Place trade if needed
                    if final_decision != 0 and allowed:
                        logger.info(f"{symbol} | Executing trade: Signal={final_decision}, Sentiment={sentiment}")
                        self.execute_trade(symbol, final_decision, sentiment, reasoning)
                    elif final_decision != 0 and not allowed:
                        logger.info(f"Trade blocked by portfolio constraints for {symbol}: {portfolio_details}")
                # Sleep between cycles
                time.sleep(self.config.get('autonomous', {}).get('cycle_interval', 60))
            except Exception as e:
                logger.error(f"Error in autonomous trading loop: {e}")
                time.sleep(30)

    def execute_trade(self, symbol, signal, sentiment, reasoning=None):
        # signal: 1=buy, -1=sell
        logger.info(f"Executing trade: {symbol} | Signal: {signal} | Sentiment: {sentiment} | Reasoning: {reasoning}")
        # Example: Use trade_manager to place order
        if self.trade_manager:
            self.trade_manager.place_order(symbol, signal, sentiment=sentiment, reasoning=reasoning)
        else:
            logger.warning("Trade manager not available. Skipping order execution.")
