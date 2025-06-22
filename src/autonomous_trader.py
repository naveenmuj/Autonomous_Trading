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
        self.strategy = EnhancedTradingStrategy(config, models=ai_model, collector=data_collector)
        self.rl_agent = TradingAgent(config)
        self.autonomous_mode = False

    def set_autonomous_mode(self, enabled: bool):
        self.autonomous_mode = enabled
        logger.info(f"Autonomous mode set to {enabled}")

    def run(self):
        logger.info("Starting Autonomous Trading Loop...")
        while self.autonomous_mode:
            try:
                symbols = self.data_collector.get_symbols_from_config()
                for symbol in symbols:
                    # 1. Get latest market data
                    df = self.data_collector.get_historical_data(symbol, interval='1d', days=30)
                    if df is None or df.empty:
                        continue
                    # 2. Technical/AI/RL signals
                    signals = self.strategy.generate_signals(df)
                    latest_signal = signals['signal'].iloc[-1] if 'signal' in signals.columns else 0
                    # 3. News sentiment (Gemini)
                    sentiment = self.sentiment_analyzer.analyze_symbol(symbol)
                    logger.info(f"{symbol} | Model Signal: {latest_signal} | News Sentiment: {sentiment}")
                    # 4. Combine signals (simple logic: require both to agree, or use weighted sum)
                    final_decision = self.combine_signals(latest_signal, sentiment)
                    # 5. Place trade if needed
                    if final_decision != 0:
                        self.execute_trade(symbol, final_decision, sentiment)
                # Sleep between cycles
                time.sleep(self.config.get('autonomous', {}).get('cycle_interval', 60))
            except Exception as e:
                logger.error(f"Error in autonomous trading loop: {e}")
                time.sleep(30)

    def combine_signals(self, model_signal, sentiment):
        # Example: Only trade if both model and sentiment agree, or use weighted sum
        sentiment_score = sentiment.get('score', 0)
        if model_signal == 0:
            return 0
        if abs(sentiment_score) < 0.2:
            return 0  # Require strong sentiment
        if (model_signal > 0 and sentiment_score > 0) or (model_signal < 0 and sentiment_score < 0):
            return model_signal
        return 0

    def execute_trade(self, symbol, signal, sentiment):
        # signal: 1=buy, -1=sell
        logger.info(f"Executing trade: {symbol} | Signal: {signal} | Sentiment: {sentiment}")
        # Example: Use trade_manager to place order
        if self.trade_manager:
            self.trade_manager.place_order(symbol, signal, sentiment=sentiment)
        else:
            logger.warning("Trade manager not available. Skipping order execution.")
