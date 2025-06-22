import logging
import pandas as pd
from src.trading.paper_trading import PaperTradingEngine

logger = logging.getLogger(__name__)

def run_paper_trading(data, strategy, ai_model=None):
    """Run paper trading simulation using strategy and (optionally) AI model."""
    engine = PaperTradingEngine(initial_balance=100000)
    engine.reset()
    signals = strategy.generate_signals(data)
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
