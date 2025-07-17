import os
import sys
from src.ai.train_models import ModelTrainer

def train_technical_model(trainer, data):
    print("Training Technical Analysis Model...")
    X_train, y_train = trainer.tech_model.prepare_data(data)
    if len(X_train.shape) == 3:
        trainer.tech_model.build_model(X_train.shape[1:], model_type='lstm')
    else:
        trainer.tech_model.build_model(X_train.shape[1], model_type='dense')
    trainer.tech_model.train_model(X_train, y_train, epochs=50, batch_size=32)
    print("Technical Analysis Model training complete.")

def train_ai_trader(trainer, data):
    print("Training AI Trader Model...")
    X_ai, y_ai = trainer.ai_trader.prepare_features(data)
    if X_ai is None or y_ai is None or len(X_ai) == 0 or len(y_ai) == 0:
        print("AI Trader features or labels are empty. Skipping AI Trader training.")
        return
    trainer.ai_trader.train(X_ai, y_ai)
    print("AI Trader Model training complete.")

def run_backtest(trainer, data):
    print("Running Backtest...")
    results = trainer.backtester.run(data)
    print("Backtest Results:", results)

def main():
    trainer = ModelTrainer()
    data = trainer.collect_training_data()
    if data.empty:
        print("No training data available.")
        sys.exit(1)
    train_technical_model(trainer, data)
    train_ai_trader(trainer, data)
    run_backtest(trainer, data)
    print("All model training steps completed.")

if __name__ == "__main__":
    main()
