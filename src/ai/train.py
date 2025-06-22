import os
import sys
import yaml
import pandas as pd
from datetime import datetime, timedelta
from data.collector import DataCollector
from ai.models import TradingEnvironment, TechnicalAnalysisModel
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
import numpy as np
import logging
from ai.meta import optimize_strategy, detect_market_regime
import torch
import torch.nn as nn
import torch.optim as optim
import math
import tensorflow as tf  # Added import for TensorFlow

class TimeSeriesTransformer(nn.Module):
    def __init__(self, input_dim, d_model=64, nhead=4, num_layers=2, output_dim=2, dropout=0.1):
        super().__init__()
        
        # Input projection with layer norm
        self.input_proj = nn.Linear(input_dim, d_model)
        self.input_norm = nn.LayerNorm(d_model)  # Changed to LayerNorm
        self.input_dropout = nn.Dropout(dropout)
        
        # Position encoding with layer norm
        self.pos_encoder = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),  # Added LayerNorm
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # Transformer encoder with improved stability
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 2,  # Reduced from 4 to 2
            dropout=dropout,
            activation=nn.GELU(),  # Use GELU instance
            batch_first=True,
            norm_first=True  # Pre-norm architecture for better stability
        )
        
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            norm=nn.LayerNorm(d_model)
        )
        
        # Output layers with residual connection
        self.fc1 = nn.Linear(d_model, d_model * 2)
        self.fc1_norm = nn.BatchNorm1d(d_model * 2)
        self.fc1_act = nn.GELU()
        self.fc1_dropout = nn.Dropout(dropout)
        
        self.fc2 = nn.Linear(d_model * 2, output_dim)
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights with Xavier/Glorot initialization"""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(self, x):
        # x: (batch, seq_len, input_dim)
        batch_size, seq_len, _ = x.shape
        
        # Initial projection
        x = self.input_proj(x.view(-1, x.size(-1)))
        x = self.input_norm(x)
        x = self.input_dropout(x)
        x = x.view(batch_size, seq_len, -1)
        
        # Add positional encoding
        pos = torch.arange(seq_len, device=x.device).unsqueeze(0).unsqueeze(-1)
        pos = pos.expand(batch_size, -1, x.size(-1)).float()
        x = x + self.pos_encoder(pos)
        
        # Transformer layers (x is already batch_first)
        x = self.transformer(x)
        
        # Take the last sequence element
        x = x[:, -1]
        
        # Output layers with residual connection
        out = self.fc1(x)
        out = self.fc1_norm(out)
        out = self.fc1_act(out)
        out = self.fc1_dropout(out)
        
        return self.fc2(out)

def train_model(config_path='config.yaml'):
    logger = logging.getLogger('ai_train')
    logger.info("Starting model training pipeline...")
    # Load configuration
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    
    # Initialize data collector
    collector = DataCollector(config)    # Get training data for multiple symbols from manual_symbols in config
    # Use DataCollector's symbol selection logic (manual/auto)
    config['skip_indices'] = True
    symbols = collector.get_symbols_from_config()
    config['skip_indices'] = False  # Reset after use
    if not symbols:
        logger.warning("No symbols discovered, using default symbols")
        symbols = ['RELIANCE-EQ', 'TCS-EQ', 'HDFCBANK-EQ', 'INFY-EQ', 'ICICIBANK-EQ']
    # Remove indices (NIFTY50, BANKNIFTY, etc.) from training symbols
    indices = set(config.get('trading', {}).get('data', {}).get('indices', []))
    symbols = [s for s in symbols if s not in indices]
    if not symbols:
        logger.warning("No symbols discovered, using default symbols")
        symbols = ['RELIANCE-EQ', 'TCS-EQ', 'HDFCBANK-EQ', 'INFY-EQ', 'ICICIBANK-EQ']

    # Prepare data collection
    all_data = []
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365)  # 1 year of data
    
    # Select only numerical features for training
    numerical_features = ['open', 'high', 'low', 'close', 'volume', 'SMA_20', 'Daily_Return']

    for symbol in symbols:
        logger.info(f"Collecting data for {symbol}...")
        # Convert symbol format from SYMBOL-EQ to SYMBOL
        base_symbol = symbol.replace('-EQ', '')
        data = collector.get_historical_data(
            symbol=base_symbol,
            start_date=start_date,
            end_date=end_date,
            interval='1d'
        )
        if data is not None and not data.empty:
            logger.info(f"Collected {len(data)} samples for {symbol}")
            logger.info(f"Data columns: {data.columns.tolist()}")
            logger.info(f"Date range: {data.index.min()} to {data.index.max()}")
            all_data.append(data)
        else:
            logger.warning(f"No data collected for {symbol}")
    
    if not all_data:
        logger.error("No data collected for any symbol. Cannot proceed with training.")
        return
      # Combine all data and sort by date
    combined_data = pd.concat(all_data)
    combined_data.sort_index(inplace=True)
    logger.info(f"Combined data shape: {combined_data.shape}")
    logger.info(f"Final date range: {combined_data.index.min()} to {combined_data.index.max()}")
    logger.info(f"Columns in training data: {combined_data.columns.tolist()}")
    
    # Select only numerical features and handle missing values
    combined_data = combined_data[numerical_features].copy()
    combined_data.fillna(method='ffill', inplace=True)  # Forward fill missing values
    combined_data.fillna(method='bfill', inplace=True)  # Backward fill any remaining missing values
    
    # Market Regime Detection
    logger.info("Detecting market regimes for training data...")
    regimes = detect_market_regime(combined_data)
    # Convert regimes to int32 first
    regimes = regimes.astype(np.int32)
    regime_counts = np.bincount(regimes)
    logger.info(f"Regime distribution: {regime_counts}")

    # Create and configure the trading environment
    logger.info("Training RL (PPO) model...")
    env = DummyVecEnv([lambda: TradingEnvironment(combined_data)])
    
    # Configure model based on config settings
    model_config = config.get('model', {})
    model_params = {
        'policy': 'MlpPolicy',
        'learning_rate': model_config.get('training', {}).get('learning_rate', 0.0003),
        'batch_size': model_config.get('training', {}).get('batch_size', 64),
        'n_steps': 2048,
        'gamma': model_config.get('training', {}).get('gamma', 0.99),
        'verbose': 1
    }
    
    # Create and train the model
    rl_model = PPO(**model_params, env=env)
    total_timesteps = config['model']['training'].get('total_timesteps', 100000)
    rl_model.learn(total_timesteps=total_timesteps)
    rl_model.save("models/trading_model_rl")
    logger.info("RL model training completed and saved.")
      # LSTM Technical Analysis Model
    logger.info("Training LSTM technical analysis model...")
    ta_model = TechnicalAnalysisModel(config)
    
    # Prepare data with lookback window for LSTM
    lookback = 60  # Increased lookback window for better context
    X, y = ta_model.prepare_data(combined_data, lookback=lookback)
    logger.info(f"Prepared {len(X)} samples for LSTM model with shape {X.shape}")
    
    # Split data into train/validation sets
    train_size = int(0.8 * len(X))
    X_train, X_val = X[:train_size], X[train_size:]
    y_train, y_val = y[:train_size], y[train_size:]
    
    # Build LSTM model
    input_shape = (lookback, X.shape[-1])
    ta_model.build_model(input_shape=input_shape, model_type='lstm')
    
    # Calculate class weights if imbalanced
    class_weights = None
    if len(np.unique(y)) > 1:
        class_counts = np.bincount(y.astype(int))
        total = len(y)
        class_weights = {i: total / (len(np.unique(y)) * count) for i, count in enumerate(class_counts)}
        logger.info(f"Using class weights: {class_weights}")
    
    # Train with validation data and class weights
    history = ta_model.train_model(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=config.get('model', {}).get('training', {}).get('epochs', 100),
        batch_size=config.get('model', {}).get('training', {}).get('batch_size', 32),
        class_weights=class_weights
    )
    
    # Save the model
    ta_model.model.save("models/ta_lstm_model.h5")
    logger.info("LSTM technical analysis model trained and saved.")
    
    # Log training metrics
    final_loss = history.get('loss', [-1])[-1]
    final_acc = history.get('accuracy', [-1])[-1]
    logger.info(f"Final training loss: {final_loss:.4f}, accuracy: {final_acc:.4f}")
    
    # AutoML/Optuna for RL hyperparameters (example search space)
    def train_rl_with_params(learning_rate, gamma):
        env = DummyVecEnv([lambda: TradingEnvironment(combined_data)])
        model = PPO('MlpPolicy', env, learning_rate=learning_rate, gamma=gamma, verbose=0)
        model.learn(total_timesteps=10000)
        # Use mean reward as score (placeholder)
        return np.random.rand()  # Replace with real evaluation
    param_space = {'learning_rate': (1e-5, 1e-2), 'gamma': (0.8, 0.999)}
    best_params = optimize_strategy(train_rl_with_params, param_space, n_trials=5)
    logger.info(f"Best RL params from AutoML: {best_params}")

    # Online/Incremental Learning (placeholder for periodic retraining)
    logger.info("[Online Learning] Add hooks for periodic model updates with new data.")
    # Example: Save model checkpoints and update with new data every month/quarter    # Transformer Model for Time Series
    logger.info("Training Transformer time series model...")    
    
    # Set device and model parameters
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Get training parameters from config
    training_config = config.get('model', {}).get('training', {})
    batch_size = training_config.get('batch_size', 64)
    n_epochs = training_config.get('epochs', 20)
    logger.info(f"Using batch size: {batch_size}")

    # Prepare data for PyTorch (use same features as LSTM)
    X_torch = torch.tensor(X, dtype=torch.float32)
    y_one_hot = torch.zeros((len(y), 2), dtype=torch.float32)
    y_one_hot[range(len(y)), y] = 1

    # Split into train and validation sets
    train_size = int(0.8 * len(X_torch))
    indices = torch.randperm(len(X_torch))
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]
    
    # Create datasets and dataloaders
    train_dataset = torch.utils.data.TensorDataset(
        X_torch[train_indices],
        y_one_hot[train_indices]
    )
    val_dataset = torch.utils.data.TensorDataset(
        X_torch[val_indices],
        y_one_hot[val_indices]
    )
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True if torch.cuda.is_available() else False
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True if torch.cuda.is_available() else False
    )

    # Initialize model and move to device
    model = TimeSeriesTransformer(
        input_dim=X.shape[2],
        output_dim=2,
        d_model=128,
        nhead=8,
        num_layers=3,
        dropout=0.2
    ).to(device)

    # Set up loss, optimizer and scheduler
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(
        model.parameters(),
        lr=1e-4,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.01
    )

    # Learning rate scheduler with warmup
    total_steps = n_epochs * len(train_loader)
    warmup_steps = total_steps // 10

    def lr_lambda(step):
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Training loop with early stopping
    best_val_loss = float('inf')
    patience = 10
    patience_counter = 0
    max_grad_norm = 1.0

    for epoch in range(n_epochs):
        # Training phase
        model.train()
        total_train_loss = 0
        n_train_batches = 0

        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            output = model(batch_X)
            loss = criterion(output, batch_y)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            scheduler.step()
            
            total_train_loss += loss.item()
            n_train_batches += 1

        # Validation phase
        model.eval()
        total_val_loss = 0
        n_val_batches = 0

        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                output = model(batch_X)
                val_loss = criterion(output, batch_y)
                total_val_loss += val_loss.item()
                n_val_batches += 1

        # Calculate average losses and log progress
        avg_train_loss = total_train_loss / n_train_batches
        avg_val_loss = total_val_loss / n_val_batches

        logger.info(f"Epoch [{epoch+1}/{n_epochs}], "
                   f"Train Loss: {avg_train_loss:.4f}, "
                   f"Val Loss: {avg_val_loss:.4f}")

        # Early stopping check
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), "models/transformer_model_best.pt")
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            logger.info("Early stopping triggered")
            break

    # Save final model
    torch.save(model.state_dict(), "models/transformer_model_final.pt")
    logger.info("Transformer model training completed. Models saved.")
    
    logger.info("All models trained and saved.")

def setup_logging():
    """Set up logging configuration"""
    # Create logs directory structure
    log_date = datetime.now().strftime('%Y-%m-%d')
    log_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'logs', log_date)
    os.makedirs(log_dir, exist_ok=True)
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),  # Log to console
            logging.FileHandler(os.path.join(log_dir, 'training.log'))
        ]
    )
    return logging.getLogger(__name__)

if __name__ == "__main__":
    try:
        # Set up logging
        logger = setup_logging()
        logger.info("Starting model training process...")
        
        # Get the absolute path to config.yaml
        config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'config.yaml')
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found at {config_path}")
        
        logger.info(f"Using config file: {config_path}")
        train_model(config_path)
        logger.info("Model training completed successfully")
        
    except Exception as e:
        logger.error(f"Training failed: {str(e)}", exc_info=True)
        sys.exit(1)
