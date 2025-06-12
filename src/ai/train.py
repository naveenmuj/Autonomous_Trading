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
    symbols = config.get('trading', {}).get('data', {}).get('manual_symbols', [])
    if not symbols:
        logger.warning("No manual symbols configured, using default symbols")
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
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Prepare data (use same features as LSTM)
    X_torch = torch.tensor(X, dtype=torch.float32)
    y_one_hot = torch.zeros((len(y), 2), dtype=torch.float32)
    y_one_hot[range(len(y)), y] = 1
      # Split into train and validation sets
    train_size = int(0.8 * len(X_torch))
    indices = torch.randperm(len(X_torch))
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]
    
    # Create TensorDatasets
    train_dataset = torch.utils.data.TensorDataset(
        X_torch[train_indices],
        y_one_hot[train_indices]
    )
    val_dataset = torch.utils.data.TensorDataset(
        X_torch[val_indices],
        y_one_hot[val_indices]
    )
    
    # Create DataLoaders
    batch_size = config.get('model', {}).get('training', {}).get('batch_size', 32)
    logger.info(f"Using batch size: {batch_size}")
    
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
    
    # Initialize model with improved architecture
    model = TimeSeriesTransformer(
        input_dim=X.shape[2],
        output_dim=2,
        d_model=128,  # Increased model capacity
        nhead=8,      # Increased number of attention heads
        num_layers=3, # Added one more transformer layer
        dropout=0.2
    )
    
    # Loss and optimizer with improved stability
    criterion = nn.BCEWithLogitsLoss(reduction='mean')
    optimizer = optim.AdamW(
        model.parameters(),
        lr=1e-4,  # Initial learning rate
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.01
    )
    
    # Learning rate scheduler with warmup and cosine decay
    total_steps = n_epochs * (len(train_dataset) // batch_size)
    warmup_steps = total_steps // 10  # 10% warmup
    
    def lr_lambda(step):
        if step < warmup_steps:
            # Linear warmup
            return float(step) / float(max(1, warmup_steps))
        else:
            # Cosine decay after warmup
            progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
            return 0.5 * (1.0 + math.cos(math.pi * progress))
            
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    # Training parameters
    max_grad_norm = 0.5  # For gradient clipping
    n_epochs = 20  
    batch_size = 32
    n_batches = len(X) // batch_size

    # Learning rate schedule parameters
    initial_learning_rate = 0.001
    decay_steps = 1000
    decay_rate = 0.9
    min_learning_rate = 1e-6
    warmup_steps = 100
    
    # Create a custom learning rate schedule that combines warmup and exponential decay
    class WarmupExponentialDecay(tf.keras.optimizers.schedules.LearningRateSchedule):
        def __init__(self, initial_lr, decay_steps, decay_rate, warmup_steps, min_lr):
            super().__init__()
            # Convert all inputs to float32 tensors
            self.initial_lr = tf.cast(initial_lr, tf.float32)
            self.decay_steps = tf.cast(decay_steps, tf.float32)
            self.decay_rate = tf.cast(decay_rate, tf.float32)
            self.warmup_steps = tf.cast(warmup_steps, tf.float32)
            self.min_lr = tf.cast(min_lr, tf.float32)

        def get_config(self):
            return {
                'initial_lr': float(self.initial_lr.numpy()),
                'decay_steps': float(self.decay_steps.numpy()),
                'decay_rate': float(self.decay_rate.numpy()),
                'warmup_steps': float(self.warmup_steps.numpy()),
                'min_lr': float(self.min_lr.numpy())
            }

        def __call__(self, step):
            # Convert step to tensor
            step = tf.cast(step, tf.float32)
            
            # Linear warmup phase
            warmup_pct = tf.minimum(step / self.warmup_steps, 1.0)
            warmup_lr = self.initial_lr * warmup_pct
            
            # Exponential decay phase
            decay_steps_remaining = tf.maximum(0.0, step - self.warmup_steps)
            decay_factor = tf.pow(self.decay_rate, decay_steps_remaining / self.decay_steps)
            decayed_lr = self.initial_lr * decay_factor
            
            # Choose between warmup and decay based on step
            final_lr = tf.cond(
                step < self.warmup_steps,
                lambda: warmup_lr,
                lambda: decayed_lr
            )
            
            # Apply minimum learning rate
            return tf.maximum(final_lr, self.min_lr)

    # Create the learning rate scheduler
    lr_schedule = WarmupExponentialDecay(
        initial_lr=initial_learning_rate,
        decay_steps=decay_steps,
        decay_rate=decay_rate,
        warmup_steps=warmup_steps,
        min_lr=min_learning_rate
    )
    
    # Create optimizer with learning rate schedule
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
    
    # Configure callbacks
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            'models/ta_lstm_model.h5',
            save_best_only=True,
            monitor='val_loss',
            mode='min',
            verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=1
        ),
        LRLogger()
    ]

    # Train model
    history = ta_model.train_model(
        X_train, y_train,
        validation_data=(X_val, y_val), 
        epochs=config.get('model', {}).get('training', {}).get('epochs', 100),
        batch_size=config.get('model', {}).get('training', {}).get('batch_size', 32),
        class_weights=class_weights,
        callbacks=callbacks
    )
    
    # Transformer Model for Time Series
    logger.info("Training Transformer time series model...")
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Prepare data (use same features as LSTM)
    X_torch = torch.tensor(X, dtype=torch.float32)
    y_one_hot = torch.zeros((len(y), 2), dtype=torch.float32)
    y_one_hot[range(len(y)), y] = 1
      # Split into train and validation sets
    train_size = int(0.8 * len(X_torch))
    indices = torch.randperm(len(X_torch))
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]
    
    # Create TensorDatasets
    train_dataset = torch.utils.data.TensorDataset(
        X_torch[train_indices],
        y_one_hot[train_indices]
    )
    val_dataset = torch.utils.data.TensorDataset(
        X_torch[val_indices],
        y_one_hot[val_indices]
    )
    
    # Create DataLoaders
    batch_size = config.get('model', {}).get('training', {}).get('batch_size', 32)
    logger.info(f"Using batch size: {batch_size}")
    
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
    
    # Initialize model with improved architecture
    model = TimeSeriesTransformer(
        input_dim=X.shape[2],
        output_dim=2,
        d_model=128,  # Increased model capacity
        nhead=8,      # Increased number of attention heads
        num_layers=3, # Added one more transformer layer
        dropout=0.2
    )
    
    # Loss and optimizer with improved stability
    criterion = nn.BCEWithLogitsLoss(reduction='mean')
    optimizer = optim.AdamW(
        model.parameters(),
        lr=1e-4,  # Initial learning rate
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.01
    )
    
    # Learning rate scheduler with warmup and cosine decay
    total_steps = n_epochs * (len(train_dataset) // batch_size)
    warmup_steps = total_steps // 10  # 10% warmup
    
    def lr_lambda(step):
        if step < warmup_steps:
            # Linear warmup
            return float(step) / float(max(1, warmup_steps))
        else:
            # Cosine decay after warmup
            progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
            return 0.5 * (1.0 + math.cos(math.pi * progress))
            
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    # Training parameters
    max_grad_norm = 0.5  # For gradient clipping
    n_epochs = 20  
    batch_size = 32
    n_batches = len(X) // batch_size

    # Learning rate schedule parameters
    initial_learning_rate = 0.001
    decay_steps = 1000
    decay_rate = 0.9
    min_learning_rate = 1e-6
    warmup_steps = 100
    
    # Create a custom learning rate schedule that combines warmup and exponential decay
    class WarmupExponentialDecay(tf.keras.optimizers.schedules.LearningRateSchedule):
        def __init__(self, initial_lr, decay_steps, decay_rate, warmup_steps, min_lr):
            super().__init__()
            # Convert all inputs to float32 tensors
            self.initial_lr = tf.cast(initial_lr, tf.float32)
            self.decay_steps = tf.cast(decay_steps, tf.float32)
            self.decay_rate = tf.cast(decay_rate, tf.float32)
            self.warmup_steps = tf.cast(warmup_steps, tf.float32)
            self.min_lr = tf.cast(min_lr, tf.float32)

        def get_config(self):
            return {
                'initial_lr': float(self.initial_lr.numpy()),
                'decay_steps': float(self.decay_steps.numpy()),
                'decay_rate': float(self.decay_rate.numpy()),
                'warmup_steps': float(self.warmup_steps.numpy()),
                'min_lr': float(self.min_lr.numpy())
            }

        def __call__(self, step):
            # Convert step to tensor
            step = tf.cast(step, tf.float32)
            
            # Linear warmup phase
            warmup_pct = tf.minimum(step / self.warmup_steps, 1.0)
            warmup_lr = self.initial_lr * warmup_pct
            
            # Exponential decay phase
            decay_steps_remaining = tf.maximum(0.0, step - self.warmup_steps)
            decay_factor = tf.pow(self.decay_rate, decay_steps_remaining / self.decay_steps)
            decayed_lr = self.initial_lr * decay_factor
            
            # Choose between warmup and decay based on step
            final_lr = tf.cond(
                step < self.warmup_steps,
                lambda: warmup_lr,
                lambda: decayed_lr
            )
            
            # Apply minimum learning rate
            return tf.maximum(final_lr, self.min_lr)

    # Create the learning rate scheduler
    lr_schedule = WarmupExponentialDecay(
        initial_lr=initial_learning_rate,
        decay_steps=decay_steps,
        decay_rate=decay_rate,
        warmup_steps=warmup_steps,
        min_lr=min_learning_rate
    )
    
    # Create optimizer with learning rate schedule
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
    
    # Add callbacks for monitoring learning rate and model checkpointing
    class LRLogger(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            logs = logs or {}
            # Cast optimizer iteration to tensor for learning rate schedule
            step = tf.cast(self.model.optimizer.iterations, tf.float32)
            # Get current learning rate directly from the schedule
            current_lr = self.model.optimizer.learning_rate(step)
            logs['learning_rate'] = float(current_lr.numpy())
            logger.info(f"Epoch {epoch}: Learning rate = {float(current_lr.numpy()):.6f}")
    
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            'models/best_model.h5',
            save_best_only=True,
            monitor='val_loss',
            mode='min'
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True
        ),
        LRLogger()
    ]
    
    # Compile the model with the new optimizer
    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy', 
        metrics=['accuracy']
    )
    
    # Training loop setup
    model.to(device)
    model.train()
    best_val_loss = float('inf')
    patience = 10
    patience_counter = 0
    
    for epoch in range(n_epochs):
        total_train_loss = 0
        total_val_loss = 0
        n_train_batches = 0
        n_val_batches = 0
        
        # Training
        model.train()
        for batch_X, batch_y in train_loader:
            # Move batch to device
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)
            
            # Forward pass
            optimizer.zero_grad()
            output = model(batch_X)
            loss = criterion(output, batch_y)            # Backward pass with gradient clipping
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            scheduler.step()
            
            total_train_loss += loss.item()
            n_train_batches += 1
        
        # Validation
        model.eval()
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                # Move batch to device
                batch_X = batch_X.to(device)
                batch_y = batch_y.to(device)
                
                output = model(batch_X)
                val_loss = criterion(output, batch_y)
                total_val_loss += val_loss.item()
                n_val_batches += 1
        
        # Calculate average losses
        avg_train_loss = total_train_loss / n_train_batches
        avg_val_loss = total_val_loss / n_val_batches
        
        # Log progress
        logger.info(f"Epoch [{epoch+1}/{n_epochs}], "
                   f"Train Loss: {avg_train_loss:.4f}, "
                   f"Val Loss: {avg_val_loss:.4f}")
        
        # Early stopping check
        if torch.isnan(torch.tensor(avg_train_loss)):
            logger.warning("NaN loss detected. Stopping training.")
            break
            
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), "models/transformer_model_best.pt")
            patience_counter = 0
        else:
            patience_counter += 1
            
        if patience_counter >= patience:
            logger.info("Early stopping triggered")
            break    # Load best model for final evaluation
    model.load_state_dict(torch.load("models/transformer_model_best.pt"))
    model.eval()
    
    total_val_loss = 0
    total_correct = 0
    total_samples = 0
    
    with torch.no_grad():
        for batch_X, batch_y in val_loader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)
            
            outputs = model(batch_X)
            val_loss = criterion(outputs, batch_y)
            total_val_loss += val_loss.item()
            
            preds = torch.argmax(outputs, dim=1)
            true_labels = torch.argmax(batch_y, dim=1)
            total_correct += (preds == true_labels).sum().item()
            total_samples += batch_y.size(0)
        
        final_val_loss = total_val_loss / len(val_loader)
        final_accuracy = total_correct / total_samples
        
        logger.info(f"Final Validation Loss: {final_val_loss:.4f}")
        logger.info(f"Final Validation Accuracy: {final_accuracy:.4f}")
    
    # Save both the best model and final model
    torch.save(model.state_dict(), "models/transformer_model_final.pt")
    logger.info("Transformer model training completed. Best and final models saved.")
    
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
