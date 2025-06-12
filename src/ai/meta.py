import optuna
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import logging
import threading

logger = logging.getLogger('ai_meta')

class SingletonMeta(type):
    """Thread-safe implementation of the Singleton pattern"""
    _instances = {}
    _lock = threading.RLock()  # Reentrant lock for thread safety

    def __call__(cls, *args, **kwargs):
        with cls._lock:
            if cls not in cls._instances:
                instance = super().__call__(*args, **kwargs)
                cls._instances[cls] = instance
            return cls._instances[cls]

    @classmethod
    def clear_instance(cls, target_class):
        """Clear an instance from the registry"""
        with cls._lock:
            if target_class in cls._instances:
                del cls._instances[target_class]

def optimize_strategy(train_func, param_space, n_trials=20):
    """Use Optuna to optimize strategy/model hyperparameters."""
    def objective(trial):
        params = {k: trial.suggest_float(k, *v) if isinstance(v, tuple) else trial.suggest_categorical(k, v) for k, v in param_space.items()}
        score = train_func(**params)
        return score
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials)
    logger.info(f'Best params: {study.best_params}, Best score: {study.best_value}')
    return study.best_params

def detect_market_regime(data, n_clusters=3):
    """Detect market regime using KMeans clustering on returns/volatility.
    Returns integer values representing different market regimes (0 to n_clusters-1)."""
    if data is None or data.empty:
        logger.error("No data provided for market regime detection")
        return np.array([], dtype=np.int32)

    try:
        scaler = StandardScaler()
        
        # Check for required columns with case-insensitive matching
        close_col = next((col for col in data.columns if col.lower() == 'close'), None)
        volume_col = next((col for col in data.columns if col.lower() == 'volume'), None)
        atr_col = next((col for col in data.columns if col.lower() == 'atr'), None)
        
        if not close_col:
            logger.error("Close price column not found in data")
            return np.zeros(len(data), dtype=np.int32)  # Return neutral regime
            
        features = []
        
        # Calculate returns using close prices
        returns = data[close_col].pct_change().fillna(0).values
        features.append(returns)
        
        # Add volume changes if available
        if volume_col:
            volume_changes = data[volume_col].pct_change().fillna(0).values
            features.append(volume_changes)
        
        # Add ATR if available
        if atr_col:
            atr = data[atr_col].fillna(0).values
            features.append(atr)
        
        features = np.column_stack(features)
        features = scaler.fit_transform(features)
          # Ensure consistent integer types
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        regimes = kmeans.fit_predict(features)
        regimes = regimes.astype(np.int32)
        unique_regimes, regime_counts = np.unique(regimes, return_counts=True)
        logger.info(f"Detected regimes (id: count): {dict(zip(unique_regimes, regime_counts))}")
        
        return regimes
        
    except Exception as e:
        logger.error(f"Error detecting market regime: {str(e)}")
        return np.zeros(len(data), dtype=np.int32)  # Return neutral regime

def select_strategy_by_regime(regime, strategies):
    """Select or weight strategies based on detected regime."""
    # Example: simple mapping, can be replaced with learned meta-policy
    if regime == 0:
        return strategies['trend']
    elif regime == 1:
        return strategies['mean_reversion']
    else:
        return strategies['ensemble']
