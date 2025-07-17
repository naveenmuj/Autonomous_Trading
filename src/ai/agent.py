import os
import logging
from typing import Dict, Any, Optional
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.vec_env import DummyVecEnv
from .models import TradingEnvironment

logger = logging.getLogger(__name__)

class TradingAgent:
    """Trading Agent using PPO for reinforcement learning"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.model_config = config.get('model', {})
        self.training_config = self.model_config.get('training', {})
        self.env = None
        self.model = None
        
    def create_env(self, data: np.ndarray, is_training: bool = True) -> DummyVecEnv:
        """Create the trading environment"""
        def make_env():
            return TradingEnvironment(data, self.config)
        return DummyVecEnv([make_env])
        
    def setup_callbacks(self, eval_env: Optional[DummyVecEnv] = None) -> list:
        """Setup training callbacks"""
        callbacks = []
        
        # Evaluation callback
        if eval_env is not None:
            eval_callback = EvalCallback(
                eval_env,
                best_model_save_path='models/',
                log_path='logs/',
                eval_freq=self.training_config.get('eval_freq', 10000),
                deterministic=True,
                render=False,
                n_eval_episodes=self.training_config.get('n_eval_episodes', 5)
            )
            callbacks.append(eval_callback)
        
        # Checkpoint callback
        checkpoint_callback = CheckpointCallback(
            save_freq=self.training_config.get('save_freq', 10000),
            save_path='models/',
            name_prefix='trading_model'
        )
        callbacks.append(checkpoint_callback)
        
        return callbacks
        
    def train(self, train_data: np.ndarray, eval_data: Optional[np.ndarray] = None) -> None:
        """Train the trading agent"""
        try:
            # Create environments
            self.env = self.create_env(train_data)
            eval_env = self.create_env(eval_data) if eval_data is not None else None
            
            # Get model parameters from config
            policy_kwargs = self.model_config.get('architecture', {}).get('policy_kwargs', {})
            if isinstance(policy_kwargs.get('net_arch', {}), dict):
                # Convert dict to list format expected by stable_baselines3
                pi_layers = policy_kwargs['net_arch'].get('pi', [64, 64])
                vf_layers = policy_kwargs['net_arch'].get('vf', [64, 64])
                policy_kwargs['net_arch'] = [dict(pi=pi_layers, vf=vf_layers)]
            
            # Initialize model
            self.model = PPO(
                policy=self.model_config.get('architecture', {}).get('policy', 'MlpPolicy'),
                env=self.env,
                learning_rate=self.training_config.get('learning_rate', 0.0003),
                n_steps=self.training_config.get('n_steps', 2048),
                batch_size=self.training_config.get('batch_size', 64),
                n_epochs=self.training_config.get('n_epochs', 10),
                gamma=self.training_config.get('gamma', 0.99),
                gae_lambda=self.training_config.get('gae_lambda', 0.95),
                clip_range=self.training_config.get('clip_range', 0.2),
                clip_range_vf=self.training_config.get('clip_range_vf', None),
                ent_coef=self.training_config.get('ent_coef', 0.01),
                max_grad_norm=self.training_config.get('max_grad_norm', 0.5),
                target_kl=self.training_config.get('target_kl', 0.015),
                tensorboard_log="logs/",
                policy_kwargs=policy_kwargs,
                device=self.training_config.get('device', 'auto'),
                verbose=1
            )
            
            # Setup callbacks
            callbacks = self.setup_callbacks(eval_env)
            
            # Train model
            self.model.learn(
                total_timesteps=self.training_config.get('total_timesteps', 100000),
                callback=callbacks,
                progress_bar=True
            )
            
            # Save final model
            self.model.save("models/trading_model_final")
            logger.info("Training completed successfully")
            
        except Exception as e:
            logger.error(f"Error training model: {str(e)}")
            raise
            
    def predict(self, state: np.ndarray, deterministic: bool = True) -> tuple:
        """Make a prediction for a given state"""
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        action, _states = self.model.predict(state, deterministic=deterministic)
        return action, _states
        
    def load(self, path: str) -> None:
        """Load a trained model"""
        try:
            self.model = PPO.load(path)
            logger.info(f"Model loaded from {path}")
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise
            
    def save(self, path: str) -> None:
        """Save the current model"""
        try:
            if self.model is not None:
                self.model.save(path)
                logger.info(f"Model saved to {path}")
            else:
                raise ValueError("No model to save")
        except Exception as e:
            logger.error(f"Error saving model: {str(e)}")
            raise
        
    def get_signal(self, df, deterministic=True):
        """
        Given a DataFrame of market data, prepare the latest state and return the RL agent's action as a signal.
        Returns: -1 (sell), 0 (hold), 1 (buy)
        """
        if self.model is None:
            logger.warning("RL model not loaded or trained. Returning hold signal (0).")
            return 0
        try:
            # Prepare the latest state (assume last row, drop non-numeric columns)
            if hasattr(df, 'select_dtypes'):
                state = df.select_dtypes(include=[np.number]).iloc[-1].values
                logger.debug(f"RL get_signal: Prepared state from DataFrame with shape {df.shape}, numeric state shape: {state.shape}")
            else:
                state = np.array(df.iloc[-1])
                logger.debug(f"RL get_signal: Used raw DataFrame row for state, shape: {state.shape if hasattr(state, 'shape') else type(state)}")
            action, _ = self.predict(state.reshape(1, -1), deterministic=deterministic)
            # Map action to signal: 0=hold, 1=buy, 2=sell (adjust if your RL env uses different mapping)
            if isinstance(action, (np.ndarray, list)):
                action = int(action[0])
            if action == 1:
                signal = 1  # buy
            elif action == 2:
                signal = -1  # sell
            else:
                signal = 0  # hold
            logger.info(f"RL get_signal: action={action}, mapped signal={signal}")
            return signal
        except Exception as e:
            logger.error(f"Error in get_signal: {e}")
            return 0
