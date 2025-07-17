import logging
import yaml
import os
import pandas as pd

def load_ml_config():
    with open(os.path.join(os.path.dirname(__file__), 'ml_config.yaml'), 'r') as f:
        return yaml.safe_load(f)

def log_action(action):
    logging.info(f"[SOTA] {action}")

def get_sota_data():
    # Always use project root data directory
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
    data_path = os.path.join(project_root, 'data', 'swing_training_data.csv')
    if not os.path.exists(data_path):
        log_action(f"SOTA data file not found: {data_path}. Skipping training.")
        return None
    df = pd.read_csv(data_path)
    # Verification log: check required columns
    required_cols = ['open', 'high', 'low', 'close', 'volume']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        logging.error(f"Missing columns {missing_cols} in SOTA swing training data. Failing function.")
        return None
    return df

# Supervised Learning

def run_supervised_training(cfg):
    if not cfg.get('enabled', False):
        log_action("Supervised learning disabled by config.")
        return
    df = get_sota_data()
    if df is None:
        return
    # PATCH: Guarantee 'pattern' column is present before feature selection
    if 'pattern' not in df.columns:
        log_action("'pattern' column missing in SOTA data. Adding default value for all rows.")
        df['pattern'] = 'None'
    # PATCH: Guarantee 'direction' column is present before feature selection
    if 'direction' not in df.columns:
        log_action("'direction' column missing in SOTA data. Adding default value for all rows.")
        df['direction'] = 0
    log_action(f"Running supervised training with model: {cfg.get('model_type')}")
    # Always include 'pattern' and technical features in training
    feature_cols = cfg.get('features', [])
    for col in df.columns:
        if col not in feature_cols and col not in ['target','returns','timestamp','symbol']:
            feature_cols.append(col)
    if 'pattern' not in feature_cols:
        feature_cols.append('pattern')
    if 'direction' not in feature_cols:
        feature_cols.append('direction')
    X = df[feature_cols]
    y = df[cfg['target']]
    log_action(f"Training features used: {feature_cols}")
    # Placeholder: Add your supervised training logic here using X, y
    # Save model if cfg['save_model']

# Unsupervised Learning

def run_unsupervised_training(cfg):
    if not cfg.get('enabled', False):
        log_action("Unsupervised learning disabled by config.")
        return
    df = get_sota_data()
    if df is None:
        return
    log_action(f"Running unsupervised clustering: {cfg['clustering']['method']} with {cfg['clustering']['n_clusters']} clusters")
    log_action(f"Running regime detection: {cfg['regime_detection']['method']} with {cfg['regime_detection']['n_components']} components")
    # Placeholder: Add your unsupervised training logic here

# Reinforcement Learning

def run_rl_training(cfg):
    if not cfg.get('enabled', False):
        log_action("Reinforcement learning disabled by config.")
        return
    df = get_sota_data()
    if df is None:
        return
    log_action(f"Running RL training with {cfg['library']} and algo {cfg['algo']}")
    # Placeholder: Add your RL training logic here
    # Save model if cfg['save_model']

# NLP/NLU

def run_nlp(cfg):
    if not cfg.get('enabled', False):
        log_action("NLP/NLU disabled by config.")
        return
    log_action(f"Running NLP with model: {cfg['sentiment_model']}")
    if cfg.get('use_llm_for_reasoning', False):
        log_action("Using LLM for trade reasoning and journaling.")
    # Placeholder: Add your NLP logic here

# Integration

def run_integration(cfg):
    log_action("Integrating supervised, RL, NLP, and unsupervised features for SOTA swing trading.")
    # Placeholder: Add your integration logic here

def main():
    ml_cfg = load_ml_config()
    logging.basicConfig(level=ml_cfg.get('logging', {}).get('level', 'INFO'),
                        filename=ml_cfg.get('logging', {}).get('log_file') if ml_cfg.get('logging', {}).get('log_to_file') else None)
    log_action("Starting SOTA swing trading pipeline...")
    run_supervised_training(ml_cfg.get('supervised', {}))
    run_unsupervised_training(ml_cfg.get('unsupervised', {}))
    run_nlp(ml_cfg.get('nlp', {}))
    run_rl_training(ml_cfg.get('reinforcement_learning', {}))
    run_integration(ml_cfg.get('integration', {}))
    log_action("SOTA swing trading pipeline completed.")

if __name__ == "__main__":
    main()
