import subprocess
import sys
from datetime import datetime

# List of training scripts to run sequentially
TRAINING_SCRIPTS = [
    'src/ai/train_models.py',    # LSTM, Transformer, etc.
    'src/ai/train_agent.py',     # RL agent
    'src/ai/train_gemini_swing.py',  # News sentiment/swing model
    'src/ai/train.py'           # Any additional training pipeline
]

def run_script(script_path):
    print(f"\n=== Running {script_path} ===")
    start = datetime.now()
    result = subprocess.run([sys.executable, script_path], capture_output=True, text=True)
    end = datetime.now()
    print(f"{script_path} started at {start.strftime('%H:%M:%S')}, ended at {end.strftime('%H:%M:%S')}")
    print(f"Return code: {result.returncode}")
    if result.stdout:
        print(f"--- STDOUT ---\n{result.stdout}")
    if result.stderr:
        print(f"--- STDERR ---\n{result.stderr}")
    if result.returncode != 0:
        print(f"ERROR: {script_path} failed. Stopping sequence.")
        sys.exit(result.returncode)

if __name__ == "__main__":
    print("=== Sequential Model Training Runner ===")
    for script in TRAINING_SCRIPTS:
        run_script(script)
    print("\nAll training scripts completed successfully.")
