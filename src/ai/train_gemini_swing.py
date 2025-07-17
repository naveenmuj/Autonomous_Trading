import logging
import requests
import time
from datetime import datetime, timedelta
import yaml
import os

logger = logging.getLogger("gemini_swing_train")
logging.basicConfig(level=logging.INFO)

# Only allow training once, within 1 day of first run (API valid for 20 days, but train only once)
TRAINING_EXPIRY_DATE = datetime.now() + timedelta(days=1)
TRAINED_FLAG_FILE = "models/gemini_swing_trained.flag"

# Load Gemini API key/config from config.yaml
CONFIG_PATH = os.path.join(os.path.dirname(__file__), '../../config.yaml')
def load_gemini_config():
    try:
        with open(CONFIG_PATH, 'r') as f:
            config = yaml.safe_load(f)
        # Only use apis.gemini for Gemini config
        return config.get('apis', {}).get('gemini', {})
    except Exception as e:
        logger.error(f"Failed to load Gemini config: {e}")
        return {}

def call_gemini_api(messages, model="gemini-2.0-flash", max_retries=5, initial_delay=2):
    """Call Gemini LLM API with a list of messages (see config.yaml for endpoint/key), with retry and rate limit handling."""
    cfg = load_gemini_config()
    api_url = cfg.get('api_url')
    api_key = cfg.get('api_key')
    if not api_url or not api_key:
        logger.error("Gemini API URL or key missing in config.yaml")
        return None
    # Compose the endpoint as in the working curl
    endpoint = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
    headers = {
        "Content-Type": "application/json"
    }
    # Convert messages to Gemini API 'contents' format
    contents = []
    for msg in messages:
        if msg['role'] == 'user':
            contents.append({"parts": [{"text": msg['content']}]} )
        # Optionally handle system/context messages if needed
    payload = {"contents": contents}
    delay = initial_delay
    for attempt in range(max_retries):
        try:
            resp = requests.post(endpoint, json=payload, headers=headers, timeout=60)
            if resp.status_code == 429:
                logger.warning(f"Gemini API rate limit hit, retrying in {delay}s (attempt {attempt+1}/{max_retries})...")
                time.sleep(delay)
                delay *= 2
                continue
            resp.raise_for_status()
            return resp.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"Gemini API call failed: {e}")
            if attempt < max_retries - 1:
                logger.info(f"Retrying Gemini API call in {delay}s...")
                time.sleep(delay)
                delay *= 2
            else:
                return None
    return None

def ask_gemini(question, context=None):
    messages = []
    if context:
        messages.append({"role": "system", "content": context})
    messages.append({"role": "user", "content": question})
    response = call_gemini_api(messages)
    # Log the full Gemini API response for transparency and debugging
    logger.info(f"Full Gemini API response for question '{question}': {response}")
    # Parse Gemini 2.0 API response
    try:
        if response and 'candidates' in response and response['candidates']:
            parts = response['candidates'][0]['content']['parts']
            if parts and 'text' in parts[0]:
                return parts[0]['text']
    except Exception as e:
        logger.error(f"Error parsing Gemini response: {e}")
    return None

def train_theory_and_patterns():
    """Ask Gemini to teach swing trading theory, patterns, and quiz the model."""
    topics = [
        "Explain advanced swing trading strategies with examples.",
        "List and describe key chart patterns for swing trading.",
        "How do RSI, MACD, and Bollinger Bands help in swing trading?",
        "Quiz me on swing trading indicators and setups.",
        "What are the best entry/exit strategies for swing trading?",
        "How to manage risk and position size in swing trading?",
        "How to recognize market regimes for swing trading?",
        "What are the most common mistakes in swing trading?",
        "How to journal and review swing trades for improvement?"
    ]
    for topic in topics:
        try:
            logger.info(f"Gemini teaching: {topic}")
            answer = ask_gemini(topic)
            if answer:
                with open("models/gemini_swing_theory.txt", "a", encoding="utf-8") as f:
                    f.write(f"Q: {topic}\nA: {answer}\n\n")
            else:
                logger.warning(f"No answer for: {topic}")
        except Exception as e:
            logger.error(f"Error in theory training: {e}")
            continue

def train_news_sentiment(news_headlines):
    """Feed news to Gemini and get swing trade sentiment labels."""
    for news in news_headlines:
        try:
            q = f"What is the likely swing trade impact of this news? Label as bullish, bearish, or neutral. News: {news}"
            logger.info(f"Gemini analyzing news: {news}")
            answer = ask_gemini(q)
            with open("models/gemini_news_sentiment.csv", "a", encoding="utf-8") as f:
                f.write(f'"{news}","{answer}"\n')
        except Exception as e:
            logger.error(f"Error in news sentiment: {e}")
            continue

def train_pattern_recognition():
    """Ask Gemini to interpret technical setups in natural language."""
    setups = [
        "RSI = 70, Price touching upper Bollinger Band, MACD crossover",
        "RSI = 30, Price at support, bullish engulfing candle",
        "MACD bearish crossover, price below 50 EMA, high volume",
        "Price breaks resistance with high volume, RSI rising"
    ]
    for setup in setups:
        try:
            q = f"Interpret this swing trading setup: {setup}"
            logger.info(f"Gemini interpreting setup: {setup}")
            answer = ask_gemini(q)
            with open("models/gemini_pattern_recognition.txt", "a", encoding="utf-8") as f:
                f.write(f"Setup: {setup}\nInterpretation: {answer}\n\n")
        except Exception as e:
            logger.error(f"Error in pattern recognition: {e}")
            continue

def train_journaling_assistant():
    """Simulate journaling and review with Gemini."""
    logs = [
        {"entry": "Bought TCS at 3500, RSI 32, support bounce", "exit": "Sold at 3600, RSI 55", "reason": "Oversold bounce, target hit"},
        {"entry": "Shorted INFY at 1500, MACD bearish, resistance fail", "exit": "Covered at 1480", "reason": "Bearish momentum, quick scalp"}
    ]
    for log in logs:
        try:
            q = f"Review this swing trade: Entry: {log['entry']}, Exit: {log['exit']}, Reason: {log['reason']}. What could be improved?"
            logger.info(f"Gemini journaling: {log}")
            answer = ask_gemini(q)
            with open("models/gemini_journal_review.txt", "a", encoding="utf-8") as f:
                f.write(f"Log: {log}\nReview: {answer}\n\n")
        except Exception as e:
            logger.error(f"Error in journaling assistant: {e}")
            continue

def train_rl_features():
    """Ask Gemini to generate RL features for swing trading agent."""
    questions = [
        "What features should a reinforcement learning agent use for swing trading?",
        "How to encode news sentiment and technical signals for RL?",
        "How to summarize market regime for RL state vector?",
        "Suggest a reward function for swing trading RL agent."
    ]
    for q in questions:
        try:
            logger.info(f"Gemini RL feature engineering: {q}")
            answer = ask_gemini(q)
            with open("models/gemini_rl_features.txt", "a", encoding="utf-8") as f:
                f.write(f"Q: {q}\nA: {answer}\n\n")
        except Exception as e:
            logger.error(f"Error in RL features: {e}")
            continue

def gemini_training_enabled():
    cfg = load_gemini_config()
    return cfg.get('enable_training', True)

def main():
    if not gemini_training_enabled():
        logger.info("Gemini swing training disabled by config. Skipping.")
        return
    if datetime.now() > TRAINING_EXPIRY_DATE:
        logger.warning("Gemini API training window expired. Skipping Gemini training.")
        return
    logger.info("Starting Gemini swing trading training (one-time)...")
    try:
        train_theory_and_patterns()
        # Example news headlines (replace with your own or fetch from news API)
        news_headlines = [
            "TCS earnings beat estimates, stock rallies 5%",
            "RBI raises interest rates by 0.25%",
            "Infosys Q4 results disappoint, shares fall",
            "US inflation data higher than expected"
        ]
        train_news_sentiment(news_headlines)
        train_pattern_recognition()
        train_journaling_assistant()
        train_rl_features()
        with open(TRAINED_FLAG_FILE, "w") as f:
            f.write(f"Trained on: {datetime.now()}\n")
        logger.info("Gemini swing trading training completed and flagged.")
    except Exception as e:
        logger.error(f"Critical error in Gemini swing training: {e}")

if __name__ == "__main__":
    main()
