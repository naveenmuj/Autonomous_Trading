import requests
import logging
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)

class NewsSentimentAnalyzer:
    """News and sentiment analyzer using Gemini LLM API."""
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.api_key = self._get_gemini_api_key()
        self.endpoint = "https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent"

    def _get_gemini_api_key(self) -> Optional[str]:
        # Look for Gemini API key in config (support nested under 'apis')
        return (
            self.config.get('gemini_api_key') or
            self.config.get('api', {}).get('gemini_api_key') or
            self.config.get('llm', {}).get('gemini_api_key') or
            self.config.get('apis', {}).get('gemini', {}).get('api_key')
        )

    def analyze_news(self, headlines: List[str], context: str = "") -> Dict[str, Any]:
        """Analyze news headlines for sentiment and trading signal using Gemini."""
        if not self.api_key:
            logger.warning("No Gemini API key found in config. Skipping news sentiment analysis.")
            return {"sentiment": "neutral", "score": 0.0, "reason": "No API key"}
        prompt = self._build_prompt(headlines, context)
        payload = {
            "contents": [{"parts": [{"text": prompt}]}]
        }
        try:
            response = requests.post(
                f"{self.endpoint}?key={self.api_key}",
                json=payload,
                timeout=20
            )
            response.raise_for_status()
            result = response.json()
            text = result['candidates'][0]['content']['parts'][0]['text']
            # Simple parsing: look for sentiment and score
            sentiment, score = self._parse_gemini_response(text)
            return {"sentiment": sentiment, "score": score, "raw": text}
        except Exception as e:
            logger.error(f"Gemini API error: {e}")
            return {"sentiment": "neutral", "score": 0.0, "reason": str(e)}

    def _build_prompt(self, headlines: List[str], context: str = "") -> str:
        prompt = (
            "You are a financial news sentiment analysis expert. "
            "Given the following news headlines, analyze the overall sentiment (bullish, bearish, neutral) "
            "and provide a confidence score between -1 (very bearish) and +1 (very bullish). "
            "If possible, suggest if a trader should buy, sell, or hold. "
        )
        if context:
            prompt += f"\nContext: {context}\n"
        prompt += "\nHeadlines:\n" + "\n".join(headlines)
        prompt += "\nRespond in the format: Sentiment: <bullish|bearish|neutral>, Score: <number>, Action: <buy|sell|hold>, Reason: <short explanation>."
        return prompt

    def _parse_gemini_response(self, text: str) -> tuple[str, float]:
        import re
        sentiment = "neutral"
        score = 0.0
        try:
            sent_match = re.search(r"Sentiment:\s*(bullish|bearish|neutral)", text, re.I)
            score_match = re.search(r"Score:\s*([-+]?\d*\.?\d+)", text)
            if sent_match:
                sentiment = sent_match.group(1).lower()
            if score_match:
                score = float(score_match.group(1))
        except Exception:
            pass
        return sentiment, score

    def get_news_headlines(self, symbol: str) -> List[str]:
        # Placeholder: In production, connect to a real news API
        # For now, return dummy headlines
        return [
            f"{symbol} reports strong quarterly earnings.",
            f"{symbol} announces new product launch.",
            f"{symbol} faces regulatory scrutiny."
        ]

    def analyze_symbol(self, symbol: str, context: str = "") -> Dict[str, Any]:
        headlines = self.get_news_headlines(symbol)
        return self.analyze_news(headlines, context)
