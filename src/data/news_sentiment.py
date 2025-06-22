import requests
import feedparser
import logging
from typing import List, Dict, Any

def fetch_yahoo_finance_news(symbol: str) -> List[Dict[str, str]]:
    """Fetch news articles from Yahoo Finance RSS feed for a given symbol."""
    try:
        url = f'https://finance.yahoo.com/rss/headline?s={symbol}'
        feed = feedparser.parse(url)
        news = []
        for entry in feed.entries:
            news.append({
                'title': entry.get('title', ''),
                'description': entry.get('summary', '')
            })
        return news
    except Exception as e:
        logging.warning(f"Yahoo Finance news fetch failed for {symbol}: {e}")
        return []

def fetch_newsapi_news(symbol: str, api_key: str) -> List[Dict[str, str]]:
    """Fetch news articles from NewsAPI for a given symbol."""
    try:
        url = f'https://newsapi.org/v2/everything?q={symbol}&apiKey={api_key}'
        resp = requests.get(url, timeout=10)
        if resp.status_code == 200:
            articles = resp.json().get('articles', [])
            return [
                {'title': a.get('title', ''), 'description': a.get('description', '')}
                for a in articles
            ]
        else:
            logging.warning(f"NewsAPI returned status {resp.status_code} for {symbol}")
            return []
    except Exception as e:
        logging.warning(f"NewsAPI fetch failed for {symbol}: {e}")
        return []

def fetch_finnhub_news(symbol: str, api_key: str) -> List[Dict[str, str]]:
    """Fetch news articles from Finnhub for a given symbol."""
    try:
        url = f'https://finnhub.io/api/v1/company-news?symbol={symbol}&from=2024-01-01&to=2025-12-31&token={api_key}'
        resp = requests.get(url, timeout=10)
        if resp.status_code == 200:
            articles = resp.json()
            return [
                {'title': a.get('headline', ''), 'description': a.get('summary', '')}
                for a in articles
            ]
        else:
            logging.warning(f"Finnhub returned status {resp.status_code} for {symbol}")
            return []
    except Exception as e:
        logging.warning(f"Finnhub fetch failed for {symbol}: {e}")
        return []

def fetch_google_news(symbol: str) -> List[Dict[str, str]]:
    """Fetch news articles from Google News RSS for a given symbol."""
    try:
        url = f'https://news.google.com/rss/search?q={symbol}'
        feed = feedparser.parse(url)
        news = []
        for entry in feed.entries:
            news.append({
                'title': entry.get('title', ''),
                'description': entry.get('summary', '')
            })
        return news
    except Exception as e:
        logging.warning(f"Google News RSS fetch failed for {symbol}: {e}")
        return []

def fetch_sentiment_news(symbol: str, config: Dict[str, Any]) -> List[Dict[str, str]]:
    """Try Yahoo Finance, then NewsAPI, then Finnhub, then Google News RSS."""
    # Yahoo Finance
    news = fetch_yahoo_finance_news(symbol)
    if news:
        return news
    # NewsAPI
    api_key = config.get('apis', {}).get('newsapi', {}).get('api_key', '')
    if api_key:
        news = fetch_newsapi_news(symbol, api_key)
        if news:
            return news
    # Finnhub
    finnhub_key = config.get('apis', {}).get('finnhub', {}).get('api_key', '')
    if finnhub_key:
        news = fetch_finnhub_news(symbol, finnhub_key)
        if news:
            return news
    # Google News RSS
    news = fetch_google_news(symbol)
    return news
