import requests
import feedparser
import logging
import json
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
        if not news:
            logging.warning(f"Yahoo Finance returned no news for {symbol}. Feed entries: {len(feed.entries)}")
        else:
            logging.info(f"Yahoo Finance returned {len(news)} articles for {symbol}")
        return news
    except Exception as e:
        logging.warning(f"Yahoo Finance news fetch failed for {symbol}: {e}")
        return []

def fetch_finnhub_news(symbol: str, api_key: str) -> List[Dict[str, str]]:
    """Fetch news articles from Finnhub for a given symbol."""
    try:
        url = f'https://finnhub.io/api/v1/company-news?symbol={symbol}&from=2024-01-01&to=2025-12-31&token={api_key}'
        resp = requests.get(url, timeout=10)
        logging.info(f"Finnhub HTTP status for {symbol}: {resp.status_code}")
        if resp.status_code == 200:
            try:
                articles = resp.json()
            except Exception as e:
                logging.warning(f"Finnhub JSON decode error for {symbol}: {e}. Response: {resp.text[:500]}")
                return []
            if not articles:
                logging.warning(f"Finnhub returned 200 but no articles for {symbol}. Response: {resp.text[:500]}")
            else:
                logging.info(f"Finnhub returned {len(articles)} articles for {symbol}")
            return [
                {'title': a.get('headline', ''), 'description': a.get('summary', '')}
                for a in articles
            ]
        else:
            logging.warning(f"Finnhub returned status {resp.status_code} for {symbol}. Response: {resp.text[:500]}")
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
        if not news:
            logging.warning(f"Google News returned no news for {symbol}. Feed entries: {len(feed.entries)}")
        else:
            logging.info(f"Google News returned {len(news)} articles for {symbol}")
        return news
    except Exception as e:
        logging.warning(f"Google News RSS fetch failed for {symbol}: {e}")
        return []

def fetch_sentiment_news(symbol: str, config: Dict[str, Any]) -> List[Dict[str, str]]:
    """Try Google News, then Yahoo Finance, then Finnhub, using symbol, base ticker, and optionally company name."""
    tried = set()
    def get_base_ticker(sym):
        # Remove exchange suffixes and segment codes
        if '-' in sym:
            sym = sym.split('-')[0]
        if '.' in sym:
            sym = sym.split('.')[0]
        return sym
    queries = [symbol]
    base_ticker = get_base_ticker(symbol)
    if base_ticker != symbol:
        queries.append(base_ticker)
    # Optionally, try company name if available in config
    symbol_map = config.get('symbol_map', {})
    company_name = symbol_map.get(symbol) or symbol_map.get(base_ticker)
    if company_name and company_name not in queries:
        queries.append(company_name)
    # Google News
    for q in queries:
        if q in tried:
            continue
        tried.add(q)
        logging.info(f"[NewsSentiment] Trying Google News for {q}")
        news = fetch_google_news(q)
        if news:
            return news
    # Yahoo Finance
    for q in queries:
        if q in tried:
            continue
        tried.add(q)
        logging.info(f"[NewsSentiment] Trying Yahoo Finance for {q}")
        news = fetch_yahoo_finance_news(q)
        if news:
            return news
    # Finnhub
    finnhub_key = config.get('apis', {}).get('finnhub', {}).get('api_key', '')
    if finnhub_key:
        for q in queries:
            if q in tried:
                continue
            tried.add(q)
            logging.info(f"[NewsSentiment] Trying Finnhub for {q}")
            news = fetch_finnhub_news(q, finnhub_key)
            if news:
                return news
    return []
