import pytest
from src.data.news_sentiment import fetch_sentiment_news

def test_fetch_sentiment_news(monkeypatch):
    # Mock config with NewsAPI key
    config = {
        'apis': {
            'newsapi': {'api_key': 'demo'},
            'finnhub': {'api_key': 'demo'}
        }
    }
    # Monkeypatch requests and feedparser to avoid real HTTP calls
    import src.data.news_sentiment as ns
    monkeypatch.setattr(ns, 'fetch_yahoo_finance_news', lambda symbol: [{'title': 'Yahoo', 'description': 'Finance'}])
    monkeypatch.setattr(ns, 'fetch_newsapi_news', lambda symbol, key: [{'title': 'NewsAPI', 'description': 'API'}])
    monkeypatch.setattr(ns, 'fetch_finnhub_news', lambda symbol, key: [{'title': 'Finnhub', 'description': 'API'}])
    monkeypatch.setattr(ns, 'fetch_google_news', lambda symbol: [{'title': 'Google', 'description': 'News'}])

    # Should return Yahoo first
    news = fetch_sentiment_news('RELIANCE.NS', config)
    assert news[0]['title'] == 'Yahoo'

    # If Yahoo fails, should try NewsAPI
    monkeypatch.setattr(ns, 'fetch_yahoo_finance_news', lambda symbol: [])
    news = fetch_sentiment_news('RELIANCE.NS', config)
    assert news[0]['title'] == 'NewsAPI'

    # If NewsAPI fails, should try Finnhub
    monkeypatch.setattr(ns, 'fetch_newsapi_news', lambda symbol, key: [])
    news = fetch_sentiment_news('RELIANCE.NS', config)
    assert news[0]['title'] == 'Finnhub'

    # If Finnhub fails, should try Google News
    monkeypatch.setattr(ns, 'fetch_finnhub_news', lambda symbol, key: [])
    news = fetch_sentiment_news('RELIANCE.NS', config)
    assert news[0]['title'] == 'Google'

    # If all fail, should return empty list
    monkeypatch.setattr(ns, 'fetch_google_news', lambda symbol: [])
    news = fetch_sentiment_news('RELIANCE.NS', config)
    assert news == []
