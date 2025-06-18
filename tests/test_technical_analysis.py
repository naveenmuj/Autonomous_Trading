import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from src.trading.analysis import TechnicalAnalyzer
from src.trading.strategy import EnhancedTradingStrategy
from src.trading.backtester import Backtester

@pytest.fixture
def sample_data():
    # Generate sample price data
    dates = pd.date_range(start='2025-01-01', periods=100, freq='D')
    data = pd.DataFrame(index=dates)
    
    # Simulate trending price data
    base_price = 100
    trend = np.linspace(0, 20, 100)  # Upward trend
    noise = np.random.normal(0, 1, 100)  # Random noise
    
    data['close'] = base_price + trend + noise
    data['high'] = data['close'] + abs(np.random.normal(0, 0.5, 100))
    data['low'] = data['close'] - abs(np.random.normal(0, 0.5, 100))
    data['open'] = data['close'].shift(1)
    data['volume'] = np.random.randint(1000, 5000, 100)
    
    return data

@pytest.fixture
def config():
    return {
        'trading': {
            'risk': {
                'max_position_size': 0.02,
                'stop_loss': 0.02,
                'max_trades': 5
            },
            'strategy': {
                'lookback_period': 20,
                'profit_target': 0.03
            }
        }
    }

@pytest.fixture
def tech_analyzer():
    return TechnicalAnalyzer()

@pytest.fixture
def strategy(config):
    return EnhancedTradingStrategy(config)

@pytest.fixture
def backtester(strategy):
    return Backtester(strategy)

class TestTechnicalAnalysis:
    def test_swing_points_detection(self, tech_analyzer, sample_data):
        """Test detection of swing highs and lows"""
        high_idx, low_idx = tech_analyzer.find_swing_points(sample_data)
        
        assert len(high_idx) > 0, "Should find swing highs"
        assert len(low_idx) > 0, "Should find swing lows"
        assert all(isinstance(idx, (int, np.integer)) for idx in high_idx)
        assert all(isinstance(idx, (int, np.integer)) for idx in low_idx)

    def test_trend_line_detection(self, tech_analyzer, sample_data):
        """Test trend line detection"""
        trend_lines = tech_analyzer.detect_trend_lines(sample_data)
        
        assert 'support' in trend_lines
        assert 'resistance' in trend_lines
        assert isinstance(trend_lines['support'], list)
        assert isinstance(trend_lines['resistance'], list)
        
        if trend_lines['support']:
            line = trend_lines['support'][0]
            assert 'slope' in line
            assert 'intercept' in line
            assert 'score' in line
            assert 'strength' in line

    def test_breakout_analysis(self, tech_analyzer, sample_data):
        """Test breakout analysis"""
        trend_lines = tech_analyzer.detect_trend_lines(sample_data)
        breakouts = tech_analyzer.analyze_breakouts(sample_data, trend_lines)
        
        assert 'support' in breakouts
        assert 'resistance' in breakouts
        assert isinstance(breakouts['support'], list)
        assert isinstance(breakouts['resistance'], list)

class TestTradingStrategy:
    def test_market_analysis(self, strategy, sample_data):
        """Test comprehensive market analysis"""
        analysis = strategy.analyze_market(sample_data)
        
        assert 'signals' in analysis
        assert 'trend_lines' in analysis
        assert 'breakouts' in analysis
        assert 'risk_levels' in analysis
        
        signals = analysis['signals']
        assert isinstance(signals, pd.DataFrame)
        assert 'signal' in signals.columns
        assert 'confidence' in signals.columns

    def test_risk_calculation(self, strategy, sample_data):
        """Test risk level calculations"""
        analysis = strategy.analyze_market(sample_data)
        risk_levels = analysis['risk_levels']
        
        assert 'stop_loss' in risk_levels
        assert 'take_profit' in risk_levels
        assert 'risk_reward_ratio' in risk_levels
        assert risk_levels['stop_loss'] < risk_levels['take_profit']

class TestBacktesting:
    def test_backtest_execution(self, backtester, sample_data):
        """Test backtest execution"""
        results = backtester.run(sample_data)
        
        assert 'total_return' in results
        assert 'sharpe_ratio' in results
        assert 'max_drawdown' in results
        assert 'win_rate' in results
        assert 'total_trades' in results
        assert 'equity_curve' in results
        assert isinstance(results['equity_curve'], list)
        assert len(results['equity_curve']) > 0

    def test_trade_execution(self, backtester, sample_data):
        """Test trade execution in backtest"""
        results = backtester.run(sample_data)
        trades = backtester.trades
        
        if trades:
            trade = trades[0]
            assert 'entry_price' in trade
            assert 'exit_price' in trade
            assert 'quantity' in trade
            assert 'pnl' in trade
            assert 'return' in trade

    def test_portfolio_value_calculation(self, backtester, sample_data):
        """Test portfolio value calculation"""
        initial_capital = backtester.initial_capital
        results = backtester.run(sample_data)
        final_value = results['equity_curve'][-1]
        
        assert final_value != initial_capital, "Portfolio value should change"
        assert final_value > 0, "Portfolio value should be positive"

    def test_risk_management(self, backtester, sample_data):
        """Test risk management in backtest"""
        results = backtester.run(sample_data)
        
        # Check max drawdown is within limits
        assert results['max_drawdown'] <= 30, "Max drawdown should be within limits"
        
        # Check position sizing
        if backtester.trades:
            for trade in backtester.trades:
                position_value = trade['entry_price'] * trade['quantity']
                assert position_value <= backtester.initial_capital * 0.02, "Position size should respect limits"
