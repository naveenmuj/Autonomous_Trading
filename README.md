# AI-based Stock Trading System

A sophisticated trading system that uses artificial intelligence, machine learning, and technical analysis to make trading decisions for NSE stocks.

## Features

- AI-based trading decisions using reinforcement learning
- Real-time stock monitoring and analysis
- Technical analysis with support/resistance detection
- News sentiment analysis
- Risk management system
- Paper trading simulation mode
- Live trading capabilities
- Interactive dashboard with Streamlit
- Angle One authentication integration
- Daily model training with historical data

## Setup

1. Install the required dependencies:
```bash
pip install -r requirements.txt
```

2. Configure your API keys in `config.yaml`:
- News API key
- Angle One API credentials

3. Initialize the environment:
```bash
python src/main.py
```

## Project Structure

```
├── config.yaml           # Configuration file
├── requirements.txt      # Python dependencies
├── src/
│   ├── ai/             # AI models and algorithms
│   │   └── models.py
│   ├── data/           # Data collection and processing
│   │   └── collector.py
│   ├── trading/        # Trading logic and risk management
│   │   └── manager.py
│   ├── ui/             # User interface
│   │   └── dashboard.py
│   └── main.py         # Application entry point
├── tests/              # Unit tests
│   └── test_trading_system.py
└── models/            # Trained model storage
```

## Configuration

The `config.yaml` file contains all the necessary settings:

- API credentials
- Trading parameters
- Risk management settings
- Model configuration
- UI preferences

## Features in Detail

### AI Trading System
- Reinforcement learning for trade decisions
- Technical analysis integration
- News sentiment analysis
- Real-time monitoring

### Risk Management
- Position sizing
- Stop-loss automation
- Portfolio risk analysis
- Maximum drawdown control

### User Interface
- Real-time trading dashboard
- Technical charts with indicators
- Portfolio performance metrics
- Trade simulation interface

### Data Collection
- Real-time market data
- Historical price data
- News feed integration
- Technical indicators

## Usage

1. Start the application:
```bash
python src/main.py
```

2. Access the dashboard:
- Open your browser
- Navigate to http://localhost:8501

3. Choose between simulation and live trading modes
4. Monitor trades and performance metrics
5. Review AI decisions and analysis

## Testing

Run the test suite:
```bash
python -m unittest tests/test_trading_system.py
```

## Safety Features

- Paper trading mode for testing
- Risk limits and position sizing
- Automated stop-loss management
- Portfolio risk monitoring

## Monitoring and Maintenance

The system includes:
- Continuous model training
- Performance monitoring
- Risk metrics tracking
- System health checks

## License

MIT License

# ai-trading-copilot