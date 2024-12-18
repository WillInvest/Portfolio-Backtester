"""Configuration module for portfolio backtester"""
from dataclasses import dataclass
from typing import List

@dataclass
class BacktestConfig:
    """Configuration for backtest"""
    start_date: str
    end_date: str
    tickers: List[str]
    risk_aversion: float = 0.5
    lookback_days_cov: int = 90
    lookback_days_ret: int = 90
    strategy: int = 1
    data_dir: str = '/Users/haofu/Desktop/portfolio_backtester/data'

    def __post_init__(self):
        """Validate configuration"""
        assert self.strategy in [1, 2], "Strategy must be 1 or 2"
        assert self.risk_aversion > 0, "Risk aversion must be positive"
        assert self.lookback_days_cov > 0, "Lookback days must be positive"
        assert self.lookback_days_ret > 0, "Lookback days must be positive"
        assert len(self.tickers) > 0, "Must provide at least one ticker"