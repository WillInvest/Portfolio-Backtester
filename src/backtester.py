"""Main backtesting module"""
import pandas as pd
import numpy as np
from typing import Tuple, Dict, Optional
from tqdm import tqdm

from .config import BacktestConfig
from .data import DataLoader
from .analytics import PortfolioAnalytics
from .optimization import PortfolioOptimizer
from .results import BacktestResults

class PortfolioBacktest:
    """Main backtesting class"""
    def __init__(self, config: BacktestConfig):
        """Initialize backtester with configuration"""
        self.config = config
        self.data_loader = DataLoader(config)
        self.analytics = PortfolioAnalytics()
        self.optimizer = PortfolioOptimizer()
        self.etf_daily_returns = None
        self.fama_french_data = None
        self.results = BacktestResults(
            strategy=config.strategy,
            start_date=config.start_date,
            end_date=config.end_date,
            risk_aversion=config.risk_aversion,
            lookback_days_cov=config.lookback_days_cov,
            lookback_days_ret=config.lookback_days_ret,
            tickers=config.tickers
        )
        self.load_data()

    def load_data(self) -> None:
        """Load and prepare data"""
        self.etf_daily_returns, self.fama_french_data = self.data_loader.load_data()

    def get_portfolio_parameters(self, data_up_to_rebalance: pd.DataFrame, 
                               ff_data_up_to_rebalance: pd.DataFrame) -> Tuple:
        """Calculate portfolio parameters for optimization"""
        # Calculate expected returns
        expected_returns = self.analytics.calculate_expected_returns(
            data_up_to_rebalance,
            self.config.lookback_days_ret
        )
        
        # Calculate Fama-French parameters
        merged_data = pd.concat([data_up_to_rebalance, ff_data_up_to_rebalance], axis=1).dropna()
        covariance = self.analytics.estimate_covariance(
            merged_data,
            self.etf_daily_returns.columns.tolist(),
            self.config.lookback_days_cov
        )
        
        # calculate the CAPM beta
        betas = self.analytics.estimate_capm_beta(
                    merged_data,
                    self.etf_daily_returns.columns.tolist(),
                    data_up_to_rebalance['SPY'],
                    self.config.lookback_days_cov
                )
    
        if self.config.strategy == 2:
            # Additional parameters for information ratio strategy
            cov_spy = self.analytics.calculate_cov_spy(
                merged_data,
                self.etf_daily_returns.columns.tolist(),
                data_up_to_rebalance['SPY'],
                self.config.lookback_days_cov
            )
            return (expected_returns, covariance, cov_spy, 
                   betas, data_up_to_rebalance['SPY'])
            
        return expected_returns, covariance, betas

    def optimize_portfolio(self, *args) -> Optional[Dict]:
        """Optimize portfolio based on strategy"""
        if self.config.strategy == 1:
            expected_returns, covariance, beta = args
            return self.optimizer.optimize_mean_variance(
                expected_returns, covariance, beta, self.config.risk_aversion)
        elif self.config.strategy == 2:
            expected_returns, covariance, cov_spy, beta, spy_returns = args
            return self.optimizer.optimize_information_ratio(
                expected_returns, spy_returns, covariance, cov_spy, 
                beta, self.config.risk_aversion)
        raise ValueError("Invalid strategy number. Must be 1 or 2.")

    def run_backtest(self) -> BacktestResults:
        """Run the backtest"""
        # Get rebalancing dates
        last_trading_days = self.etf_daily_returns.groupby(pd.Grouper(freq='W-FRI')).last()
        rebalance_dates = last_trading_days.index[15:]
        
        portfolio_weights = {}
        previous_weights = None
        
        # Run through rebalancing dates
        pbar = tqdm(rebalance_dates)
        for rebalance_date in pbar:
            pbar.set_description(f"Processing {rebalance_date.strftime('%Y-%m-%d')}")
             # Get data up to rebalance date
            data_up_to_rebalance = self.etf_daily_returns.loc[:rebalance_date]
            ff_data_up_to_rebalance = self.fama_french_data.loc[:rebalance_date]
            
            # Calculate parameters and optimize
            params = self.get_portfolio_parameters(data_up_to_rebalance, ff_data_up_to_rebalance)
            weights = self.optimize_portfolio(*params)
            
            if weights is None:
                # print(f"Using previous weights on {rebalance_date}")
                portfolio_weights[rebalance_date] = previous_weights
                self.results.update(rebalance_date, failure="Optimization failed")
            else:
                previous_weights = weights
                portfolio_weights[rebalance_date] = weights
                self.results.update(rebalance_date, weights=weights)

        # Calculate returns
        self._calculate_portfolio_returns(portfolio_weights)
        
        return self.results

    def _calculate_portfolio_returns(self, portfolio_weights: Dict) -> None:
        """Calculate portfolio returns"""
        start_date = min(portfolio_weights.keys())
        trading_days = self.etf_daily_returns[
            self.etf_daily_returns.index >= start_date
        ].index
        
        current_weights = None
        wipe_out = False
        pbar = tqdm(trading_days, desc="Calculating daily returns")
        for date in pbar:
            # Update progress bar description with current date
            pbar.set_description(f"Calculating {date.strftime('%Y-%m-%d')}")
            if not wipe_out:
                if date in portfolio_weights:
                    current_weights = portfolio_weights[date]
                
                if current_weights is not None:
                    daily_return = sum(
                        weight * self.etf_daily_returns.loc[date, asset]
                        for asset, weight in current_weights.items()
                    )
                    if daily_return < -1:
                        wipe_out = True
                    
                self.results.update(date, returns=daily_return)
            else:
                print(f"Portfolio wiped out on {date}")
                return None


def run_strategy(config: BacktestConfig) -> BacktestResults:
    """Helper function to run a backtest with given configuration"""
    backtest = PortfolioBacktest(config)
    results = backtest.run_backtest()
    results.save()
    return results