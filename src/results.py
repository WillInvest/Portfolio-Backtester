"""Results tracking module"""
from dataclasses import dataclass, asdict
import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import json
from datetime import datetime
import os
from copy import deepcopy

@dataclass
class BacktestResults:
    """Store and manage backtest results"""
    strategy: int
    start_date: str
    end_date: str
    risk_aversion: float
    lookback_days_cov: int
    lookback_days_ret: int
    tickers: List[str]

    def __post_init__(self):
        self.weights_history: Dict = {}
        self.daily_returns: List = []
        self.dates: List = []
        self.failures: List = []
        self.cumulative_returns: Optional[pd.Series] = None
        self.metrics = {
            'total_return': None,
            'annual_return': None,
            'volatility': None,
            'sharpe_ratio': None,
            'max_drawdown': None,
            'daily_mean_arithmetic': None,
            'daily_mean_geometric': None,
            'daily_min_return': None,
            'max_10day_drawdown': None,
            'skewness': None,
            'kurtosis': None,
            'modified_var_95': None,
            'cvar_95': None,
            'num_trades': 0,
            'num_failures': 0
        }

    def update(self, date: pd.Timestamp, weights: Optional[Dict] = None,
               returns: Optional[float] = None, failure: Optional[str] = None):
        """Update results with new data"""
        if weights is not None:
            self.weights_history[date.strftime('%Y-%m-%d')] = deepcopy(weights)
            self.metrics['num_trades'] += 1

        if returns is not None:
            self.daily_returns.append(returns)
            self.dates.append(date)
            # Update cumulative returns and deanualize the anualized returns to daily returns
            returns_series = pd.Series(self.daily_returns, index=self.dates)
            self.cumulative_returns = (1 + returns_series).cumprod()

        if failure is not None:
            self.failures.append({
                'date': date.strftime('%Y-%m-%d'),
                'error': failure
            })
            self.metrics['num_failures'] += 1

    def calculate_metrics(self):
        """Calculate comprehensive performance metrics"""
        # Convert to Series
        returns_series = pd.Series(self.daily_returns, index=self.dates)
        cum_returns = self.cumulative_returns

        # Basic metrics
        num_days = (self.dates[-1] - self.dates[0]).days
        self.metrics['total_return'] = cum_returns.iloc[-1]
        self.metrics['annual_return'] = (cum_returns.iloc[-1]) ** (250 / num_days)
        self.metrics['volatility'] = returns_series.std() * np.sqrt(250)
        
        # Risk-adjusted returns
        risk_free_rate = 0.0
        excess_returns = returns_series - risk_free_rate/250
        self.metrics['sharpe_ratio'] = excess_returns.mean() / excess_returns.std() * np.sqrt(250)
        
        # Return statistics
        self.metrics['daily_mean_arithmetic'] = returns_series.mean()
        self.metrics['daily_mean_geometric'] = (np.prod(1 + returns_series) ** (1/len(returns_series))) - 1
        self.metrics['daily_min_return'] = returns_series.min()
        
        # Drawdown analysis
        rolling_max = cum_returns.cummax()
        drawdowns = cum_returns / rolling_max - 1
        self.metrics['max_drawdown'] = drawdowns.min()
        
        # 10-day drawdown
        rolling_drawdown = drawdowns.rolling(window=10).min()
        self.metrics['max_10day_drawdown'] = rolling_drawdown.min()
        
        # Higher moments
        from scipy.stats import skew, kurtosis
        self.metrics['skewness'] = skew(returns_series, nan_policy='omit')
        self.metrics['kurtosis'] = kurtosis(returns_series, nan_policy='omit')
        
        # Risk metrics
        z = -1.65  # 95% confidence
        self.metrics['modified_var_95'] = (z * returns_series.std() + 
                                         ((z**2 - 1) * self.metrics['skewness'] / 6) + 
                                         ((z**3 - 3*z) * self.metrics['kurtosis'] / 24))
        
        var_95 = np.percentile(returns_series, 5)
        self.metrics['cvar_95'] = returns_series[returns_series <= var_95].mean()

    def save(self, output_dir: str = 'backtest_results'):
        """
        Save results to files with a timestamped folder and subfolders.

        Args:
            output_dir: Root directory to save the backtest results.
        """
        # Generate a unique timestamp for this run
        timestamp = datetime.now().strftime('%Y%m%d')
    
        # Create the main folder name: strategy number + lookback days
        folder_name = (f'strategy_{self.strategy}_'
                       f'lambda_{self.risk_aversion}_'
                       f'ret_{self.lookback_days_ret}_'
                       f'cov_{self.lookback_days_cov}')
    
        # Combine output_dir, timestamp, and folder_name to form the path
        save_dir = os.path.join(output_dir, timestamp, folder_name)
        os.makedirs(save_dir, exist_ok=True)

        # Save configuration
        config = asdict(self)
        with open(os.path.join(save_dir, 'config.json'), 'w') as f:
            json.dump(config, f, indent=4)

        # Save weights
        weights_df = pd.DataFrame.from_dict(self.weights_history, orient='index')
        weights_df.to_csv(os.path.join(save_dir, 'weights.csv'))

        # Save returns
        returns_df = pd.DataFrame({
            'date': self.dates,
            'returns': self.daily_returns,
            'cumulative_returns': self.cumulative_returns.values if self.cumulative_returns is not None else None
        }).set_index('date')
        returns_df.to_csv(os.path.join(save_dir, 'returns.csv'))

        # Calculate and save metrics
        self.calculate_metrics()
        with open(os.path.join(save_dir, 'metrics.json'), 'w') as f:
            json.dump(self.metrics, f, indent=4)
        
        # Generate and save plots
        self.plot_and_save_results(save_dir)

    def print_metrics(self):
        """Print metrics in a formatted way"""
        print("\nPortfolio Performance Metrics:")
        print("=" * 40)
        for metric, value in self.metrics.items():
            if isinstance(value, (int, float)) and value is not None:
                print(f"{metric.replace('_', ' ').title()}: {value:.4f}")
            else:
                print(f"{metric.replace('_', ' ').title()}: {value}")
                
    def plot_and_save_results(self, save_dir: str) -> None:
        """Create and save analysis plots"""
        import matplotlib.pyplot as plt
        import seaborn as sns
    
        # Create plots directory
        plots_dir = os.path.join(save_dir, 'plots')
        os.makedirs(plots_dir, exist_ok=True)

        plt.figure(figsize=(12, 6))
        self.cumulative_returns.plot()
        plt.title(f"Portfolio Cumulative Returns - Strategy {self.strategy}")
        plt.xlabel("Date")
        plt.ylabel("Cumulative Returns")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, 'cumulative_returns.png'))
        plt.close()

