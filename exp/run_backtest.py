# run_backtest.py (in portfolio_backtester/)
import sys
sys.path.append('/Users/haofu/Desktop/portfolio_backtester')
from src.config import BacktestConfig
from src.backtester import run_strategy

def main():
    for strategy in [1, 2]:
        for LAMBDA in [0.1, 0.5, 1.0]:
            for lb_days_cov in [40, 60, 90, 180]:
                for lb_days_ret in [40, 60, 90, 180]:
                    print(f"Running backtest with lb_days_cov={lb_days_cov} and lb_days_ret={lb_days_ret}, strategy={strategy}, lambda={LAMBDA}")
                    # Create configuration
                    config = BacktestConfig(
                        start_date="2007-03-01",
                        end_date="2024-10-31",
                        tickers=["FXE", "EWJ", "GLD", "QQQ", "SPY", "SHV", "DBA", "USO", "XBI", "ILF", "EPP", "FEZ"],
                        risk_aversion=LAMBDA,
                        lookback_days_cov=lb_days_cov,
                        lookback_days_ret=lb_days_ret,
                        strategy=strategy
                    )
                    
                    # Run backtest
                    results = run_strategy(config)

if __name__ == "__main__":
    main()