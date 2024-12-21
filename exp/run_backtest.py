import os
import sys
from datetime import datetime
sys.path.append('/Users/haofu/Desktop/portfolio_backtester')
from src.config import BacktestConfig
from src.backtester import run_strategy

def main():
    # Base output directory
    output_dir = 'backtest_results'
    timestamp = datetime.now().strftime('%Y%m%d')
    
    for strategy in [1]:
        for LAMBDA in [0.1, 0.5, 1.0]:
            for lb_days_cov in [40, 60, 90, 180]:
                for lb_days_ret in [40, 60, 90, 180]:
                    # Generate folder name using the same format as in save()
                    folder_name = (f'strategy_{strategy}_'
                                 f'lambda_{LAMBDA}_'
                                 f'ret_{lb_days_ret}_'
                                 f'cov_{lb_days_cov}')
                    
                    # Check if this combination has already been run
                    result_path = os.path.join(output_dir, timestamp, folder_name)
                    if os.path.exists(result_path):
                        print(f"Skipping existing combination: {folder_name}")
                        continue
                    
                    print(f"Running backtest with parameters:")
                    print(f"- Strategy: {strategy}")
                    print(f"- Lambda: {LAMBDA}")
                    print(f"- Lookback days (cov): {lb_days_cov}")
                    print(f"- Lookback days (ret): {lb_days_ret}")
                    
                    # Create configuration
                    config = BacktestConfig(
                        start_date="2007-03-01",
                        end_date="2024-10-31",
                        tickers=["FXE", "EWJ", "GLD", "QQQ", "SPY", "SHV", 
                                "DBA", "USO", "XBI", "ILF", "EPP", "FEZ"],
                        risk_aversion=LAMBDA,
                        lookback_days_cov=lb_days_cov,
                        lookback_days_ret=lb_days_ret,
                        strategy=strategy
                    )
                    
                    # Run the backtest
                    try:
                        results = run_strategy(config)
                        print(f"Successfully completed: {folder_name}\n")
                    except Exception as e:
                        print(f"Error running {folder_name}: {str(e)}\n")
                        continue

if __name__ == "__main__":
    main()

if __name__ == "__main__":
    main()