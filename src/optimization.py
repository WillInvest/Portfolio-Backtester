"""Portfolio optimization module"""
import numpy as np
import pandas as pd
import cvxpy as cp
from scipy.optimize import minimize
from typing import Dict, Optional

class PortfolioOptimizer:
    """Portfolio optimization strategies"""
    @staticmethod
    def optimize_mean_variance(rho: pd.Series,
                             cov: pd.DataFrame,
                             beta: pd.Series,
                             risk_aversion: float) -> Optional[Dict]:
        """Mean-variance optimization strategy"""
        try:
            tickers = rho.index.tolist()
            num_assets = len(rho)
            weights = cp.Variable(num_assets)
            rho_np = rho.to_numpy()
            beta_np = beta.loc[rho.index].to_numpy()
            
            # Get Cholesky decomposition of covariance matrix
            L = np.linalg.cholesky(cov)
            obj = cp.Maximize(rho_np.T @ weights - risk_aversion * cp.norm(L @ weights))
            # obj = cp.Maximize(rho_np.T @ weights - risk_aversion * cp.quad_form(weights, cov))
            
            constraints = [
                cp.sum(weights) == 1,
                -2 <= weights,
                weights <= 2,
                -0.5 <= beta_np @ weights,
                beta_np @ weights <= 0.5
            ]

            prob = cp.Problem(obj, constraints)
            prob.solve()

            if prob.status in ['optimal', 'optimal_inaccurate']:
                weights_dict = dict(zip(tickers, weights.value))
                # Validate weights
                if all(np.isfinite(w) for w in weights.value):
                    return weights_dict
            return None
            
        except Exception as e:
            print(f"Optimization error: {str(e)}")
            return None

    @staticmethod
    def optimize_information_ratio(rho: pd.Series,
                                 spy_returns: pd.Series,
                                 cov: pd.DataFrame,
                                 cov_with_spy: pd.Series,
                                 beta: pd.Series,
                                 risk_aversion: float) -> Optional[Dict]:
        """Information ratio optimization strategy"""
        tickers = rho.index.tolist()
        num_assets = len(rho)
        spy_var = np.var(spy_returns)

        def objective(weights):
            port_var = weights.T @ cov @ weights
            tev_squared = port_var - 2 * weights.T @ cov_with_spy + spy_var
            tev = np.sqrt(max(tev_squared, 1e-8))
            expected_return = rho @ weights
            penalty = risk_aversion * np.sqrt(port_var)
            return -(expected_return / tev - penalty)

        constraints = [
            {'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1},
            {'type': 'ineq', 'fun': lambda weights: 2 + weights},
            {'type': 'ineq', 'fun': lambda weights: 2 - weights},
            {'type': 'ineq', 'fun': lambda weights: 2 + beta @ weights},
            {'type': 'ineq', 'fun': lambda weights: 2 - beta @ weights}
        ]

        initial_weights = np.ones(num_assets) / num_assets
        result = minimize(objective, initial_weights, constraints=constraints, method='SLSQP')

        if result.success:
            return dict(zip(tickers, result.x))
        return None