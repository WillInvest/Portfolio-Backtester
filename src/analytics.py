# analytics.py
"""Portfolio analytics module"""
import pandas as pd
import numpy as np
import statsmodels.api as sm
from typing import Tuple, Dict, List

class PortfolioAnalytics:
    """Portfolio analytics calculations"""
    @staticmethod
    def estimate_capm_beta(data: pd.DataFrame, 
                           returns_cols: List[str], 
                           spy_returns: pd.Series,
                         lookback_days: int) -> pd.Series:
        """
        Estimate CAPM betas for assets using OLS regression
        
        Args:
            returns: DataFrame of asset returns
            market_returns: Series of market returns (typically SPY)
            lookback_days: Number of days for estimation
            
        Returns:
            Series of CAPM betas for each asset
        """
        # Get recent data
        recent_data = data.iloc[-lookback_days-1:-1]
        recent_spy = spy_returns.iloc[-lookback_days-1:-1]
        etf_excess_returns = recent_data.subtract(recent_data['RF'], axis=0)
        spy_excess_returns = recent_spy - recent_data['RF']
        
        betas = {}
        for column in returns_cols:
            # Calculate beta using covariance method
            covariance = np.cov(etf_excess_returns[column], spy_excess_returns)[0, 1]
            market_variance = np.var(spy_excess_returns)
            beta = covariance / market_variance
            betas[column] = beta
            
        return pd.Series(betas)
    
    
    @staticmethod
    def estimate_covariance(data: pd.DataFrame, 
                           returns_cols: List[str], 
                           lookback_days: int,
                           use_ff: bool = True) -> pd.DataFrame:
        """
        Estimate covariance matrix using either Fama-French factors or direct calculation
    
        Args:
            data: DataFrame containing returns and FF factors
            returns_cols: List of return column names
            lookback_days: Number of days for estimation
            use_ff: If True, use Fama-French factors; if False, use direct calculation
        
        Returns:
            DataFrame of covariance matrix
        """
        recent_data = data.iloc[-lookback_days-1:-1]
    
        # Fama-French factor model
        etf_excess_returns = recent_data[returns_cols].subtract(recent_data['RF'], axis=0)
        F = recent_data[['Mkt-RF', 'SMB', 'HML']]

        betas = {}
        residual_variances = {}

        # Estimate betas and residuals
        for ticker in returns_cols:
            # Get excess returns for this ETF
            Ri = etf_excess_returns[ticker].dropna()

            # Get factor data aligned with this ETF's dates
            F_aligned = F.loc[Ri.index]
            X = sm.add_constant(F_aligned)

            # First run OLS to estimate rho
            ols_model = sm.OLS(Ri, X)
            ols_results = ols_model.fit()
            rho_est = np.corrcoef(ols_results.resid[:-1], ols_results.resid[1:])[0,1]

            # Use GLSAR with estimated rho
            model = sm.GLSAR(Ri, X, rho=rho_est)
            results = model.iterative_fit(maxiter=50)
            
            betas[ticker] = {
                'alpha': results.params['const'],
                'beta_market': results.params['Mkt-RF'],
                'beta_smb': results.params['SMB'],
                'beta_hml': results.params['HML']
            }
            residual_variances[ticker] = results.scale

        # Convert betas to DataFrame
        betas_df = pd.DataFrame(betas).T

        # Calculate factor covariance
        factor_cov = F.cov()
    
        # Create coefficient matrix
        B = betas_df[['beta_market', 'beta_smb', 'beta_hml']].values
    
        # Create residual variance matrix
        D = np.diag(list(residual_variances.values()))
    
        # Calculate final covariance matrix
        BΩB = B @ factor_cov.values @ B.T
        cov_matrix = BΩB + D
    
        return pd.DataFrame(cov_matrix, index=returns_cols, columns=returns_cols)
    
    @staticmethod
    def calculate_expected_returns(returns_data: pd.DataFrame, 
                                 lookback_days: int) -> pd.Series:
        """Calculate expected returns"""
        recent_returns = returns_data.iloc[-lookback_days-1:-1]
        return recent_returns.mean()

    @staticmethod
    def calculate_cov_spy(data: pd.DataFrame,
                          returns_cols: List[str],
                          spy_returns: pd.Series,
                          lookback_days: int) -> pd.Series:
        """
        Calculate covariance between each asset and SPY
        
        Parameters:
        -----------
        recent_data : pd.DataFrame
            Recent return data for all assets
        spy_returns : pd.Series
            SPY returns for the same period
            
        Returns:
        --------
        pd.Series
            Covariance between each asset and SPY
        """
        recent_data = data.iloc[-lookback_days-1:-1]
        recent_spy = spy_returns.iloc[-lookback_days-1:-1]


        # Calculate covariance for each column with SPY
        covs = {}
        for ticker in returns_cols:
            cov_matrix = np.cov(recent_data[ticker], recent_spy, ddof=0)
            covs[ticker] = cov_matrix[0, 1]  # Get the covariance value (off-diagonal element)
                
        return pd.Series(covs)  # Return as pandas Series
