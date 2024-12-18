from setuptools import setup, find_packages

setup(
    name="portfolio_backtester",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        'pandas',
        'numpy',
        'yfinance',
        'statsmodels',
        'cvxpy',
        'scipy',
        'matplotlib',
        'tqdm'
    ],
    description="A portfolio backtesting system",
    keywords="finance, portfolio, backtest",
)