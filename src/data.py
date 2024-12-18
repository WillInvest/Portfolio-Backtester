"""Data loading and processing module"""
import pandas as pd
import yfinance as yf
import os
from typing import Tuple
import numpy as np

class DataLoader:
    """Data loading and processing class"""
    def __init__(self, config):
        self.config = config
        self.etf_daily_returns = None
        self.fama_french_data = None

    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load and prepare all data"""
        self._load_etf_data()
        self._load_fama_french_data()
        self._align_data()
        return self.etf_daily_returns, self.fama_french_data

    def _load_etf_data(self):
        """Load ETF data from Yahoo Finance"""
        etf_data = yf.download(
            self.config.tickers,
            start=self.config.start_date,
            end=self.config.end_date,
            progress=False
        )['Adj Close']
        self.etf_daily_returns = etf_data.pct_change().dropna()

    def _load_fama_french_data(self):
        """Load Fama-French factors data"""
        ff_path = os.path.join(self.config.data_dir, 'F-F_Research_Data_Factors_daily.CSV')
        self.fama_french_data = pd.read_csv(
            ff_path,
            skiprows=3,
            skipfooter=1,
            engine='python'
        )
        self.fama_french_data = self.fama_french_data.rename(columns={'Unnamed: 0': 'Date'})
        self.fama_french_data['Date'] = pd.to_datetime(self.fama_french_data['Date'], format='%Y%m%d')
        self.fama_french_data.set_index('Date', inplace=True)
        self.fama_french_data = self.fama_french_data[
            self.fama_french_data.index >= self.config.start_date
        ]

    def _align_data(self):
        """Align ETF and Fama-French data"""
        self.etf_daily_returns.index = self.etf_daily_returns.index.tz_localize(None)
        self.fama_french_data.index = self.fama_french_data.index.tz_localize(None)
        
