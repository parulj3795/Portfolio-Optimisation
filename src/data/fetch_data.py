import pandas as pd
import yfinance as yf

def fetch_data(tickers, start_date, end_date):
    '''
    Fetch historical data for a list of tickers from Yahnoo Finance.
    Returns a DataFrame with adjusted close prices.
    '''
    data = yf.download(tickers, start=start_date, end=end_date)['Adj Close']
    return data

def compute_daily_returns(data):
    '''
    Compute daily percentage returns from price data.
    '''
    return data.pct_change().dropna()