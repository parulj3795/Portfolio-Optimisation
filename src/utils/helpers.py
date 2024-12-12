import pandas as pd

def compute_expected_returns(daily_returns):
    '''
    Compute expected/mean returns from the price data.
    The expected return is the mean of historial daily return of each asset.
    '''
    return daily_returns.mean()

def compute_covariance_matrix(daily_returns):
    '''
    Compute the covariance matrix of asset returns. 
    The covariance matrix measures how asset returns move relative to each other.
    This will be used to calculate portfolio volatility.
    '''
    return daily_returns.cov()