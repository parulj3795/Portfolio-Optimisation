import numpy as np
from scipy.optimize import minimize
import pandas as pd

def monte_carlo_optimisation(expected_return, covariance_matrix, risk_free_rate, num_simulations=10000):
    '''
    Using MC optimisation for better portfolio optimisation and diversification.
    '''
    num_assets = len(expected_return)
    results = {'Returns': [], 'Volatility': [], 'Sharpe_Ratio': [], 'Weights': []}

    for _ in range(num_simulations):

        # Adding weight caps to ensure no asset gets an outsized allocation
        weights = np.random.dirichlet(np.ones(num_assets))
        #weights = np.clip(weights, 0, 0.2)  # Cap weights to 20% max
        weights = weights / np.sum(weights)  # Normalize weights

        # Portfolio return
        portfolio_return = np.dot(weights, expected_return)

        # Portfolio volatility
        portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(covariance_matrix, weights)))

        # Sharpe ratio
        sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_volatility

        # Results
        results["Returns"].append(portfolio_return)
        results["Volatility"].append(portfolio_volatility)
        results["Sharpe_Ratio"].append(sharpe_ratio)
        results["Weights"].append(weights)

    return pd.DataFrame(results)

def generate_efficient_frontier(expected_returns, covariance_matrix, num_point=50):
    '''
    The Efficient Frontier is a concept from Modern Portfolio Theory (MPT) 
    that represents a set of optimal portfolios offering the highest 
    expected return for a given level of risk. 

    Args:
        expected_returns (np.array): Expected returns vector.
        cov_matrix (np.array): Covariance matrix of returns.
        num_points (int): Number of points to calculate on the frontier.
    Returns:
        np.array: Risk (volatility) and return for portfolios on the frontier.
    '''
    results = []

    target_returns = np.linspace(min(expected_returns), max(expected_returns), num_point)

    # Loop through target returns and find the portfolio with minimum risk
    for target in target_returns:
        constraints = [
            {'type':'eq', 'fun': lambda w: np.sum(w) - 1},
            {'type':'eq', 'fun': lambda w: np.dot(w, expected_returns) - target}
        ]

        num_assets = len(expected_returns)
        bounds = [(0,1) for _ in range(num_assets)]

        # Objective: Minimise risk
        result = minimize(
            lambda w: np.sqrt(np.dot(w.T, np.dot(covariance_matrix, w))),
            np.ones(num_assets) / num_assets,
            bounds=bounds,
            constraints=constraints
        )

        portfolio_volatility = np.sqrt(np.dot(result.x.T, np.dot(covariance_matrix, result.x)))
        results.append((portfolio_volatility, target))

    return np.array(results)