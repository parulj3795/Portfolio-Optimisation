import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def backtest_portfolio(weights, daily_returns):
    
    portfolio_returns = daily_returns.dot(weights)

    # cumprod = cumulative product
    cumulative_returns = (1 + portfolio_returns).cumprod()

    return cumulative_returns

def plot_backtest(cumulative_returns, filename='outputs/Portfolio_Backtest.png'):

    plt.figure(figsize=(10,6))
    plt.plot(cumulative_returns, label='Portfolio')
    plt.title('Portfolio Backtest: Cumulative Returns')
    plt.xlabel('Date')
    plt.ylabel('Cumulative Returns')
    plt.legend()
    plt.grid()
    plt.savefig(filename, bbox_inches='tight')

def compute_portfolio_value(weights, test_returns, initial_investment):
    """
    Compute the final portfolio value based on the test returns and initial investment.

    Args:
        weights (np.array): Optimized portfolio weights.
        test_returns (np.array): Asset returns during the testing period (T x N matrix).
        initial_investment (float): User's total investment.

    Returns:
        final_value (float): Final portfolio value.
        cumulative_return (float): Portfolio cumulative return.
    """
    # Portfolio returns during the test period
    portfolio_returns = np.dot(test_returns, weights)

    # Cumulative portfolio return
    cumulative_return = np.prod(1 + portfolio_returns) - 1

    # Final portfolio value
    final_value = initial_investment * (1 + cumulative_return)

    return final_value, cumulative_return