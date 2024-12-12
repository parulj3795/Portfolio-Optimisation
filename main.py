# main.py

import os
import sys
import json
import logging
from logging.handlers import RotatingFileHandler
import numpy as np
import yfinance as yf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "src")))

from src.data.fetch_data import fetch_data, compute_daily_returns
from src.optimisation.portfolio_opt import monte_carlo_optimisation, generate_efficient_frontier
from src.backtesting.backtest import backtest_portfolio, plot_backtest, compute_portfolio_value
from src.utils.helpers import compute_expected_returns, compute_covariance_matrix
from src.visualisations.plot_results import plot_correlation_matrix, plot_weight_distribution

# Ensure outputs and logs directories exist
os.makedirs("logs", exist_ok=True)
os.makedirs("outputs", exist_ok=True)

# Create a rotating file handler
handler = RotatingFileHandler(
    "logs/app.log",       # Log file location
    maxBytes=5 * 1024 * 1024,  # Maximum file size: 5 MB
    backupCount=10         # Keep up to 5 backup log files
)

# Set the logging format
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)

# Add the handler to the root logger
logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(handler)

# Log example
logger.info("Application started")

# Load config
with open("config/config.json") as f:
    config = json.load(f)

# Access parameters from config
tickers = config["tickers"]
start_date = config["start_date"]
end_date = config["end_date"]
risk_free_rate = config.get("risk_free_rate", 0.01)  # Default to 1% if not specified
num_simulations = config.get("num_simulations", 10000)
backtest_start_date = config.get("backtest_start_date")
backtest_end_date = config.get("backtest_end_date")
benchmark_symbol = config.get("benchmark_symbol", "^GSPC")  # Default to S&P 500

# Set numpy print options
np.set_printoptions(precision=3, suppress=True)

logger.info(f"Fetching data for tickers: {tickers}")

# Fetch data
data = fetch_data(tickers, start_date, end_date)
data.plot()
plt.title('Adjusted Close Data')
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.savefig('outputs/Data.png', bbox_inches='tight')
plt.close()

logger.info("Computing daily returns")

# Compute daily returns
daily_returns = compute_daily_returns(data)

logger.info("Computing expected returns and covariance matrix")

# Compute expected returns and covariance matrix
expected_return = compute_expected_returns(daily_returns)
covariance_matrix = compute_covariance_matrix(daily_returns)

# Normalize expected returns
expected_returns_scaled = (expected_return - np.mean(expected_return)) / np.std(expected_return)
expected_return = expected_returns_scaled

# Plot correlation matrix
plot_correlation_matrix(daily_returns, filename='outputs/Correlation_Matrix.png')

logger.info("Optimizing portfolio using Monte Carlo simulation")

# Optimize portfolio using Monte Carlo
mc_results = monte_carlo_optimisation(expected_return, covariance_matrix, risk_free_rate, num_simulations)

# Find the portfolio with the maximum Sharpe ratio
optimal_portfolio = mc_results.loc[mc_results["Sharpe_Ratio"].idxmax()]
optimal_weights = optimal_portfolio["Weights"]

logger.info(f"Optimal portfolio weights: {optimal_weights}")

print("Optimal Portfolio Return:", np.round(optimal_portfolio["Returns"],2))
print("Optimal Portfolio Volatility:", np.round(optimal_portfolio["Volatility"],2))
print("Optimal Sharpe Ratio:", np.round(optimal_portfolio["Sharpe_Ratio"],2))

# Plot portfolio weight distribution
plot_weight_distribution(optimal_weights, tickers, filename='outputs/Portfolio_Weight_Distribution_MC.png')

logger.info("Generating efficient frontier")

# Generate efficient frontier
frontier = generate_efficient_frontier(expected_return, covariance_matrix)

# Plot efficient frontier and Monte Carlo results
plt.figure(figsize=(10,6))
plt.plot(frontier[:,0], frontier[:,1], label='Efficient Frontier (Theoretical)')

plt.scatter(
    mc_results["Volatility"],
    mc_results["Returns"],
    c=mc_results["Sharpe_Ratio"],
    cmap="viridis",
    alpha=0.5
)

plt.colorbar(label="Sharpe Ratio")
plt.xlabel("Portfolio Volatility")
plt.ylabel("Portfolio Return")
plt.title("Monte Carlo Simulation: Efficient Frontier")
plt.scatter(
    optimal_portfolio["Volatility"],
    optimal_portfolio["Returns"],
    color="red",
    label="Optimal Portfolio (Monte Carlo)",
    marker="*",
    s=200
)
plt.legend()
plt.grid(True)
plt.savefig('outputs/MC_Optimisation.png', bbox_inches='tight')
plt.close()

# Backtesting
if backtest_start_date and backtest_end_date:
    logger.info(f"Backtesting portfolio from {backtest_start_date} to {backtest_end_date}")

    # Fetch backtest data
    backtest_data = fetch_data(tickers, backtest_start_date, backtest_end_date)
    backtest_daily_returns = compute_daily_returns(backtest_data)

    # Perform backtest
    cumulative_returns = backtest_portfolio(optimal_weights, backtest_daily_returns)
    plot_backtest(cumulative_returns, filename='outputs/Portfolio_Backtest.png')

    benchmark_data = yf.download(benchmark_symbol, start=backtest_start_date, end=backtest_end_date)['Adj Close']
    benchmark_returns = benchmark_data.pct_change().dropna()
    benchmark_cumulative = (1 + benchmark_returns).cumprod()

    # Plot portfolio vs benchmark
    plt.figure(figsize=(10, 6))
    plt.plot(cumulative_returns, label="Optimized Portfolio")
    plt.plot(benchmark_cumulative, label=f"{benchmark_symbol} Benchmark", linestyle="--")
    plt.title("Portfolio vs Benchmark")
    plt.xlabel("Date")
    plt.ylabel("Cumulative Returns")
    plt.legend()
    plt.grid()
    plt.savefig('outputs/Portfolio_vs_Benchmark.png', bbox_inches='tight')
    plt.close()

else:
    logger.warning("Backtest start and end dates not provided. Skipping backtesting.")

# User Input: Investment Amount
initial_investment = float(input("Enter your total investment amount (e.g., 10000): "))

# Backtesting Period: Use test data (e.g., 2021-2024 returns)
test_data = fetch_data(tickers, "2023-01-01", "2024-12-31")
test_returns = compute_daily_returns(test_data)

# Calculate final portfolio value and cumulative return
final_value, cumulative_return = compute_portfolio_value(optimal_weights, test_returns, initial_investment)

# Display the result
print(f"Your initial investment of £{initial_investment:.2f} would grow to £{final_value:.2f} over the testing period.")
print(f"Cumulative portfolio return: {cumulative_return * 100:.2f}%")

logger.info("Application finished successfully.")
