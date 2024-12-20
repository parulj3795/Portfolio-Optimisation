# Portfolio Optimisation

A Python-based application for portfolio optimisation, risk management, and backtesting, leveraging Monte Carlo simulations and the Efficient Frontier from Modern Portfolio Theory.

## Features

- Fetches historical stock data using Yahoo Finance (`yfinance`).
- Computes expected returns, daily returns, and covariance matrices.
- Performs portfolio optimisation using:
  - **Monte Carlo simulations** for risk-return trade-off.
  - **Efficient Frontier** to visualise optimal portfolios.
- Backtests portfolio performance against a benchmark index (e.g., S&P 500).
- Generates insightful visualisations:
  - Adjusted close prices.
  - Correlation matrix.
  - Portfolio weight distribution.
  - Portfolio performance vs benchmark.
- Supports user-defined configurations via JSON files.

## Project Structure

```
├── src/
│   ├── data/
│   │   ├── fetch_data.py
│   │   └── compute_daily_returns.py
│   ├── optimisation/
│   │   └── portfolio_opt.py
│   ├── visualisations/
│   │   └── plot_results.py
│   ├── backtesting/
│   │   └── backtest.py
│   └── utils/
│       └── helpers.py
├── config/
│   └── config.json
├── outputs/
│   └── [Generated plots and results]
├── logs/
│   └── app.log
├── main.py
└── README.md
```

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/portfolio-optimisation.git
   cd portfolio-optimisation
   ```

2. Set up a virtual environment and activate it:
   ```bash
   python -m venv env
   source env/bin/activate  # On Windows: .\env\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Create necessary directories:
   ```bash
   mkdir logs outputs
   ```

5. Update the configuration file `config/config.json` with your desired parameters.

## Usage

1. Run the main script:
   ```bash
   python main.py
   ```

2. Follow the prompts to input your initial investment amount.

3. Review generated outputs in the `outputs` directory.

### Example Configuration (`config/config.json`):
```json
{
  "tickers": ["AAPL", "GOOGL", "AMZN"],
  "start_date": "2018-01-01",
  "end_date": "2023-01-01",
  "risk_free_rate": 0.02,
  "num_simulations": 10000,
  "backtest_start_date": "2021-01-01",
  "backtest_end_date": "2023-01-01",
  "benchmark_symbol": "^GSPC"
}
```

## Dependencies

- `numpy`
- `pandas`
- `matplotlib`
- `yfinance`
- `scipy`
- `logging`

Install all dependencies via `requirements.txt`.

## Licence

This project is licensed under the MIT Licence. See `LICENCE` for more details.

## Acknowledgements

- [Yahoo Finance](https://finance.yahoo.com/) for providing historical stock data.
- Modern Portfolio Theory for inspiring the project’s core optimisation algorithms.