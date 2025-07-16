import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Portfolio definitions (update tickers/weights as needed)
portfolios = {
    "Portfolio 1": {"RELIANCE.NS": 0.179, "HDFCBANK.NS": 0.168, "ITC.NS": 0.170},
    "Portfolio 2": {"ITC.NS": 0.195, "RELIANCE.NS": 0.150, "HDFCBANK.NS": 0.145},
    "Portfolio 3": {"ITC.NS": 0.166, "RELIANCE.NS": 0.128, "HDFCBANK.NS": 0.123}
}

# Download stock data
tickers = list({ticker for p in portfolios.values() for ticker in p})
data = yf.download(tickers, start="2022-07-10", end="2025-07-10", auto_adjust=True)["Close"]

# Check for missing tickers (if any)
missing = set(tickers) - set(data.columns)
if missing:
    print(f"⚠️ Missing tickers: {missing}. Proceeding with available data.")

# Clean data, compute returns, and annualize
data = data.dropna(axis=1, how='all')  
returns = np.log(data / data.shift(1)).dropna()
mean_returns = returns.mean() * 252
cov_matrix = returns.cov() * 252

# Monte Carlo simulation for efficient frontier
num_portfolios = 10000
results = np.zeros((3, num_portfolios))
for i in range(num_portfolios):
    weights = np.random.random(len(mean_returns))
    weights /= np.sum(weights)
    port_return = np.dot(weights, mean_returns)
    port_vol = np.sqrt(weights.T @ cov_matrix @ weights)
    results[0, i], results[1, i], results[2, i] = port_vol, port_return, (port_return - 0.07) / port_vol

# Plotting
plt.figure(figsize=(10, 6))
plt.scatter(results[0], results[1], c=results[2], cmap='viridis', s=10, alpha=0.3)
plt.colorbar(label='Sharpe Ratio')
plt.title("Efficient Frontier with Simulated Portfolios")
plt.xlabel("Annualized Volatility (σ)")
plt.ylabel("Annualized Return (μ)")
plt.grid(True)

# Save the plot to a file
plt.savefig("efficient_frontier.png", dpi=300, bbox_inches='tight')
plt.show()
