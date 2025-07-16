import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Your 3 portfolios
portfolios = {
    "Portfolio 1": {
        "RELIANCE.NS": 0.179,
        "HDFCBANK.NS": 0.168,
        "ITC.NS": 0.170,
        "IDFCFIRSTB.NS": 0.080,
        "BSOFT.NS": 0.081,
        "TVSMOTOR.NS": 0.085,
        "AARTIIND.NS": 0.080,
        "CASTROLIND.NS": 0.079,
        "GLENMARK.NS": 0.078
    },
    "Portfolio 2": {
        "ITC.NS": 0.195,
        "RELIANCE.NS": 0.150,
        "HDFCBANK.NS": 0.145,
        "CASTROLIND.NS": 0.095,
        "BSOFT.NS": 0.085,
        "TVSMOTOR.NS": 0.085,
        "IDFCFIRSTB.NS": 0.080,
        "AARTIIND.NS": 0.080,
        "GLENMARK.NS": 0.080
    },
    "Portfolio 3": {
        "ITC.NS": 0.166,
        "RELIANCE.NS": 0.128,
        "HDFCBANK.NS": 0.123,
        "SBIN.NS": 0.084,
        "CASTROLIND.NS": 0.081,
        "BSOFT.NS": 0.073,
        "TVSMOTOR.NS": 0.073,
        "IDFCFIRSTB.NS": 0.068,
        "AARTIIND.NS": 0.068
    }
}

# All unique tickers
tickers = list({ticker for p in portfolios.values() for ticker in p})

# Download 3 years of data
data = yf.download(tickers, start="2022-07-10", end="2025-07-10", auto_adjust=True)

# Check which tickers are available
available = [t for t in tickers if t in data.columns]
missing = set(tickers) - set(available)
if missing:
    print("⚠️ Warning: Missing tickers:", missing)

# Filter only valid tickers
adj_close = data[available]

# Clean up any zero or missing prices
adj_close = adj_close.replace(0, np.nan).dropna(axis=0, how='any')

# Compute log returns
log_returns = np.log(adj_close / adj_close.shift(1)).dropna()

# Annualized mean returns and covariance matrix
mean_returns = log_returns.mean() * 252
cov_matrix = log_returns.cov() * 252
rf = 0.07  # 7% risk-free rate

# Portfolio metrics
results = []
tickers_used = mean_returns.index.tolist()
for name, weights_dict in portfolios.items():
    weights = np.array([weights_dict.get(t, 0) for t in tickers_used])
    port_return = np.dot(weights, mean_returns[tickers_used])
    port_vol = np.sqrt(weights.T @ cov_matrix.loc[tickers_used, tickers_used].values @ weights)
    sharpe = (port_return - rf) / port_vol
    results.append((port_vol, port_return, sharpe, name))

# Efficient Frontier
num_points = 10000
all_weights = np.random.dirichlet(np.ones(len(tickers_used)), num_points)
rets = []
vols = []
for w in all_weights:
    ret = np.dot(w, mean_returns[tickers_used])
    vol = np.sqrt(w.T @ cov_matrix.loc[tickers_used, tickers_used].values @ w)
    rets.append(ret)
    vols.append(vol)

# Plot
plt.figure(figsize=(10, 6))
plt.scatter(vols, rets, c=(np.array(rets)-rf)/np.array(vols), cmap='viridis', s=2, alpha=0.6)
plt.colorbar(label='Sharpe Ratio')

# Overlay real portfolios
for vol, ret, sharpe, name in results:
    plt.scatter(vol, ret, c='red', s=100, edgecolors='black')
    plt.text(vol + 0.002, ret, name, fontsize=9)

plt.title("Efficient Frontier with 3 Portfolios (2022–2025)")
plt.xlabel("Annualized Volatility (σ)")
plt.ylabel("Annualized Expected Return (μ)")
plt.grid(True)
plt.tight_layout()
plt.show()
