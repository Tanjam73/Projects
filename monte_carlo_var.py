
import numpy as np
import pandas as pd
import yfinance as yf
from pathlib import Path
from scipy.stats import norm

# ----------------------------- USER SETTINGS ----------------------------- #
TICKERS = [
    "AARTIIND.NS", "BSOFT.NS", "CASTROLIND.NS", "GLENMARK.NS",
    "HDFCBANK.NS", "IDFCFIRSTB.NS", "ITC.NS", "RELIANCE.NS", "TVSMOTOR.NS"
]

QUANTITIES = np.array([18, 16, 60, 12, 17, 149, 58, 9, 6], dtype=float)

START_DATE = "2022-01-01"       # 3-year look-back
CONF_LEVELS = [0.95, 0.99]      # VaR confidence levels
N_SIM       = 50_000            # number of Monte Carlo paths
SEED        = 42                # reproducibility
# ------------------------------------------------------------------------ #

def download_prices(tickers, start):
    """Fetch adjusted close prices."""
    data = yf.download(tickers, start=start)["Adj Close"]
    return data.dropna(how="all")

def build_returns(price_df):
    """Compute daily log returns."""
    return np.log(price_df / price_df.shift(1)).dropna()

def portfolio_statistics(returns, weights):
    """Annualised mean vector and covariance matrix."""
    mu = returns.mean().values          # daily mean
    cov = returns.cov().values          # daily covariance
    return mu, cov

def simulate_paths(mu, cov, weights, paths, seed=None):
    """
    Generate correlated daily P/L for the portfolio.
    Returns array of simulated P/L values of length = paths.
    """
    n_assets = len(mu)
    rng = np.random.default_rng(seed)
    # Cholesky factorisation for speed & stability
    chol = np.linalg.cholesky(cov)
    # draw iid standard normals
    z = rng.standard_normal((n_assets, paths))
    # correlated returns
    correlated = mu.reshape(-1, 1) + chol @ z
    # convert to portfolio P/L
    port_returns = weights @ correlated     # shape: (paths,)
    return port_returns

def var_cvar(pl_array, alpha):
    """Compute VaR and CVaR at confidence level alpha."""
    var = -np.quantile(pl_array, 1 - alpha)
    cvar = -pl_array[pl_array <= -var].mean()
    return var, cvar

def backtest_var(historical_pl, var_series):
    """Kupiec unconditional coverage test of VaR exceptions."""
    exceptions = historical_pl < -var_series
    n = len(exceptions)
    x = exceptions.sum()
    # Kupiec LR statistic
    p = 1 - CONF_LEVELS[0]   # use 95 % test by default
    lr_uc = -2 * (x*np.log((1-p)/p) + (n-x)*np.log(1 - (1-p)/(1-p)))
    return x, n, lr_uc

def main():
    # 1. Download prices
    prices = download_prices(TICKERS, START_DATE)

    # 2. Build returns & current market values
    rets = build_returns(prices)
    latest_prices = prices.iloc[-1].values
    position_values = QUANTITIES * latest_prices
    port_value = position_values.sum()
    weights = position_values / port_value

    # 3. Estimate moments
    mu, cov = portfolio_statistics(rets, weights)

    # 4. Monte Carlo simulation
    sim_pl = simulate_paths(mu, cov, weights, N_SIM, SEED) * port_value

    # 5. VaR / CVaR
    results = []
    for cl in CONF_LEVELS:
        var, cvar = var_cvar(sim_pl, cl)
        results.append((cl, var, cvar))

    # 6. Output
    print(f"\nOne-Day Monte Carlo VaR (paths = {N_SIM:,})")
    print(f"Portfolio market value: ₹ {port_value:,.0f}\n")
    for cl, var, cvar in results:
        print(f"{int(cl*100):>3}% VaR  : ₹ {var:,.0f}")
        print(f"{int(cl*100):>3}% CVaR : ₹ {cvar:,.0f}\n")

    # 7. Optional back-test (95 % VaR vs last 250 days)
    hist_pl = (rets @ position_values).iloc[-250:]
    hist_var = np.repeat(results[0][1], len(hist_pl))
    exc, n, lr = backtest_var(hist_pl.values, hist_var)
    print("Back-test (last 250 trading days, 95 % VaR):")
    print(f"Exceptions: {exc} of {n}  (expected ≈ {0.05*n:.1f})")
    print(f"Kupiec LR statistic: {lr:.2f}\n")

    # 8. Save simulated distribution for audit
    out = Path("simulated_pl.csv")
    pd.Series(sim_pl, name="Simulated_PnL").to_csv(out, index=False)
    print(f"Simulated P/L distribution saved to {out.resolve()}")

if __name__ == "__main__":
    main()
