import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt

ticker = "AAPL"
start = "2020-01-01"
end = "2024-01-01"

data = yf.download(ticker, start=start, end=end)

data["SMA20"] = data["Close"].rolling(20).mean()
data["SMA50"] = data["Close"].rolling(50).mean()

cash = 100000
shares = 0

portfolio_values = []
signals = []

for i in range(50, len(data)):

    price = float(data["Close"].iloc[i])

    sma20 = float(data["SMA20"].iloc[i])
    sma50 = float(data["SMA50"].iloc[i])

    if sma20 > sma50 and shares == 0:

        shares = int(cash // price)
        cash -= shares * price
        signals.append(("BUY", data.index[i], price))

    elif sma20 < sma50 and shares > 0:

        cash += shares * price
        shares = 0
        signals.append(("SELL", data.index[i], price))

    portfolio = cash + shares * price
    portfolio_values.append(portfolio)

final_price = float(data["Close"].iloc[-1])
final_value = cash + shares * final_price

returns = (final_value - 100000) / 100000 * 100

print("Initial Capital :", 100000)
print("Final Portfolio :", round(final_value, 2))
print("Return (%)      :", round(returns, 2))

print("\nTrades:")
for signal in signals:
    print(signal)

plt.figure(figsize=(12,6))
plt.plot(portfolio_values)
plt.title("Portfolio Value Over Time")
plt.xlabel("Trading Days")
plt.ylabel("Portfolio Value")
plt.grid(True)
plt.show()
