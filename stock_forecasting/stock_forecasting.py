"""
Next-day stock closing price forecasting using Linear Regression on Yahoo Finance historical data.
Downloads full-year 2022 OHLCV data for AAPL, MSFT, and GOOGL via yfinance, creates a next-day
target via shift(-1), performs an 80/20 chronological train/test split, trains LinearRegression,
evaluates with MSE and R2, plots actual vs predicted with a split marker, then produces a
side-by-side bar chart comparing both metrics across all three stocks saved as stocks_comparison.png.
"""

import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

warnings.filterwarnings("ignore")


def forecast_stock(ticker, start="2022-01-01", end="2023-01-01"):
    print(f"\n=== {ticker} ===")

    raw = yf.download(ticker, start=start, end=end, progress=False)
    if raw.empty:
        print(f"No data returned for {ticker}")
        return None

    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)

    data = raw[['Close']].copy()
    close = data['Close'].squeeze()

    print(f"Trading days : {len(data)}")
    print(f"Price range  : ${float(close.min()):.2f} - ${float(close.max()):.2f}")

    plt.figure(figsize=(12, 4))
    plt.plot(data.index, close, color='steelblue', linewidth=1.5)
    plt.title(f"{ticker} Closing Prices (2022)")
    plt.xlabel("Date")
    plt.ylabel("Price (USD)")
    plt.tight_layout()
    plt.savefig(f"{ticker}_prices.png", dpi=150)
    plt.show()
    print(f"Saved: {ticker}_prices.png")

    data = data.copy()
    data['Tomorrow'] = close.shift(-1)
    data = data.dropna()

    X = data[['Close']].values
    y = data['Tomorrow'].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred_all = model.predict(X)
    y_pred_test = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred_test)
    r2 = r2_score(y_test, y_pred_test)
    print(f"Test MSE : {mse:.4f}")
    print(f"Test R2  : {r2:.4f}")

    split_idx = len(X_train)
    plt.figure(figsize=(12, 5))
    plt.plot(data.index, y, label="Actual Next-Day Price",
             color='royalblue', linewidth=1.5)
    plt.plot(data.index, y_pred_all, label="Predicted Next-Day Price",
             color='orangered', linestyle='--', linewidth=1.5)
    plt.axvline(data.index[split_idx], color='gray', linestyle=':',
                linewidth=1.5, label='Train/Test Split')
    plt.title(f"{ticker} - Actual vs Predicted Next-Day Closing Price")
    plt.xlabel("Date")
    plt.ylabel("Price (USD)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{ticker}_prediction.png", dpi=150)
    plt.show()
    print(f"Saved: {ticker}_prediction.png")

    return {"MSE": round(mse, 4), "R2": round(r2, 4)}


results = {}
for ticker in ["AAPL", "MSFT", "GOOGL"]:
    metrics = forecast_stock(ticker)
    if metrics:
        results[ticker] = metrics

print("\n=== Cross-Stock Performance Comparison ===")
comparison = pd.DataFrame(results).T
print(comparison)

colors = ['steelblue', 'seagreen', 'tomato']
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

comparison['MSE'].plot(kind='bar', ax=axes[0], color=colors, edgecolor='black', width=0.5)
axes[0].set_title("Test MSE by Stock (lower is better)")
axes[0].set_ylabel("Mean Squared Error")
axes[0].tick_params(axis='x', rotation=0)

comparison['R2'].plot(kind='bar', ax=axes[1], color=colors, edgecolor='black', width=0.5)
axes[1].set_title("Test R2 Score by Stock (higher is better)")
axes[1].set_ylabel("R2 Score")
axes[1].tick_params(axis='x', rotation=0)

plt.tight_layout()
plt.savefig("stocks_comparison.png", dpi=150)
plt.show()
print("Comparison chart saved: stocks_comparison.png")
