import yfinance as yf
import pandas as pd
import os

print("Downloading S&P 500 data...")

# Download S&P 500 price data from 2000 to 2024
sp500 = yf.download("^GSPC", start="2000-01-01", end="2024-01-01")
sp500 = sp500[["Open", "High", "Low", "Close", "Volume"]]

# Add features that capture market behavior
sp500["Daily_Return"] = sp500["Close"].pct_change()
sp500["Volatility_30d"] = sp500["Daily_Return"].rolling(30).std()
sp500["MA_50"] = sp500["Close"].rolling(50).mean()
sp500["MA_200"] = sp500["Close"].rolling(200).mean()
sp500["Golden_Cross"] = (sp500["MA_50"] > sp500["MA_200"]).astype(int)

# Label market regimes based on real historical events
def label_regime(date):
    if date < pd.Timestamp("2007-10-01"):
        return "bull"        # Pre-crisis bull market
    elif date < pd.Timestamp("2009-03-01"):
        return "bear"        # 2008 financial crisis
    elif date < pd.Timestamp("2020-02-01"):
        return "bull"        # Recovery + long bull market
    elif date < pd.Timestamp("2020-04-01"):
        return "bear"        # COVID crash
    elif date < pd.Timestamp("2022-01-01"):
        return "bull"        # Stimulus recovery
    else:
        return "volatile"    # Rate hikes era

sp500["Regime"] = sp500.index.map(label_regime)

# Drop NaN rows from rolling calculations
sp500 = sp500.dropna()

# Save as parquet (optimized for Spark)
os.makedirs("data/raw", exist_ok=True)
sp500.to_parquet("data/raw/sp500_with_regimes.parquet")

print(f"Done! Dataset shape: {sp500.shape}")
print(f"Regime distribution:\n{sp500['Regime'].value_counts()}")
print("Saved to data/raw/sp500_with_regimes.parquet")
