import yfinance as yf
import pandas as pd
import os

print("Loading S&P 500 ticker list...")

# Hardcoded S&P 500 tickers with sectors (avoids Wikipedia SSL issues)
stocks = {
    # Technology
    "AAPL": "Technology", "MSFT": "Technology", "NVDA": "Technology",
    "GOOGL": "Technology", "META": "Technology", "AVGO": "Technology",
    "ORCL": "Technology", "CSCO": "Technology", "ADBE": "Technology",
    "CRM": "Technology", "AMD": "Technology", "INTC": "Technology",
    "QCOM": "Technology", "TXN": "Technology", "IBM": "Technology",
    "NOW": "Technology", "INTU": "Technology", "AMAT": "Technology",
    "MU": "Technology", "LRCX": "Technology",
    # Financials
    "JPM": "Financials", "BAC": "Financials", "WFC": "Financials",
    "GS": "Financials", "MS": "Financials", "BLK": "Financials",
    "C": "Financials", "AXP": "Financials", "USB": "Financials",
    "PNC": "Financials", "SCHW": "Financials", "COF": "Financials",
    "CB": "Financials", "MMC": "Financials", "AON": "Financials",
    # Healthcare
    "UNH": "Healthcare", "JNJ": "Healthcare", "LLY": "Healthcare",
    "ABBV": "Healthcare", "MRK": "Healthcare", "TMO": "Healthcare",
    "ABT": "Healthcare", "DHR": "Healthcare", "PFE": "Healthcare",
    "AMGN": "Healthcare", "BSX": "Healthcare", "SYK": "Healthcare",
    "ISRG": "Healthcare", "MDT": "Healthcare", "CVS": "Healthcare",
    # Consumer
    "AMZN": "Consumer", "TSLA": "Consumer", "HD": "Consumer",
    "MCD": "Consumer", "NKE": "Consumer", "SBUX": "Consumer",
    "TGT": "Consumer", "LOW": "Consumer", "BKNG": "Consumer",
    "CMG": "Consumer", "YUM": "Consumer", "ORLY": "Consumer",
    # Energy
    "XOM": "Energy", "CVX": "Energy", "COP": "Energy",
    "EOG": "Energy", "SLB": "Energy", "MPC": "Energy",
    "PSX": "Energy", "VLO": "Energy", "PXD": "Energy",
    "OXY": "Energy",
    # Industrials
    "CAT": "Industrials", "DE": "Industrials", "BA": "Industrials",
    "HON": "Industrials", "UPS": "Industrials", "RTX": "Industrials",
    "LMT": "Industrials", "GE": "Industrials", "MMM": "Industrials",
    "FDX": "Industrials",
    # Communications
    "NFLX": "Communications", "DIS": "Communications", "CMCSA": "Communications",
    "T": "Communications", "VZ": "Communications", "TMUS": "Communications",
    # Utilities & Real Estate
    "NEE": "Utilities", "DUK": "Utilities", "SO": "Utilities",
    "AMT": "Real Estate", "PLD": "Real Estate", "EQIX": "Real Estate",
}

symbols = list(stocks.keys())
print(f"Downloading {len(symbols)} stocks from 2000-2024...")
print("This will take 3-5 minutes, please wait...")

# Download all stocks
raw = yf.download(
    symbols,
    start="2000-01-01",
    end="2024-01-01",
    group_by="ticker",
    auto_adjust=True,
    threads=True
)

print("Download complete! Processing data...")

# Reshape into long format
frames = []
for ticker in symbols:
    try:
        df = raw[ticker][["Open", "High", "Low", "Close", "Volume"]].copy()
        df = df.dropna()
        df["Ticker"] = ticker
        df["Sector"] = stocks[ticker]
        df["Daily_Return"] = df["Close"].pct_change()
        df["Volatility_30d"] = df["Daily_Return"].rolling(30).std()
        df["MA_50"] = df["Close"].rolling(50).mean()
        df["MA_200"] = df["Close"].rolling(200).mean()
        df["Golden_Cross"] = (df["MA_50"] > df["MA_200"]).astype(int)
        df = df.dropna()
        frames.append(df)
    except Exception as e:
        print(f"Skipping {ticker}: {e}")
        continue

# Combine all stocks
full_df = pd.concat(frames)
full_df.index.name = "Date"
full_df = full_df.reset_index()

# Label market regimes
def label_regime(date):
    if date < pd.Timestamp("2007-10-01"):
        return "bull"
    elif date < pd.Timestamp("2009-03-01"):
        return "bear"        # 2008 financial crisis
    elif date < pd.Timestamp("2020-02-01"):
        return "bull"
    elif date < pd.Timestamp("2020-04-01"):
        return "bear"        # COVID crash
    elif date < pd.Timestamp("2022-01-01"):
        return "bull"
    else:
        return "volatile"    # Rate hikes era

full_df["Regime"] = full_df["Date"].map(label_regime)

# Save as parquet
os.makedirs("data/raw", exist_ok=True)
full_df.to_parquet("data/raw/sp500_all_stocks.parquet", index=False)

print(f"\n✅ Done!")
print(f"Dataset shape: {full_df.shape}")
print(f"Unique stocks: {full_df['Ticker'].nunique()}")
print(f"Date range: {full_df['Date'].min()} to {full_df['Date'].max()}")
print(f"Regime distribution:\n{full_df['Regime'].value_counts()}")
print(f"Sectors covered:\n{full_df['Sector'].value_counts()}")
print(f"Saved to data/raw/sp500_all_stocks.parquet")
