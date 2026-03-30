import pandas as pd
import numpy as np
import os

print("Loading dataset...")
df = pd.read_parquet("data/raw/sp500_all_stocks.parquet")
df["Date"] = pd.to_datetime(df["Date"])
df = df.sort_values(["Ticker", "Date"]).reset_index(drop=True)

# ── Define drift windows based on real historical events ──────────────────────
WINDOWS = {
    "pretrain": {
        "start": "2000-01-01",
        "end":   "2007-09-30",
        "label": 0,
        "description": "Normal bull market — model trains here"
    },
    "drift_2008": {
        "start": "2007-10-01",
        "end":   "2009-03-01",
        "label": 1,
        "description": "2008 Financial Crisis — first drift event"
    },
    "recovery": {
        "start": "2009-03-02",
        "end":   "2020-01-31",
        "label": 0,
        "description": "Long bull market recovery"
    },
    "drift_covid": {
        "start": "2020-02-01",
        "end":   "2020-04-01",
        "label": 1,
        "description": "COVID crash — second drift event"
    },
    "post_covid": {
        "start": "2020-04-02",
        "end":   "2021-12-31",
        "label": 0,
        "description": "Stimulus recovery"
    },
    "drift_volatile": {
        "start": "2022-01-01",
        "end":   "2024-01-01",
        "label": 1,
        "description": "Rate hike era — third drift event"
    },
}

# ── Tag each row with its window ──────────────────────────────────────────────
def assign_window(date):
    for window_name, info in WINDOWS.items():
        if pd.Timestamp(info["start"]) <= date <= pd.Timestamp(info["end"]):
            return window_name, info["label"]
    return "unknown", -1

print("Assigning drift windows...")
df[["Window", "Drift_Label"]] = df["Date"].apply(
    lambda d: pd.Series(assign_window(d))
)

# ── Add additional drift features ─────────────────────────────────────────────
print("Engineering drift features...")

# Rolling z-score of daily return (measures how abnormal returns are)
df["Return_Zscore"] = (
    df.groupby("Ticker")["Daily_Return"]
    .transform(lambda x: (x - x.rolling(60).mean()) / (x.rolling(60).std() + 1e-9))
)

# Volatility ratio (current vol vs historical vol — key drift signal)
df["Volatility_Ratio"] = (
    df.groupby("Ticker")["Volatility_30d"]
    .transform(lambda x: x / (x.rolling(120).mean() + 1e-9))
)

# Volume spike (abnormal volume = market stress)
df["Volume_Zscore"] = (
    df.groupby("Ticker")["Volume"]
    .transform(lambda x: (x - x.rolling(60).mean()) / (x.rolling(60).std() + 1e-9))
)

# Drop NaN rows from rolling calculations
df = df.dropna()

# ── Split into train / drift windows and save ─────────────────────────────────
os.makedirs("data/processed", exist_ok=True)

# Save full structured dataset
df.to_parquet("data/processed/structured_dataset.parquet", index=False)

# Save pretrain data separately (XGBoost trains on this)
pretrain = df[df["Window"] == "pretrain"]
pretrain.to_parquet("data/processed/pretrain_data.parquet", index=False)

# Save all drift periods separately (detectors tested on this)
drift = df[df["Drift_Label"] == 1]
drift.to_parquet("data/processed/drift_data.parquet", index=False)

# Save summary
print("\n✅ Done!")
print(f"Full dataset:     {df.shape}")
print(f"Pretrain window:  {pretrain.shape} ({pretrain['Date'].min().date()} to {pretrain['Date'].max().date()})")
print(f"Drift windows:    {drift.shape}")
print(f"\nWindow distribution:")
print(df.groupby(['Window', 'Drift_Label']).size().reset_index(name='rows'))
print(f"\nSaved to data/processed/")
