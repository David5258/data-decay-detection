import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import joblib
import json
from scipy import stats
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import uvicorn
import warnings
warnings.filterwarnings("ignore")

# Input schema
class MarketData(BaseModel):
    features: List[float]
    ticker: str = "UNKNOWN"
    date: str = "UNKNOWN"

# Response schema
class DriftResponse(BaseModel):
    ticker: str
    date: str
    ks_score: float
    psi_score: float
    autoencoder_score: float
    drift_detected: bool
    confidence: str
    details: dict

# Autoencoder definition
class DriftAutoencoder(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, 8),
        )
        self.decoder = nn.Sequential(
            nn.Linear(8, 32), nn.ReLU(),
            nn.Linear(32, 64), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(64, input_dim),
        )
    def forward(self, x):
        return self.decoder(self.encoder(x))
    def reconstruction_error(self, x):
        with torch.no_grad():
            return torch.mean((x - self.forward(x)) ** 2, dim=1).item()

# Load models and reference data
print("Loading models and reference data...")

FEATURES = [
    "Daily_Return", "Volatility_30d", "Return_Zscore",
    "Volatility_Ratio", "Volume_Zscore", "Momentum_5d",
    "Momentum_20d", "Drawdown", "Volume_Ratio",
    "Sector_Avg_Return", "Sector_Avg_Volatility",
    "Return_vs_Sector", "Golden_Cross"
]

# Load reference data (pretrain window)
df = pd.read_parquet("data/spark_output/featured_dataset.parquet")
reference_data = df[df["Window"] == "pretrain"][FEATURES].dropna()
print(f"Reference data loaded: {reference_data.shape}")

# Load autoencoder
autoencoder = DriftAutoencoder(len(FEATURES))
autoencoder.load_state_dict(
    torch.load("models/autoencoder/model.pth", map_location="cpu")
)
autoencoder.eval()

# Load scaler
scaler = joblib.load("models/autoencoder/scaler.pkl")

print("All models loaded successfully!")

# Thresholds
KS_THRESHOLD  = 0.15
PSI_THRESHOLD = 0.20
AE_THRESHOLD  = 0.10

# Helper functions
def compute_ks_score(reference, current_value, feature_idx):
    ref_values = reference.iloc[:, feature_idx].values
    ks_stat, _ = stats.ks_2samp(ref_values, [current_value] * 100)
    return float(ks_stat)

def compute_psi_score(reference, current_value, feature_idx, bins=10):
    ref_values = reference.iloc[:, feature_idx].values
    breakpoints = np.linspace(ref_values.min(), ref_values.max(), bins + 1)
    ref_counts, _ = np.histogram(ref_values, bins=breakpoints)
    cur_counts, _ = np.histogram([current_value], bins=breakpoints)
    ref_pct = ref_counts / len(ref_values) + 1e-6
    cur_pct = cur_counts / 1 + 1e-6
    psi = np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct))
    return float(abs(psi))

# Initialize FastAPI app
app = FastAPI(
    title="Data Decay Detection API",
    description="Real-time market regime drift detection using KS Test, PSI and PyTorch Autoencoder",
    version="1.0.0"
)

@app.get("/")
def root():
    return {
        "message": "Data Decay Detection API",
        "version": "1.0.0",
        "endpoints": ["/detect", "/health", "/features"]
    }

@app.get("/health")
def health():
    return {
        "status": "healthy",
        "models_loaded": True,
        "reference_rows": len(reference_data),
        "features": len(FEATURES)
    }

@app.get("/features")
def get_features():
    return {
        "features": FEATURES,
        "count": len(FEATURES),
        "description": "Expected feature order for /detect endpoint"
    }

@app.post("/detect", response_model=DriftResponse)
def detect_drift(data: MarketData):
    if len(data.features) != len(FEATURES):
        raise HTTPException(
            status_code=400,
            detail=f"Expected {len(FEATURES)} features, got {len(data.features)}"
        )

    features_array = np.array(data.features)

    # KS score (mean across all features)
    ks_scores = [
        compute_ks_score(reference_data, features_array[i], i)
        for i in range(len(FEATURES))
    ]
    ks_score = float(np.mean(ks_scores))

    # PSI score (mean across all features)
    psi_scores = [
        compute_psi_score(reference_data, features_array[i], i)
        for i in range(len(FEATURES))
    ]
    psi_score = float(np.mean(psi_scores))

    # Autoencoder reconstruction error
    scaled = scaler.transform([features_array]).astype(np.float32)
    x_tensor = torch.FloatTensor(scaled)
    ae_score = autoencoder.reconstruction_error(x_tensor)

    # Drift decision
    ks_drift  = ks_score  > KS_THRESHOLD
    psi_drift = psi_score > PSI_THRESHOLD
    ae_drift  = ae_score  > AE_THRESHOLD

    votes = sum([ks_drift, psi_drift, ae_drift])
    drift_detected = votes >= 2

    if votes == 3:
        confidence = "high"
    elif votes == 2:
        confidence = "medium"
    elif votes == 1:
        confidence = "low"
    else:
        confidence = "none"

    return DriftResponse(
        ticker=data.ticker,
        date=data.date,
        ks_score=round(ks_score, 6),
        psi_score=round(psi_score, 6),
        autoencoder_score=round(ae_score, 6),
        drift_detected=drift_detected,
        confidence=confidence,
        details={
            "ks_drift":    ks_drift,
            "psi_drift":   psi_drift,
            "ae_drift":    ae_drift,
            "votes":       votes,
            "thresholds": {
                "ks":  KS_THRESHOLD,
                "psi": PSI_THRESHOLD,
                "ae":  AE_THRESHOLD
            }
        }
    )

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)

    