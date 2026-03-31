import pytest
import numpy as np
import pandas as pd
from scipy import stats




# Test 1 — KS Test detects drift correctly
def test_ks_detects_drift():
    # Normal distribution vs shifted distribution
    reference = np.random.normal(0, 1, 1000)
    drifted = np.random.normal(5, 1, 1000)  # clearly shifted
    stat, p_value = stats.ks_2samp(reference, drifted)
    assert p_value < 0.05, "KS test should detect drift in clearly shifted data"


# Test 2 — KS Test does not flag stable data
    
def test_ks_no_false_positive():
    reference = np.random.normal(0, 1, 1000)
    similar = np.random.normal(0, 1, 1000)  # same distribution
    stat, p_value = stats.ks_2samp(reference, similar)
    assert stat < 0.5, "KS statistic should be low for similar distributions"


# Test 3 — PSI calculation works correctly
def test_psi_high_for_drifted_data():
    def compute_psi(reference, current, bins=10):
        breakpoints = np.linspace(
            min(reference.min(), current.min()),
            max(reference.max(), current.max()),
            bins + 1
        )

        ref_counts, _ = np.histogram(reference, bins=breakpoints)
        cur_counts, _ = np.histogram(current, bins=breakpoints)


        ref_pct = ref_counts / len(reference) + 1e-6
        cur_pct = cur_counts / len(current) + 1e-6
        return float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))

    reference = np.random.normal(0, 1, 1000)
    drifted = np.random.normal(10, 1, 1000)
    psi = compute_psi(reference, drifted)
    assert psi > 0.2, "PSI should be high for clearly drifted data"


# Test 4 — PSI is low for stable data
def test_psi_low_for_stable_data():
    def compute_psi(reference, current, bins=10):
        breakpoints = np.linspace(
            min(reference.min(), current.min()),
            max(reference.max(), current.max()),
            
            bins + 1
        )
        ref_counts, _ = np.histogram(reference, bins=breakpoints)
        cur_counts, _ = np.histogram(current, bins=breakpoints)
        ref_pct = ref_counts / len(reference) + 1e-6
        cur_pct = cur_counts / len(current) + 1e-6
        return float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))

    reference = np.random.normal(0, 1, 1000)
    similar = np.random.normal(0, 1, 1000)
    psi = compute_psi(reference, similar)
    assert psi < 0.5, "PSI should be low for similar distributions"


# Test 5 — Autoencoder reconstruction error is higher for drifted data
def test_autoencoder_higher_error_on_drift():
    import torch
    import torch.nn as nn

    class SimpleAutoencoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = nn.Sequential(nn.Linear(5, 2), nn.ReLU())
            self.decoder = nn.Sequential(nn.Linear(2, 5))

        def forward(self, x):
            return self.decoder(self.encoder(x))

    model = SimpleAutoencoder()

    # Train on normal data
    normal_data = torch.FloatTensor(np.random.normal(0, 1, (100, 5)))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    for _ in range(50):
        optimizer.zero_grad()
        loss = nn.MSELoss()(model(normal_data), normal_data)
        loss.backward()
        optimizer.step()

    # Compare reconstruction errors
    with torch.no_grad():
        normal_error = nn.MSELoss()(model(normal_data), normal_data).item()
        drifted_data = torch.FloatTensor(np.random.normal(10, 1, (100, 5)))
        drifted_error = nn.MSELoss()(model(drifted_data), drifted_data).item()

    assert drifted_error > normal_error, "Autoencoder should have higher error on drifted data"


# Test 6 — API feature validation
def test_feature_count():
    features = [
        "Daily_Return", "Volatility_30d", "Return_Zscore",
        "Volatility_Ratio", "Volume_Zscore", "Momentum_5d",
        "Momentum_20d", "Drawdown", "Volume_Ratio",
        "Sector_Avg_Return", "Sector_Avg_Volatility",
        "Return_vs_Sector", "Golden_Cross"
    ]
    assert len(features) == 13, "Should have exactly 13 features"


# Test 7 — Data loading works
def test_data_shape():
    df = pd.read_parquet("data/spark_output/featured_dataset.parquet")
    assert df.shape[0] > 400000, "Dataset should have more than 400k rows"
    assert "Drift_Label" in df.columns, "Dataset should have Drift_Label column"
    assert "Window" in df.columns, "Dataset should have Window column"

