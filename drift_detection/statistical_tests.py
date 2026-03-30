import pandas as pd
import numpy as np
from scipy import stats
import mlflow
import matplotlib.pyplot as plt
import json
import os
import warnings
warnings.filterwarnings("ignore")

# Config
FEATURES = [
    "Daily_Return", "Volatility_30d", "Return_Zscore",
    "Volatility_Ratio", "Volume_Zscore", "Momentum_5d",
    "Momentum_20d", "Drawdown", "Volume_Ratio",
    "Sector_Avg_Return", "Sector_Avg_Volatility",
    "Return_vs_Sector", "Golden_Cross"
]

PSI_BINS = 10
PSI_DRIFT_THRESHOLD    = 0.2
KS_DRIFT_THRESHOLD     = 0.05  # p-value threshold

# PSI Function
def compute_psi(reference, current, bins=10):
    """
    Population Stability Index — industry standard drift metric.
    PSI < 0.1  : no drift
    PSI 0.1-0.2: moderate drift
    PSI > 0.2  : significant drift
    """
    # Create bins from reference distribution
    breakpoints = np.linspace(
        min(reference.min(), current.min()),
        max(reference.max(), current.max()),
        bins + 1
    )

    ref_counts, _ = np.histogram(reference, bins=breakpoints)
    cur_counts, _ = np.histogram(current,   bins=breakpoints)

    # Convert to proportions, avoid division by zero
    ref_pct = ref_counts / len(reference) + 1e-6
    cur_pct = cur_counts / len(current)   + 1e-6

    psi = np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct))
    return float(psi)

# KS Test Function
def compute_ks(reference, current):
    """
    Kolmogorov-Smirnov test.
    Returns statistic and p-value.
    Low p-value = distributions are different = drift detected.
    """
    statistic, p_value = stats.ks_2samp(reference, current)
    return float(statistic), float(p_value)

# Load data
print("Loading dataset...")
df = pd.read_parquet("data/spark_output/featured_dataset.parquet")
df["Date"] = pd.to_datetime(df["Date"])
print(f"Dataset shape: {df.shape}")

# Reference distribution = pretrain window (2000-2007)
reference = df[df["Window"] == "pretrain"][FEATURES].dropna()
print(f"Reference (pretrain) shape: {reference.shape}")

# Windows to test against
test_windows = ["recovery", "post_covid",
                "drift_2008", "drift_covid", "drift_volatile"]

# Run Both Tests on All Windows
print("\n🔍 Running statistical drift tests...")

mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("statistical_drift_detectors")

all_results = {}

for window in test_windows:
    current = df[df["Window"] == window][FEATURES].dropna()
    if len(current) == 0:
        continue

    window_results = {
        "ks":  {},
        "psi": {},
        "drift_detected_ks":  False,
        "drift_detected_psi": False
    }

    ks_stats   = []
    ks_pvals   = []
    psi_scores = []

    # Run tests per feature
    for feature in FEATURES:
        ref_vals = reference[feature].values
        cur_vals = current[feature].values

        # KS test
        ks_stat, ks_pval = compute_ks(ref_vals, cur_vals)
        window_results["ks"][feature] = {
            "statistic": ks_stat,
            "p_value":   ks_pval,
            "drift":     ks_pval < KS_DRIFT_THRESHOLD
        }
        ks_stats.append(ks_stat)
        ks_pvals.append(ks_pval)

        # PSI
        psi = compute_psi(ref_vals, cur_vals, bins=PSI_BINS)
        window_results["psi"][feature] = {
            "psi":   psi,
            "drift": psi > PSI_DRIFT_THRESHOLD
        }
        psi_scores.append(psi)

    # Aggregate scores
    mean_ks_stat  = float(np.mean(ks_stats))
    mean_psi      = float(np.mean(psi_scores))
    pct_ks_drift  = float(np.mean([v["drift"] for v in window_results["ks"].values()]))
    pct_psi_drift = float(np.mean([v["drift"] for v in window_results["psi"].values()]))

    window_results["mean_ks_statistic"]  = mean_ks_stat
    window_results["mean_psi"]           = mean_psi
    window_results["pct_features_drift_ks"]  = pct_ks_drift
    window_results["pct_features_drift_psi"] = pct_psi_drift
    window_results["drift_detected_ks"]  = pct_ks_drift  > 0.5
    window_results["drift_detected_psi"] = pct_psi_drift > 0.5

    all_results[window] = window_results

    # Log to MLflow
    with mlflow.start_run(run_name=f"stats_{window}"):
        mlflow.log_param("window", window)
        mlflow.log_param("n_samples", len(current))
        mlflow.log_metric("mean_ks_statistic",      mean_ks_stat)
        mlflow.log_metric("mean_psi",               mean_psi)
        mlflow.log_metric("pct_features_drift_ks",  pct_ks_drift)
        mlflow.log_metric("pct_features_drift_psi", pct_psi_drift)
        mlflow.log_metric("drift_detected_ks",  int(window_results["drift_detected_ks"]))
        mlflow.log_metric("drift_detected_psi", int(window_results["drift_detected_psi"]))

    print(f"\n{window}:")
    print(f"  KS  → mean stat: {mean_ks_stat:.4f} | {pct_ks_drift*100:.0f}% features drifted | Drift: {window_results['drift_detected_ks']}")
    print(f"  PSI → mean PSI:  {mean_psi:.4f}      | {pct_psi_drift*100:.0f}% features drifted | Drift: {window_results['drift_detected_psi']}")

# Visualization
print("\nGenerating visualizations...")

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
windows_list = list(all_results.keys())
colors = ["red" if "drift" in w else "green" for w in windows_list]

# KS scores
ks_scores = [all_results[w]["mean_ks_statistic"] for w in windows_list]
axes[0].bar(windows_list, ks_scores, color=colors, edgecolor="none")
axes[0].axhline(0.3, color="black", linestyle="--", linewidth=1.5, label="Drift threshold")
axes[0].set_title("KS Test — Mean Statistic by Window", fontsize=13)
axes[0].set_xlabel("Window")
axes[0].set_ylabel("KS Statistic")
axes[0].tick_params(axis="x", rotation=30)
axes[0].legend()

# PSI scores
psi_scores_plot = [all_results[w]["mean_psi"] for w in windows_list]
axes[1].bar(windows_list, psi_scores_plot, color=colors, edgecolor="none")
axes[1].axhline(PSI_DRIFT_THRESHOLD, color="black", linestyle="--",
                linewidth=1.5, label=f"PSI threshold ({PSI_DRIFT_THRESHOLD})")
axes[1].set_title("PSI Score by Window", fontsize=13)
axes[1].set_xlabel("Window")
axes[1].set_ylabel("Mean PSI Score")
axes[1].tick_params(axis="x", rotation=30)
axes[1].legend()

plt.suptitle("Statistical Drift Detection — KS Test & PSI", fontsize=15)
plt.tight_layout()

os.makedirs("drift_detection", exist_ok=True)
plot_path = "drift_detection/statistical_tests_results.png"
plt.savefig(plot_path, dpi=150)
plt.show()
print(f"Plot saved to {plot_path}")

# Save results
results_path = "drift_detection/statistical_results.json"
with open(results_path, "w") as f:
    json.dump(all_results, f, indent=2)

print(f"\n Results saved to {results_path}")
print(f"Logged to MLflow — check http://127.0.0.1:5000")
print(f"\n Step 8 complete!")
