import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import joblib
import json
import mlflow
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats
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

NORMAL_WINDOWS = ["recovery", "post_covid"]
DRIFT_WINDOWS  = ["drift_2008", "drift_covid", "drift_volatile"]

# Load autoencoder
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
            return torch.mean((x - self.forward(x)) ** 2, dim=1).numpy()

# Helper functions
def compute_psi(reference, current, bins=10):
    breakpoints = np.linspace(
        min(reference.min(), current.min()),
        max(reference.max(), current.max()),
        bins + 1
    )
    ref_counts, _ = np.histogram(reference, bins=breakpoints)
    cur_counts, _ = np.histogram(current,   bins=breakpoints)
    ref_pct = ref_counts / len(reference) + 1e-6
    cur_pct = cur_counts / len(current)   + 1e-6
    return float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))

def compute_ks(reference, current):
    stat, pval = stats.ks_2samp(reference, current)
    return float(stat), float(pval)

def get_autoencoder_score(model, scaler, data):
    X = scaler.transform(data.values).astype(np.float32)
    X_tensor = torch.FloatTensor(X)
    return model.reconstruction_error(X_tensor).mean()

# Load data
print("Loading data and models...")
df = pd.read_parquet("data/spark_output/featured_dataset.parquet")
df["Date"] = pd.to_datetime(df["Date"])

reference = df[df["Window"] == "pretrain"][FEATURES].dropna()

autoencoder = DriftAutoencoder(len(FEATURES))
autoencoder.load_state_dict(
    torch.load("models/autoencoder/model.pth", map_location="cpu")
)
autoencoder.eval()
scaler = joblib.load("models/autoencoder/scaler.pkl")

print("All models loaded!")

# Compute scores for all windows
print("Computing scores for all methods across all windows...")

results = {}
for window in NORMAL_WINDOWS + DRIFT_WINDOWS:
    current = df[df["Window"] == window][FEATURES].dropna()
    if len(current) == 0:
        continue

    ks_scores = [compute_ks(reference[f].values, current[f].values)[0]
                 for f in FEATURES]
    ks_score = float(np.mean(ks_scores))

    psi_scores = [compute_psi(reference[f].values, current[f].values)
                  for f in FEATURES]
    psi_score = float(np.mean(psi_scores))

    ae_score = float(get_autoencoder_score(autoencoder, scaler, current))

    is_drift = window in DRIFT_WINDOWS
    results[window] = {
        "ks_score":  ks_score,
        "psi_score": psi_score,
        "ae_score":  ae_score,
        "is_drift":  is_drift
    }
    print(f"{window:20s} | KS: {ks_score:.4f} | PSI: {psi_score:.4f} | AE: {ae_score:.4f} | Drift: {is_drift}")

# Normalize scores to 0-1 for fair comparison
print("Normalizing scores...")
for method in ["ks_score", "psi_score", "ae_score"]:
    values = np.array([results[w][method] for w in results])
    min_v, max_v = values.min(), values.max()
    for window in results:
        results[window][f"{method}_norm"] = float(
            (results[window][method] - min_v) / (max_v - min_v + 1e-9)
        )

# A/B Test Evaluation
print("Running A/B test evaluation...")

def evaluate_method(score_key, threshold=0.5):
    true_labels  = [results[w]["is_drift"] for w in results]
    norm_scores  = [results[w][f"{score_key}_norm"] for w in results]
    predictions  = [s > threshold for s in norm_scores]

    tp = sum(p and t for p, t in zip(predictions, true_labels))
    fp = sum(p and not t for p, t in zip(predictions, true_labels))
    tn = sum(not p and not t for p, t in zip(predictions, true_labels))
    fn = sum(not p and t for p, t in zip(predictions, true_labels))

    detection_rate = tp / (tp + fn + 1e-9)
    false_pos_rate = fp / (fp + tn + 1e-9)
    precision      = tp / (tp + fp + 1e-9)
    f1             = 2 * precision * detection_rate / (precision + detection_rate + 1e-9)

    return {
        "detection_rate": detection_rate,
        "false_pos_rate": false_pos_rate,
        "precision":      precision,
        "f1_score":       f1,
        "tp": tp, "fp": fp, "tn": tn, "fn": fn
    }

methods = {
    "KS Test":     "ks_score",
    "PSI":         "psi_score",
    "Autoencoder": "ae_score"
}

method_results = {}
for method_name, score_key in methods.items():
    eval_results = evaluate_method(score_key)
    method_results[method_name] = eval_results
    print(f"{method_name}:")
    print(f"  Detection Rate: {eval_results['detection_rate']:.2f}")
    print(f"  False Positive: {eval_results['false_pos_rate']:.2f}")
    print(f"  Precision:      {eval_results['precision']:.2f}")
    print(f"  F1 Score:       {eval_results['f1_score']:.2f}")

# Statistical significance test
print("Testing statistical significance...")

ae_scores   = [results[w]["ae_score_norm"]  for w in results]
psi_scores  = [results[w]["psi_score_norm"] for w in results]
ks_scores   = [results[w]["ks_score_norm"]  for w in results]
true_labels = [float(results[w]["is_drift"]) for w in results]

ae_corr,  ae_p  = stats.pearsonr(ae_scores,  true_labels)
psi_corr, psi_p = stats.pearsonr(psi_scores, true_labels)
ks_corr,  ks_p  = stats.pearsonr(ks_scores,  true_labels)

print(f"Correlation with true drift labels:")
print(f"  Autoencoder: r={ae_corr:.4f}  p={ae_p:.4f}")
print(f"  PSI:         r={psi_corr:.4f}  p={psi_p:.4f}")
print(f"  KS Test:     r={ks_corr:.4f}  p={ks_p:.4f}")

# MLflow Logging
mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("ab_test_drift_detectors")

with mlflow.start_run(run_name="ab_test_all_methods"):
    for method_name, eval_r in method_results.items():
        prefix = method_name.lower().replace(" ", "_")
        mlflow.log_metric(f"{prefix}_detection_rate", eval_r["detection_rate"])
        mlflow.log_metric(f"{prefix}_false_pos_rate", eval_r["false_pos_rate"])
        mlflow.log_metric(f"{prefix}_f1_score",       eval_r["f1_score"])

    mlflow.log_metric("ae_drift_correlation",  ae_corr)
    mlflow.log_metric("psi_drift_correlation", psi_corr)
    mlflow.log_metric("ks_drift_correlation",  ks_corr)

# Visualization
print("Generating A/B test visualizations...")

fig = plt.figure(figsize=(18, 12))
gs  = gridspec.GridSpec(2, 3, figure=fig)

ax1 = fig.add_subplot(gs[0, :])
windows_list = list(results.keys())
x = np.arange(len(windows_list))
width = 0.25

ax1.bar(x - width, [results[w]["ks_score_norm"]  for w in windows_list],
        width, label="KS Test",     color="steelblue",  alpha=0.8)
ax1.bar(x,         [results[w]["psi_score_norm"] for w in windows_list],
        width, label="PSI",         color="darkorange",  alpha=0.8)
ax1.bar(x + width, [results[w]["ae_score_norm"]  for w in windows_list],
        width, label="Autoencoder", color="green",       alpha=0.8)

ax1.axhline(0.5, color="black", linestyle="--", linewidth=1.5, label="Threshold")
ax1.set_xticks(x)
ax1.set_xticklabels(windows_list, rotation=15)
ax1.set_title("Normalized Drift Scores by Method and Window", fontsize=13)
ax1.set_ylabel("Normalized Score")
ax1.legend()

metrics    = ["f1_score", "detection_rate", "false_pos_rate"]
titles     = ["F1 Score", "Detection Rate", "False Positive Rate"]
colors_bar = ["steelblue", "darkorange", "green"]

for i, (metric, title) in enumerate(zip(metrics, titles)):
    ax = fig.add_subplot(gs[1, i])
    method_names = list(method_results.keys())
    values = [method_results[m][metric] for m in method_names]
    ax.bar(method_names, values, color=colors_bar, alpha=0.8, edgecolor="none")
    ax.set_title(title, fontsize=12)
    ax.set_ylim(0, 1.1)
    for j, v in enumerate(values):
        ax.text(j, v + 0.02, f"{v:.2f}", ha="center", fontsize=11)

plt.suptitle("A/B Test - Drift Detection Method Comparison", fontsize=15)
plt.tight_layout()

plot_path = "drift_detection/ab_test_results.png"
plt.savefig(plot_path, dpi=150)
plt.show()
print(f"Plot saved to {plot_path}")

# Final verdict
best_method = max(method_results, key=lambda m: method_results[m]["f1_score"])
print(f"A/B Test Winner: {best_method}")
print(f"F1 Score: {method_results[best_method]['f1_score']:.4f}")

with open("drift_detection/ab_test_results.json", "w") as f:
    json.dump({
        "method_results": method_results,
        "correlations": {
            "autoencoder": {"r": ae_corr, "p": ae_p},
            "psi":         {"r": psi_corr, "p": psi_p},
            "ks_test":     {"r": ks_corr,  "p": ks_p}
        },
        "winner": best_method
    }, f, indent=2)
