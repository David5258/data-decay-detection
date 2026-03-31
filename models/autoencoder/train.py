import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import mlflow
import mlflow.pytorch
import matplotlib.pyplot as plt
import os
import json
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

BATCH_SIZE   = 512
EPOCHS       = 50
LEARNING_RATE = 0.001
RANDOM_STATE  = 42

torch.manual_seed(RANDOM_STATE)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load data 
print("Loading dataset...")
df = pd.read_parquet("data/spark_output/featured_dataset.parquet")
df["Date"] = pd.to_datetime(df["Date"])

# Train ONLY on pretrain window — no drift examples
pretrain = df[df["Window"] == "pretrain"][FEATURES].dropna()
print(f"Pretrain shape: {pretrain.shape}")

# Normalize features
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(pretrain.values).astype(np.float32)

# Save scaler for inference later
import joblib
os.makedirs("models/autoencoder", exist_ok=True)
joblib.dump(scaler, "models/autoencoder/scaler.pkl")

# Convert to PyTorch tensors
X_tensor = torch.FloatTensor(X_train).to(device)
dataset  = TensorDataset(X_tensor, X_tensor)
loader   = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

print(f"Training samples: {len(X_train)}")

# Autoencoder Architecture
class DriftAutoencoder(nn.Module):
    def __init__(self, input_dim):
        super(DriftAutoencoder, self).__init__()

        # Encoder — compresses normal patterns into a small bottleneck
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 8),   # bottleneck — 8 dimensions
        )

        # Decoder — reconstructs input from bottleneck
        self.decoder = nn.Sequential(
            nn.Linear(8, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, input_dim),
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

    def reconstruction_error(self, x):
        with torch.no_grad():
            reconstructed = self.forward(x)
            error = torch.mean((x - reconstructed) ** 2, dim=1)
        return error.cpu().numpy()


input_dim = len(FEATURES)
model     = DriftAutoencoder(input_dim).to(device)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
criterion = nn.MSELoss()

print(f"\nModel architecture:")
print(model)
print(f"\nTotal parameters: {sum(p.numel() for p in model.parameters()):,}")

# Training
mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("pytorch_autoencoder")

with mlflow.start_run(run_name="autoencoder_v1"):

    mlflow.log_param("epochs", EPOCHS)
    mlflow.log_param("batch_size", BATCH_SIZE)
    mlflow.log_param("learning_rate", LEARNING_RATE)
    mlflow.log_param("bottleneck_dim", 8)
    mlflow.log_param("input_dim", input_dim)
    mlflow.log_param("train_window", "2000-2007")

    print("\nTraining autoencoder...")
    train_losses = []

    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0
        for batch_x, _ in loader:
            optimizer.zero_grad()
            reconstructed = model(batch_x)
            loss = criterion(reconstructed, batch_x)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(loader)
        train_losses.append(avg_loss)
        mlflow.log_metric("train_loss", avg_loss, step=epoch)

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1:3d}/{EPOCHS} | Loss: {avg_loss:.6f}")

    print("\nTraining complete!")

    # Plot training loss
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(train_losses, color="steelblue", linewidth=2)
    ax.set_title("Autoencoder Training Loss", fontsize=14)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE Loss")
    plt.tight_layout()
    loss_plot_path = "models/autoencoder/training_loss.png"
    plt.savefig(loss_plot_path, dpi=150)
    plt.show()
    mlflow.log_artifact(loss_plot_path)

    # Compute reconstruction errors per window
    print("\n🔍 Computing reconstruction errors per market regime...")
    model.eval()

    results = {}
    windows = ["pretrain", "recovery", "post_covid",
               "drift_2008", "drift_covid", "drift_volatile"]

    for window in windows:
        window_df = df[df["Window"] == window][FEATURES].dropna()
        if len(window_df) == 0:
            continue

        X_window = scaler.transform(window_df.values).astype(np.float32)
        X_tensor_w = torch.FloatTensor(X_window).to(device)

        errors = model.reconstruction_error(X_tensor_w)
        results[window] = {
            "mean_error": float(errors.mean()),
            "std_error":  float(errors.std()),
            "p95_error":  float(np.percentile(errors, 95))
        }

        mlflow.log_metric(f"{window}_mean_error", results[window]["mean_error"])
        print(f"{window:20s} → Mean error: {results[window]['mean_error']:.6f} | P95: {results[window]['p95_error']:.6f}")

    # Plot reconstruction errors by window
    fig, ax = plt.subplots(figsize=(12, 6))
    windows_list = list(results.keys())
    mean_errors  = [results[w]["mean_error"] for w in windows_list]
    colors = ["green" if "drift" not in w else "red" for w in windows_list]

    ax.bar(windows_list, mean_errors, color=colors, edgecolor="none")
    ax.set_title("Reconstruction Error by Market Regime\n(Red = Drift, Green = Normal)", fontsize=14)
    ax.set_xlabel("Window")
    ax.set_ylabel("Mean Reconstruction Error")
    ax.tick_params(axis="x", rotation=30)
    plt.tight_layout()
    error_plot_path = "models/autoencoder/reconstruction_errors.png"
    plt.savefig(error_plot_path, dpi=150)
    plt.show()
    mlflow.log_artifact(error_plot_path)

    # Save model
    torch.save(model.state_dict(), "models/autoencoder/model.pth")
    mlflow.pytorch.log_model(model, "autoencoder_model")

    # Save results
    with open("models/autoencoder/reconstruction_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"Model saved to models/autoencoder/model.pth")
    print(f"Logged to MLflow — check http://127.0.0.1:5000")
    print(f"Step 7 complete!")
