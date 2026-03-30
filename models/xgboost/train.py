import pandas as pd
import numpy as np
import xgboost as xgb
import mlflow
import mlflow.xgboost
import os
import json
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    average_precision_score,
    confusion_matrix
)
import matplotlib.pyplot as plt
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
TARGET = "Drift_Label"
RANDOM_STATE = 42

print("Loading Spark featured dataset...")
df = pd.read_parquet("data/spark_output/featured_dataset.parquet")
df["Date"] = pd.to_datetime(df["Date"])
print(f"Full dataset: {df.shape}")

# Train only on pretrain window (2000-2007)
pretrain = df[df["Window"] == "pretrain"].copy()
print(f"Pretrain window: {pretrain.shape}")
print(f"Label distribution:\n{pretrain[TARGET].value_counts()}")

X = pretrain[FEATURES]
y = pretrain[TARGET]

# Train/validation split (time-aware — no shuffling)
split_idx = int(len(pretrain) * 0.8)
X_train = X.iloc[:split_idx]
X_val   = X.iloc[split_idx:]
y_train = y.iloc[:split_idx]
y_val   = y.iloc[split_idx:]

print(f"\nTrain size: {X_train.shape}, Val size: {X_val.shape}")

# MLflow experiment
mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("xgboost_baseline")

with mlflow.start_run(run_name="xgboost_pretrain_v1"):

    # Model params
    params = {
        "n_estimators":     300,
        "max_depth":        6,
        "learning_rate":    0.05,
        "subsample":        0.8,
        "colsample_bytree": 0.8,
        "scale_pos_weight": len(y_train[y_train==0]) / (len(y_train[y_train==1]) + 1),
        "random_state":     RANDOM_STATE,
        "eval_metric":      "logloss",
        "early_stopping_rounds": 20,
    }

    # Log params to MLflow
    mlflow.log_params(params)
    mlflow.log_param("train_window", "2000-2007")
    mlflow.log_param("n_features", len(FEATURES))
    mlflow.log_param("train_size", len(X_train))

    # Train
    print("\nTraining XGBoost...")
    model = xgb.XGBClassifier(**params)
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=50
    )

    # Evaluate
    y_pred      = model.predict(X_val)
    y_pred_prob = model.predict_proba(X_val)[:, 1]

    if len(y_val.unique()) > 1:
        auc_roc = roc_auc_score(y_val, y_pred_prob)
        auc_pr  = average_precision_score(y_val, y_pred_prob)
    else:
        auc_roc = 0.0
        auc_pr  = 0.0
        print("Note: only one class in validation set — AUC not defined (expected for pretrain window)")

    print(f"\n📊 Validation Results:")
    print(f"AUC-ROC:  {auc_roc:.4f}")
    print(f"AUC-PR:   {auc_pr:.4f}")
    print(f"\nClassification Report:")
    print(classification_report(y_val, y_pred))

    # Log metrics to MLflow
    mlflow.log_metric("auc_roc", auc_roc)
    mlflow.log_metric("auc_pr", auc_pr)

    # Feature importance plot
    importance = pd.Series(
        model.feature_importances_,
        index=FEATURES
    ).sort_values(ascending=True)

    fig, ax = plt.subplots(figsize=(10, 7))
    importance.plot(kind="barh", ax=ax, color="steelblue")
    ax.set_title("XGBoost Feature Importance", fontsize=14)
    ax.set_xlabel("Importance Score")
    plt.tight_layout()

    os.makedirs("models/xgboost", exist_ok=True)
    plot_path = "models/xgboost/feature_importance.png"
    plt.savefig(plot_path, dpi=150)
    plt.show()

    # Log plot to MLflow
    mlflow.log_artifact(plot_path)

    # Save model
    model_path = "models/xgboost/model.json"
    model.save_model(model_path)
    mlflow.xgboost.log_model(model, "xgboost_model")

    # Save feature list for later use
    with open("models/xgboost/features.json", "w") as f:
        json.dump(FEATURES, f)

    print(f"\n✅ Model saved to {model_path}")
    print(f"✅ Logged to MLflow — check http://127.0.0.1:5000")

# Now test on drift periods
print("\n🔍 Testing model on drift periods...")

mlflow.set_experiment("xgboost_drift_evaluation")

for window_name in ["drift_2008", "drift_covid", "drift_volatile"]:
    drift_df = df[df["Window"] == window_name].copy()
    if len(drift_df) == 0:
        continue

    X_drift = drift_df[FEATURES]
    y_drift = drift_df[TARGET]

    y_drift_prob = model.predict_proba(X_drift)[:, 1]
    y_drift_pred = model.predict(X_drift)

    auc = roc_auc_score(y_drift, y_drift_prob) if len(y_drift.unique()) > 1 else 0

    with mlflow.start_run(run_name=f"drift_eval_{window_name}"):
        mlflow.log_param("window", window_name)
        mlflow.log_metric("auc_roc", auc)
        mlflow.log_metric("mean_drift_prob", float(y_drift_prob.mean()))
        mlflow.log_metric("n_rows", len(drift_df))

    print(f"{window_name:20s} → AUC: {auc:.4f} | Mean drift prob: {y_drift_prob.mean():.4f}")

print("\n✅ Step 6 complete!")
