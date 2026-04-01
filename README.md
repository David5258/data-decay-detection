# Market Regime Drift Detection System

> Real-time detection of financial market regime changes using statistical tests, deep learning, and production ML infrastructure — built to catch what traditional models miss.

[![CI Pipeline](https://github.com/David5258/data-decay-detection/actions/workflows/ci.yml/badge.svg)](https://github.com/David5258/data-decay-detection/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/Python-3.11-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.2-red)
![XGBoost](https://img.shields.io/badge/XGBoost-2.0-green)
![Spark](https://img.shields.io/badge/Apache%20Spark-3.5-orange)
![AWS](https://img.shields.io/badge/AWS-EC2%20%7C%20S3%20%7C%20ECR-yellow)

---

## Key Results

| Metric | Value |
|--------|-------|
| Dataset size | 510,000+ rows, 92 stocks, 24 years |
| Drift events detected | 3 real historical events (2008, COVID, Rate Hikes) |
| Autoencoder error on COVID crash | **6.5x higher** than normal market |
| A/B Test winner | KS Test — F1 Score: **0.80** |
| False positive rate | **0%** across all methods |
| CI/CD pipeline | 7 automated tests, runs in **58 seconds** |

---

## The Problem

ML models trained on historical data silently fail when market conditions change. A model trained on pre-2008 bull market data has **no idea** a financial crisis is coming — it keeps predicting "everything is normal" while the market collapses.

This project builds a system that automatically detects when market data has drifted from training distribution, triggering alerts before model performance degrades.

---

## Architecture
```
Raw Market Data (92 S&P 500 stocks, 2000-2024)
        ↓
Apache Spark — Feature Engineering at Scale (510k rows, 28 features)
        ↓
┌─────────────────────────────────────────────┐
│           Drift Detection Layer              │
│                                             │
│  XGBoost        KS Test        PyTorch      │
│  Baseline    Statistical    Autoencoder     │
│  (blind to   (F1 = 0.80)   (6.5x error     │
│   drift)                    on crashes)     │
│                                             │
│         A/B Test — KS Test Wins             │
└─────────────────────────────────────────────┘
        ↓
FastAPI REST API — Real-time drift scoring
        ↓
Docker → AWS ECR → EC2 Deployment
        ↓
MLflow Experiment Tracking + GitHub Actions CI/CD
```

---

## Tech Stack

| Layer | Tools |
|-------|-------|
| Data Processing | Apache Spark, PySpark, Pandas |
| ML Models | XGBoost, PyTorch, Scikit-learn |
| Drift Detection | KS Test, PSI, Autoencoder |
| Experiment Tracking | MLflow |
| API Serving | FastAPI, Uvicorn |
| Containerization | Docker |
| Cloud | AWS S3, ECR, EC2, IAM |
| CI/CD | GitHub Actions |
| Testing | Pytest |
| Data Validation | Great Expectations |

---

## Dataset

- **Source:** Yahoo Finance via yfinance
- **Coverage:** 92 S&P 500 stocks across 9 sectors (2000-2024)
- **Size:** 510,039 rows, 28 engineered features
- **Drift Events:** 3 real historical events used as natural drift labels

| Window | Period | Label | Rows |
|--------|--------|-------|------|
| Pretrain | 2000-2007 | Normal | 132,077 |
| 2008 Crisis | Oct 2007 - Mar 2009 | Drift | 30,056 |
| Recovery | 2009-2020 | Normal | 246,338 |
| COVID Crash | Feb-Apr 2020 | Drift | 3,864 |
| Post-COVID | 2020-2021 | Normal | 40,664 |
| Rate Hike Era | 2022-2024 | Drift | 46,092 |

---

## Key Findings

### 1. XGBoost is Blind to Drift
A model trained exclusively on pre-2008 bull market data predicts "normal" on every single drift period, confirming why drift detection is necessary.

### 2. Autoencoder Catches What XGBoost Misses
The PyTorch autoencoder, trained unsupervised on normal data, detects drift through reconstruction error without ever seeing a labeled drift example.

| Market Period | Reconstruction Error |
|--------------|---------------------|
| Pretrain (normal) | 0.035 |
| Recovery (normal) | 0.026 |
| 2008 Crisis | 0.083 **(2.4x normal)** |
| COVID Crash | 0.229 **(6.5x normal)** |
| Rate Hike Era | 0.028 |

### 3. A/B Test Results
Three drift detection methods compared across 5 market windows:

| Method | Detection Rate | False Positive Rate | F1 Score |
|--------|---------------|--------------------| ---------|
| KS Test | 0.67 | 0.00 | **0.80** |
| PSI | 0.33 | 0.00 | 0.50 |
| Autoencoder | 0.33 | 0.00 | 0.50 |

**Winner: KS Test** with the highest F1 score and zero false positives.

---

## API Usage

The drift detection system is served as a REST API:
```bash
curl -X POST "http://localhost:8000/detect" \
  -H "Content-Type: application/json" \
  -d '{
    "ticker": "AAPL",
    "date": "2008-10-15",
    "features": [-0.08, 0.04, -3.2, 2.8, 1.9,
                 -0.12, -0.18, -0.09, 1.7,
                 -0.05, 0.03, -0.02, 0.0]
  }'
```

Response:
```json
{
  "ticker": "AAPL",
  "date": "2008-10-15",
  "drift_detected": true,
  "confidence": "high",
  "ks_score": 0.928,
  "psi_score": 12.33,
  "autoencoder_score": 0.31,
  "details": {
    "votes": 3,
    "ks_drift": true,
    "psi_drift": true,
    "ae_drift": true
  }
}
```

---

## How to Run

### 1. Clone and setup
```bash
git clone https://github.com/David5258/data-decay-detection.git
cd data-decay-detection
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Download data
```bash
python3 data/download_data.py
python3 data/generate_drift.py
```

### 3. Run Spark pipeline
```bash
python3 spark/feature_pipeline.py
```

### 4. Train models
```bash
python3 models/xgboost/train.py
python3 models/autoencoder/train.py
```

### 5. Run drift detectors
```bash
python3 drift_detection/statistical_tests.py
python3 drift_detection/ab_test.py
```

### 6. Start API
```bash
python3 api/main.py
# Visit http://localhost:8000/docs
```

### 7. Run tests
```bash
pytest tests/ -v
```

### 8. Docker
```bash
docker build -t data-decay-detection -f docker/Dockerfile .
docker run -p 8000:8000 data-decay-detection
```

---

## Project Structure
```
data-decay-detection/
├── .github/workflows/      # CI/CD pipeline
├── data/
│   ├── download_data.py    # Download 92 stocks from Yahoo Finance
│   └── generate_drift.py  # Label drift periods
├── spark/
│   └── feature_pipeline.py # Engineer 28 features at scale
├── models/
│   ├── xgboost/            # Baseline production model
│   └── autoencoder/        # PyTorch drift detector
├── drift_detection/
│   ├── statistical_tests.py # KS Test + PSI
│   ├── autoencoder_detector.py
│   └── ab_test.py          # Compare all methods
├── api/
│   └── main.py             # FastAPI serving layer
├── docker/                 # Containerization
├── tests/                  # 7 automated tests
├── notebooks/
│   └── eda.ipynb           # Exploratory analysis
└── aws/                    # Deployment infrastructure
```

---

## Content

- **End-to-end ML system** — from raw data ingestion to production API
- **Production thinking** — drift detection, experiment tracking, automated testing
- **Scale** — Spark pipeline processing 510k rows across 92 stocks
- **Rigorous evaluation** — A/B testing with statistical significance
- **MLOps** — MLflow, Docker, AWS, CI/CD with GitHub Actions
