# 🏥 Hospital Cost & Charge Estimation System

> End-to-end ML system that predicts hospital inpatient discharge **costs** and **charges** using only admission-time parameters — before treatment begins.

[![Live Demo](https://img.shields.io/badge/Live_Demo-Streamlit-FF4B4B?logo=streamlit&style=for-the-badge)](https://cost-estimation-06.streamlit.app/)
[![API](https://img.shields.io/badge/API-Render-46E3B7?logo=render&style=for-the-badge)](https://cost-estimation-g6ny.onrender.com/docs)
[![Python](https://img.shields.io/badge/Python-3.13-3776AB?logo=python&style=flat-square)](https://python.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.8.0-F7931E?logo=scikit-learn&style=flat-square)](https://scikit-learn.org)

---

## 📊 Problem Statement

Hospitals need to estimate the cost and charges of an inpatient stay **at the time of admission** — before any treatment is delivered. Accurate early estimates support:

- **Financial planning** for patients and insurers
- **Resource allocation** for hospital administrators
- **Anomaly detection** for unusually high/low cost cases

This project builds **8 specialized ML models** (4 for cost, 4 for charge estimation) with a dual-model risk classification pipeline, and deploys them as a production API + interactive web dashboard.

## 🏗️ Architecture

```
┌─────────────────────────┐       HTTPS        ┌──────────────────────────────┐
│   Streamlit Frontend    │ ◄──────────────────►│     FastAPI Backend          │
│   (Streamlit Cloud)     │                     │     (Render Free Tier)       │
│                         │                     │                              │
│  • Interactive form     │                     │  • 8 ML models in memory     │
│  • Dropdown fields      │                     │  • Hybrid feature pipeline   │
│  • Risk filter presets  │                     │  • Risk classification       │
│  • What-if analysis     │                     │  • Keep-alive mechanism      │
│  • Randomize test data  │                     │  • 500 sample test records   │
└─────────────────────────┘                     └──────────────────────────────┘
```

## 🧠 Model Pipeline

### Dual-Model Architecture

Each estimation target (cost/charge) uses **4 models** working together:

| Model | Role | Algorithm |
|-------|------|-----------|
| **V1 (History)** | Regressor trained with target-encoded historical aggregates | `HistGradientBoostingRegressor` |
| **V2 (Specialist)** | Regressor trained on cleaner feature set | `HistGradientBoostingRegressor` |
| **Risk Classifier (Under)** | Flags likely under-predictions | `HistGradientBoostingClassifier` |
| **Risk Classifier (Over)** | Flags likely over-predictions | `HistGradientBoostingClassifier` |

### Hybrid Feature Engineering

```
Raw Patient Data
       │
       ├──► V1 Pipeline (feature_engineer → regressor) ──► pred_v1
       │         └──► target-encoded features (MEDIAN_COST, DIH)
       │
       ├──► V2 Pipeline (feature_engineer → regressor) ──► pred_v2
       │         └──► ordinal + target-encoded features
       │
       └──► Hybrid Features = [V2_features + V1_target_enc + pred_v1 + pred_v2 + diff]
                    │
                    ├──► Risk Classifier (Under) ──► P(under-prediction)
                    └──► Risk Classifier (Over)  ──► P(over-prediction)
```

### Configurable Risk Filtering

5 preset threshold levels from **Lenient** (fewer flags) to **Aggressive** (strictest):

| Level | Cost Thresholds (Under, Over) | Charge Thresholds (Under, Over) |
|-------|:---:|:---:|
| Lenient | (0.70, 0.95) | (0.70, 0.85) |
| Moderate | (0.65, 0.90) | (0.65, 0.80) |
| **Balanced** | **(0.60, 0.85)** | **(0.60, 0.75)** |
| Cautious | (0.55, 0.80) | (0.55, 0.70) |
| Aggressive | (0.50, 0.75) | (0.50, 0.65) |

## 🔧 Feature Engineering

The `AutomatedFeatureEngineer` (custom sklearn transformer) performs:

- **Ordinal Encoding** — 14 categorical features (hospital, demographics, clinical codes)
- **Target Encoding** — 3 high-cardinality diagnosis/procedure fields against cost/charge target
- **Length of Stay Encoding** — Same 3 fields target-encoded against days-in-hospital (estimated, not direct input)

### Input Features (16 admission-time parameters)

| Category | Features |
|----------|----------|
| **Hospital** | Service Area, County, Facility Name |
| **Demographics** | Age Group, Gender, Race, Ethnicity |
| **Admission** | Type of Admission, Patient Disposition, Emergency Dept Indicator |
| **Clinical** | CCSR Diagnosis, CCSR Procedure, APR DRG, APR MDC |
| **Severity** | APR Severity of Illness, APR Risk of Mortality, Medical/Surgical |

## 📁 Project Structure

```
├── app/                          # FastAPI backend
│   ├── main.py                   # App entry point + lifespan
│   ├── inference.py              # Core prediction pipeline
│   ├── model_loader.py           # Loads 8 .joblib models at startup
│   ├── custom_transformers.py    # AutomatedFeatureEngineer class
│   ├── models.py                 # Pydantic schemas + threshold presets
│   ├── keep_alive.py             # Background self-ping (14 min interval)
│   └── routers/
│       ├── cost.py               # /api/v1/cost/* endpoints
│       ├── charge.py             # /api/v1/charge/* endpoints
│       └── data.py               # /api/v1/data/* endpoints
│
├── frontend/                     # Streamlit dashboard
│   ├── streamlit_app.py          # Main UI with what-if analysis
│   ├── requirements.txt          # Frontend-only dependencies
│   └── .streamlit/config.toml    # Dark theme configuration
│
├── train_notebooks/              # Full training notebooks
│   ├── cost_final.ipynb          # Cost estimation model training
│   └── charge_final.ipynb        # Charge estimation model training
│
├── models_new/                   # 8 trained .joblib model files
├── data_sample/                  # 500 test records + field options
├── scripts/prepare_sample_data.py
├── Dockerfile                    # Python 3.13-slim production image
├── render.yaml                   # Render Blueprint
└── requirements.txt              # Backend dependencies
```

## 🚀 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check + model count |
| `/api/v1/cost/predict` | POST | Single cost prediction |
| `/api/v1/cost/predict/batch` | POST | Batch cost predictions |
| `/api/v1/cost/thresholds` | GET | Available risk threshold presets |
| `/api/v1/charge/predict` | POST | Single charge prediction |
| `/api/v1/charge/predict/batch` | POST | Batch charge predictions |
| `/api/v1/charge/thresholds` | GET | Available risk threshold presets |
| `/api/v1/data/random` | GET | Random patient from test set |
| `/api/v1/data/field-options` | GET | Unique values per field |

## 🖥️ Frontend Features

- **Interactive form** — 17 dropdown fields populated from the API
- **Randomize** — Load random patients from 420K+ test records
- **What-if analysis** — Modify a field and see the prediction delta (↑/↓ with %)
- **Dual prediction** — Cost & Charge estimated simultaneously
- **Risk visualization** — Color-coded badges, confidence levels, probability bars
- **Actual vs Predicted** — When using test data, see how close the model gets

## 🛠️ Tech Stack

| Component | Technology |
|-----------|-----------|
| ML Framework | scikit-learn 1.8.0 |
| Data Processing | pandas 3.0.2, numpy |
| Backend API | FastAPI, Uvicorn |
| Frontend | Streamlit |
| Containerization | Docker (Python 3.13-slim) |
| Backend Hosting | Render (Free Tier) |
| Frontend Hosting | Streamlit Community Cloud |

## 📈 Dataset

- **Source**: NY SPARCS Inpatient Discharge Dataset
- **Records**: 420,687 test samples (used for evaluation)
- **Target Variables**: `Total Costs` and `Total Charges`

## ⚡ Local Development

```bash
# Backend
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000

# Frontend
cd frontend
pip install -r requirements.txt
streamlit run streamlit_app.py
```

## 📝 Training Notebooks

Full model development process is documented in:
- [`train_notebooks/cost_final.ipynb`](train_notebooks/cost_final.ipynb) — Cost estimation pipeline
- [`train_notebooks/charge_final.ipynb`](train_notebooks/charge_final.ipynb) — Charge estimation pipeline

Includes: EDA, feature engineering, model training, hyperparameter tuning, risk classifier development, and evaluation metrics.

---

*Built by [Priyanshu Verma](https://github.com/Priyanshukv06)*
