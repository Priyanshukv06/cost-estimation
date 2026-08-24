# Hospital Cost & Charge Estimation System

> End-to-end ML system that predicts hospital inpatient discharge costs and charges using only admission-time parameters, before treatment begins.

[![Live Demo](https://img.shields.io/badge/Live_Demo-Streamlit-FF4B4B?logo=streamlit&style=for-the-badge)](https://cost-estimation-06.streamlit.app/)
[![API Docs](https://img.shields.io/badge/API_Docs-Render-46E3B7?logo=render&style=for-the-badge)](https://cost-estimation-g6ny.onrender.com/docs)
[![Python](https://img.shields.io/badge/Python-3.13-3776AB?logo=python&style=flat-square)](https://python.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.8.0-F7931E?logo=scikit-learn&style=flat-square)](https://scikit-learn.org)

---

## Problem Statement

Hospitals need to estimate the cost and charges of an inpatient stay at the time of admission, before any treatment is delivered. Accurate early estimates support:

- Financial planning for patients and insurers
- Resource allocation for hospital administrators
- Anomaly detection for unusually high or low cost cases

This project builds 8 specialized machine learning models (4 for cost estimation, 4 for charge estimation) with a dual-model risk classification pipeline, deployed as a production API and interactive web dashboard.

---

## Architecture

The system consists of two deployed services communicating over HTTPS:

```
+---------------------------------------------------+       HTTPS        +--------------------------------------------------+
|               Streamlit Frontend                  | <----------------> |                 FastAPI Backend                  |
|               (Streamlit Cloud)                   |                    |                (Render Free Tier)                |
|                                                   |                    |                                                  |
|  - Interactive admission input form (17 fields)   |                    |  - 8 ML models loaded in memory (.joblib)        |
|  - Dynamic field options from backend             |                    |  - Hybrid feature engineering pipeline          |
|  - Dual prediction (Cost & Charge)                |                    |  - Dual risk classification (Under / Over)       |
|  - Configurable risk filter presets               |                    |  - Configurable risk threshold presets           |
|  - Real-time what-if scenario analysis            |                    |  - 500 test patient sample records               |
|  - Random test data loader                        |                    |  - Precomputed test statistics & evaluation data |
|  - Model monitoring & evaluation dashboard        |                    |  - Input validation with Pydantic Literal types  |
+---------------------------------------------------+                    +--------------------------------------------------+
```

---

## Model Pipeline

### Dual-Model Architecture

Each estimation target (cost and charge) uses 4 specialized models working together, resulting in a total of 8 trained models:

| Model | Role | Algorithm |
|-------|------|-----------|
| V1 (History) | Regressor trained with target-encoded historical aggregates | HistGradientBoostingRegressor |
| V2 (Specialist) | Regressor trained on cleaner feature set — produces the final prediction | HistGradientBoostingRegressor |
| Risk Classifier (Under) | Flags likely under-predictions using hybrid features | HistGradientBoostingClassifier |
| Risk Classifier (Over) | Flags likely over-predictions using hybrid features | HistGradientBoostingClassifier |

### Hybrid Feature Engineering Pipeline

```
Raw Patient Data
       |
       +---> V1 Pipeline (feature_engineer -> regressor) ---> pred_v1
       |          |
       |          +-> Target-encoded features (MEDIAN_COST, DIH)
       |
       +---> V2 Pipeline (feature_engineer -> regressor) ---> pred_v2
       |          |
       |          +-> Ordinal + target-encoded features
       |
       +---> Hybrid Features = [V2_features + V1_target_enc + pred_v1 + pred_v2 + pred_diff]
                    |
                    +---> Risk Classifier (Under) ---> P(under-prediction)
                    |
                    +---> Risk Classifier (Over)  ---> P(over-prediction)
```

### Configurable Risk Filtering

Predictions are evaluated against risk thresholds across 5 preset sensitivity levels from Lenient (fewer flags) to Aggressive (strictest). Filter levels are validated at the API schema level using Pydantic `Literal` types (invalid values return HTTP 422).

#### Cost Thresholds

| Level | Under Threshold | Over Threshold |
|-------|:-:|:-:|
| Lenient | 0.70 | 0.95 |
| Moderate | 0.65 | 0.90 |
| Balanced (default) | 0.60 | 0.85 |
| Cautious | 0.55 | 0.80 |
| Aggressive | 0.50 | 0.75 |

#### Charge Thresholds

| Level | Under Threshold | Over Threshold |
|-------|:-:|:-:|
| Lenient | 0.70 | 0.85 |
| Moderate | 0.65 | 0.80 |
| Balanced (default) | 0.60 | 0.75 |
| Cautious | 0.55 | 0.70 |
| Aggressive | 0.50 | 0.65 |

---

## Feature Engineering

The custom scikit-learn transformer `AutomatedFeatureEngineer` executes the feature transformation pipeline:

1. **Ordinal Encoding**: 14 categorical features (hospital attributes, patient demographics, clinical descriptors).
2. **Target Encoding**: 3 high-cardinality diagnosis and procedure fields against the cost/charge target (smooth=50.0).
3. **Length of Stay Encoding**: The same 3 fields target-encoded against estimated days in hospital (smooth=20.0).

The 3 target-encoded fields are `CCSR Diagnosis Description`, `CCSR Procedure Description`, and `APR DRG Description`.

### Input Features (17 admission-time parameters)

| Category | Features |
|----------|----------|
| Hospital | Hospital Service Area, Hospital County, Facility Name |
| Demographics | Age Group, Gender, Race, Ethnicity |
| Admission | Type of Admission, Patient Disposition, Emergency Department Indicator |
| Clinical | CCSR Diagnosis Description, CCSR Procedure Description, APR DRG Description, APR MDC Description |
| Severity | APR Severity of Illness Description, APR Risk of Mortality, APR Medical Surgical Description |

*Note: `Birth Weight` is included in the pipeline interface but defaults to 0 and carries no predictive weight. All input strings are validated with `min_length=1` and field-specific `max_length` bounds.*

---

## Project Structure

```
project-root/
|-- app/                              # FastAPI backend
|   |-- main.py                       # App entry point, lifespan, CORS, health check
|   |-- inference.py                  # Core prediction pipeline (hybrid features, risk assessment)
|   |-- model_loader.py              # Loads 8 .joblib models at startup
|   |-- custom_transformers.py       # AutomatedFeatureEngineer (custom sklearn transformer)
|   |-- models.py                    # Pydantic schemas, threshold presets, input validation
|   +-- routers/
|       |-- cost.py                  # /api/v1/cost/* endpoints
|       |-- charge.py                # /api/v1/charge/* endpoints
|       |-- data.py                  # /api/v1/data/* endpoints (random patients, field options)
|       +-- stats.py                 # /api/v1/stats/* endpoints (test statistics)
|
|-- frontend/                         # Streamlit dashboard
|   |-- streamlit_app.py             # Main UI (patient evaluation + model monitoring)
|   |-- requirements.txt             # Frontend dependencies
|   +-- .streamlit/config.toml       # Dark theme configuration
|
|-- models_new/                       # 8 trained .joblib model files
|   |-- cost_prediction_regressor_v1_history.joblib
|   |-- cost_prediction_regressor_v2_specialist.joblib
|   |-- cost_prediction_risk_classifier_under.joblib
|   |-- cost_prediction_risk_classifier_over.joblib
|   |-- charge_prediction_regressor_v1_history.joblib
|   |-- charge_prediction_regressor_v2_specialist.joblib
|   |-- charge_prediction_risk_classifier_under.joblib
|   +-- charge_prediction_risk_classifier_over.joblib
|
|-- data_sample/                      # Precomputed data served by the API
|   |-- test_sample.json             # 500 test patient records with actuals
|   |-- field_options.json           # Unique values per field (for dropdowns)
|   +-- test_stats.json              # PR curves, threshold/buffer stats, feature importance
|
|-- train_notebooks/                  # Model training notebooks
|   |-- cost_final.ipynb             # Cost estimation pipeline
|   +-- charge_final.ipynb           # Charge estimation pipeline
|
|-- scripts/                          # Data preparation and UI build tools
|   |-- prepare_sample_data.py       # Generates test_sample.json and field_options.json
|   |-- generate_test_stats.py       # Generates test_stats.json from full test dataset
|   |-- add_tab2.py                  # Adds Model Monitoring tab to frontend
|   |-- inject_tabs.py              # Injects tab structure into frontend
|   +-- refactor_ui.py              # Refactors frontend from tabs to sidebar navigation
|
|-- Dockerfile                        # Python 3.13-slim production image
|-- render.yaml                       # Render Blueprint (web service config)
|-- requirements.txt                  # Backend Python dependencies
+-- .dockerignore                     # Excludes training data, notebooks, scripts from image
```

---

## API Endpoints

Interactive Swagger documentation is available at `/docs` when the API is running.

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check — returns status and model counts |
| `/` | GET | API root — basic info and endpoint links |
| `/api/v1/cost/predict` | POST | Single cost prediction with risk assessment |
| `/api/v1/cost/predict/batch` | POST | Batch cost predictions (multiple patients) |
| `/api/v1/cost/thresholds` | GET | Available risk threshold presets for cost |
| `/api/v1/charge/predict` | POST | Single charge prediction with risk assessment |
| `/api/v1/charge/predict/batch` | POST | Batch charge predictions (multiple patients) |
| `/api/v1/charge/thresholds` | GET | Available risk threshold presets for charge |
| `/api/v1/data/random` | GET | Random patient record from test dataset |
| `/api/v1/data/random/batch` | GET | Multiple random patient records (max 50) |
| `/api/v1/data/field-options` | GET | Unique values per input field (for dropdowns) |
| `/api/v1/stats/test_stats` | GET | Precomputed PR curves, threshold stats, buffer analysis, feature importance |

---

## Frontend Features

The Streamlit dashboard provides two primary views toggled via the navigation selector:

### Patient Evaluation View

- **Interactive Admission Form**: 17 dropdown input fields dynamically populated from the API.
- **Randomize Patient**: Loads a random patient record from the 500 test sample dataset.
- **What-If Scenario Analysis**: Modify any admission parameter to observe the predicted cost and charge deltas with absolute dollar changes, percentage shifts, and direction indicators.
- **Dual Simultaneous Prediction**: Evaluates cost and charge models in parallel with a single action.
- **Risk Visualization**: Displays risk factor categorization (Low, Medium, High, Critical), classification badges (Safe, Risky), and calibrated under/over prediction risk probability bars.
- **Actual vs. Predicted Comparison**: When viewing unmodified test sample records, shows ground truth values and variance metrics.
- **Summary Metrics**: Displays estimated cost, estimated charge, net margin, and margin percentage.
- **Configurable Risk Filtering**: Adjusts risk sensitivity levels directly via the sidebar.

### Model Monitoring View

- **Dual Risk Impact Analysis**: Tabular breakdown across all threshold presets evaluating filtering rates, residual MAE, under-prediction percentages, and over-prediction percentages.
- **Buffer Strategy Analysis**: Evaluates the effect of applying financial buffers to claims that clear risk thresholds to mitigate under-prediction exposure.
- **Precision-Recall Curves**: Visualizes precision, recall, and F1-score across classification thresholds with area under the curve (AUC) metrics for under- and over-prediction classifiers.
- **Feature Importance**: Permutation importance rankings for V1 (History) and V2 (Specialist) regressors, visualized as sorted bar charts.
- **Model Selector**: Toggle analysis views between Cost and Charge pipelines.

### Sidebar (Patient Evaluation View)

- Risk filter preset selector
- Randomize patient trigger
- Cached backend connectivity status indicator (evaluated every 60 seconds)
- Project repository link
- Automatically hidden in the Model Monitoring view

---

## Tech Stack

| Component | Technology |
|-----------|-----------|
| ML Framework | scikit-learn 1.8.0 |
| Data Processing | pandas 3.0.2, numpy |
| Backend API | FastAPI, Uvicorn |
| Frontend | Streamlit |
| HTTP Client | httpx |
| Model Serialization | joblib |
| Containerization | Docker (Python 3.13-slim) |
| Backend Hosting | Render (Free Tier) |
| Frontend Hosting | Streamlit Community Cloud |
| Input Validation | Pydantic v2 with Literal types |

---

## Dataset

- **Source**: New York State Department of Health Statewide Planning and Research Cooperative System (SPARCS) Inpatient Discharge Dataset.
- **Target Variables**: `Total Costs` and `Total Charges`.
- **Sample Records**: 500 test patient records bundled with the application for interactive sampling.
- **Full Test Set**: Used offline to compute static evaluation metrics, precision-recall curves, buffer strategies, and feature importances stored in `data_sample/test_stats.json`.
- **Training Data**: Raw and processed training CSV files are excluded from the repository.

---

## Local Development

### Backend Setup

```bash
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

### Frontend Setup

```bash
cd frontend
pip install -r requirements.txt
streamlit run streamlit_app.py
```

*Note: The backend API must be running for the frontend to retrieve field options and execute inference requests.*

---

## Training Notebooks

Comprehensive exploratory data analysis, feature engineering, model training, hyperparameter optimization, and evaluation pipelines are documented in:

- [`train_notebooks/cost_final.ipynb`](train_notebooks/cost_final.ipynb) — Cost estimation pipeline
- [`train_notebooks/charge_final.ipynb`](train_notebooks/charge_final.ipynb) — Charge estimation pipeline

---

## Deployment

- **Backend (Render)**: Packaged via multi-stage Docker build using `python:3.13-slim`. Copies only `app/`, `models_new/`, and `data_sample/`. Configured with a single Uvicorn worker for memory optimization on Render free tier. Health check endpoint: `/health`.
- **Frontend (Streamlit Cloud)**: Connected directly to the repository and configured to communicate with the Render API backend via HTTPS.

---

## Author

Built by [Priyanshu Verma](https://github.com/Priyanshukv06)
