# Project Progress - Cost & Charge Estimation Deployment

## Current Status: ✅ Backend Complete & Tested Locally

---

## Phase 1: Backend API (FastAPI + Docker)

| Task | Status | Notes |
|------|--------|-------|
| Analyze existing ML pipelines | ✅ Done | 8 models (4 cost + 4 charge) understood |
| Create implementation plan | ✅ Done | Approved by user with comments |
| Create FastAPI app structure | ✅ Done | app/main.py, models.py, inference.py, model_loader.py, custom_transformers.py |
| Implement inference pipeline | ✅ Done | Hybrid feature engineering + dual risk classifiers |
| Implement API routers (cost/charge/data) | ✅ Done | Single + batch predict, random data, field options |
| Add keep-alive mechanism | ✅ Done | Self-ping every 14 min via SERVICE_URL |
| Create sample data prep script | ✅ Done | 500 samples + 17 field dropdown options |
| Dockerfile + deployment configs | ✅ Done | Python 3.13-slim, render.yaml |
| Local testing | ✅ Done | All endpoints verified working |
| Deploy to Render | ⬜ Pending | Push to GitHub, connect to Render |

## Phase 2: Frontend (Streamlit) — Not Started

| Task | Status | Notes |
|------|--------|-------|
| Streamlit UI for predictions | ⬜ Pending | Will be hosted on streamlit.io |
| Randomize button for test data | ⬜ Pending | Backend endpoint ready |
| Editable fields with dropdowns | ⬜ Pending | Backend field-options endpoint ready |
| Display predictions + risk flags | ⬜ Pending | |

---

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/api/v1/cost/predict` | POST | Predict total cost for a patient |
| `/api/v1/cost/predict/batch` | POST | Batch cost predictions |
| `/api/v1/cost/thresholds` | GET | Available cost risk threshold presets |
| `/api/v1/charge/predict` | POST | Predict total charge for a patient |
| `/api/v1/charge/predict/batch` | POST | Batch charge predictions |
| `/api/v1/charge/thresholds` | GET | Available charge risk threshold presets |
| `/api/v1/data/random` | GET | Random patient from test set |
| `/api/v1/data/random/batch` | GET | Multiple random patients |
| `/api/v1/data/field-options` | GET | Unique values per field (for dropdowns) |

## Key Design Decisions
- **Risk thresholds**: Configurable presets from "Lenient" to "Aggressive" filtering
- **No buffer**: Buffer strategy removed from deployment
- **Length of Stay**: Estimated via target encoding (not user input)
- **Keep-alive**: Backend self-pings every 14 min to prevent Render spin-down
- **Custom transformer**: `AutomatedFeatureEngineer` class registered at startup for joblib deserialization

## Test Results (Local)
- **Cost prediction**: `$15,708.34` for Cesarean Delivery, Moderate severity → `is_safe: true`
- **Charge prediction**: `$53,565.91` for same patient with aggressive filtering → `is_safe: false` (under-prediction risk flagged)
- Both models running simultaneously ✅
