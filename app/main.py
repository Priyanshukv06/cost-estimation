"""
FastAPI application entry point.

Cost & Charge Estimation API — predicts hospital inpatient discharge
costs and charges using admission-time parameters.
"""

import asyncio
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.model_loader import load_all_models
from app.routers import cost, charge, data, stats
from app.routers.data import load_sample_data
from app.routers.stats import load_test_stats
from app.keep_alive import keep_alive_loop
from app.models import HealthResponse

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)


# ─── Lifespan ─────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup: load models + sample data. Shutdown: cleanup."""
    logger.info("=" * 60)
    logger.info("  Starting Cost & Charge Estimation API")
    logger.info("=" * 60)

    # Load ML models into memory
    load_all_models()

    # Load sample data for the randomize feature
    load_sample_data()

    # Load precomputed test statistics
    load_test_stats()

    # Start keep-alive background task
    keep_alive_task = asyncio.create_task(keep_alive_loop())

    logger.info("=" * 60)
    logger.info("  ✅ API Ready — all models loaded")
    logger.info("=" * 60)

    yield

    # Shutdown
    keep_alive_task.cancel()
    logger.info("API shutting down.")


# ─── App ──────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Hospital Cost & Charge Estimation API",
    description=(
        "Predicts hospital inpatient discharge costs and charges "
        "using admission-time parameters. Powered by dual-model "
        "pipelines with risk classification."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# CORS — allow Streamlit frontend and local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:8501",       # Local Streamlit
        "https://*.streamlit.app",     # Streamlit Cloud
        "*",                           # Allow all for development
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount routers
app.include_router(cost.router)
app.include_router(charge.router)
app.include_router(data.router)
app.include_router(stats.router)

# ─── Health Check ─────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """Health check endpoint. Also used by the keep-alive mechanism."""
    from app.model_loader import _models

    return HealthResponse(
        status="healthy",
        models_loaded=len(_models) > 0,
        cost_models_count=sum(1 for k in _models if k.startswith("cost_")),
        charge_models_count=sum(1 for k in _models if k.startswith("charge_")),
    )


@app.get("/", tags=["System"])
async def root():
    """API root — basic info and links."""
    return {
        "name": "Hospital Cost & Charge Estimation API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health",
        "endpoints": {
            "cost_predict": "/api/v1/cost/predict",
            "charge_predict": "/api/v1/charge/predict",
            "cost_thresholds": "/api/v1/cost/thresholds",
            "charge_thresholds": "/api/v1/charge/thresholds",
            "random_patient": "/api/v1/data/random",
            "field_options": "/api/v1/data/field-options",
        },
    }
