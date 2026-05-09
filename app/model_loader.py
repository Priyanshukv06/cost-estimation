"""
Model loader — loads all 8 joblib model files at startup and stores them in memory.
"""

import os
import logging
from joblib import load
from app.custom_transformers import register_custom_classes

logger = logging.getLogger(__name__)

MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models_new")

# Global model storage
_models: dict = {}


def get_models() -> dict:
    """Returns the loaded models dict. Raises if not loaded."""
    if not _models:
        raise RuntimeError("Models not loaded. Call load_all_models() first.")
    return _models


def load_all_models():
    """
    Loads all 8 model files into memory.
    Called once at application startup via FastAPI lifespan.
    """
    global _models

    # Register custom sklearn classes so pickle can find them
    register_custom_classes()

    model_files = {
        # Cost estimation models
        "cost_v1": "cost_prediction_regressor_v1_history.joblib",
        "cost_v2": "cost_prediction_regressor_v2_specialist.joblib",
        "cost_risk_under": "cost_prediction_risk_classifier_under.joblib",
        "cost_risk_over": "cost_prediction_risk_classifier_over.joblib",
        # Charge estimation models
        "charge_v1": "charge_prediction_regressor_v1_history.joblib",
        "charge_v2": "charge_prediction_regressor_v2_specialist.joblib",
        "charge_risk_under": "charge_prediction_risk_classifier_under.joblib",
        "charge_risk_over": "charge_prediction_risk_classifier_over.joblib",
    }

    logger.info(f"Loading models from: {MODELS_DIR}")

    for key, filename in model_files.items():
        filepath = os.path.join(MODELS_DIR, filename)
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found: {filepath}")

        logger.info(f"  Loading {key}: {filename}")
        _models[key] = load(filepath)

    logger.info(f"✅ All {len(_models)} models loaded successfully.")
    return _models


def get_cost_models() -> dict:
    """Returns the 4 cost estimation models."""
    m = get_models()
    return {
        "v1": m["cost_v1"],
        "v2": m["cost_v2"],
        "risk_under": m["cost_risk_under"],
        "risk_over": m["cost_risk_over"],
    }


def get_charge_models() -> dict:
    """Returns the 4 charge estimation models."""
    m = get_models()
    return {
        "v1": m["charge_v1"],
        "v2": m["charge_v2"],
        "risk_under": m["charge_risk_under"],
        "risk_over": m["charge_risk_over"],
    }
