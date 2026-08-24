"""
Charge estimation API router.
"""

from fastapi import APIRouter
from app.models import (
    PredictionRequest, PredictionResponse,
    BatchPredictionRequest, BatchPredictionResponse,
    CHARGE_THRESHOLD_PRESETS,
)
from app.model_loader import get_charge_models
from app.inference import predict_single, predict_batch

router = APIRouter(prefix="/api/v1/charge", tags=["Charge Estimation"])


def _resolve_thresholds(level: str) -> tuple[float, float]:
    """Resolves a named filter level to actual (under, over) threshold values."""
    return CHARGE_THRESHOLD_PRESETS[level]


@router.post("/predict", response_model=PredictionResponse)
async def predict_charge(request: PredictionRequest):
    """
    Predict total charge for a single patient using admission-time parameters.

    Risk filter levels (lenient → aggressive):
    - **lenient**: (0.70, 0.85) — fewer cases flagged as risky
    - **moderate**: (0.65, 0.80)
    - **balanced**: (0.60, 0.75) — default
    - **cautious**: (0.55, 0.70)
    - **aggressive**: (0.50, 0.65) — more cases flagged as risky
    """
    thresholds = _resolve_thresholds(request.risk_filter_level)
    models = get_charge_models()

    result = predict_single(
        patient=request.patient,
        models=models,
        thresholds=thresholds,
        model_type="charge",
    )
    result["risk_filter_level"] = request.risk_filter_level
    return PredictionResponse(**result)


@router.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_charge_batch(request: BatchPredictionRequest):
    """Predict total charge for multiple patients in a single request."""
    thresholds = _resolve_thresholds(request.risk_filter_level)
    models = get_charge_models()

    results = predict_batch(
        patients=request.patients,
        models=models,
        thresholds=thresholds,
        model_type="charge",
    )
    for r in results:
        r["risk_filter_level"] = request.risk_filter_level

    return BatchPredictionResponse(
        predictions=[PredictionResponse(**r) for r in results]
    )


@router.get("/thresholds")
async def get_charge_thresholds():
    """Returns available risk threshold presets for charge estimation."""
    return {
        "presets": {
            name: {"under": t[0], "over": t[1]}
            for name, t in CHARGE_THRESHOLD_PRESETS.items()
        },
        "description": "From 'lenient' (fewer flags) to 'aggressive' (more flags)"
    }
