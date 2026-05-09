"""
Pydantic request/response schemas for the Cost & Charge Estimation API.
"""

from pydantic import BaseModel, Field
from typing import Optional


# ─── Risk Threshold Presets ───────────────────────────────────────────────────
# Named from "Lenient" (fewer rejections) to "Aggressive" (more rejections)

COST_THRESHOLD_PRESETS = {
    "lenient":      (0.70, 0.95),
    "moderate":     (0.65, 0.90),
    "balanced":     (0.60, 0.85),
    "cautious":     (0.55, 0.80),
    "aggressive":   (0.50, 0.75),
}

CHARGE_THRESHOLD_PRESETS = {
    "lenient":      (0.70, 0.85),
    "moderate":     (0.65, 0.80),
    "balanced":     (0.60, 0.75),
    "cautious":     (0.55, 0.70),
    "aggressive":   (0.50, 0.65),
}


# ─── Request Schemas ──────────────────────────────────────────────────────────

class PatientInput(BaseModel):
    """All fields available at admission time for prediction."""

    hospital_service_area: str = Field(..., description="Hospital Service Area")
    hospital_county: str = Field(..., description="Hospital County")
    facility_name: str = Field(..., description="Facility Name")
    age_group: str = Field(..., description="Age Group (e.g. '30 to 49')")
    gender: str = Field(..., description="Gender (M/F/U)")
    race: str = Field(..., description="Race")
    ethnicity: str = Field(..., description="Ethnicity")
    type_of_admission: str = Field(..., description="Type of Admission (e.g. 'Emergency', 'Elective')")
    patient_disposition: str = Field(..., description="Patient Disposition")
    ccsr_diagnosis_description: str = Field(..., description="CCSR Diagnosis Description")
    ccsr_procedure_description: str = Field(..., description="CCSR Procedure Description")
    apr_drg_description: str = Field(..., description="APR DRG Description")
    apr_mdc_description: str = Field(..., description="APR MDC Description")
    apr_severity_of_illness_description: str = Field(..., description="APR Severity of Illness Description")
    apr_risk_of_mortality: str = Field(..., description="APR Risk of Mortality")
    apr_medical_surgical_description: str = Field(..., description="APR Medical Surgical Description")
    birth_weight: float = Field(0, description="Birth Weight (0 if not applicable, -1 if unknown)")
    emergency_department_indicator: str = Field(..., description="Emergency Department Indicator (Y/N)")

    # Optional: actual values for comparison (populated from test data samples)
    actual_total_cost: Optional[float] = Field(None, description="Actual Total Cost (for comparison)")
    actual_total_charge: Optional[float] = Field(None, description="Actual Total Charge (for comparison)")


class PredictionRequest(BaseModel):
    """Single prediction request with configurable risk filtering."""

    patient: PatientInput
    risk_filter_level: str = Field(
        "balanced",
        description="Risk filtering level: 'lenient', 'moderate', 'balanced', 'cautious', 'aggressive'"
    )


class BatchPredictionRequest(BaseModel):
    """Batch prediction request."""

    patients: list[PatientInput]
    risk_filter_level: str = Field("balanced")


# ─── Response Schemas ─────────────────────────────────────────────────────────

class PredictionResponse(BaseModel):
    """Response for a single prediction."""

    predicted_amount: float = Field(..., description="Predicted amount from V2 specialist model")
    risk_probability_under: float = Field(..., description="Probability of under-prediction")
    risk_probability_over: float = Field(..., description="Probability of over-prediction")
    risk_flag_under: bool = Field(..., description="True if flagged as high under-prediction risk")
    risk_flag_over: bool = Field(..., description="True if flagged as high over-prediction risk")
    is_safe: bool = Field(..., description="True if cleared both risk thresholds")
    confidence_level: str = Field(..., description="HIGH / MEDIUM / LOW")
    risk_filter_level: str = Field(..., description="Risk filter level used")
    risk_thresholds_used: dict = Field(..., description="Actual (under, over) thresholds applied")
    model_type: str = Field(..., description="'cost' or 'charge'")

    # Optional: actual values for comparison
    actual_amount: Optional[float] = Field(None, description="Actual amount if provided")


class BatchPredictionResponse(BaseModel):
    """Response for batch predictions."""
    predictions: list[PredictionResponse]


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    models_loaded: bool
    cost_models_count: int
    charge_models_count: int


class FieldOptionsResponse(BaseModel):
    """Available values for each input field (for dropdown population)."""
    field_options: dict[str, list[str]]


class RandomPatientResponse(BaseModel):
    """A random patient record from the test dataset."""
    patient: dict
