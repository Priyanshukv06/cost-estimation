"""
Core inference logic — extracted from the training notebooks.

Replicates the exact prediction pipeline:
1. Build input DataFrame from patient data
2. Generate hybrid features from V1 + V2 models
3. Run risk classifiers for under/over prediction probabilities
4. Return prediction with risk assessment
"""

import pandas as pd
import numpy as np
from app.models import PatientInput


# ─── Column Definitions ──────────────────────────────────────────────────────
# These must match exactly what the trained AutomatedFeatureEngineer expects.

FEATURE_COLUMNS = [
    'Hospital Service Area', 'Hospital County', 'Facility Name',
    'Age Group', 'Gender', 'Race', 'Ethnicity',
    'Type of Admission', 'Patient Disposition',
    'CCSR Diagnosis Description', 'CCSR Procedure Description',
    'APR DRG Description', 'APR MDC Description',
    'APR Severity of Illness Description', 'APR Risk of Mortality',
    'APR Medical Surgical Description', 'Birth Weight',
    'Emergency Department Indicator',
]


def _patient_to_dict(p: PatientInput) -> dict:
    """
    Converts a PatientInput Pydantic model to a dict
    with the exact column names the model pipelines expect.
    """
    return {
        'Hospital Service Area': p.hospital_service_area,
        'Hospital County': p.hospital_county,
        'Facility Name': p.facility_name,
        'Age Group': p.age_group,
        'Gender': p.gender,
        'Race': p.race,
        'Ethnicity': p.ethnicity,
        'Type of Admission': p.type_of_admission,
        'Patient Disposition': p.patient_disposition,
        'CCSR Diagnosis Description': p.ccsr_diagnosis_description,
        'CCSR Procedure Description': p.ccsr_procedure_description,
        'APR DRG Description': p.apr_drg_description,
        'APR MDC Description': p.apr_mdc_description,
        'APR Severity of Illness Description': p.apr_severity_of_illness_description,
        'APR Risk of Mortality': p.apr_risk_of_mortality,
        'APR Medical Surgical Description': p.apr_medical_surgical_description,
        'Birth Weight': p.birth_weight,
        'Emergency Department Indicator': p.emergency_department_indicator,
    }


def patients_to_dataframe(patients: list[PatientInput]) -> pd.DataFrame:
    """Converts a list of PatientInput to a multi-row DataFrame."""
    return pd.DataFrame([_patient_to_dict(p) for p in patients])


def patient_to_dataframe(patient: PatientInput) -> pd.DataFrame:
    """
    Converts a PatientInput Pydantic model to a single-row DataFrame
    with the exact column names the model pipelines expect.
    """
    return patients_to_dataframe([patient])


def create_hybrid_features(X_raw: pd.DataFrame, model_v1, model_v2):
    """
    Builds the combined feature set used by the risk classifiers.
    Replicates the exact logic from the training notebooks.

    Returns:
        (X_hybrid DataFrame, V2 predictions array)
    """
    # Get transformed features from both models
    X_v2 = model_v2.named_steps['feature_engineer'].transform(X_raw)
    X_v1_all = model_v1.named_steps['feature_engineer'].transform(X_raw)

    # Keep only target-encoded columns from V1 (MEDIAN_COST + DIH)
    cols_to_keep = [c for c in X_v1_all.columns if 'MEDIAN_COST' in c or 'DIH' in c]
    X_v1_subset = X_v1_all[cols_to_keep].copy()
    X_v1_subset.columns = [c + '_M1_HIST' for c in X_v1_subset.columns]

    # Get predictions from both models
    preds_v1 = model_v1.predict(X_raw)
    preds_v2 = model_v2.predict(X_raw)

    # Combine into hybrid feature set
    X_hybrid = pd.concat([X_v2, X_v1_subset], axis=1)
    X_hybrid['PRED_M1_DIRTY'] = preds_v1
    X_hybrid['PRED_M2_CLEAN'] = preds_v2
    X_hybrid['PRED_DIFF'] = preds_v1 - preds_v2

    return X_hybrid.fillna(0), preds_v2


def _build_prediction_result(
    predicted_amount: float,
    prob_under: float,
    prob_over: float,
    thresholds: tuple[float, float],
    model_type: str,
    patient: PatientInput,
) -> dict:
    """
    Builds a standardized prediction result dict from raw model outputs.
    Shared by both predict_single and predict_batch.
    """
    thresh_u, thresh_o = thresholds

    risk_flag_under = prob_under >= thresh_u
    risk_flag_over = prob_over >= thresh_o
    is_safe = not risk_flag_under and not risk_flag_over

    # Confidence level based on risk probabilities
    max_risk = max(prob_under, prob_over)
    if max_risk < 0.3:
        confidence = "HIGH"
    elif max_risk < 0.6:
        confidence = "MEDIUM"
    else:
        confidence = "LOW"

    # Get actual value for comparison if provided
    actual_amount = None
    if model_type == "cost" and patient.actual_total_cost is not None:
        actual_amount = patient.actual_total_cost
    elif model_type == "charge" and patient.actual_total_charge is not None:
        actual_amount = patient.actual_total_charge

    return {
        "predicted_amount": round(predicted_amount, 2),
        "risk_probability_under": round(prob_under, 4),
        "risk_probability_over": round(prob_over, 4),
        "risk_flag_under": risk_flag_under,
        "risk_flag_over": risk_flag_over,
        "is_safe": is_safe,
        "confidence_level": confidence,
        "risk_filter_level": "",  # Filled by the router
        "risk_thresholds_used": {"under": thresh_u, "over": thresh_o},
        "model_type": model_type,
        "actual_amount": actual_amount,
    }


def predict_single(
    patient: PatientInput,
    models: dict,
    thresholds: tuple[float, float],
    model_type: str,
) -> dict:
    """
    Full inference pipeline for a single patient.

    Args:
        patient: PatientInput with all admission-time fields
        models: dict with keys 'v1', 'v2', 'risk_under', 'risk_over'
        thresholds: (under_threshold, over_threshold)
        model_type: 'cost' or 'charge'

    Returns:
        dict with prediction results
    """
    X = patient_to_dataframe(patient)

    # Build hybrid features and get V2 prediction
    X_hybrid, preds_v2 = create_hybrid_features(X, models['v1'], models['v2'])
    predicted_amount = float(preds_v2[0])

    # Get risk probabilities
    prob_under = float(models['risk_under'].predict_proba(X_hybrid)[0, 1])
    prob_over = float(models['risk_over'].predict_proba(X_hybrid)[0, 1])

    return _build_prediction_result(
        predicted_amount, prob_under, prob_over,
        thresholds, model_type, patient,
    )


def predict_batch(
    patients: list[PatientInput],
    models: dict,
    thresholds: tuple[float, float],
    model_type: str,
) -> list[dict]:
    """
    Batch inference for multiple patients.
    More efficient than calling predict_single in a loop
    because it processes all patients through the pipeline at once.
    """
    if not patients:
        return []

    X = patients_to_dataframe(patients)

    # Build hybrid features and get V2 predictions
    X_hybrid, preds_v2 = create_hybrid_features(X, models['v1'], models['v2'])

    # Get risk probabilities for all patients
    probs_under = models['risk_under'].predict_proba(X_hybrid)[:, 1]
    probs_over = models['risk_over'].predict_proba(X_hybrid)[:, 1]

    return [
        _build_prediction_result(
            float(preds_v2[i]), float(probs_under[i]), float(probs_over[i]),
            thresholds, model_type, patient,
        )
        for i, patient in enumerate(patients)
    ]
