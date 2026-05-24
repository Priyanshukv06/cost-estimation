"""
Hospital Cost & Charge Estimation — Streamlit Frontend

A polished UI for predicting hospital inpatient discharge costs and charges
using admission-time parameters. Connects to the FastAPI backend on Render.
"""

import streamlit as st
import httpx
import json

# ─── Configuration ────────────────────────────────────────────────────────────

API_BASE = "https://cost-estimation-g6ny.onrender.com"
REQUEST_TIMEOUT = 60  # seconds (cold start can take up to 60s)

# Risk filter presets with display labels
FILTER_LEVELS = {
    "🟢 Lenient — Fewer flags, wider acceptance": "lenient",
    "🔵 Moderate — Slightly tighter filtering": "moderate",
    "⚖️ Balanced — Default recommended setting": "balanced",
    "🟠 Cautious — More cases flagged as risky": "cautious",
    "🔴 Aggressive — Strictest filtering": "aggressive",
}

# Column name mapping: API field → Display label
FIELD_LABELS = {
    "Hospital Service Area": "Hospital Service Area",
    "Hospital County": "Hospital County",
    "Facility Name": "Facility Name",
    "Age Group": "Age Group",
    "Gender": "Gender",
    "Race": "Race",
    "Ethnicity": "Ethnicity",
    "Type of Admission": "Type of Admission",
    "Patient Disposition": "Patient Disposition",
    "CCSR Diagnosis Description": "CCSR Diagnosis",
    "CCSR Procedure Description": "CCSR Procedure",
    "APR DRG Description": "APR DRG",
    "APR MDC Description": "APR MDC",
    "APR Severity of Illness Description": "Severity of Illness",
    "APR Risk of Mortality": "Risk of Mortality",
    "APR Medical Surgical Description": "Medical / Surgical",
    "Emergency Department Indicator": "Emergency Dept (Y/N)",
}

# Pydantic field name mapping for API requests
FIELD_TO_API = {
    "Hospital Service Area": "hospital_service_area",
    "Hospital County": "hospital_county",
    "Facility Name": "facility_name",
    "Age Group": "age_group",
    "Gender": "gender",
    "Race": "race",
    "Ethnicity": "ethnicity",
    "Type of Admission": "type_of_admission",
    "Patient Disposition": "patient_disposition",
    "CCSR Diagnosis Description": "ccsr_diagnosis_description",
    "CCSR Procedure Description": "ccsr_procedure_description",
    "APR DRG Description": "apr_drg_description",
    "APR MDC Description": "apr_mdc_description",
    "APR Severity of Illness Description": "apr_severity_of_illness_description",
    "APR Risk of Mortality": "apr_risk_of_mortality",
    "APR Medical Surgical Description": "apr_medical_surgical_description",
    "Emergency Department Indicator": "emergency_department_indicator",
}


# ─── Custom CSS ───────────────────────────────────────────────────────────────

def inject_css():
    st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

        /* Global font */
        html, body, [class*="css"] {
            font-family: 'Inter', sans-serif;
        }

        /* Header gradient */
        .main-header {
            background: linear-gradient(135deg, #6C5CE7 0%, #a29bfe 50%, #74b9ff 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-size: 2.8rem;
            font-weight: 700;
            margin-bottom: 0;
            letter-spacing: -0.02em;
        }

        .sub-header {
            color: #a0a0b0;
            font-size: 1.05rem;
            margin-top: -8px;
            margin-bottom: 24px;
            font-weight: 300;
        }

        /* Metric cards */
        .metric-card {
            background: linear-gradient(145deg, #1e2130 0%, #252839 100%);
            border-radius: 16px;
            padding: 24px;
            border: 1px solid rgba(108, 92, 231, 0.15);
            box-shadow: 0 4px 24px rgba(0, 0, 0, 0.2);
            margin-bottom: 16px;
            transition: transform 0.2s ease, box-shadow 0.2s ease;
        }

        .metric-card:hover {
            transform: translateY(-2px);
            box-shadow: 0 8px 32px rgba(108, 92, 231, 0.15);
        }

        .metric-label {
            color: #8a8a9a;
            font-size: 0.8rem;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            font-weight: 500;
            margin-bottom: 4px;
        }

        .metric-value {
            font-size: 2rem;
            font-weight: 700;
            margin: 0;
            line-height: 1.2;
        }

        .metric-value.cost { color: #00cec9; }
        .metric-value.charge { color: #6C5CE7; }
        .metric-value.safe { color: #00b894; }
        .metric-value.risky { color: #ff7675; }

        /* Risk badges */
        .badge {
            display: inline-block;
            padding: 4px 14px;
            border-radius: 20px;
            font-size: 0.78rem;
            font-weight: 600;
            letter-spacing: 0.04em;
        }

        .badge-safe {
            background: rgba(0, 184, 148, 0.15);
            color: #00b894;
            border: 1px solid rgba(0, 184, 148, 0.3);
        }

        .badge-risky {
            background: rgba(255, 118, 117, 0.15);
            color: #ff7675;
            border: 1px solid rgba(255, 118, 117, 0.3);
        }

        .badge-medium {
            background: rgba(253, 203, 110, 0.15);
            color: #fdcb6e;
            border: 1px solid rgba(253, 203, 110, 0.3);
        }

        /* Confidence indicators */
        .confidence-high { color: #00b894; }
        .confidence-medium { color: #fdcb6e; }
        .confidence-low { color: #ff7675; }

        /* Risk probability bar */
        .risk-bar-container {
            background: #1a1d26;
            border-radius: 8px;
            height: 8px;
            overflow: hidden;
            margin: 6px 0;
        }

        .risk-bar {
            height: 100%;
            border-radius: 8px;
            transition: width 0.5s ease;
        }

        /* Section dividers */
        .section-title {
            font-size: 1.2rem;
            font-weight: 600;
            color: #e0e0e8;
            margin: 24px 0 12px 0;
            padding-bottom: 8px;
            border-bottom: 2px solid rgba(108, 92, 231, 0.3);
        }

        /* Comparison card */
        .comparison-row {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 8px 0;
            border-bottom: 1px solid rgba(255,255,255,0.05);
        }

        .comparison-label { color: #8a8a9a; font-size: 0.85rem; }
        .comparison-value { font-weight: 600; font-size: 0.95rem; }

        /* Hide default streamlit elements for cleaner look */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}

        /* Selectbox styling */
        div[data-baseweb="select"] {
            border-radius: 10px !important;
        }

        /* Button styling */
        .stButton > button {
            border-radius: 12px !important;
            font-weight: 600 !important;
            letter-spacing: 0.02em !important;
            transition: all 0.2s ease !important;
        }

        .stButton > button:hover {
            transform: translateY(-1px) !important;
            box-shadow: 0 4px 16px rgba(108, 92, 231, 0.3) !important;
        }
    </style>
    """, unsafe_allow_html=True)


# ─── API Helpers ──────────────────────────────────────────────────────────────

@st.cache_data(ttl=300)
def fetch_field_options():
    """Fetches dropdown options for all fields from the API."""
    try:
        r = httpx.get(f"{API_BASE}/api/v1/data/field-options", timeout=REQUEST_TIMEOUT)
        r.raise_for_status()
        return r.json().get("field_options", {})
    except Exception as e:
        st.error(f"Failed to load field options: {e}")
        return {}


def fetch_random_patient():
    """Fetches a random patient record from the test dataset."""
    try:
        r = httpx.get(f"{API_BASE}/api/v1/data/random", timeout=REQUEST_TIMEOUT)
        r.raise_for_status()
        return r.json().get("patient", {})
    except Exception as e:
        st.error(f"Failed to fetch random patient: {e}")
        return {}


def predict(endpoint: str, patient_data: dict, risk_level: str):
    """Sends a prediction request to the API."""
    payload = {
        "patient": patient_data,
        "risk_filter_level": risk_level,
    }
    try:
        r = httpx.post(
            f"{API_BASE}/api/v1/{endpoint}/predict",
            json=payload,
            timeout=REQUEST_TIMEOUT,
        )
        r.raise_for_status()
        return r.json()
    except httpx.HTTPStatusError as e:
        st.error(f"API error ({e.response.status_code}): {e.response.text}")
        return None
    except Exception as e:
        st.error(f"Request failed: {e}")
        return None


# ─── UI Rendering Helpers ─────────────────────────────────────────────────────

def render_metric_card(label: str, value: str, css_class: str = ""):
    st.markdown(f"""
    <div class="metric-card">
        <p class="metric-label">{label}</p>
        <p class="metric-value {css_class}">{value}</p>
    </div>
    """, unsafe_allow_html=True)


def render_risk_bar(probability: float, label: str, color: str):
    width_pct = min(probability * 100, 100)
    st.markdown(f"""
    <div style="margin-bottom: 12px;">
        <div style="display: flex; justify-content: space-between; margin-bottom: 4px;">
            <span style="color: #8a8a9a; font-size: 0.8rem;">{label}</span>
            <span style="color: {color}; font-weight: 600; font-size: 0.85rem;">{probability:.1%}</span>
        </div>
        <div class="risk-bar-container">
            <div class="risk-bar" style="width: {width_pct}%; background: {color};"></div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_prediction_result(result: dict, model_type: str,
                             show_actuals: bool = True,
                             baseline_amount: float | None = None):
    """Renders a prediction result card with all details.

    Args:
        result: prediction response dict from the API
        model_type: 'cost' or 'charge'
        show_actuals: whether to show actual values (False when user edited fields)
        baseline_amount: original prediction amount for delta display
    """
    if not result:
        return

    is_cost = model_type == "cost"
    accent = "#00cec9" if is_cost else "#6C5CE7"
    title = "💰 Cost Estimation" if is_cost else "🏷️ Charge Estimation"
    css_class = "cost" if is_cost else "charge"

    st.markdown(f'<div class="section-title">{title}</div>', unsafe_allow_html=True)

    # Main prediction amount
    amount = result["predicted_amount"]
    render_metric_card(
        f"Predicted {'Cost' if is_cost else 'Charge'}",
        f"${amount:,.2f}",
        css_class
    )

    # Delta from baseline (what-if analysis)
    if baseline_amount is not None and baseline_amount != amount:
        delta = amount - baseline_amount
        delta_pct = (delta / baseline_amount * 100) if baseline_amount != 0 else 0
        delta_color = "#ff7675" if delta > 0 else "#00b894"
        delta_sign = "+" if delta > 0 else ""
        arrow = "↑" if delta > 0 else "↓"

        st.markdown(f"""
        <div class="metric-card" style="border-left: 3px solid {delta_color};">
            <p class="metric-label">Change from Original Patient</p>
            <p style="font-size: 1.3rem; font-weight: 700; color: {delta_color}; margin: 4px 0;">
                {arrow} {delta_sign}${abs(delta):,.2f}
            </p>
            <p style="color: #8a8a9a; font-size: 0.85rem; margin: 0;">
                {delta_sign}{delta_pct:.1f}% &nbsp;|&nbsp; Baseline: ${baseline_amount:,.2f}
            </p>
        </div>
        """, unsafe_allow_html=True)

    # Actual vs Predicted comparison (only for unmodified random data)
    if show_actuals:
        actual = result.get("actual_amount")
        if actual is not None:
            diff = amount - actual
            diff_pct = (diff / actual * 100) if actual != 0 else 0
            diff_color = "#ff7675" if diff > 0 else "#00b894"
            diff_sign = "+" if diff > 0 else ""

            st.markdown(f"""
            <div class="metric-card">
                <p class="metric-label">Actual {'Cost' if is_cost else 'Charge'}</p>
                <p class="metric-value" style="color: #e0e0e8;">${actual:,.2f}</p>
                <p style="color: {diff_color}; font-size: 0.9rem; margin-top: 4px; font-weight: 600;">
                    {diff_sign}${diff:,.2f} ({diff_sign}{diff_pct:.1f}%)
                </p>
            </div>
            """, unsafe_allow_html=True)

    # Risk Assessment
    is_safe = result["is_safe"]
    confidence = result["confidence_level"]

    badge_class = "badge-safe" if is_safe else "badge-risky"
    status_text = "SAFE — Cleared Risk Filters" if is_safe else "RISKY — Flagged by Risk Filters"

    conf_class = f"confidence-{confidence.lower()}"
    conf_emoji = {"HIGH": "🟢", "MEDIUM": "🟡", "LOW": "🔴"}.get(confidence, "⚪")

    st.markdown(f"""
    <div class="metric-card">
        <p class="metric-label">Risk Assessment</p>
        <div style="margin: 8px 0;">
            <span class="badge {badge_class}">{status_text}</span>
        </div>
        <p style="margin-top: 12px; font-size: 0.85rem;">
            Confidence: <span class="{conf_class}" style="font-weight: 600;">{conf_emoji} {confidence}</span>
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Risk probability bars
    st.markdown(f"""
    <div class="metric-card">
        <p class="metric-label">Risk Probabilities</p>
    """, unsafe_allow_html=True)

    render_risk_bar(
        result["risk_probability_under"],
        "Under-prediction Risk",
        "#fdcb6e" if result["risk_probability_under"] < 0.5 else "#ff7675"
    )
    render_risk_bar(
        result["risk_probability_over"],
        "Over-prediction Risk",
        "#fdcb6e" if result["risk_probability_over"] < 0.5 else "#ff7675"
    )

    thresholds = result.get("risk_thresholds_used", {})
    st.markdown(f"""
        <p style="color: #606070; font-size: 0.75rem; margin-top: 8px;">
            Thresholds: Under={thresholds.get('under', 'N/A')}, Over={thresholds.get('over', 'N/A')}
            ({result.get('risk_filter_level', 'balanced')})
        </p>
    </div>
    """, unsafe_allow_html=True)


# ─── Main App ─────────────────────────────────────────────────────────────────

def main():
    st.set_page_config(
        page_title="Hospital Cost & Charge Estimator",
        page_icon="🏥",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    inject_css()

    # Header
    st.markdown('<h1 class="main-header">Hospital Cost & Charge Estimator</h1>', unsafe_allow_html=True)
    st.markdown(
        '<p class="sub-header">Predict inpatient discharge costs and charges using admission-time parameters</p>',
        unsafe_allow_html=True,
    )

    # ── Load Field Options (cached) ───────────────────────────────────────────
    field_options = fetch_field_options()

    # Navigation
    app_mode = st.radio("Navigation", ["Patient Evaluation", "Model Monitoring"], horizontal=True, label_visibility="collapsed")
    st.divider()

    if app_mode == "Patient Evaluation":
        # ── Sidebar ───────────────────────────────────────────────────────────────
        with st.sidebar:
            st.markdown("### ⚙️ Settings")

            # Risk filter selector
            filter_label = st.selectbox(
                "Risk Filter Level",
                options=list(FILTER_LEVELS.keys()),
                index=2,  # Default: Balanced
                help="Controls how strictly the model flags risky predictions. Lenient = fewer flags, Aggressive = more flags.",
            )
            risk_level = FILTER_LEVELS[filter_label]

            st.divider()

            # Randomize button
            st.markdown("### 🎲 Sample Data")
            st.caption("Load a random patient from the test dataset")
            if st.button("🔀 Randomize Patient", use_container_width=True, type="primary"):
                random_patient = fetch_random_patient()
                if random_patient:
                    st.session_state["patient_data"] = random_patient
                    st.session_state["has_actuals"] = True

                    # Directly SET each widget key to the new patient's value.
                    for field_name in FIELD_LABELS:
                        wkey = f"field_{field_name}"
                        new_val = str(random_patient.get(field_name, ""))
                        opts = field_options.get(field_name, [])
                        if opts and new_val in opts:
                            st.session_state[wkey] = new_val
                        elif opts:
                            st.session_state[wkey] = opts[0]

                    # Auto-predict after randomize
                    st.session_state["auto_predict"] = True
                    st.rerun()

            st.divider()

            # Backend status
            st.markdown("### 📡 Backend Status")
            try:
                health = httpx.get(f"{API_BASE}/health", timeout=10).json()
                st.success(f"Connected — {health.get('cost_models_count', 0) + health.get('charge_models_count', 0)} models loaded")
            except Exception:
                st.error("Backend unreachable (may be cold-starting, wait ~30s)")

            st.divider()

            # Portfolio / GitHub links
            st.markdown("### 📂 Project")
            st.markdown(
                "[![GitHub](https://img.shields.io/badge/GitHub-Source_Code-181717?logo=github&style=for-the-badge)]"
                "(https://github.com/Priyanshukv06/cost-estimation)"
            )
            st.caption("Built with FastAPI + scikit-learn")
            st.caption(f"Backend: `{API_BASE}`")
    else:
        # Provide dummy risk_level just in case, though it's not used in Monitoring tab
        risk_level = "balanced"
        
        # Hide the sidebar completely in Model Monitoring tab using CSS
        st.markdown(
            '<style>[data-testid="stSidebar"] {display: none;}</style>',
            unsafe_allow_html=True,
        )
    # ── Guard: field options required ─────────────────────────────────────────
    if not field_options:
        st.warning("⏳ Loading field options from backend... If this persists, the server may be cold-starting (takes ~30-60s on free tier).")
        st.stop()

    # ── Initialize defaults on first load ─────────────────────────────────────
    if "patient_data" not in st.session_state:
        st.session_state["patient_data"] = {}
        st.session_state["has_actuals"] = False

    patient = st.session_state.get("patient_data", {})

    if app_mode == "Patient Evaluation":
        pass # Used to be tab1
        # ── Patient Input Form ────────────────────────────────────────────────────
        st.markdown('<div class="section-title">📋 Patient Information</div>', unsafe_allow_html=True)
        st.caption("Edit any field below or click **Randomize Patient** in the sidebar to load test data")

        cols_per_row = 3
        field_names = list(FIELD_LABELS.keys())
        current_values = {}

        for i in range(0, len(field_names), cols_per_row):
            cols = st.columns(cols_per_row)
            for j, col in enumerate(cols):
                idx = i + j
                if idx >= len(field_names):
                    break

                field = field_names[idx]
                label = FIELD_LABELS[field]
                options = field_options.get(field, [])

                with col:
                    if options:
                        # Compute initial index (only used on first render
                        # when the key doesn't exist in session_state yet)
                        default_val = str(patient.get(field, ""))
                        default_idx = 0
                        if default_val in options:
                            default_idx = options.index(default_val)

                        selected = st.selectbox(
                            label,
                            options=options,
                            index=default_idx,
                            key=f"field_{field}",
                        )
                        current_values[field] = selected
                    else:
                        current_values[field] = st.text_input(
                            label,
                            value=str(patient.get(field, "")),
                            key=f"field_{field}",
                        )



        # ── Detect Manual Edits ────────────────────────────────────────────────────
        # Compare current widget values to stored patient_data to detect changes
        fields_modified = False
        changed_fields = []
        if patient:  # only check if we have a randomized patient
            for field in FIELD_LABELS:
                original_val = str(patient.get(field, ""))
                current_val = str(current_values.get(field, ""))
                if original_val and current_val and original_val != current_val:
                    fields_modified = True
                    changed_fields.append(FIELD_LABELS[field])

        # Show actuals only when patient data is unmodified
        show_actuals = st.session_state.get("has_actuals", False) and not fields_modified

        # ── Build API Payload ─────────────────────────────────────────────────────
        api_payload = {}
        for field, api_name in FIELD_TO_API.items():
            api_payload[api_name] = current_values.get(field, "")
        api_payload["birth_weight"] = 0  # Not used by model, always default

        # Only include actual values when patient data hasn't been manually edited
        if show_actuals:
            actual_cost = patient.get("Total Costs")
            actual_charge = patient.get("Total Charges")
            if actual_cost is not None:
                api_payload["actual_total_cost"] = float(actual_cost)
            if actual_charge is not None:
                api_payload["actual_total_charge"] = float(actual_charge)

        # Show edit indicator
        if fields_modified:
            st.info(
                f"✏️ **Modified fields:** {', '.join(changed_fields[:5])}"
                + (f" +{len(changed_fields) - 5} more" if len(changed_fields) > 5 else "")
                + " — Actuals hidden. Click Predict to see the impact."
            )

        # ── Predict Button / Auto-predict ─────────────────────────────────────────
        st.markdown("")
        predict_col1, predict_col2, predict_col3 = st.columns([1, 2, 1])
        with predict_col2:
            predict_clicked = st.button(
                "🚀 Predict Cost & Charge",
                use_container_width=True,
                type="primary",
            )

        # Auto-predict flag set by Randomize
        auto_predict = st.session_state.pop("auto_predict", False)
        should_predict = predict_clicked or auto_predict

        if should_predict:
            with st.spinner("Running predictions through both models..."):
                cost_result = predict("cost", api_payload, risk_level)
                charge_result = predict("charge", api_payload, risk_level)

            # Store results in session_state so they persist across reruns
            st.session_state["cost_result"] = cost_result
            st.session_state["charge_result"] = charge_result

            # If this is an auto-predict (from Randomize), save as baseline
            if auto_predict:
                st.session_state["baseline_cost"] = cost_result["predicted_amount"] if cost_result else None
                st.session_state["baseline_charge"] = charge_result["predicted_amount"] if charge_result else None

        # ── Results (show if available) ───────────────────────────────────────────
        cost_result = st.session_state.get("cost_result")
        charge_result = st.session_state.get("charge_result")

        # Get baseline amounts for delta display (only show when fields modified)
        baseline_cost = st.session_state.get("baseline_cost") if fields_modified else None
        baseline_charge = st.session_state.get("baseline_charge") if fields_modified else None

        if cost_result or charge_result:
            st.markdown("---")
            st.markdown('<div class="section-title">📊 Prediction Results</div>', unsafe_allow_html=True)

            col_cost, col_charge = st.columns(2)

            with col_cost:
                render_prediction_result(
                    cost_result, "cost",
                    show_actuals=show_actuals,
                    baseline_amount=baseline_cost,
                )

            with col_charge:
                render_prediction_result(
                    charge_result, "charge",
                    show_actuals=show_actuals,
                    baseline_amount=baseline_charge,
                )

            # Summary comparison
            if cost_result and charge_result:
                st.markdown("---")
                st.markdown('<div class="section-title">📈 Summary Comparison</div>', unsafe_allow_html=True)

                summary_cols = st.columns(4)
                with summary_cols[0]:
                    render_metric_card("Cost Estimate", f"${cost_result['predicted_amount']:,.2f}", "cost")
                with summary_cols[1]:
                    render_metric_card("Charge Estimate", f"${charge_result['predicted_amount']:,.2f}", "charge")
                with summary_cols[2]:
                    margin = charge_result['predicted_amount'] - cost_result['predicted_amount']
                    render_metric_card("Estimated Margin", f"${margin:,.2f}", "")
                with summary_cols[3]:
                    margin_pct = (margin / charge_result['predicted_amount'] * 100) if charge_result['predicted_amount'] != 0 else 0
                    render_metric_card("Margin %", f"{margin_pct:.1f}%", "")
    elif app_mode == "Model Monitoring":
        st.markdown('<div class="section-title">🔍 Model Monitoring & Test Statistics</div>', unsafe_allow_html=True)
        st.caption("Insights based on the full external test dataset.")
        
        # Load test stats
        @st.cache_data(ttl=300)
        def fetch_test_stats():
            try:
                r = httpx.get(f"{API_BASE}/api/v1/stats/test_stats", timeout=REQUEST_TIMEOUT)
                r.raise_for_status()
                return r.json()
            except Exception as e:
                st.error(f"Failed to fetch test stats: {e}")
                return {}
                
        test_stats = fetch_test_stats()
        
        if not test_stats:
            st.warning("Test stats not available. Ensure the backend has loaded them.")
        else:
            model_toggle = st.radio("Select Model", ["Cost Estimation", "Charge Estimation"], horizontal=True)
            mt = "cost" if "Cost" in model_toggle else "charge"
            stats = test_stats.get(mt, {})
            
            if stats:
                st.markdown("### 📊 Dual Risk Impact Analysis")
                
                # Format threshold labels
                thresh_stats = stats.get("threshold_stats", {})
                
                # Render table with all presets
                import pandas as pd
                
                table_rows = []
                for level, data in thresh_stats.items():
                    table_rows.append({
                        "Preset": level.capitalize(),
                        "U-Thresh": f"{data['threshold_u']:.2f}",
                        "O-Thresh": f"{data['threshold_o']:.2f}",
                        "Filtered %": f"{data['filtering_pct']:.1f}%",
                        "Overall MAE": f"${data['overall_mae']:,.0f}",
                        "Under-Pred %": f"{data['under_pct']:.1f}%",
                        "Under MAE": f"${data['under_mae']:,.0f}",
                        "Over-Pred %": f"{data['over_pct']:.1f}%",
                        "Over MAE": f"${data['over_mae']:,.0f}"
                    })
                    
                df_thresh = pd.DataFrame(table_rows)
                # Sort putting No Filter first, then Aggressive, Cautious, Balanced, Moderate, Lenient.
                # Actually, threshold_u increases, so let's just sort by U-Thresh
                df_thresh = df_thresh.sort_values(by="U-Thresh", ascending=False)
                st.dataframe(df_thresh, use_container_width=True, hide_index=True)
                
                st.markdown("---")
                st.markdown("### 🛡️ Buffer Strategy Analysis (Under-Prediction Risk Only)")
                
                thresh_options = []
                for level, data in thresh_stats.items():
                    label = f"{level.capitalize()} (U: {data['threshold_u']:.2f}, O: {data['threshold_o']:.2f})"
                    thresh_options.append((level, label))
                    
                selected_label = st.selectbox(
                    "Select Risk Threshold Preset for Buffer Analysis", 
                    [lbl for _, lbl in thresh_options],
                    index=3 # Default to Balanced assuming No Filter is now 0
                )
                selected_level = next(lvl for lvl, lbl in thresh_options if lbl == selected_label)
                
                st.caption(f"Applies a financial buffer scaled by the under-prediction probability to claims that cleared the {selected_level.capitalize()} thresholds.")
                
                buf_stats = stats.get("buffer_stats", {}).get(selected_level, [])
                if buf_stats:
                    buf_df = pd.DataFrame(buf_stats)
                    # Format DataFrame
                    display_df = buf_df.copy()
                    display_df['buffer'] = display_df['buffer'].apply(lambda x: f"${x:,}")
                    display_df['overall_mae'] = display_df['overall_mae'].apply(lambda x: f"${x:,.0f}")
                    display_df['under_pct'] = display_df['under_pct'].apply(lambda x: f"{x:.1f}%")
                    display_df['under_mae'] = display_df['under_mae'].apply(lambda x: f"${x:,.0f}")
                    display_df['over_pct'] = display_df['over_pct'].apply(lambda x: f"{x:.1f}%")
                    display_df['over_mae'] = display_df['over_mae'].apply(lambda x: f"${x:,.0f}")
                    
                    display_df.columns = [
                        "Buffer Max", "Overall MAE", 
                        "Under-Pred %", "Avg Deficit (Under MAE)", 
                        "Over-Pred %", "Avg Excess (Over MAE)",
                        "Flipped Cases (Under → Over)"
                    ]
                    
                    st.dataframe(display_df, use_container_width=True, hide_index=True)
                
                st.markdown("---")
                st.markdown("### 🎯 Precision-Recall Curves")
                pr_cols = st.columns(2)
                
                pr_under = stats.get("pr_curves", {}).get("under", {})
                pr_over = stats.get("pr_curves", {}).get("over", {})
                
                with pr_cols[0]:
                    st.markdown("**Under-Prediction Risk Classifier**")
                    if pr_under:
                        df_pu = pd.DataFrame({
                            "Precision": pr_under["precision"],
                            "Recall": pr_under["recall"],
                        }, index=pr_under["thresholds"]).sort_index()
                        df_pu["F1 Score"] = 2 * (df_pu["Precision"] * df_pu["Recall"]) / (df_pu["Precision"] + df_pu["Recall"] + 1e-9)
                        st.line_chart(df_pu, x_label="Threshold", y_label="Score", color=["#0984e3", "#d63031", "#00b894"])
                        st.caption(f"AUC: {pr_under.get('auc', 0):.3f}")
                
                with pr_cols[1]:
                    st.markdown("**Over-Prediction Risk Classifier**")
                    if pr_over:
                        df_po = pd.DataFrame({
                            "Precision": pr_over["precision"],
                            "Recall": pr_over["recall"],
                        }, index=pr_over["thresholds"]).sort_index()
                        df_po["F1 Score"] = 2 * (df_po["Precision"] * df_po["Recall"]) / (df_po["Precision"] + df_po["Recall"] + 1e-9)
                        st.line_chart(df_po, x_label="Threshold", y_label="Score", color=["#0984e3", "#d63031", "#00b894"])
                        st.caption(f"AUC: {pr_over.get('auc', 0):.3f}")
                        
                st.markdown("---")
                st.markdown("### 🔑 Feature Importance (Top Drivers)")
                st.caption("Derived via permutation importance from the estimation regressors.")
                
                fi_v1 = stats.get("feature_importance", {}).get("v1", {})
                fi_v2 = stats.get("feature_importance", {}).get("v2", {})
                
                fi_toggle = st.radio(
                    "Select Model for Feature Importance", 
                    options=["V1 (History Based)", "V2 (Specialist Based)"], 
                    horizontal=True,
                    key="fi_toggle"
                )
                
                if "V1" in fi_toggle:
                    fi_data = fi_v1
                    color = "#6c5ce7"
                else:
                    fi_data = fi_v2
                    color = "#00cec9"
                    
                if fi_data:
                    import altair as alt
                    df_fi = pd.DataFrame(list(fi_data.items()), columns=["Feature", "Importance"])
                    # Use Altair to ensure it's sorted visually descending (highest at top)
                    chart = alt.Chart(df_fi).mark_bar(color=color).encode(
                        x=alt.X("Importance:Q", title="Importance (MAE Impact)"),
                        y=alt.Y("Feature:N", sort="-x", title="")
                    ).properties(height=450)
                    st.altair_chart(chart, use_container_width=True)


if __name__ == "__main__":
    main()
