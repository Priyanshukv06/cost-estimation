import os
import sys
import json
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, precision_recall_curve, auc
from sklearn.inspection import permutation_importance
import joblib

# Fix for deserializing custom transformer
from app.custom_transformers import AutomatedFeatureEngineer

def __sklearn_is_fitted__(self):
    return True
AutomatedFeatureEngineer.__sklearn_is_fitted__ = __sklearn_is_fitted__

sys.modules['__main__'].AutomatedFeatureEngineer = AutomatedFeatureEngineer

# Configuration
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
TEST_FILE = os.path.join(PROJECT_ROOT, "data", "processed", "test_data.csv")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models_new")
OUTPUT_FILE = os.path.join(PROJECT_ROOT, "data_sample", "test_stats.json")

# Threshold presets mapped exactly from app/models.py
COST_THRESHOLDS = {
    "lenient": (0.70, 0.95),
    "moderate": (0.65, 0.90),
    "balanced": (0.60, 0.85),
    "cautious": (0.55, 0.80),
    "aggressive": (0.50, 0.75),
}

CHARGE_THRESHOLDS = {
    "lenient": (0.70, 0.85),
    "moderate": (0.65, 0.80),
    "balanced": (0.60, 0.75),
    "cautious": (0.55, 0.70),
    "aggressive": (0.50, 0.65),
}

BUFFERS = [0, 2000, 5000, 7500, 10000, 15000]

from app.inference import create_hybrid_features

def subsample_curve(precision, recall, thresholds, max_points=100):
    if len(thresholds) <= max_points:
        return precision.tolist(), recall.tolist(), thresholds.tolist()
    indices = np.linspace(0, len(thresholds) - 1, max_points, dtype=int)
    return precision[indices].tolist(), recall[indices].tolist(), thresholds[indices].tolist()

def generate_stats_for_model(model_type, df):
    print(f"\nProcessing {model_type.upper()} models...")
    
    # Load models
    try:
        model_v1 = joblib.load(os.path.join(MODELS_DIR, f"{model_type}_prediction_regressor_v1_history.joblib"))
        model_v2 = joblib.load(os.path.join(MODELS_DIR, f"{model_type}_prediction_regressor_v2_specialist.joblib"))
        clf_under = joblib.load(os.path.join(MODELS_DIR, f"{model_type}_prediction_risk_classifier_under.joblib"))
        clf_over = joblib.load(os.path.join(MODELS_DIR, f"{model_type}_prediction_risk_classifier_over.joblib"))
    except Exception as e:
        print(f"Error loading {model_type} models: {e}")
        return None

    target_col = "Total Costs" if model_type == "cost" else "Total Charges"
    if target_col not in df.columns:
        print(f"Target column {target_col} missing from data!")
        return None
        
    # Drop rows with NaN target
    df_clean = df.dropna(subset=[target_col]).copy()
    y_true = df_clean[target_col].values
    
    # Run V1 & V2
    print("Generating hybrid features...")
    X_hybrid, y_pred_v2 = create_hybrid_features(df_clean, model_v1, model_v2)
    
    # Get probabilities
    print("Predicting probabilities...")
    probs_under = clf_under.predict_proba(X_hybrid)[:, 1]
    probs_over = clf_over.predict_proba(X_hybrid)[:, 1]
    
    # Calculate true risk labels for PR curves
    # (residuals > 0 means predicted > actual -> over-prediction)
    # (residuals < 0 means predicted < actual -> under-prediction)
    residuals = y_pred_v2 - y_true
    std_resid = np.std(residuals)
    y_true_under = (residuals < -0.5 * std_resid).astype(int)
    y_true_over = (residuals > 0.5 * std_resid).astype(int)
    
    print("Calculating PR curves...")
    pu, ru, tu = precision_recall_curve(y_true_under, probs_under)
    po, ro, to = precision_recall_curve(y_true_over, probs_over)
    
    pu_sub, ru_sub, tu_sub = subsample_curve(pu, ru, tu)
    po_sub, ro_sub, to_sub = subsample_curve(po, ro, to)
    
    stats = {
        "pr_curves": {
            "under": {"precision": pu_sub, "recall": ru_sub, "thresholds": tu_sub, "auc": float(auc(ru, pu))},
            "over": {"precision": po_sub, "recall": ro_sub, "thresholds": to_sub, "auc": float(auc(ro, po))}
        },
        "threshold_stats": {},
        "buffer_stats": {},
        "feature_importance": {}
    }
    
    thresholds_dict = COST_THRESHOLDS if model_type == "cost" else CHARGE_THRESHOLDS
    
    print("Calculating threshold and buffer impacts...")
    for level, (thresh_u, thresh_o) in thresholds_dict.items():
        # A claim is safe if it clears BOTH thresholds
        safe_mask = (probs_under < thresh_u) & (probs_over < thresh_o)
        y_true_safe = y_true[safe_mask]
        y_pred_safe = y_pred_v2[safe_mask]
        
        n_kept = len(y_true_safe)
        pct_filtered = 100.0 - ((n_kept / len(y_true)) * 100.0) if len(y_true) > 0 else 0
        
        if n_kept > 0:
            resid_safe = y_pred_safe - y_true_safe
            mae = float(mean_absolute_error(y_true_safe, y_pred_safe))
            
            under_mask = resid_safe < 0
            over_mask = resid_safe > 0
            
            n_under = under_mask.sum()
            n_over = over_mask.sum()
            
            pct_under = float(n_under / n_kept * 100)
            pct_over = float(n_over / n_kept * 100)
            
            mae_under = float(abs(resid_safe[under_mask]).mean()) if n_under > 0 else 0.0
            mae_over = float(resid_safe[over_mask].mean()) if n_over > 0 else 0.0
        else:
            mae = pct_under = pct_over = mae_under = mae_over = 0.0
            
        stats["threshold_stats"][level] = {
            "threshold_u": thresh_u,
            "threshold_o": thresh_o,
            "filtering_pct": pct_filtered,
            "overall_mae": mae,
            "under_pct": pct_under,
            "over_pct": pct_over,
            "under_mae": mae_under,
            "over_mae": mae_over,
        }
        
        # Buffer Stats
        stats["buffer_stats"][level] = []
        if n_kept > 0:
            y_safe_probs_under = probs_under[safe_mask]
            was_under = resid_safe < 0
            
            for buf in BUFFERS:
                padding = y_safe_probs_under * buf
                y_safe_pred_adj = y_pred_safe + padding
                residuals_adj = y_safe_pred_adj - y_true_safe
                
                buf_mae = float(mean_absolute_error(y_true_safe, y_safe_pred_adj))
                u_mask_adj = residuals_adj < 0
                o_mask_adj = residuals_adj > 0
                
                n_u_adj = u_mask_adj.sum()
                n_o_adj = o_mask_adj.sum()
                
                pct_u_adj = float(n_u_adj / n_kept * 100)
                pct_o_adj = float(n_o_adj / n_kept * 100)
                
                avg_u_adj = float(abs(residuals_adj[u_mask_adj]).mean()) if n_u_adj > 0 else 0.0
                avg_o_adj = float(residuals_adj[o_mask_adj].mean()) if n_o_adj > 0 else 0.0
                
                n_flipped = int((was_under & o_mask_adj).sum())
                
                stats["buffer_stats"][level].append({
                    "buffer": buf,
                    "overall_mae": buf_mae,
                    "under_pct": pct_u_adj,
                    "under_mae": avg_u_adj,
                    "over_pct": pct_o_adj,
                    "over_mae": avg_o_adj,
                    "flipped_cases": n_flipped
                })
                
    # Feature Importance (using permutation importance on a subset for speed)
    print("Calculating feature importance (sample=5000)...")
    sample_size = min(5000, len(X_hybrid))
    sample_idx = np.random.choice(len(X_hybrid), sample_size, replace=False)
    X_sample = X_hybrid.iloc[sample_idx]
    y_u_sample = y_true_under[sample_idx]
    y_o_sample = y_true_over[sample_idx]
    
    # Under model
    res_under = permutation_importance(clf_under, X_sample, y_u_sample, n_repeats=3, random_state=42, scoring='f1', n_jobs=-1)
    imp_u = pd.Series(res_under.importances_mean, index=X_hybrid.columns).sort_values(ascending=False).head(15)
    stats["feature_importance"]["under"] = imp_u.to_dict()
    
    # Over model
    res_over = permutation_importance(clf_over, X_sample, y_o_sample, n_repeats=3, random_state=42, scoring='f1', n_jobs=-1)
    imp_o = pd.Series(res_over.importances_mean, index=X_hybrid.columns).sort_values(ascending=False).head(15)
    stats["feature_importance"]["over"] = imp_o.to_dict()
    
    return stats

def main():
    print("Loading test dataset...")
    df = pd.read_csv(TEST_FILE)
    
    # Clean numeric columns (same as prepare_sample_data.py)
    if 'Birth Weight' in df.columns:
        df['Birth Weight'] = df['Birth Weight'].apply(lambda x: -1 if str(x).strip().upper() == "UNKN" else x)
        df['Birth Weight'] = pd.to_numeric(df['Birth Weight'], errors='coerce').fillna(0)
    if 'Length of Stay' in df.columns:
        df['Length of Stay'] = df['Length of Stay'].apply(lambda x: 120 if str(x).strip() == "120 +" else x)
        df['Length of Stay'] = pd.to_numeric(df['Length of Stay'], errors='coerce').fillna(0)
    for col in ['Total Charges', 'Total Costs']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col].astype(str).str.replace(',', ''), errors='coerce')

    final_stats = {}
    for mt in ["cost", "charge"]:
        s = generate_stats_for_model(mt, df)
        if s:
            final_stats[mt] = s
            
    # Save output
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    with open(OUTPUT_FILE, "w") as f:
        json.dump(final_stats, f, indent=2)
    print(f"\n[OK] Saved test statistics to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
