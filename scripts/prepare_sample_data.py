"""
Script to prepare sample data for the API.

Reads the test dataset, samples 500 rows, and extracts unique values
for each field (for dropdown population in the frontend).

Run once locally before building the Docker image:
    python scripts/prepare_sample_data.py

Outputs:
    data_sample/test_sample.json   — 500 random patient records
    data_sample/field_options.json — unique values per field
"""

import os
import sys
import json
import pandas as pd

# Configuration
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
TEST_FILE = os.path.join(PROJECT_ROOT, "data", "processed", "test_data.csv")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "data_sample")
SAMPLE_SIZE = 500

# Columns to keep in sample data
KEEP_COLUMNS = [
    'Hospital Service Area', 'Hospital County', 'Facility Name',
    'Age Group', 'Gender', 'Race', 'Ethnicity',
    'Type of Admission', 'Patient Disposition',
    'CCSR Diagnosis Description', 'CCSR Procedure Description',
    'APR DRG Description', 'APR MDC Description',
    'APR Severity of Illness Description', 'APR Risk of Mortality',
    'APR Medical Surgical Description', 'Birth Weight',
    'Emergency Department Indicator', 'Total Charges', 'Total Costs',
]

# Categorical fields for dropdown options
CATEGORICAL_FIELDS = [
    'Hospital Service Area', 'Hospital County', 'Facility Name',
    'Age Group', 'Gender', 'Race', 'Ethnicity',
    'Type of Admission', 'Patient Disposition',
    'CCSR Diagnosis Description', 'CCSR Procedure Description',
    'APR DRG Description', 'APR MDC Description',
    'APR Severity of Illness Description', 'APR Risk of Mortality',
    'APR Medical Surgical Description',
    'Emergency Department Indicator',
]


def main():
    print(f"Loading test data from: {TEST_FILE}")
    if not os.path.exists(TEST_FILE):
        print(f"ERROR: Test file not found: {TEST_FILE}")
        sys.exit(1)

    df = pd.read_csv(TEST_FILE)
    print(f"Loaded {len(df)} rows, {len(df.columns)} columns")

    # Keep only relevant columns
    available_cols = [c for c in KEEP_COLUMNS if c in df.columns]
    df = df[available_cols]

    # Clean numeric columns
    if 'Birth Weight' in df.columns:
        df['Birth Weight'] = df['Birth Weight'].apply(lambda x: -1 if str(x).strip().upper() == "UNKN" else x)
        df['Birth Weight'] = pd.to_numeric(df['Birth Weight'], errors='coerce').fillna(0)

    if 'Length of Stay' in df.columns:
        df['Length of Stay'] = df['Length of Stay'].apply(lambda x: 120 if str(x).strip() == "120 +" else x)
        df['Length of Stay'] = pd.to_numeric(df['Length of Stay'], errors='coerce').fillna(0)

    for col in ['Total Charges', 'Total Costs']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col].astype(str).str.replace(',', ''), errors='coerce')

    # Drop rows with missing targets
    df = df.dropna(subset=[c for c in ['Total Charges', 'Total Costs'] if c in df.columns])
    print(f"After cleaning: {len(df)} rows")

    # ─── Sample Data ──────────────────────────────────────────────────────────
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    sample_df = df.sample(n=min(SAMPLE_SIZE, len(df)), random_state=42)
    sample_records = sample_df.to_dict(orient='records')

    # Clean NaN values for JSON serialization
    for record in sample_records:
        for key, val in record.items():
            if pd.isna(val):
                record[key] = None

    sample_file = os.path.join(OUTPUT_DIR, "test_sample.json")
    with open(sample_file, "w") as f:
        json.dump(sample_records, f, indent=2)
    print(f"[OK] Saved {len(sample_records)} sample records to {sample_file}")

    # ─── Field Options ────────────────────────────────────────────────────────
    field_options = {}
    for col in CATEGORICAL_FIELDS:
        if col in df.columns:
            unique_vals = df[col].dropna().unique().tolist()
            unique_vals = sorted([str(v) for v in unique_vals])
            field_options[col] = unique_vals
            print(f"     {col}: {len(unique_vals)} unique values")

    options_file = os.path.join(OUTPUT_DIR, "field_options.json")
    with open(options_file, "w") as f:
        json.dump(field_options, f, indent=2)
    print(f"[OK] Saved field options for {len(field_options)} fields to {options_file}")

    print("\n[OK] Sample data preparation complete!")
    print(f"     Output directory: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
