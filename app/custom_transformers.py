"""
Custom sklearn transformers used in the trained pipelines.

These classes were defined in the training notebooks (cost_estimation.py,
charge_estimation.py) and are embedded in the serialized .joblib pipelines.
They MUST be importable at deserialization time for joblib.load() to work.

We register them in __main__ so pickle can find them when loading models.
"""

import sys
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OrdinalEncoder, TargetEncoder


class AutomatedFeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Custom feature engineering transformer used inside the sklearn Pipeline.

    Performs:
    - Ordinal encoding of categorical features
    - Target encoding of diagnosis/procedure codes against cost/charge target
    - Target encoding of diagnosis/procedure codes against Length of Stay (DIH)
    """

    def __init__(self):
        # Column Groups
        # Note: the cost model uses 'te_cost_cols' and the charge model uses
        # 'te_charge_cols', but both have the same columns. The joblib files
        # will have whichever attribute was defined during training.
        self.te_cost_cols = ['CCSR Diagnosis Description', 'CCSR Procedure Description', 'APR DRG Description']
        self.te_charge_cols = ['CCSR Diagnosis Description', 'CCSR Procedure Description', 'APR DRG Description']
        self.te_days_in_hosp_cols = ['CCSR Diagnosis Description', 'CCSR Procedure Description', 'APR DRG Description']

        self.ordinal_cols = [
            'Hospital Service Area', 'Hospital County', 'Facility Name',
            'Age Group', 'Gender', 'Race', 'Ethnicity',
            'Type of Admission', 'Patient Disposition',
            'APR MDC Description', 'APR Severity of Illness Description',
            'APR Risk of Mortality', 'APR Medical Surgical Description',
            'Emergency Department Indicator'
        ]

        self.numerical_cols = []

        # Encoders
        self.ord_enc = OrdinalEncoder(
            handle_unknown='use_encoded_value', unknown_value=-1,
            encoded_missing_value=-1, max_categories=250
        )
        self.cost_enc = TargetEncoder(smooth=50.0, target_type='continuous')
        self.charge_enc = TargetEncoder(smooth=50.0, target_type='continuous')
        self.dih_enc = TargetEncoder(smooth=20.0, target_type='continuous')

    def fit(self, X, y=None):
        self.ord_enc.fit(X[self.ordinal_cols].astype(str))

        # Fit cost/charge encoder
        te_cols = getattr(self, 'te_cost_cols', None) or getattr(self, 'te_charge_cols', [])
        encoder = getattr(self, 'cost_enc', None) or getattr(self, 'charge_enc', None)
        if te_cols and y is not None and encoder is not None:
            encoder.fit(X[te_cols], y)

        if self.te_days_in_hosp_cols and 'Length of Stay' in X.columns:
            self.dih_enc.fit(X[self.te_days_in_hosp_cols], X['Length of Stay'])

        return self

    def fit_transform(self, X, y=None):
        X_out = X.copy()
        X_out[self.ordinal_cols] = self.ord_enc.fit_transform(X_out[self.ordinal_cols].astype(str))

        # Determine which target encoder to use
        te_cols = getattr(self, 'te_cost_cols', None) or getattr(self, 'te_charge_cols', [])
        encoder = getattr(self, 'cost_enc', None) or getattr(self, 'charge_enc', None)

        if te_cols and y is not None and encoder is not None:
            trans_arr = encoder.fit_transform(X_out[te_cols], y)
            trans_df = pd.DataFrame(
                trans_arr,
                columns=[c + '_MEDIAN_COST' for c in te_cols],
                index=X.index
            )
        else:
            trans_df = pd.DataFrame(index=X.index)

        if self.te_days_in_hosp_cols and 'Length of Stay' in X_out.columns:
            dih_arr = self.dih_enc.fit_transform(X_out[self.te_days_in_hosp_cols], X_out['Length of Stay'])
            dih_df = pd.DataFrame(
                dih_arr,
                columns=[c + '_DIH' for c in self.te_days_in_hosp_cols],
                index=X.index
            )
        else:
            dih_df = pd.DataFrame(index=X.index)

        X_final = pd.concat([
            X_out[self.numerical_cols], X_out[self.ordinal_cols], trans_df, dih_df
        ], axis=1)
        return X_final.fillna(0)

    def transform(self, X):
        X_out = X.copy()
        X_out[self.ordinal_cols] = self.ord_enc.transform(X[self.ordinal_cols].astype(str))

        te_cols = getattr(self, 'te_cost_cols', None) or getattr(self, 'te_charge_cols', [])
        encoder = getattr(self, 'cost_enc', None) or getattr(self, 'charge_enc', None)

        if te_cols and encoder is not None:
            trans_arr = encoder.transform(X[te_cols])
            trans_df = pd.DataFrame(
                trans_arr,
                columns=[c + '_MEDIAN_COST' for c in te_cols],
                index=X.index
            )
        else:
            trans_df = pd.DataFrame(index=X.index)

        if self.te_days_in_hosp_cols:
            dih_arr = self.dih_enc.transform(X[self.te_days_in_hosp_cols])
            dih_df = pd.DataFrame(
                dih_arr,
                columns=[c + '_DIH' for c in self.te_days_in_hosp_cols],
                index=X.index
            )
        else:
            dih_df = pd.DataFrame(index=X.index)

        X_final = pd.concat([
            X_out[self.numerical_cols], X_out[self.ordinal_cols], trans_df, dih_df
        ], axis=1)
        return X_final.fillna(0)


def register_custom_classes():
    """
    Registers custom classes in the __main__ module namespace
    so that joblib/pickle can find them during deserialization.

    Must be called BEFORE loading any .joblib model files.
    """
    main_module = sys.modules.get('__main__')
    if main_module is None:
        main_module = sys.modules.get('__mp_main__')

    if main_module and not hasattr(main_module, 'AutomatedFeatureEngineer'):
        main_module.AutomatedFeatureEngineer = AutomatedFeatureEngineer

    # Also register in common module paths pickle might look for
    for module_name in ['__main__', '__mp_main__', 'uvicorn.__main__', 'uvicorn']:
        mod = sys.modules.get(module_name)
        if mod and not hasattr(mod, 'AutomatedFeatureEngineer'):
            mod.AutomatedFeatureEngineer = AutomatedFeatureEngineer
