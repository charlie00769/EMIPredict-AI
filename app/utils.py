"""
Shared utilities for EMIPredict AI Streamlit app.
Loads trained XGBoost models + encoders and reproduces the exact
preprocessing pipeline used during training in Colab.
"""

import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st

MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")

# Column order must exactly match training: num_cols + cat_cols
CAT_COLS = ['gender', 'marital_status', 'education', 'employment_type',
            'company_type', 'house_type', 'existing_loans', 'emi_scenario']

NUM_COLS = ['age', 'monthly_salary', 'years_of_employment', 'monthly_rent',
            'family_size', 'dependents', 'school_fees', 'college_fees',
            'travel_expenses', 'groceries_utilities', 'other_monthly_expenses',
            'current_emi_amount', 'credit_score', 'bank_balance', 'emergency_fund',
            'requested_amount', 'requested_tenure',
            'total_monthly_expenses', 'debt_to_income', 'emi_burden_ratio',
            'expense_ratio', 'savings_ratio', 'loan_to_income',
            'financial_buffer', 'dependents_per_income']

FEATURE_COLS = NUM_COLS + CAT_COLS


@st.cache_resource
def load_artifacts():
    """Load all trained models and encoders once, cached across reruns."""
    clf = joblib.load(os.path.join(MODELS_DIR, "best_classifier.pkl"))
    reg = joblib.load(os.path.join(MODELS_DIR, "best_regressor.pkl"))
    encoders = joblib.load(os.path.join(MODELS_DIR, "label_encoders.pkl"))
    target_encoder = joblib.load(os.path.join(MODELS_DIR, "target_encoder.pkl"))
    return clf, reg, encoders, target_encoder


def engineer_features(raw: dict) -> dict:
    """Reproduce the exact ratio features created in Colab Cell 3."""
    total_expenses = (
        raw['monthly_rent'] + raw['school_fees'] + raw['college_fees'] +
        raw['travel_expenses'] + raw['groceries_utilities'] + raw['other_monthly_expenses']
    )
    salary = raw['monthly_salary'] if raw['monthly_salary'] != 0 else np.nan

    out = dict(raw)
    out['total_monthly_expenses'] = total_expenses
    out['debt_to_income'] = (raw['current_emi_amount'] + total_expenses) / salary
    out['emi_burden_ratio'] = raw['current_emi_amount'] / salary
    out['expense_ratio'] = total_expenses / salary
    out['savings_ratio'] = raw['emergency_fund'] / salary
    out['loan_to_income'] = raw['requested_amount'] / (raw['monthly_salary'] * 12 or np.nan)
    out['financial_buffer'] = raw['monthly_salary'] - raw['current_emi_amount'] - total_expenses
    out['dependents_per_income'] = raw['dependents'] / ((raw['monthly_salary'] / 10000) or np.nan)

    for k in ['debt_to_income', 'emi_burden_ratio', 'expense_ratio', 'savings_ratio',
              'loan_to_income', 'dependents_per_income']:
        if out[k] is None or (isinstance(out[k], float) and (np.isnan(out[k]) or np.isinf(out[k]))):
            out[k] = 0.0
    return out


def encode_categoricals(row: dict, encoders: dict) -> dict:
    """Label-encode categorical fields; unseen values fall back to the first known class."""
    row = dict(row)
    for col in CAT_COLS:
        le = encoders[col]
        val = row[col]
        if val in le.classes_:
            row[col] = int(le.transform([val])[0])
        else:
            row[col] = 0  # unseen category fallback
    return row


def build_feature_row(raw_input: dict) -> pd.DataFrame:
    """Take raw form input -> engineered + encoded single-row DataFrame in training column order."""
    featured = engineer_features(raw_input)
    encoded = encode_categoricals(featured, load_artifacts()[2])
    row = {c: encoded[c] for c in FEATURE_COLS}
    return pd.DataFrame([row])[FEATURE_COLS]


def predict_eligibility(raw_input: dict):
    clf, _, _, target_encoder = load_artifacts()
    X = build_feature_row(raw_input)
    pred_idx = clf.predict(X)[0]
    proba = clf.predict_proba(X)[0]
    label = target_encoder.inverse_transform([pred_idx])[0]
    proba_dict = dict(zip(target_encoder.classes_, proba))
    return label, proba_dict


def predict_max_emi(raw_input: dict):
    _, reg, _, _ = load_artifacts()
    X = build_feature_row(raw_input)
    return float(reg.predict(X)[0])


# ---- Dropdown options (verified against actual training data unique values) ----
GENDER_OPTIONS = ['Male', 'Female']
MARITAL_OPTIONS = ['Single', 'Married']
EDUCATION_OPTIONS = ['High School', 'Graduate', 'Post Graduate', 'Professional']
EMPLOYMENT_OPTIONS = ['Government', 'Private', 'Self-employed']
COMPANY_OPTIONS = ['MNC', 'Large Indian', 'Mid-size', 'Small', 'Startup']
HOUSE_OPTIONS = ['Own', 'Rented', 'Family']
EXISTING_LOAN_OPTIONS = ['Yes', 'No']
SCENARIO_OPTIONS = ['E-commerce Shopping EMI', 'Education EMI', 'Home Appliances EMI',
                     'Personal Loan EMI', 'Vehicle EMI']
