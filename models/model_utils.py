"""
model_utils.py — Load saved models and run predictions
"""

import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import LabelEncoder

MODELS_DIR = Path(__file__).parent / "saved"

# ── Label encoders (refit from known categories) ────────────────────────────
INSURANCE_PROVIDERS = ["BCBS", "Medicare", "Medicaid", "Aetna", "UHC", "Cigna", "Other"]
PAYMENT_METHODS     = ["Insurance", "Self-Pay", "Government", "Other"]
GENDERS             = ["Male", "Female", "Other"]
INSURANCE_TYPES     = ["BCBS", "Medicare", "Medicaid", "Aetna", "UHC", "Cigna", "Other"]
VISIT_TYPES         = ["Inpatient", "Outpatient", "Emergency", "Telehealth", "Observation"]
DEPARTMENTS         = ["Emergency Department", "Cardiology", "Orthopedics", "Neurology",
                       "Oncology", "Obstetrics & Gynecology", "Gastroenterology",
                       "Pulmonology", "Radiology", "General Surgery", "Other"]
MARITAL_STATUSES    = ["Single", "Married", "Divorced", "Widowed", "Other"]
ADMISSION_TYPES     = ["Emergency", "Elective", "Urgent", "Newborn", "Other", ""]


def _le(categories):
    le = LabelEncoder()
    le.fit(categories)
    return le

def safe_transform(le, val):
    val = str(val).strip()
    if val in le.classes_:
        return le.transform([val])[0]
    return 0


# ── Load model helper ────────────────────────────────────────────────────────
def load_model(name):
    path = MODELS_DIR / f"{name}.pkl"
    if not path.exists():
        raise FileNotFoundError(f"Model '{name}' not found. Run train_models.py first.")
    return joblib.load(path)


# ── 1. Denial Prediction ─────────────────────────────────────────────────────
def predict_denial(insurance_provider, payment_method, billed_amount):
    model = load_model("denial_model")

    le_ins = _le(INSURANCE_PROVIDERS)
    le_pay = _le(PAYMENT_METHODS)

    X = np.array([[
        safe_transform(le_ins, insurance_provider),
        safe_transform(le_pay, payment_method),
        float(billed_amount)
    ]])
    proba = model.predict_proba(X)[0]
    pred  = model.predict(X)[0]
    return {
        "prediction": "Denied" if pred == 1 else "Approved",
        "denial_probability": round(proba[1] * 100, 1),
        "approval_probability": round(proba[0] * 100, 1),
    }


# ── 2. Readmission Prediction ─────────────────────────────────────────────────
def predict_readmission(age, gender, insurance_type, visit_type, department, length_of_stay):
    model = load_model("readmission_model")

    le_gen = _le(GENDERS)
    le_ins = _le(INSURANCE_TYPES)
    le_vis = _le(VISIT_TYPES)
    le_dep = _le(DEPARTMENTS)

    X = np.array([[
        float(age),
        safe_transform(le_gen, gender),
        safe_transform(le_ins, insurance_type),
        safe_transform(le_vis, visit_type),
        safe_transform(le_dep, department),
        float(length_of_stay),
    ]])
    proba = model.predict_proba(X)[0]
    pred  = model.predict(X)[0]
    return {
        "prediction": "High Risk" if pred == 1 else "Low Risk",
        "readmission_probability": round(proba[1] * 100, 1),
    }


# ── 3. Fraud Detection ────────────────────────────────────────────────────────
def predict_fraud(billed_amount, total_proc_cost, proc_count, insurance_provider, payment_method):
    model = load_model("fraud_model")

    le_ins = _le(INSURANCE_PROVIDERS)
    le_pay = _le(PAYMENT_METHODS)

    cost_ratio = float(billed_amount) / (float(total_proc_cost) + 1)
    X = np.array([[
        float(billed_amount),
        float(total_proc_cost),
        float(proc_count),
        cost_ratio,
        safe_transform(le_ins, insurance_provider),
        safe_transform(le_pay, payment_method),
    ]])
    score = model.decision_function(X)[0]
    pred  = model.predict(X)[0]
    # Normalize score to 0-100 risk (lower decision_function = more anomalous)
    fraud_risk = round(max(0, min(100, (1 - (score + 0.5)) * 100)), 1)
    return {
        "prediction": "⚠️ Suspicious" if pred == -1 else "✅ Normal",
        "fraud_risk_score": fraud_risk,
        "is_fraud": pred == -1,
    }


# ── 4. High-Cost Patient Prediction ──────────────────────────────────────────
def predict_high_cost(age, gender, insurance_type, marital_status, enc_count):
    model = load_model("high_cost_model")

    le_gen = _le(GENDERS)
    le_ins = _le(INSURANCE_TYPES)
    le_mar = _le(MARITAL_STATUSES)

    X = np.array([[
        float(age),
        safe_transform(le_gen, gender),
        safe_transform(le_ins, insurance_type),
        safe_transform(le_mar, marital_status),
        float(enc_count),
    ]])
    proba = model.predict_proba(X)[0]
    pred  = model.predict(X)[0]
    return {
        "prediction": "High Cost" if pred == 1 else "Standard Cost",
        "high_cost_probability": round(proba[1] * 100, 1),
    }


# ── 5. Length of Stay Prediction ─────────────────────────────────────────────
def predict_los(age, gender, insurance_type, visit_type, department, admission_type, has_chronic):
    model = load_model("los_model")

    le_gen = _le(GENDERS)
    le_ins = _le(INSURANCE_TYPES)
    le_vis = _le(VISIT_TYPES)
    le_dep = _le(DEPARTMENTS)
    le_adm = _le(ADMISSION_TYPES)

    X = np.array([[
        float(age),
        safe_transform(le_gen, gender),
        safe_transform(le_ins, insurance_type),
        safe_transform(le_vis, visit_type),
        safe_transform(le_dep, department),
        safe_transform(le_adm, admission_type),
        int(has_chronic),
    ]])
    days = model.predict(X)[0]
    return {
        "predicted_los_days": round(max(1, days), 1),
    }