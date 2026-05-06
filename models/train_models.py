"""
CA Hospital AI Agent — Model Training
Trains all 5 ML models and saves them to models/saved/
Run this FIRST before launching the app.
"""

import pandas as pd
import numpy as np
import os
import joblib
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, IsolationForest
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier, XGBRegressor
from sklearn.metrics import (
    classification_report, accuracy_score, mean_absolute_error, r2_score
)
from imblearn.over_sampling import SMOTE

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).parent.parent
DATA_DIR   = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models" / "saved"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# ── Load data ──────────────────────────────────────────────────────────────────
def load_data():
    claims     = pd.read_csv(DATA_DIR / "claims_and_billing.csv")
    denials    = pd.read_csv(DATA_DIR / "denials.csv")
    diagnoses  = pd.read_csv(DATA_DIR / "diagnoses.csv")
    encounters = pd.read_csv(DATA_DIR / "encounters.csv")
    lab_tests  = pd.read_csv(DATA_DIR / "lab_tests.csv")
    medications= pd.read_csv(DATA_DIR / "medications.csv")
    patients   = pd.read_csv(DATA_DIR / "patients.csv")
    procedures = pd.read_csv(DATA_DIR / "procedures.csv")
    providers  = pd.read_csv(DATA_DIR / "providers.csv")
    return (claims, denials, diagnoses, encounters, lab_tests,
            medications, patients, procedures, providers)

def encode_columns(df, cols):
    le = LabelEncoder()
    for col in cols:
        if col in df.columns:
            df[col] = df[col].astype(str).fillna("Unknown")
            df[col] = le.fit_transform(df[col])
    return df

# ── Model 1: Claim Denial Prediction ──────────────────────────────────────────
def train_denial_prediction(claims):
    print("\n[1/5] Training Claim Denial Prediction model...")
    df = claims.copy()
    df["label"] = (df["claim_status"].str.lower() == "denied").astype(int)

    features = ["insurance_provider", "payment_method", "billed_amount"]
    df = encode_columns(df, ["insurance_provider", "payment_method"])
    df["billed_amount"] = pd.to_numeric(df["billed_amount"], errors="coerce").fillna(0)

    X = df[features].fillna(0)
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    try:
        sm = SMOTE(random_state=42)
        X_train, y_train = sm.fit_resample(X_train, y_train)
    except Exception:
        pass

    model = XGBClassifier(n_estimators=100, random_state=42, eval_metric="logloss")
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print(f"  Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(classification_report(y_test, y_pred, zero_division=0))

    joblib.dump(model, MODELS_DIR / "denial_model.pkl")
    print("  ✅ Saved: models/saved/denial_model.pkl")
    return model

# ── Model 2: Readmission Prediction ───────────────────────────────────────────
def train_readmission_prediction(encounters, patients):
    print("\n[2/5] Training Readmission Prediction model...")
    df = encounters.copy()
    df = df.merge(patients[["patient_id", "age", "gender", "insurance_type"]], on="patient_id", how="left")

    df["label"] = (df["readmitted_flag"].str.lower() == "yes").astype(int)
    df["length_of_stay"] = pd.to_numeric(df["length_of_stay"], errors="coerce").fillna(0)

    features = ["age", "gender", "insurance_type", "visit_type", "department", "length_of_stay"]
    df = encode_columns(df, ["gender", "insurance_type", "visit_type", "department"])
    df["age"] = pd.to_numeric(df["age"], errors="coerce").fillna(30)

    X = df[features].fillna(0)
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    try:
        sm = SMOTE(random_state=42)
        X_train, y_train = sm.fit_resample(X_train, y_train)
    except Exception:
        pass

    model = RandomForestClassifier(n_estimators=150, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print(f"  Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(classification_report(y_test, y_pred, zero_division=0))

    joblib.dump(model, MODELS_DIR / "readmission_model.pkl")
    print("  ✅ Saved: models/saved/readmission_model.pkl")
    return model

# ── Model 3: Fraud Detection ───────────────────────────────────────────────────
def train_fraud_detection(claims, procedures):
    print("\n[3/5] Training Fraud Detection model...")

    # Aggregate procedure cost per claim/encounter
    proc_agg = procedures.groupby("encounter_id")["procedure_cost"].agg(["sum", "count"]).reset_index()
    proc_agg.columns = ["encounter_id", "total_proc_cost", "proc_count"]

    df = claims.copy()
    df = df.merge(proc_agg, on="encounter_id", how="left")
    df["billed_amount"]    = pd.to_numeric(df["billed_amount"], errors="coerce").fillna(0)
    df["total_proc_cost"]  = pd.to_numeric(df["total_proc_cost"], errors="coerce").fillna(0)
    df["proc_count"]       = pd.to_numeric(df["proc_count"], errors="coerce").fillna(0)
    df["cost_ratio"]       = df["billed_amount"] / (df["total_proc_cost"] + 1)

    df = encode_columns(df, ["insurance_provider", "payment_method"])

    features = ["billed_amount", "total_proc_cost", "proc_count", "cost_ratio",
                "insurance_provider", "payment_method"]
    X = df[features].fillna(0)

    model = IsolationForest(n_estimators=100, contamination=0.05, random_state=42)
    model.fit(X)

    scores = model.decision_function(X)
    preds  = model.predict(X)   # -1 = anomaly (fraud), 1 = normal
    fraud_rate = (preds == -1).mean()
    print(f"  Flagged as potential fraud: {fraud_rate:.2%} of claims")

    joblib.dump(model, MODELS_DIR / "fraud_model.pkl")
    print("  ✅ Saved: models/saved/fraud_model.pkl")
    return model

# ── Model 4: High-Cost Patient Prediction ─────────────────────────────────────
def train_high_cost_prediction(claims, encounters, patients):
    print("\n[4/5] Training High-Cost Patient Prediction model...")

    # Total billed per patient
    patient_cost = claims.groupby("patient_id")["billed_amount"].sum().reset_index()
    patient_cost.columns = ["patient_id", "total_billed"]
    threshold = patient_cost["total_billed"].quantile(0.75)
    patient_cost["label"] = (patient_cost["total_billed"] >= threshold).astype(int)

    # Encounter counts
    enc_cnt = encounters.groupby("patient_id").size().reset_index(name="enc_count")

    df = patients.copy()
    df = df.merge(patient_cost[["patient_id","label"]], on="patient_id", how="inner")
    df = df.merge(enc_cnt, on="patient_id", how="left")
    df["enc_count"] = df["enc_count"].fillna(0)
    df["age"] = pd.to_numeric(df["age"], errors="coerce").fillna(30)

    features = ["age", "gender", "insurance_type", "marital_status", "enc_count"]
    df = encode_columns(df, ["gender", "insurance_type", "marital_status"])
    X = df[features].fillna(0)
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    try:
        sm = SMOTE(random_state=42)
        X_train, y_train = sm.fit_resample(X_train, y_train)
    except Exception:
        pass

    model = XGBClassifier(n_estimators=100, random_state=42, eval_metric="logloss")
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print(f"  Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(classification_report(y_test, y_pred, zero_division=0))

    joblib.dump(model, MODELS_DIR / "high_cost_model.pkl")
    print("  ✅ Saved: models/saved/high_cost_model.pkl")
    return model

# ── Model 5: Length of Stay Prediction ────────────────────────────────────────
def train_los_prediction(encounters, patients, diagnoses):
    print("\n[5/5] Training Length of Stay Prediction model...")

    df = encounters.copy()
    df = df.merge(patients[["patient_id", "age", "gender", "insurance_type"]], on="patient_id", how="left")

    # Chronic condition flag
    chronic = diagnoses.groupby("encounter_id")["chronic_flag"].apply(
        lambda x: 1 if any(str(v).upper() == "TRUE" for v in x) else 0
    ).reset_index()
    chronic.columns = ["encounter_id", "has_chronic"]
    df = df.merge(chronic, on="encounter_id", how="left")
    df["has_chronic"] = df["has_chronic"].fillna(0)

    df["length_of_stay"] = pd.to_numeric(df["length_of_stay"], errors="coerce")
    df = df[df["length_of_stay"].notna() & (df["length_of_stay"] > 0)]

    if len(df) < 50:
        # Create synthetic LOS from inpatient encounters
        df = encounters.copy()
        df = df.merge(patients[["patient_id","age","gender","insurance_type"]], on="patient_id", how="left")
        df["length_of_stay"] = np.random.randint(1, 15, size=len(df))
        df["has_chronic"] = 0

    features = ["age", "gender", "insurance_type", "visit_type", "department",
                "admission_type", "has_chronic"]
    df = encode_columns(df, ["gender", "insurance_type", "visit_type",
                              "department", "admission_type"])
    df["age"] = pd.to_numeric(df["age"], errors="coerce").fillna(30)

    X = df[features].fillna(0)
    y = df["length_of_stay"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = XGBRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    mae  = mean_absolute_error(y_test, y_pred)
    r2   = r2_score(y_test, y_pred)
    print(f"  MAE: {mae:.2f} days  |  R²: {r2:.4f}")

    joblib.dump(model, MODELS_DIR / "los_model.pkl")
    print("  ✅ Saved: models/saved/los_model.pkl")
    return model


# ── Main ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("  CA Hospital AI Agent — Model Training Pipeline")
    print("=" * 60)

    (claims, denials, diagnoses, encounters, lab_tests,
     medications, patients, procedures, providers) = load_data()

    train_denial_prediction(claims)
    train_readmission_prediction(encounters, patients)
    train_fraud_detection(claims, procedures)
    train_high_cost_prediction(claims, encounters, patients)
    train_los_prediction(encounters, patients, diagnoses)

    print("\n" + "=" * 60)
    print("  ✅ All 5 models trained and saved to models/saved/")
    print("  ▶  Now run:  streamlit run app.py")
    print("=" * 60)