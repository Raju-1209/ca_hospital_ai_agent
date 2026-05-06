"""
CA Hospital AI Agent — Streamlit App
Run: streamlit run app.py
"""

import os
import sys
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ── Path setup ─────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))
DATA_DIR = BASE_DIR / "data"

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CA Hospital AI Agent",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=JetBrains+Mono:wght@400;600&display=swap');

html, body, [class*="css"] {
    font-family: 'Space Grotesk', sans-serif;
}

.stApp {
    background: #0a0f1e;
    color: #e2e8f0;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: #0d1425 !important;
    border-right: 1px solid #1e2d4a;
}

/* Cards */
.metric-card {
    background: linear-gradient(135deg, #0f1e3d, #1a2a4a);
    border: 1px solid #2a4080;
    border-radius: 12px;
    padding: 20px;
    margin: 8px 0;
}

.result-card {
    background: linear-gradient(135deg, #0a1628, #0f2040);
    border: 1px solid #1e4080;
    border-radius: 16px;
    padding: 24px;
    margin: 16px 0;
}

.risk-high {
    background: linear-gradient(135deg, #2d0a0a, #4a0f0f);
    border-color: #ff4444;
    border-radius: 12px;
    padding: 20px;
}

.risk-low {
    background: linear-gradient(135deg, #0a2d0a, #0f4a1e);
    border-color: #44ff88;
    border-radius: 12px;
    padding: 20px;
}

/* Buttons */
.stButton > button {
    background: linear-gradient(135deg, #1a56db, #0e9de0) !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    font-size: 15px !important;
    padding: 12px 32px !important;
    transition: all 0.2s ease !important;
    width: 100%;
}

.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 25px rgba(26, 86, 219, 0.4) !important;
}

/* Inputs */
.stSelectbox > div, .stNumberInput > div, .stTextInput > div {
    background: #0f1e3d !important;
    border: 1px solid #2a4080 !important;
    border-radius: 8px !important;
    color: #e2e8f0 !important;
}

/* Section header */
.section-title {
    font-size: 28px;
    font-weight: 700;
    color: #60a5fa;
    margin-bottom: 4px;
}

.section-sub {
    font-size: 14px;
    color: #64748b;
    margin-bottom: 24px;
}

/* Nav pill */
.nav-pill {
    display: inline-block;
    background: #1e3a6e;
    border-radius: 24px;
    padding: 6px 16px;
    font-size: 13px;
    font-weight: 600;
    color: #93c5fd;
    margin-right: 8px;
}

/* Chat bubbles */
.chat-user {
    background: #1a3a6e;
    border-radius: 16px 16px 4px 16px;
    padding: 12px 16px;
    margin: 8px 0;
    color: #e2e8f0;
}

.chat-ai {
    background: #0f2040;
    border: 1px solid #2a4080;
    border-radius: 16px 16px 16px 4px;
    padding: 12px 16px;
    margin: 8px 0;
    color: #e2e8f0;
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    background: #0d1425;
    border-radius: 12px;
    padding: 4px;
    gap: 4px;
}
.stTabs [data-baseweb="tab"] {
    background: transparent;
    color: #64748b;
    border-radius: 8px;
    font-weight: 600;
}
.stTabs [aria-selected="true"] {
    background: #1a3a6e !important;
    color: #93c5fd !important;
}

/* Plotly backgrounds */
.js-plotly-plot {
    border-radius: 12px;
    overflow: hidden;
}
</style>
""", unsafe_allow_html=True)


# ── Data loaders (cached) ──────────────────────────────────────────────────────
@st.cache_data
def load_all_data():
    return {
        "claims":      pd.read_csv(DATA_DIR / "claims_and_billing.csv"),
        "denials":     pd.read_csv(DATA_DIR / "denials.csv"),
        "diagnoses":   pd.read_csv(DATA_DIR / "diagnoses.csv"),
        "encounters":  pd.read_csv(DATA_DIR / "encounters.csv"),
        "lab_tests":   pd.read_csv(DATA_DIR / "lab_tests.csv"),
        "medications": pd.read_csv(DATA_DIR / "medications.csv"),
        "patients":    pd.read_csv(DATA_DIR / "patients.csv"),
        "procedures":  pd.read_csv(DATA_DIR / "procedures.csv"),
        "providers":   pd.read_csv(DATA_DIR / "providers.csv"),
    }


def models_available():
    saved = BASE_DIR / "models" / "saved"
    required = ["denial_model", "readmission_model", "fraud_model",
                "high_cost_model", "los_model"]
    return all((saved / f"{m}.pkl").exists() for m in required)


# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding: 16px 0 8px;'>
        <span style='font-size:36px'>🏥</span>
        <div style='font-size:20px; font-weight:700; color:#60a5fa; margin-top:8px;'>CA Hospital</div>
        <div style='font-size:13px; color:#64748b;'>AI Agent Platform</div>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    pages = {
        "📊 Dashboard":              "Dashboard",
        "🚫 Denial Prediction":      "Denial Prediction",
        "🔄 Readmission Prediction": "Readmission Prediction",
        "🔍 Fraud Detection":        "Fraud Detection",
        "💰 High-Cost Patient":      "High-Cost Patient",
        "🛏️ Length of Stay":         "Length of Stay",
        "🤖 Ask AI Agent":           "Ask AI Agent",
    }

    page = st.radio("Navigation", list(pages.keys()), label_visibility="collapsed")
    active = pages[page]

    st.divider()
    # Groq API key loaded securely from .env — never shown in UI
    groq_key = os.getenv("GROQ_API_KEY", "")  # Set this in your .env file

    if not models_available():
        st.error("⚠️ Models not trained yet!\n\nRun:\n```\npython models/train_models.py\n```")
    else:
        st.success("✅ All 5 models loaded")

data = load_all_data()
PLOT_THEME = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(15,30,63,0.6)",
    font=dict(color="#94a3b8", family="Space Grotesk"),
    xaxis=dict(gridcolor="#1e3a6e", linecolor="#2a4080"),
    yaxis=dict(gridcolor="#1e3a6e", linecolor="#2a4080"),
)
COLORS = ["#3b82f6","#10b981","#f59e0b","#ef4444","#8b5cf6","#06b6d4"]


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════
if active == "Dashboard":
    st.markdown('<div class="section-title">📊 Hospital Intelligence Dashboard</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Real-time overview of claims, patients, and clinical data</div>', unsafe_allow_html=True)

    claims    = data["claims"]
    patients  = data["patients"]
    encounters= data["encounters"]
    denials   = data["denials"]

    claims["billed_amount"] = pd.to_numeric(claims["billed_amount"], errors="coerce").fillna(0)
    claims["paid_amount"]   = pd.to_numeric(claims["paid_amount"],   errors="coerce").fillna(0)

    total_billed   = claims["billed_amount"].sum()
    total_paid     = claims["paid_amount"].sum()
    denial_rate    = (claims["claim_status"].str.lower() == "denied").mean() * 100
    total_patients = patients["patient_id"].nunique()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("💵 Total Billed",   f"${total_billed:,.0f}")
    c2.metric("✅ Total Paid",     f"${total_paid:,.0f}")
    c3.metric("🚫 Denial Rate",    f"{denial_rate:.1f}%")
    c4.metric("👥 Total Patients", f"{total_patients:,}")

    st.markdown("---")
    col1, col2 = st.columns(2)

    # Claims by status
    with col1:
        status_cnt = claims["claim_status"].value_counts().reset_index()
        fig = px.pie(status_cnt, names="claim_status", values="count",
                     title="Claim Status Distribution",
                     color_discrete_sequence=COLORS, hole=0.45)
        fig.update_layout(**PLOT_THEME, title_font_color="#93c5fd")
        st.plotly_chart(fig, use_container_width=True)

    # Top denial reasons
    with col2:
        denied = claims[claims["claim_status"].str.lower() == "denied"]
        dr = denied["denial_reason"].value_counts().head(6).reset_index()
        fig = px.bar(dr, x="count", y="denial_reason", orientation="h",
                     title="Top Denial Reasons",
                     color_discrete_sequence=["#ef4444"])
        fig.update_layout(**PLOT_THEME, title_font_color="#93c5fd",
                          yaxis_title="", xaxis_title="Count")
        st.plotly_chart(fig, use_container_width=True)

    col3, col4 = st.columns(2)

    # Billed by insurance provider
    with col3:
        ins = claims.groupby("insurance_provider")["billed_amount"].sum().reset_index()
        fig = px.bar(ins, x="insurance_provider", y="billed_amount",
                     title="Billed Amount by Insurance Provider",
                     color_discrete_sequence=["#3b82f6"])
        fig.update_layout(**PLOT_THEME, title_font_color="#93c5fd",
                          xaxis_title="", yaxis_title="Amount ($)")
        st.plotly_chart(fig, use_container_width=True)

    # Patient age distribution
    with col4:
        patients["age"] = pd.to_numeric(patients["age"], errors="coerce")
        fig = px.histogram(patients, x="age", nbins=20,
                           title="Patient Age Distribution",
                           color_discrete_sequence=["#10b981"])
        fig.update_layout(**PLOT_THEME, title_font_color="#93c5fd",
                          xaxis_title="Age", yaxis_title="Count")
        st.plotly_chart(fig, use_container_width=True)

    # Department visits
    st.markdown("#### 🏥 Visits by Department")
    dept = encounters["department"].value_counts().head(10).reset_index()
    fig = px.bar(dept, x="department", y="count",
                 color_discrete_sequence=COLORS)
    fig.update_layout(**PLOT_THEME, xaxis_title="", yaxis_title="Visits")
    st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: DENIAL PREDICTION
# ══════════════════════════════════════════════════════════════════════════════
elif active == "Denial Prediction":
    st.markdown('<div class="section-title">🚫 Claim Denial Prediction</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Predict whether a claim will be denied before submission</div>', unsafe_allow_html=True)

    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown("#### 📋 Claim Details")
        insurance_provider = st.selectbox("Insurance Provider",
            ["BCBS", "Medicare", "Medicaid", "Aetna", "UHC", "Cigna", "Other"])
        payment_method = st.selectbox("Payment Method",
            ["Insurance", "Self-Pay", "Government", "Other"])
        billed_amount = st.number_input("Billed Amount ($)", min_value=0.0,
                                        max_value=500000.0, value=1500.0, step=50.0)

        predict_btn = st.button("🔮 Predict Denial Risk", use_container_width=True)

    with col2:
        if predict_btn:
            if not models_available():
                st.error("Run `python models/train_models.py` first.")
            else:
                from models.model_utils import predict_denial
                result = predict_denial(insurance_provider, payment_method, billed_amount)

                is_denied = result["prediction"] == "Denied"
                color = "#ff4444" if is_denied else "#44ff88"
                icon  = "🚫" if is_denied else "✅"

                st.markdown(f"""
                <div class="result-card">
                    <div style='font-size:48px; text-align:center'>{icon}</div>
                    <div style='font-size:28px; font-weight:700; text-align:center; color:{color}'>
                        {result['prediction']}
                    </div>
                    <hr style='border-color:#2a4080; margin:16px 0'>
                    <div style='display:flex; justify-content:space-between;'>
                        <span>Denial Probability</span>
                        <span style='color:#ef4444; font-weight:700'>{result['denial_probability']}%</span>
                    </div>
                    <div style='display:flex; justify-content:space-between; margin-top:8px;'>
                        <span>Approval Probability</span>
                        <span style='color:#10b981; font-weight:700'>{result['approval_probability']}%</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)

                # Gauge chart
                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=result["denial_probability"],
                    title={"text": "Denial Risk %", "font": {"color": "#93c5fd"}},
                    gauge={
                        "axis": {"range": [0, 100], "tickcolor": "#64748b"},
                        "bar":  {"color": "#ef4444" if is_denied else "#10b981"},
                        "steps": [
                            {"range": [0,  33], "color": "#0a2d1a"},
                            {"range": [33, 66], "color": "#2d2d0a"},
                            {"range": [66,100], "color": "#2d0a0a"},
                        ],
                        "threshold": {"line": {"color":"white","width":3}, "value":50}
                    },
                    number={"suffix": "%", "font": {"color": "#e2e8f0"}}
                ))
                fig.update_layout(paper_bgcolor="rgba(0,0,0,0)",
                                  font={"color":"#94a3b8"}, height=250)
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("👈 Fill in the claim details and click **Predict Denial Risk**")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: READMISSION PREDICTION
# ══════════════════════════════════════════════════════════════════════════════
elif active == "Readmission Prediction":
    st.markdown('<div class="section-title">🔄 Readmission Prediction</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Predict 30-day readmission risk for a patient encounter</div>', unsafe_allow_html=True)

    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown("#### 👤 Patient & Visit Info")
        age = st.slider("Patient Age", 1, 100, 55)
        gender = st.selectbox("Gender", ["Male", "Female", "Other"])
        insurance_type = st.selectbox("Insurance Type",
            ["BCBS", "Medicare", "Medicaid", "Aetna", "UHC", "Cigna", "Other"])
        visit_type = st.selectbox("Visit Type",
            ["Inpatient", "Outpatient", "Emergency", "Telehealth", "Observation"])
        department = st.selectbox("Department", [
            "Emergency Department","Cardiology","Orthopedics","Neurology",
            "Oncology","Obstetrics & Gynecology","Gastroenterology",
            "Pulmonology","Radiology","General Surgery","Other"])
        length_of_stay = st.number_input("Length of Stay (days)", 0, 365, 3)

        predict_btn = st.button("🔮 Predict Readmission Risk", use_container_width=True)

    with col2:
        if predict_btn:
            if not models_available():
                st.error("Run `python models/train_models.py` first.")
            else:
                from models.model_utils import predict_readmission
                result = predict_readmission(age, gender, insurance_type,
                                             visit_type, department, length_of_stay)
                is_high = result["prediction"] == "High Risk"
                color = "#ff4444" if is_high else "#44ff88"
                icon  = "⚠️" if is_high else "✅"

                st.markdown(f"""
                <div class="result-card">
                    <div style='font-size:48px; text-align:center'>{icon}</div>
                    <div style='font-size:28px; font-weight:700; text-align:center; color:{color}'>
                        {result['prediction']}
                    </div>
                    <hr style='border-color:#2a4080; margin:16px 0'>
                    <div style='display:flex; justify-content:space-between;'>
                        <span>Readmission Probability</span>
                        <span style='color:{color}; font-weight:700'>{result['readmission_probability']}%</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)

                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=result["readmission_probability"],
                    title={"text": "Readmission Risk %", "font": {"color": "#93c5fd"}},
                    gauge={
                        "axis": {"range": [0, 100], "tickcolor":"#64748b"},
                        "bar":  {"color": "#ef4444" if is_high else "#10b981"},
                        "steps": [
                            {"range":[0, 33],"color":"#0a2d1a"},
                            {"range":[33,66],"color":"#2d2d0a"},
                            {"range":[66,100],"color":"#2d0a0a"},
                        ],
                    },
                    number={"suffix":"%","font":{"color":"#e2e8f0"}}
                ))
                fig.update_layout(paper_bgcolor="rgba(0,0,0,0)",
                                  font={"color":"#94a3b8"}, height=250)
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("👈 Fill in patient details and click **Predict Readmission Risk**")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: FRAUD DETECTION
# ══════════════════════════════════════════════════════════════════════════════
elif active == "Fraud Detection":
    st.markdown('<div class="section-title">🔍 Fraud Detection</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Detect anomalous billing patterns that may indicate fraud</div>', unsafe_allow_html=True)

    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown("#### 💳 Claim Billing Details")
        billed_amount    = st.number_input("Billed Amount ($)",    0.0, 500000.0, 2000.0, 100.0)
        total_proc_cost  = st.number_input("Total Procedure Cost ($)", 0.0, 500000.0, 1200.0, 100.0)
        proc_count       = st.number_input("Number of Procedures", 0, 50, 2)
        ins_prov         = st.selectbox("Insurance Provider",
            ["BCBS","Medicare","Medicaid","Aetna","UHC","Cigna","Other"])
        pay_method       = st.selectbox("Payment Method",
            ["Insurance","Self-Pay","Government","Other"])

        predict_btn = st.button("🔍 Detect Fraud Risk", use_container_width=True)

    with col2:
        if predict_btn:
            if not models_available():
                st.error("Run `python models/train_models.py` first.")
            else:
                from models.model_utils import predict_fraud
                result = predict_fraud(billed_amount, total_proc_cost, proc_count,
                                       ins_prov, pay_method)
                is_fraud = result["is_fraud"]
                color = "#ff4444" if is_fraud else "#44ff88"

                st.markdown(f"""
                <div class="result-card">
                    <div style='font-size:36px; text-align:center'>{result['prediction']}</div>
                    <hr style='border-color:#2a4080; margin:16px 0'>
                    <div style='display:flex; justify-content:space-between;'>
                        <span>Fraud Risk Score</span>
                        <span style='color:{color}; font-weight:700; font-size:24px'>
                            {result['fraud_risk_score']}/100
                        </span>
                    </div>
                    <div style='margin-top:12px; font-size:13px; color:#64748b'>
                        {"🚨 This claim has unusual billing patterns. Flag for manual review." if is_fraud
                         else "✅ Billing patterns appear consistent with standard claims."}
                    </div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("👈 Enter billing details and click **Detect Fraud Risk**")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: HIGH-COST PATIENT
# ══════════════════════════════════════════════════════════════════════════════
elif active == "High-Cost Patient":
    st.markdown('<div class="section-title">💰 High-Cost Patient Prediction</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Identify patients likely to incur high healthcare costs</div>', unsafe_allow_html=True)

    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown("#### 👤 Patient Profile")
        age            = st.slider("Patient Age", 1, 100, 45)
        gender         = st.selectbox("Gender", ["Male","Female","Other"])
        insurance_type = st.selectbox("Insurance Type",
            ["BCBS","Medicare","Medicaid","Aetna","UHC","Cigna","Other"])
        marital_status = st.selectbox("Marital Status",
            ["Single","Married","Divorced","Widowed","Other"])
        enc_count      = st.number_input("Number of Past Encounters", 0, 200, 5)

        predict_btn = st.button("💰 Predict Cost Category", use_container_width=True)

    with col2:
        if predict_btn:
            if not models_available():
                st.error("Run `python models/train_models.py` first.")
            else:
                from models.model_utils import predict_high_cost
                result = predict_high_cost(age, gender, insurance_type,
                                           marital_status, enc_count)
                is_high = result["prediction"] == "High Cost"
                color = "#f59e0b" if is_high else "#10b981"
                icon  = "💸" if is_high else "✅"

                st.markdown(f"""
                <div class="result-card">
                    <div style='font-size:48px; text-align:center'>{icon}</div>
                    <div style='font-size:28px; font-weight:700; text-align:center; color:{color}'>
                        {result['prediction']}
                    </div>
                    <hr style='border-color:#2a4080; margin:16px 0'>
                    <div style='display:flex; justify-content:space-between;'>
                        <span>High Cost Probability</span>
                        <span style='color:{color}; font-weight:700'>{result['high_cost_probability']}%</span>
                    </div>
                    <div style='margin-top:12px; font-size:13px; color:#64748b'>
                        {"⚠️ Consider proactive care management for this patient." if is_high
                         else "Patient falls within standard cost expectations."}
                    </div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("👈 Fill in patient profile and click **Predict Cost Category**")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: LENGTH OF STAY
# ══════════════════════════════════════════════════════════════════════════════
elif active == "Length of Stay":
    st.markdown('<div class="section-title">🛏️ Length of Stay Prediction</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Estimate how many days a patient will stay admitted</div>', unsafe_allow_html=True)

    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown("#### 🏥 Admission Details")
        age            = st.slider("Patient Age", 1, 100, 50)
        gender         = st.selectbox("Gender", ["Male","Female","Other"])
        insurance_type = st.selectbox("Insurance Type",
            ["BCBS","Medicare","Medicaid","Aetna","UHC","Cigna","Other"])
        visit_type     = st.selectbox("Visit Type",
            ["Inpatient","Outpatient","Emergency","Telehealth","Observation"])
        department     = st.selectbox("Department", [
            "Emergency Department","Cardiology","Orthopedics","Neurology",
            "Oncology","Obstetrics & Gynecology","Gastroenterology",
            "Pulmonology","Radiology","General Surgery","Other"])
        admission_type = st.selectbox("Admission Type",
            ["Emergency","Elective","Urgent","Newborn","Other"])
        has_chronic    = st.checkbox("Patient has chronic condition(s)")

        predict_btn = st.button("🛏️ Predict Length of Stay", use_container_width=True)

    with col2:
        if predict_btn:
            if not models_available():
                st.error("Run `python models/train_models.py` first.")
            else:
                from models.model_utils import predict_los
                result = predict_los(age, gender, insurance_type, visit_type,
                                     department, admission_type, int(has_chronic))
                days = result["predicted_los_days"]

                st.markdown(f"""
                <div class="result-card" style='text-align:center'>
                    <div style='font-size:64px'>🛏️</div>
                    <div style='font-size:52px; font-weight:700; color:#60a5fa'>{days}</div>
                    <div style='font-size:20px; color:#94a3b8'>Predicted Days</div>
                    <hr style='border-color:#2a4080; margin:16px 0'>
                    <div style='font-size:13px; color:#64748b'>
                        Based on patient demographics, visit type, and clinical factors.
                    </div>
                </div>
                """, unsafe_allow_html=True)

                # Visual bar
                fig = go.Figure(go.Bar(
                    x=["Predicted LOS"],
                    y=[days],
                    marker_color="#3b82f6",
                    text=[f"{days} days"],
                    textposition="outside",
                    textfont={"color":"#93c5fd","size":16}
                ))
                fig.update_layout(**PLOT_THEME, yaxis_title="Days",
                                  yaxis_range=[0, max(20, days+3)],
                                  height=300)
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("👈 Fill in admission details and click **Predict Length of Stay**")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: ASK AI AGENT
# ══════════════════════════════════════════════════════════════════════════════
elif active == "Ask AI Agent":
    st.markdown('<div class="section-title">🤖 Ask AI Agent</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Ask any question about patients, claims, diagnoses, or billing</div>', unsafe_allow_html=True)

    # Example questions
    st.markdown("#### 💡 Example Questions")
    examples = [
        "Why was claim CLM000001 denied?",
        "What are the top denial reasons?",
        "Which patients have the highest billed amounts?",
        "What medications were most commonly prescribed?",
        "Show abnormal lab results summary",
        "What is the readmission rate by department?",
    ]
    cols = st.columns(3)
    for i, ex in enumerate(examples):
        if cols[i % 3].button(ex, key=f"ex_{i}", use_container_width=True):
            st.session_state["question"] = ex

    st.markdown("---")

    # Chat history
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    question = st.text_input(
        "Ask a question about your hospital data:",
        value=st.session_state.get("question", ""),
        placeholder="e.g. Why was claim CLM000001 denied?",
        key="question_input"
    )

    ask_btn = st.button("🚀 Ask AI Agent", use_container_width=False)

    if ask_btn and question.strip():
        if not groq_key:
            st.error("⚠️ Groq API key not found. Please add GROQ_API_KEY=your_key to your .env file.")
        else:
            with st.spinner("🤖 Searching knowledge base and generating answer..."):
                try:
                    from rag.rag_engine import ask_ai
                    answer = ask_ai(question, groq_key)
                    st.session_state.chat_history.append({
                        "q": question, "a": answer
                    })
                    st.session_state["question"] = ""
                except Exception as e:
                    st.error(f"Error: {e}")

    # Display chat history
    if st.session_state.chat_history:
        st.markdown("#### 💬 Conversation")
        for turn in reversed(st.session_state.chat_history):
            st.markdown(f"""
            <div class="chat-user">
                <strong>🧑 You:</strong><br>{turn['q']}
            </div>
            <div class="chat-ai">
                <strong>🤖 AI Agent:</strong><br>{turn['a']}
            </div>
            """, unsafe_allow_html=True)

        if st.button("🗑️ Clear Chat History"):
            st.session_state.chat_history = []
            st.rerun()
    else:
        if not ask_btn:
            st.info("Ask a question above to get started. The AI agent has access to all 9 hospital datasets.")