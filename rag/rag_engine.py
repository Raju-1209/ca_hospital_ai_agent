"""
rag_engine.py — Lightweight RAG using TF-IDF + Groq LLM
Fixed: NaN cleaning, better chunk text, richer context retrieval
"""

import os
import json
import pickle
import pandas as pd
import numpy as np
from pathlib import Path
from groq import Groq
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

BASE_DIR    = Path(__file__).resolve().parent.parent
DATA_DIR    = BASE_DIR / "data"
CHUNKS_PATH = BASE_DIR / "rag" / "chunks.json"
TFIDF_PATH  = BASE_DIR / "rag" / "tfidf.pkl"

_vectorizer   = None
_tfidf_matrix = None
_chunks       = None


def clean(val, default="N/A"):
    """Convert any value to clean string, replacing nan/None."""
    if val is None:
        return default
    s = str(val).strip()
    if s.lower() in ("nan", "none", "", "nat"):
        return default
    return s


def build_knowledge_base():
    print("Building RAG knowledge base with TF-IDF...")
    chunks = []

    # ── Claims & Billing ──────────────────────────────────────────────────────
    df = pd.read_csv(DATA_DIR / "claims_and_billing.csv")
    # Pre-compute denial rate by insurance for aggregated chunks
    denied_df = df[df["claim_status"].str.lower() == "denied"]
    denial_by_ins = denied_df.groupby("insurance_provider").size()
    total_by_ins  = df.groupby("insurance_provider").size()
    denial_rate   = (denial_by_ins / total_by_ins * 100).fillna(0).round(1)

    # Add aggregated insurance denial rate chunks
    for ins, rate in denial_rate.items():
        chunks.append(
            f"Insurance denial rate: {ins} has a denial rate of {rate}% "
            f"({denial_by_ins.get(ins,0)} denied out of {total_by_ins.get(ins,0)} claims)."
        )

    # Add denial reason aggregated chunks
    denial_reasons = df[df["denial_reason"].notna() & (df["denial_reason"].str.lower() != "nan")]
    reason_counts  = denial_reasons["denial_reason"].value_counts()
    for reason, count in reason_counts.items():
        chunks.append(
            f"Top denial reason: '{reason}' occurred {count} times across all claims."
        )

    # Individual claim chunks
    for _, row in df.iterrows():
        denial_reason = clean(row.get("denial_reason"), "None")
        chunks.append(
            f"Claim {clean(row.get('claim_id'))}: "
            f"Patient {clean(row.get('patient_id'))} "
            f"billed ${clean(row.get('billed_amount'),'0')} "
            f"to {clean(row.get('insurance_provider'))}. "
            f"Payment method: {clean(row.get('payment_method'))}. "
            f"Status: {clean(row.get('claim_status'))}. "
            f"Denial reason: {denial_reason}. "
            f"Amount paid: ${clean(row.get('paid_amount'),'0')}. "
            f"Encounter: {clean(row.get('encounter_id'))}."
        )

    # ── Denials ───────────────────────────────────────────────────────────────
    df = pd.read_csv(DATA_DIR / "denials.csv")
    # Aggregated denial code chunks
    code_counts = df["denial_reason_description"].value_counts()
    for desc, count in code_counts.items():
        chunks.append(
            f"Denial reason summary: '{desc}' is responsible for {count} denials."
        )
    for _, row in df.iterrows():
        chunks.append(
            f"Denial record: Claim {clean(row.get('claim_id'))} "
            f"was denied with code {clean(row.get('denial_reason_code'))} "
            f"— reason: {clean(row.get('denial_reason_description'))}. "
            f"Denied amount: ${clean(row.get('denied_amount'),'0')}. "
            f"Denial date: {clean(row.get('denial_date'))}. "
            f"Appeal filed: {clean(row.get('appeal_filed'))}. "
            f"Appeal status: {clean(row.get('appeal_status'))}. "
            f"Final outcome: {clean(row.get('final_outcome'),'Pending')}."
        )

    # ── Patients ──────────────────────────────────────────────────────────────
    df = pd.read_csv(DATA_DIR / "patients.csv")
    for _, row in df.iterrows():
        chunks.append(
            f"Patient {clean(row.get('patient_id'))} "
            f"{clean(row.get('first_name'))} {clean(row.get('last_name'))}: "
            f"Age {clean(row.get('age'))}, "
            f"Gender {clean(row.get('gender'))}, "
            f"Ethnicity {clean(row.get('ethnicity'))}, "
            f"Insurance {clean(row.get('insurance_type'))}, "
            f"Marital status {clean(row.get('marital_status'))}. "
            f"City: {clean(row.get('city'))}, State: {clean(row.get('state'))}."
        )

    # ── Encounters ────────────────────────────────────────────────────────────
    df = pd.read_csv(DATA_DIR / "encounters.csv")
    # Readmission rate by department aggregated
    dept_total    = df.groupby("department").size()
    dept_readmit  = df[df["readmitted_flag"].str.lower() == "yes"].groupby("department").size()
    dept_rate     = (dept_readmit / dept_total * 100).fillna(0).round(1)
    for dept, rate in dept_rate.items():
        chunks.append(
            f"Readmission rate by department: {dept} has a readmission rate of {rate}% "
            f"({dept_readmit.get(dept,0)} readmitted out of {dept_total.get(dept,0)} encounters)."
        )
    for _, row in df.iterrows():
        chunks.append(
            f"Encounter {clean(row.get('encounter_id'))} "
            f"for patient {clean(row.get('patient_id'))} "
            f"on {clean(row.get('visit_date'))}. "
            f"Visit type: {clean(row.get('visit_type'))}. "
            f"Department: {clean(row.get('department'))}. "
            f"Reason: {clean(row.get('reason_for_visit'))}. "
            f"Diagnosis code: {clean(row.get('diagnosis_code'))}. "
            f"Admission type: {clean(row.get('admission_type'))}. "
            f"Discharge date: {clean(row.get('discharge_date'))}. "
            f"Length of stay: {clean(row.get('length_of_stay'),'0')} days. "
            f"Readmitted: {clean(row.get('readmitted_flag'))}."
        )

    # ── Diagnoses ─────────────────────────────────────────────────────────────
    df = pd.read_csv(DATA_DIR / "diagnoses.csv")
    # Top diagnoses aggregated
    top_diag = df["diagnosis_description"].value_counts().head(20)
    for diag, count in top_diag.items():
        chunks.append(
            f"Top diagnosis: '{diag}' appears {count} times across all encounters."
        )
    for _, row in df.iterrows():
        chunks.append(
            f"Diagnosis in encounter {clean(row.get('encounter_id'))}: "
            f"Code {clean(row.get('diagnosis_code'))} — {clean(row.get('diagnosis_description'))}. "
            f"Primary diagnosis: {clean(row.get('primary_flag'))}. "
            f"Chronic condition: {clean(row.get('chronic_flag'))}."
        )

    # ── Lab Tests ─────────────────────────────────────────────────────────────
    df = pd.read_csv(DATA_DIR / "lab_tests.csv")
    # Abnormal lab summary
    abnormal = df[df["test_result"].str.lower() == "abnormal"]
    abnormal_counts = abnormal["test_name"].value_counts()
    for test, count in abnormal_counts.items():
        chunks.append(
            f"Abnormal lab result summary: '{test}' returned abnormal results {count} times."
        )
    for _, row in df.iterrows():
        chunks.append(
            f"Lab test in encounter {clean(row.get('encounter_id'))}: "
            f"{clean(row.get('test_name'))} "
            f"result is {clean(row.get('test_result'))}. "
            f"Units: {clean(row.get('units'))}. "
            f"Normal range: {clean(row.get('normal_range'))}. "
            f"Test date: {clean(row.get('test_date'))}. "
            f"Status: {clean(row.get('status'))}."
        )

    # ── Medications ───────────────────────────────────────────────────────────
    df = pd.read_csv(DATA_DIR / "medications.csv")
    # Most prescribed drugs
    top_drugs = df["drug_name"].value_counts().head(20)
    for drug, count in top_drugs.items():
        chunks.append(
            f"Most prescribed medication: '{drug}' was prescribed {count} times."
        )
    for _, row in df.iterrows():
        chunks.append(
            f"Medication in encounter {clean(row.get('encounter_id'))}: "
            f"{clean(row.get('drug_name'))} "
            f"{clean(row.get('dosage'))} via {clean(row.get('route'))}, "
            f"{clean(row.get('frequency'))} for {clean(row.get('duration'))}. "
            f"Prescribed on {clean(row.get('prescribed_date'))}. "
            f"Cost: ${clean(row.get('cost'),'0')}."
        )

    # ── Procedures ────────────────────────────────────────────────────────────
    df = pd.read_csv(DATA_DIR / "procedures.csv")
    for _, row in df.iterrows():
        chunks.append(
            f"Procedure in encounter {clean(row.get('encounter_id'))}: "
            f"{clean(row.get('procedure_description'))} "
            f"(code {clean(row.get('procedure_code'))}). "
            f"Date: {clean(row.get('procedure_date'))}. "
            f"Cost: ${clean(row.get('procedure_cost'),'0')}."
        )

    # ── Providers ─────────────────────────────────────────────────────────────
    df = pd.read_csv(DATA_DIR / "providers.csv")
    for _, row in df.iterrows():
        chunks.append(
            f"Provider {clean(row.get('provider_id'))}: "
            f"Dr. {clean(row.get('name'))}. "
            f"Department: {clean(row.get('department'))}. "
            f"Specialty: {clean(row.get('specialty'))}. "
            f"Experience: {clean(row.get('years_experience'))} years. "
            f"Location: {clean(row.get('location'))}. "
            f"In-house: {clean(row.get('inhouse'))}."
        )

    print(f"  Total chunks built: {len(chunks)}")

    # Fit TF-IDF
    vectorizer = TfidfVectorizer(
        max_features=30000,
        ngram_range=(1, 2),
        stop_words="english",
        sublinear_tf=True,
    )
    tfidf_matrix = vectorizer.fit_transform(chunks)

    # Save
    with open(CHUNKS_PATH, "w", encoding="utf-8") as f:
        json.dump(chunks, f)
    with open(TFIDF_PATH, "wb") as f:
        pickle.dump({"vectorizer": vectorizer, "matrix": tfidf_matrix}, f)

    print("  TF-IDF index saved successfully.")
    return vectorizer, tfidf_matrix, chunks


def load_rag():
    global _vectorizer, _tfidf_matrix, _chunks
    if _vectorizer is None:
        if TFIDF_PATH.exists() and CHUNKS_PATH.exists():
            with open(TFIDF_PATH, "rb") as f:
                data = pickle.load(f)
            _vectorizer   = data["vectorizer"]
            _tfidf_matrix = data["matrix"]
            with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
                _chunks = json.load(f)
        else:
            _vectorizer, _tfidf_matrix, _chunks = build_knowledge_base()
    return _vectorizer, _tfidf_matrix, _chunks


def retrieve_context(query: str, top_k: int = 12) -> list:
    vectorizer, tfidf_matrix, chunks = load_rag()
    q_vec   = vectorizer.transform([query])
    scores  = cosine_similarity(q_vec, tfidf_matrix).flatten()
    top_idx = scores.argsort()[::-1][:top_k]
    return [chunks[i] for i in top_idx if scores[i] > 0]


def ask_ai(question: str, api_key: str) -> str:
    context_chunks = retrieve_context(question, top_k=12)

    if not context_chunks:
        context = "No directly matching records found in the hospital database."
    else:
        context = "\n".join(f"- {c}" for c in context_chunks)

    system_prompt = (
        "You are CA Hospital AI Agent — an expert medical data analyst.\n"
        "You have access to real hospital records covering: patients, encounters, "
        "claims, diagnoses, lab tests, medications, procedures, providers, and denials.\n\n"
        "STRICT RULES:\n"
        "1. Answer ONLY from the context below — never say 'context does not include' "
        "   if relevant data IS present.\n"
        "2. For aggregated questions (top denial reasons, rates, counts), "
        "   look for summary chunks that start with 'Top denial reason:', "
        "   'Insurance denial rate:', 'Readmission rate:', etc.\n"
        "3. Be specific: include IDs, amounts, dates, percentages when available.\n"
        "4. Format your answer clearly with bullet points or numbered lists when listing items.\n"
        "5. Always give a direct answer — never say data is unavailable if the context has it."
    )

    user_message = (
        f"Hospital database context:\n{context}\n\n"
        f"Question: {question}\n\n"
        "Give a clear, specific, data-driven answer."
    )

    client   = Groq(api_key=api_key)
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_message},
        ],
        temperature=0.1,
        max_tokens=1024,
    )
    return response.choices[0].message.content


def rebuild_index():
    global _vectorizer, _tfidf_matrix, _chunks
    for path in [CHUNKS_PATH, TFIDF_PATH]:
        if path.exists():
            path.unlink()
    _vectorizer = _tfidf_matrix = _chunks = None
    return build_knowledge_base()


if __name__ == "__main__":
    print("=" * 50)
    print("  Rebuilding RAG Knowledge Base (cleaned)")
    print("=" * 50)
    build_knowledge_base()
    print("\nDone! Run: streamlit run app.py")