"""
text_to_sql.py — Text-to-SQL Engine
User asks a question → Groq generates SQL → Execute on data → Perfect answer
Works with both PostgreSQL (live DB) and CSV (pandas fallback)
"""

import os
import re
import pandas as pd
from pathlib import Path
from groq import Groq

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"

# ── Database Schema (tells the LLM what tables/columns exist) ─────────────────
DB_SCHEMA = """
DATABASE SCHEMA — CA Hospital AI Agent
=======================================
IMPORTANT: Use EXACT column names as listed. Do NOT invent column names.

TABLE: claims_and_billing
  Columns: claim_id, patient_id, encounter_id, insurance_provider,
           payment_method, billed_amount, paid_amount, claim_status,
           denial_reason, claim_date

TABLE: denials
  Columns: denial_id, claim_id, denial_reason_code,
           denial_reason_description, denied_amount, denial_date,
           appeal_filed, appeal_status, final_outcome

TABLE: patients
  Columns: patient_id, first_name, last_name, dob, age, gender,
           ethnicity, insurance_type, marital_status, address,
           city, state, zip, phone, email, registration_date

TABLE: encounters
  Columns: encounter_id, patient_id, provider_id, visit_date,
           visit_type, department, reason_for_visit, diagnosis_code,
           admission_type, discharge_date, length_of_stay, status,
           readmitted_flag

TABLE: diagnoses
  Columns: diagnosis_id, encounter_id, diagnosis_code,
           diagnosis_description, primary_flag, chronic_flag

TABLE: lab_tests
  Columns: lab_test_id, encounter_id, test_name, test_result,
           units, normal_range, test_date, status

TABLE: medications
  Columns: medication_id, encounter_id, drug_name, dosage, route,
           frequency, duration, prescribed_date, cost

TABLE: procedures
  Columns: procedure_id, encounter_id, procedure_code,
           procedure_description, procedure_date, provider_id,
           procedure_cost

TABLE: providers
  Columns: provider_id, name, department, specialty, npi,
           inhouse, location, years_experience, contact_info, email

KEY RELATIONSHIPS:
  encounters.patient_id   → patients.patient_id
  encounters.provider_id  → providers.provider_id
  encounters.encounter_id → claims_and_billing.encounter_id
  encounters.encounter_id → diagnoses.encounter_id
  encounters.encounter_id → lab_tests.encounter_id
  encounters.encounter_id → medications.encounter_id
  encounters.encounter_id → procedures.encounter_id
  procedures.provider_id  → providers.provider_id
  claims_and_billing.claim_id → denials.claim_id

EXAMPLES OF CORRECT QUERIES:
-- Most repeated procedure:
SELECT procedure_description, COUNT(*) as count
FROM procedures
GROUP BY procedure_description
ORDER BY count DESC LIMIT 10;

-- Doctor who handled most patients:
SELECT p.name, COUNT(DISTINCT e.patient_id) as patient_count
FROM encounters e
JOIN providers p ON e.provider_id = p.provider_id
GROUP BY p.name
ORDER BY patient_count DESC LIMIT 10;

-- Top denial reasons:
SELECT denial_reason_description, COUNT(*) as count
FROM denials
GROUP BY denial_reason_description
ORDER BY count DESC LIMIT 10;

-- Insurance with highest denial rate:
SELECT insurance_provider,
       COUNT(*) as total_claims,
       SUM(CASE WHEN claim_status = 'Denied' THEN 1 ELSE 0 END) as denied,
       ROUND(100.0 * SUM(CASE WHEN claim_status = 'Denied' THEN 1 ELSE 0 END) / COUNT(*), 1) as denial_rate
FROM claims_and_billing
GROUP BY insurance_provider
ORDER BY denial_rate DESC LIMIT 10;
"""


# ── Load all CSVs into pandas (used as in-memory SQL engine) ──────────────────
_dataframes = {}

def get_dataframes() -> dict:
    global _dataframes
    if not _dataframes:
        files = {
            "claims_and_billing": "claims_and_billing.csv",
            "denials":            "denials.csv",
            "patients":           "patients.csv",
            "encounters":         "encounters.csv",
            "diagnoses":          "diagnoses.csv",
            "lab_tests":          "lab_tests.csv",
            "medications":        "medications.csv",
            "procedures":         "procedures.csv",
            "providers":          "providers.csv",
        }
        for table, fname in files.items():
            path = DATA_DIR / fname
            if path.exists():
                _dataframes[table] = pd.read_csv(path)
    return _dataframes


# ── Step 1: Generate SQL from natural language ────────────────────────────────
def generate_sql(question: str, api_key: str) -> str:
    client = Groq(api_key=api_key)

    prompt = f"""You are an expert SQL query generator for a hospital database.

{DB_SCHEMA}

STRICT RULES:
1. Output ONLY a single valid SQL SELECT query — no explanations, no markdown, no backticks
2. Use ONLY column names exactly as listed in the schema above
3. Use standard SQLite-compatible SQL (no ILIKE, no FILTER, use CASE WHEN instead)
4. When joining tables, always verify the join column exists in BOTH tables
5. For procedures questions: use procedure_description column (NOT proc_description)
6. For doctor/provider questions: JOIN encounters e ON e.provider_id = providers.provider_id
7. Always add LIMIT 10 unless user asks for all
8. Use COUNT(*) for counting rows, COUNT(DISTINCT col) for unique values
9. Return ONLY the raw SQL query — nothing else at all

Question: {question}

SQL:"""

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,
        max_tokens=512,
    )

    sql = response.choices[0].message.content.strip()
    # Clean up any markdown if model adds it
    sql = re.sub(r"```sql|```", "", sql).strip()
    # Take only first statement
    sql = sql.split(";")[0].strip() + ";"
    return sql


# ── Step 2: Execute SQL using pandasql ───────────────────────────────────────
def execute_sql(sql: str) -> pd.DataFrame:
    """Execute SQL query using pandasql (works on CSV data like a real DB)."""
    try:
        import pandasql as psql
        dfs = get_dataframes()
        # Make dataframes available as local variables for pandasql
        env = {**dfs}
        result = psql.sqldf(sql, env)
        return result
    except ImportError:
        # Fallback: try with direct PostgreSQL if available
        try:
            from database.postgres_connector import get_connection
            import pandas as pd
            conn = get_connection()
            result = pd.read_sql(sql.rstrip(";"), conn)
            conn.close()
            return result
        except Exception as e:
            raise Exception(f"SQL execution failed. Install pandasql: pip install pandasql. Error: {e}")


# ── Step 3: Format result into natural language ───────────────────────────────
def format_answer(question: str, sql: str, result: pd.DataFrame, api_key: str) -> str:
    if result is None or result.empty:
        return "The query returned no results. The data may not contain matching records."

    # Limit to first 20 rows for context
    result_str = result.head(20).to_string(index=False)
    row_count  = len(result)

    client = Groq(api_key=api_key)

    prompt = f"""You are CA Hospital AI Agent. A user asked a question and we ran a SQL query to get real data.

User Question: {question}

SQL Query Used:
{sql}

Query Results ({row_count} rows returned):
{result_str}

Instructions:
- Give a clear, concise answer using the REAL data above
- Format numbers nicely (add commas, $ for currency, % for percentages)
- Use bullet points or numbered lists when listing multiple items
- Be specific — mention actual names, IDs, and numbers from the data
- If the result has many rows, summarize the top findings
- Do NOT mention SQL or technical details in your answer"""

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
        max_tokens=1024,
    )
    return response.choices[0].message.content.strip()


# ── Main function: question → SQL → execute → answer ─────────────────────────
def ask_with_sql(question: str, api_key: str) -> dict:
    """
    Full pipeline:
    1. Generate SQL from question
    2. Execute SQL on real data
    3. Format result as natural language answer
    Returns dict with answer, sql, and result dataframe
    """
    try:
        # Step 1: Generate SQL
        sql = generate_sql(question, api_key)

        # Step 2: Execute SQL
        result_df = execute_sql(sql)

        # Step 3: Format answer
        answer = format_answer(question, sql, result_df, api_key)

        return {
            "success": True,
            "answer":  answer,
            "sql":     sql,
            "rows":    len(result_df) if result_df is not None else 0,
            "data":    result_df,
        }

    except Exception as e:
        return {
            "success": False,
            "answer":  f"I encountered an issue processing your question: {str(e)}",
            "sql":     "",
            "rows":    0,
            "data":    None,
        }


if __name__ == "__main__":
    import os
    from dotenv import load_dotenv
    load_dotenv()

    api_key = os.getenv("GROQ_API_KEY", "")
    if not api_key:
        print("Set GROQ_API_KEY in .env first!")
        exit()

    questions = [
        "Which doctor handled the most patients?",
        "What are the top 5 denial reasons?",
        "Which insurance has the highest denial rate?",
        "What are the most prescribed medications?",
        "Which department has the highest readmission rate?",
    ]

    for q in questions:
        print(f"\nQ: {q}")
        result = ask_with_sql(q, api_key)
        print(f"SQL: {result['sql']}")
        print(f"A: {result['answer'][:200]}...")
        print("-" * 60)
