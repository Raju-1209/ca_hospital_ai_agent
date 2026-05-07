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

TABLE: claims_and_billing
  claim_id           VARCHAR  -- unique claim ID (e.g. CLM000001)
  patient_id         VARCHAR  -- patient ID (e.g. PAT000001)
  encounter_id       VARCHAR  -- encounter ID (e.g. ENC000001)
  insurance_provider VARCHAR  -- BCBS, Medicare, Medicaid, Aetna, UHC, Cigna
  payment_method     VARCHAR  -- Insurance, Self-Pay, Government
  billed_amount      DECIMAL  -- total amount billed
  paid_amount        DECIMAL  -- amount actually paid
  claim_status       VARCHAR  -- Approved, Denied, Pending
  denial_reason      VARCHAR  -- reason if denied
  claim_date         DATE

TABLE: denials
  denial_id                 VARCHAR
  claim_id                  VARCHAR
  denial_reason_code        VARCHAR
  denial_reason_description TEXT    -- full description of denial reason
  denied_amount             DECIMAL
  denial_date               DATE
  appeal_filed              VARCHAR -- Yes/No
  appeal_status             VARCHAR
  final_outcome             VARCHAR

TABLE: patients
  patient_id     VARCHAR
  first_name     VARCHAR
  last_name      VARCHAR
  age            INTEGER
  gender         VARCHAR -- Male, Female, Other
  ethnicity      VARCHAR
  insurance_type VARCHAR
  marital_status VARCHAR
  city           VARCHAR
  state          VARCHAR

TABLE: encounters
  encounter_id     VARCHAR
  patient_id       VARCHAR
  provider_id      VARCHAR  -- links to providers table
  visit_date       DATE
  discharge_date   DATE
  visit_type       VARCHAR  -- Inpatient, Outpatient, Emergency, Telehealth
  department       VARCHAR  -- Cardiology, Emergency Department, etc.
  reason_for_visit TEXT
  diagnosis_code   VARCHAR
  admission_type   VARCHAR  -- Emergency, Elective, Urgent
  length_of_stay   INTEGER  -- days
  readmitted_flag  VARCHAR  -- Yes/No

TABLE: diagnoses
  diagnosis_id          VARCHAR
  encounter_id          VARCHAR
  diagnosis_code        VARCHAR
  diagnosis_description TEXT
  primary_flag          VARCHAR -- Yes/No
  chronic_flag          VARCHAR -- Yes/No (True/False)

TABLE: lab_tests
  lab_test_id  VARCHAR
  encounter_id VARCHAR
  test_name    VARCHAR
  test_result  VARCHAR  -- Normal, Abnormal, Critical
  units        VARCHAR
  normal_range VARCHAR
  test_date    DATE
  status       VARCHAR

TABLE: medications
  medication_id   VARCHAR
  encounter_id    VARCHAR
  drug_name       VARCHAR
  dosage          VARCHAR
  route           VARCHAR
  frequency       VARCHAR
  duration        VARCHAR
  prescribed_date DATE
  cost            DECIMAL

TABLE: procedures
  procedure_id          VARCHAR
  encounter_id          VARCHAR
  procedure_code        VARCHAR
  procedure_description TEXT
  procedure_date        DATE
  procedure_cost        DECIMAL

TABLE: providers
  provider_id      VARCHAR
  name             VARCHAR  -- doctor/provider name
  department       VARCHAR
  specialty        VARCHAR
  years_experience INTEGER
  location         VARCHAR
  inhouse          VARCHAR

KEY RELATIONSHIPS:
  encounters.patient_id  → patients.patient_id
  encounters.provider_id → providers.provider_id
  encounters.encounter_id → claims_and_billing.encounter_id
  encounters.encounter_id → diagnoses.encounter_id
  encounters.encounter_id → lab_tests.encounter_id
  encounters.encounter_id → medications.encounter_id
  encounters.encounter_id → procedures.encounter_id
  claims_and_billing.claim_id → denials.claim_id
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

RULES:
1. Generate ONLY a single valid SQL SELECT query — no explanations, no markdown, no ```sql blocks
2. Use standard SQL compatible with SQLite/PostgreSQL
3. Always use table aliases for clarity
4. For "which doctor" questions, JOIN encounters with providers on provider_id
5. For counting patients per doctor: COUNT(DISTINCT patient_id)
6. Use LOWER() for case-insensitive string comparisons
7. Always add LIMIT 10 unless asking for all records
8. For denial reasons, use the denials table denial_reason_description column
9. Return ONLY the SQL query, nothing else

Question: {question}

SQL Query:"""

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
