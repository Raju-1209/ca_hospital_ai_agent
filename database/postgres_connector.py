"""
postgres_connector.py — PostgreSQL connection and data loader
Connects to PostgreSQL and loads all 9 hospital tables.
"""

import os
import pandas as pd
from pathlib import Path

# Try importing psycopg2
try:
    import psycopg2
    from psycopg2.extras import RealDictCursor
    PSYCOPG2_AVAILABLE = True
except ImportError:
    PSYCOPG2_AVAILABLE = False

# ── Connection config ─────────────────────────────────────────────────────────
def get_pg_config():
    return {
        "host":     os.getenv("POSTGRES_HOST",     "localhost"),
        "port":     int(os.getenv("POSTGRES_PORT", "5432")),
        "database": os.getenv("POSTGRES_DB",       "hospital_db"),
        "user":     os.getenv("POSTGRES_USER",     "postgres"),
        "password": os.getenv("POSTGRES_PASSWORD", "postgres"),
    }


def get_connection():
    if not PSYCOPG2_AVAILABLE:
        raise ImportError("psycopg2 not installed. Run: pip install psycopg2-binary")
    config = get_pg_config()
    return psycopg2.connect(**config)


def test_connection() -> bool:
    """Returns True if PostgreSQL is reachable."""
    try:
        conn = get_connection()
        conn.close()
        return True
    except Exception:
        return False


# ── Load any table as DataFrame ───────────────────────────────────────────────
def load_table(table_name: str) -> pd.DataFrame:
    """Load a full table from PostgreSQL into a DataFrame."""
    conn = get_connection()
    try:
        df = pd.read_sql(f"SELECT * FROM {table_name}", conn)
        return df
    finally:
        conn.close()


def load_table_filtered(table_name: str, where: str = "", limit: int = None) -> pd.DataFrame:
    """Load table with optional WHERE clause and LIMIT."""
    query = f"SELECT * FROM {table_name}"
    if where:
        query += f" WHERE {where}"
    if limit:
        query += f" LIMIT {limit}"
    conn = get_connection()
    try:
        return pd.read_sql(query, conn)
    finally:
        conn.close()


# ── All 9 table loaders ───────────────────────────────────────────────────────
def load_claims():
    return load_table("claims_and_billing")

def load_denials():
    return load_table("denials")

def load_diagnoses():
    return load_table("diagnoses")

def load_encounters():
    return load_table("encounters")

def load_lab_tests():
    return load_table("lab_tests")

def load_medications():
    return load_table("medications")

def load_patients():
    return load_table("patients")

def load_procedures():
    return load_table("procedures")

def load_providers():
    return load_table("providers")


# ── Upload CSVs to PostgreSQL ─────────────────────────────────────────────────
def upload_csv_to_postgres(data_dir: str = None):
    """
    One-time migration: Upload all 9 CSVs into PostgreSQL tables.
    Run this once to populate your database from CSV files.
    """
    if data_dir is None:
        data_dir = Path(__file__).resolve().parent.parent / "data"
    else:
        data_dir = Path(data_dir)

    tables = {
        "claims_and_billing": "claims_and_billing.csv",
        "denials":            "denials.csv",
        "diagnoses":          "diagnoses.csv",
        "encounters":         "encounters.csv",
        "lab_tests":          "lab_tests.csv",
        "medications":        "medications.csv",
        "patients":           "patients.csv",
        "procedures":         "procedures.csv",
        "providers":          "providers.csv",
    }

    conn = get_connection()
    try:
        from sqlalchemy import create_engine
        cfg = get_pg_config()
        engine = create_engine(
            f"postgresql://{cfg['user']}:{cfg['password']}@"
            f"{cfg['host']}:{cfg['port']}/{cfg['database']}"
        )
        for table, csv_file in tables.items():
            csv_path = data_dir / csv_file
            if csv_path.exists():
                df = pd.read_csv(csv_path)
                df.to_sql(table, engine, if_exists="replace", index=False)
                print(f"  ✅ Uploaded {table} ({len(df)} rows)")
            else:
                print(f"  ⚠️  {csv_file} not found, skipping")
        print("\n✅ All tables uploaded to PostgreSQL!")
    finally:
        conn.close()


if __name__ == "__main__":
    print("Testing PostgreSQL connection...")
    if test_connection():
        print("✅ Connected!")
        print("\nUploading CSVs to PostgreSQL...")
        upload_csv_to_postgres()
    else:
        print("❌ Connection failed. Check your .env settings.")
