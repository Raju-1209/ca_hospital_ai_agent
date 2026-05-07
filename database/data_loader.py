"""
data_loader.py — Smart Data Loader
Priority: PostgreSQL → BigQuery → CSV fallback
Automatically picks the best available data source.
"""

import os
import pandas as pd
from pathlib import Path
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"


def get_data_source() -> str:
    """
    Detect which data source is available.
    Returns: 'postgresql', 'bigquery', or 'csv'
    """
    # Check PostgreSQL
    pg_host = os.getenv("POSTGRES_HOST", "")
    if pg_host and pg_host not in ("", "localhost", "your-postgres-host"):
        try:
            from database.postgres_connector import test_connection
            if test_connection():
                return "postgresql"
        except Exception:
            pass

    # Check BigQuery
    bq_project = os.getenv("BIGQUERY_PROJECT_ID", "")
    if bq_project and bq_project not in ("", "your-project-id"):
        try:
            from database.bigquery_connector import test_connection
            if test_connection():
                return "bigquery"
        except Exception:
            pass

    # Fallback to CSV
    return "csv"


# Cache the source detection
_data_source = None

def data_source() -> str:
    global _data_source
    if _data_source is None:
        _data_source = get_data_source()
    return _data_source


def source_badge() -> str:
    """Return a colored badge showing current data source."""
    src = data_source()
    badges = {
        "postgresql": "🐘 PostgreSQL",
        "bigquery":   "☁️ GCP BigQuery",
        "csv":        "📁 CSV Files",
    }
    return badges.get(src, "📁 CSV Files")


# ── Generic loader ────────────────────────────────────────────────────────────
def load(table_name: str) -> pd.DataFrame:
    """
    Load a table from the best available source.
    PostgreSQL → BigQuery → CSV
    """
    src = data_source()

    if src == "postgresql":
        try:
            from database.postgres_connector import load_table
            return load_table(table_name)
        except Exception as e:
            print(f"PostgreSQL failed ({e}), trying BigQuery...")

    if src == "bigquery":
        try:
            from database.bigquery_connector import load_table
            return load_table(table_name)
        except Exception as e:
            print(f"BigQuery failed ({e}), falling back to CSV...")

    # CSV fallback
    csv_path = DATA_DIR / f"{table_name}.csv"
    if csv_path.exists():
        return pd.read_csv(csv_path)
    raise FileNotFoundError(f"No data source available for table: {table_name}")


# ── Individual table loaders ──────────────────────────────────────────────────
def load_claims():
    return load("claims_and_billing")

def load_denials():
    return load("denials")

def load_diagnoses():
    return load("diagnoses")

def load_encounters():
    return load("encounters")

def load_lab_tests():
    return load("lab_tests")

def load_medications():
    return load("medications")

def load_patients():
    return load("patients")

def load_procedures():
    return load("procedures")

def load_providers():
    return load("providers")


def load_all() -> dict:
    """Load all 9 tables and return as a dictionary of DataFrames."""
    return {
        "claims":      load_claims(),
        "denials":     load_denials(),
        "diagnoses":   load_diagnoses(),
        "encounters":  load_encounters(),
        "lab_tests":   load_lab_tests(),
        "medications": load_medications(),
        "patients":    load_patients(),
        "procedures":  load_procedures(),
        "providers":   load_providers(),
    }


if __name__ == "__main__":
    print(f"Data source: {data_source()}")
    print(f"Badge: {source_badge()}")
    claims = load_claims()
    print(f"Claims loaded: {len(claims)} rows")
