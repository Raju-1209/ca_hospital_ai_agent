"""
bigquery_connector.py — GCP BigQuery connection and data loader
Connects to BigQuery and loads all 9 hospital tables.
"""

import os
import pandas as pd
from pathlib import Path

# Try importing BigQuery
try:
    from google.cloud import bigquery
    from google.oauth2 import service_account
    BIGQUERY_AVAILABLE = True
except ImportError:
    BIGQUERY_AVAILABLE = False


# ── Connection config ─────────────────────────────────────────────────────────
def get_bq_client():
    if not BIGQUERY_AVAILABLE:
        raise ImportError("google-cloud-bigquery not installed. Run: pip install google-cloud-bigquery")

    credentials_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS", "")
    project_id       = os.getenv("BIGQUERY_PROJECT_ID", "")

    if not project_id:
        raise ValueError("BIGQUERY_PROJECT_ID not set in .env")

    if credentials_path and Path(credentials_path).exists():
        credentials = service_account.Credentials.from_service_account_file(
            credentials_path,
            scopes=["https://www.googleapis.com/auth/cloud-platform"],
        )
        return bigquery.Client(credentials=credentials, project=project_id)
    else:
        # Use Application Default Credentials (works on GCP Cloud Run, etc.)
        return bigquery.Client(project=project_id)


def get_dataset_id():
    return os.getenv("BIGQUERY_DATASET_ID", "hospital_dataset")


def test_connection() -> bool:
    """Returns True if BigQuery is reachable."""
    try:
        client = get_bq_client()
        list(client.list_datasets())
        return True
    except Exception:
        return False


# ── Load any table as DataFrame ───────────────────────────────────────────────
def load_table(table_name: str) -> pd.DataFrame:
    """Load a full table from BigQuery into a DataFrame."""
    client     = get_bq_client()
    dataset_id = get_dataset_id()
    project_id = os.getenv("BIGQUERY_PROJECT_ID", "")
    query      = f"SELECT * FROM `{project_id}.{dataset_id}.{table_name}`"
    return client.query(query).to_dataframe()


def load_table_filtered(table_name: str, where: str = "", limit: int = None) -> pd.DataFrame:
    """Load table with optional WHERE and LIMIT."""
    project_id = os.getenv("BIGQUERY_PROJECT_ID", "")
    dataset_id = get_dataset_id()
    query      = f"SELECT * FROM `{project_id}.{dataset_id}.{table_name}`"
    if where:
        query += f" WHERE {where}"
    if limit:
        query += f" LIMIT {limit}"
    client = get_bq_client()
    return client.query(query).to_dataframe()


# ── All 9 table loaders ───────────────────────────────────────────────────────
def load_claims():      return load_table("claims_and_billing")
def load_denials():     return load_table("denials")
def load_diagnoses():   return load_table("diagnoses")
def load_encounters():  return load_table("encounters")
def load_lab_tests():   return load_table("lab_tests")
def load_medications(): return load_table("medications")
def load_patients():    return load_table("patients")
def load_procedures():  return load_table("procedures")
def load_providers():   return load_table("providers")


# ── Upload CSVs to BigQuery ───────────────────────────────────────────────────
def upload_csv_to_bigquery(data_dir: str = None):
    """
    One-time migration: Upload all 9 CSVs into BigQuery tables.
    Run this once to populate your BigQuery dataset from CSV files.
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

    client     = get_bq_client()
    project_id = os.getenv("BIGQUERY_PROJECT_ID", "")
    dataset_id = get_dataset_id()

    # Create dataset if not exists
    dataset_ref = bigquery.Dataset(f"{project_id}.{dataset_id}")
    dataset_ref.location = "US"
    try:
        client.create_dataset(dataset_ref, exists_ok=True)
        print(f"✅ Dataset '{dataset_id}' ready.")
    except Exception as e:
        print(f"⚠️  Dataset creation: {e}")

    for table, csv_file in tables.items():
        csv_path = data_dir / csv_file
        if not csv_path.exists():
            print(f"  ⚠️  {csv_file} not found, skipping")
            continue

        df        = pd.read_csv(csv_path)
        table_ref = f"{project_id}.{dataset_id}.{table}"

        job_config = bigquery.LoadJobConfig(
            write_disposition=bigquery.WriteDisposition.WRITE_TRUNCATE,
            autodetect=True,
        )
        job = client.load_table_from_dataframe(df, table_ref, job_config=job_config)
        job.result()
        print(f"  ✅ Uploaded {table} ({len(df)} rows) → BigQuery")

    print("\n✅ All tables uploaded to BigQuery!")


if __name__ == "__main__":
    print("Testing BigQuery connection...")
    if test_connection():
        print("✅ Connected to BigQuery!")
        print("\nUploading CSVs to BigQuery...")
        upload_csv_to_bigquery()
    else:
        print("❌ Connection failed. Check BIGQUERY_PROJECT_ID and credentials.")
