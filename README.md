# 🏥 CA Hospital AI Agent — End-to-End Healthcare AI Platform

[![Live Demo](https://img.shields.io/badge/Live%20Demo-Streamlit-FF4B4B?style=for-the-badge&logo=streamlit)](https://cahospitalaiagent-4bphnlqclazayudtedf5qp.streamlit.app/)
[![Kaggle Dataset](https://img.shields.io/badge/Kaggle-Dataset-20BEFF?style=for-the-badge&logo=kaggle)](https://www.kaggle.com/datasets/rajkumarpadmanabhan/ca-hospital-dataset-q1-2025)
[![GitHub](https://img.shields.io/badge/GitHub-Raju--1209-181717?style=for-the-badge&logo=github)](https://github.com/Raju-1209/ca_hospital_ai_agent)
[![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=for-the-badge&logo=python)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-REST%20API-009688?style=for-the-badge&logo=fastapi)](https://fastapi.tiangolo.com)
[![Docker](https://img.shields.io/badge/Docker-Containerized-2496ED?style=for-the-badge&logo=docker)](https://docker.com)
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-Database-316192?style=for-the-badge&logo=postgresql)](https://postgresql.org)
[![BigQuery](https://img.shields.io/badge/GCP-BigQuery-4285F4?style=for-the-badge&logo=google-cloud)](https://cloud.google.com/bigquery)

> **A true End-to-End Healthcare AI Project** — from raw data creation and engineering, through PostgreSQL and GCP BigQuery ingestion, to a production-deployed AI platform with 5 ML models, Text-to-SQL AI Agent, Role-Based Authentication, FastAPI REST backend, and Docker containerization.

---

## 🔄 What Makes This Truly End-to-End?

Most projects start with someone else's dataset. This project starts from **zero** — the dataset itself was designed and built from scratch by the same author.

```
PHASE 1 — Data Engineering          PHASE 2 — Data Infrastructure
──────────────────────────          ──────────────────────────────
Designed 9-table schema        →    Loaded into PostgreSQL
Generated 126K+ records        →    Loaded into GCP BigQuery
Published on Kaggle            →    Smart fallback to CSV
SQL + Python (Faker, Pandas)        Production DB architecture

PHASE 3 — AI & ML Platform          PHASE 4 — Deployment
───────────────────────────         ────────────────────
5 ML Models trained            →    Dockerized (Dockerfile)
Text-to-SQL AI Agent           →    REST API (FastAPI)
Role-Based Auth System         →    Streamlit Cloud (Live)
RAG Pipeline (Groq LLaMA)      →    GitHub CI/CD
Interactive Dashboard               Public URL + Demo Credentials
```

---

## 🌐 Live Demo

**https://cahospitalaiagent-4bphnlqclazayudtedf5qp.streamlit.app/**

| Role | Username | Password | Access |
|------|----------|----------|--------|
| Admin | `admin` | `admin@123` | Full access to all features |
| Doctor | `doctor` | `doctor@123` | Clinical pages only |
| Analyst | `analyst` | `analyst@123` | Analytics and billing pages |

---

## Phase 1 — Dataset Creation (Data Engineering)

> **This dataset was designed, generated, and published by the same author — not downloaded from an external source.**

**Kaggle:** https://www.kaggle.com/datasets/rajkumarpadmanabhan/ca-hospital-dataset-q1-2025

### What Was Built

- Designed a **9-table relational schema** covering the full patient journey — from registration and visits to billing, claims, and denials
- Generated **126,000+ synthetic records** using Python (Faker, Pandas) with realistic ICD-10 codes, CPT procedure codes, drug names, and date ranges
- Ensured **referential integrity** across all 9 tables (patient → encounter → claim → denial chain)
- Published publicly on **Kaggle** for the healthcare ML and data science community

### Dataset Tables

| Table | Records | Description |
|-------|---------|-------------|
| `patients` | 60,000 | Demographics, insurance, location |
| `encounters` | 100,000+ | Hospital visits, departments, LOS |
| `claims_and_billing` | 100,000+ | Insurance claims, billed/paid amounts |
| `denials` | ~8,600 | Denied claims with reason codes |
| `diagnoses` | 150,000+ | ICD-10 diagnosis codes |
| `lab_tests` | 200,000+ | Lab results with normal ranges |
| `medications` | 180,000+ | Drug names, dosages, costs |
| `procedures` | 150,000+ | CPT procedure codes and costs |
| `providers` | 1,000+ | Doctor profiles and specialties |

**Tools Used:** Python, Faker, Pandas, SQL, Excel, Jupyter Notebook, VS Code

---

## Phase 2 — Data Infrastructure

After creating the dataset, all 9 tables were loaded into two production-grade cloud databases:

### PostgreSQL
- All 9 tables created with proper schema, indexes, and constraints
- Optimized queries with indexes on patient_id, encounter_id, claim_id

### GCP BigQuery
- All 9 tables uploaded to BigQuery dataset `hospital_dataset`
- Supports large-scale analytics and BI dashboard queries

### Smart Data Loader
The platform automatically detects the best available data source:

```
PostgreSQL available?  → Use PostgreSQL
BigQuery available?    → Use BigQuery
Neither connected?     → Use CSV files (never crashes)
```

---

## Phase 3 — AI and ML Platform

### 5 Machine Learning Models

| # | Model | Algorithm | Purpose |
|---|-------|-----------|---------|
| 1 | Claim Denial Prediction | XGBoost Classifier | Predict if a claim will be denied before submission |
| 2 | Readmission Prediction | Random Forest Classifier | Predict 30-day patient readmission risk |
| 3 | Fraud Detection | Isolation Forest | Detect anomalous billing patterns |
| 4 | High-Cost Patient | XGBoost Classifier | Identify patients likely to incur high costs |
| 5 | Length of Stay | XGBoost Regressor | Estimate days a patient will stay admitted |

- SMOTE applied for class imbalance handling
- Models auto-train on first launch — no manual setup needed
- All models saved as .pkl files via joblib

### Text-to-SQL AI Agent

Converts natural language to SQL, executes on real data, returns accurate answers:

```
User: "Which doctor handled the most patients?"

Groq LLaMA 3.3 70B generates:
  SELECT p.name, COUNT(DISTINCT e.patient_id) as patients
  FROM encounters e
  JOIN providers p ON e.provider_id = p.provider_id
  GROUP BY p.name ORDER BY patients DESC LIMIT 10;

Result: "Dr. Michael Hernandez handled the most patients with 342 visits"
```

Example questions:
- "Which insurance company has the highest denial rate?"
- "What are the top 5 denial reasons?"
- "Which department has the most readmissions?"
- "What medications were most commonly prescribed?"
- "Show abnormal lab results summary"
- "What is the most repeated procedure code?"

### Role-Based Authentication

```
Admin   → Full access: all 7 pages
Doctor  → Clinical: Dashboard, Readmission, High-Cost, LOS, Ask AI
Analyst → Analytics: Dashboard, Denial, Fraud, High-Cost, Ask AI
```

---

## Phase 4 — Deployment and DevOps

### REST API (FastAPI)

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | API health check |
| GET | `/model-status` | Check all 5 models |
| POST | `/predict-denial` | Claim denial prediction |
| POST | `/predict-readmission` | Readmission risk |
| POST | `/predict-fraud` | Fraud detection |
| POST | `/predict-high-cost` | High-cost patient flag |
| POST | `/predict-los` | Length of stay estimate |
| POST | `/ask` | AI Agent question answering |

Full interactive docs at: `http://localhost:8000/docs`

### Docker

```bash
# Launch full stack: Streamlit + FastAPI + PostgreSQL
docker-compose up --build

# Streamlit App  → http://localhost:8501
# FastAPI Docs   → http://localhost:8000/docs
# PostgreSQL     → localhost:5432
```

### Cloud Deployment
- Streamlit Cloud — auto-deploys on every GitHub push
- GitHub CI/CD — push to main branch triggers live update
- Secrets management — API keys stored in Streamlit Cloud secrets, never in code

---

## Project Structure

```
ca_hospital_ai_agent/
│
├── app.py                    # Main Streamlit application
├── auth.py                   # Login system and role management
├── api.py                    # FastAPI REST backend
├── Dockerfile                # Docker container
├── docker-compose.yml        # Full stack orchestration
├── requirements.txt          # Python dependencies
├── packages.txt              # System dependencies
├── .env.example              # Environment variables template
│
├── data/                     # 9 hospital CSV datasets (created for Kaggle)
│   ├── claims_and_billing.csv
│   ├── denials.csv
│   ├── diagnoses.csv
│   ├── encounters.csv
│   ├── lab_tests.csv
│   ├── medications.csv
│   ├── patients.csv
│   ├── procedures.csv
│   └── providers.csv
│
├── models/                   # ML layer
│   ├── train_models.py       # Training pipeline (auto-runs on first launch)
│   ├── model_utils.py        # Prediction utilities
│   └── saved/                # Trained .pkl files (auto-generated)
│
├── rag/                      # AI Agent layer
│   ├── rag_engine.py         # TF-IDF RAG engine
│   └── text_to_sql.py        # Text-to-SQL engine (Groq LLaMA 3.3 70B)
│
└── database/                 # Data infrastructure layer
    ├── postgres_connector.py # PostgreSQL integration
    ├── bigquery_connector.py # GCP BigQuery integration
    ├── data_loader.py        # Smart auto-fallback loader
    └── init.sql              # PostgreSQL schema and indexes
```

---

## Quick Start

### Run Locally

```bash
# 1. Clone the repo
git clone https://github.com/Raju-1209/ca_hospital_ai_agent.git
cd ca_hospital_ai_agent

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure environment
cp .env.example .env
# Edit .env and add your GROQ_API_KEY

# 4. Train ML models (first time only, takes 3-5 minutes)
python models/train_models.py

# 5. Launch the app
streamlit run app.py
```

### Run with Docker

```bash
git clone https://github.com/Raju-1209/ca_hospital_ai_agent.git
cd ca_hospital_ai_agent
cp .env.example .env
# Add your GROQ_API_KEY to .env

docker-compose up --build
```

### Connect Your Database

```env
# .env file
GROQ_API_KEY=your_groq_key

# PostgreSQL
POSTGRES_HOST=your-host
POSTGRES_DB=hospital_db
POSTGRES_USER=postgres
POSTGRES_PASSWORD=your-password

# GCP BigQuery
BIGQUERY_PROJECT_ID=your-project-id
BIGQUERY_DATASET_ID=hospital_dataset
GOOGLE_APPLICATION_CREDENTIALS=path/to/service-account.json
```

---

## Full Tech Stack

**Data Engineering:** Python, Faker, Pandas, SQL, Jupyter Notebook, Excel

**Machine Learning:** XGBoost, Scikit-learn, Random Forest, Isolation Forest, SMOTE, Joblib

**AI and NLP:** Groq API, LLaMA 3.3 70B, Text-to-SQL, TF-IDF, RAG Pipeline, pandasql

**Backend:** FastAPI, Uvicorn, Pydantic, REST API, Bearer Token Auth

**Frontend:** Streamlit, Plotly, Custom CSS, Role-Based UI

**Database:** PostgreSQL, GCP BigQuery, SQLAlchemy, psycopg2, google-cloud-bigquery

**DevOps and Cloud:** Docker, docker-compose, Streamlit Cloud, GitHub CI/CD, Secrets Management

---

## Security

- Role-based access control with 3 roles and different page permissions
- Password hashing with SHA-256
- API key authentication for all REST endpoints
- Secrets managed via environment variables — never hardcoded
- .env file excluded from version control via .gitignore
- GitHub secret scanning compliant

---

## Author

**Rajkumar Padmanabhan** — [@Raju-1209](https://github.com/Raju-1209)

- Kaggle Dataset: https://www.kaggle.com/datasets/rajkumarpadmanabhan/ca-hospital-dataset-q1-2025
- Live App: https://cahospitalaiagent-4bphnlqclazayudtedf5qp.streamlit.app/

---

## License

MIT License — free to use, modify, and distribute.

---

> Star this repo if you found it useful!
> This project demonstrates a complete pipeline from dataset creation to live production deployment — entirely built from scratch.
