"""
api.py — FastAPI REST API Layer
Endpoints:
  POST /predict-denial
  POST /predict-readmission
  POST /predict-fraud
  POST /predict-high-cost
  POST /predict-los
  POST /ask
  GET  /health
  GET  /model-status

Run: uvicorn api:app --host 0.0.0.0 --port 8000 --reload
Docs: http://localhost:8000/docs
"""

import os
import sys
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

# ── Path setup ────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

# ── App setup ─────────────────────────────────────────────────────────────────
app = FastAPI(
    title="CA Hospital AI Agent API",
    description="REST API for Hospital ML predictions and AI Q&A",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

security = HTTPBearer(auto_error=False)

# Simple API key auth (in production use JWT)
VALID_API_KEYS = {
    "admin-key-001":   "Admin",
    "doctor-key-002":  "Doctor",
    "analyst-key-003": "Analyst",
}


def verify_api_key(credentials: HTTPAuthorizationCredentials = Depends(security)):
    if credentials is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="API key required. Pass as Bearer token.",
        )
    if credentials.credentials not in VALID_API_KEYS:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key.",
        )
    return VALID_API_KEYS[credentials.credentials]


# ── Request / Response Models ─────────────────────────────────────────────────

class DenialRequest(BaseModel):
    insurance_provider: str = "BCBS"
    payment_method: str = "Insurance"
    billed_amount: float = 1500.0

    class Config:
        json_schema_extra = {
            "example": {
                "insurance_provider": "Medicare",
                "payment_method": "Insurance",
                "billed_amount": 2500.0,
            }
        }


class ReadmissionRequest(BaseModel):
    age: int = 55
    gender: str = "Male"
    insurance_type: str = "Medicare"
    visit_type: str = "Inpatient"
    department: str = "Cardiology"
    length_of_stay: int = 5


class FraudRequest(BaseModel):
    billed_amount: float = 2000.0
    total_proc_cost: float = 1200.0
    proc_count: int = 2
    insurance_provider: str = "BCBS"
    payment_method: str = "Insurance"


class HighCostRequest(BaseModel):
    age: int = 45
    gender: str = "Female"
    insurance_type: str = "Medicaid"
    marital_status: str = "Single"
    enc_count: int = 8


class LOSRequest(BaseModel):
    age: int = 60
    gender: str = "Male"
    insurance_type: str = "Medicare"
    visit_type: str = "Inpatient"
    department: str = "Cardiology"
    admission_type: str = "Emergency"
    has_chronic: bool = True


class AskRequest(BaseModel):
    question: str = "What are the top denial reasons?"
    groq_api_key: Optional[str] = None


class PredictionResponse(BaseModel):
    status: str
    result: dict
    model: str


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health", tags=["System"])
def health_check():
    """Check if API is running."""
    return {"status": "healthy", "service": "CA Hospital AI Agent API", "version": "1.0.0"}


@app.get("/model-status", tags=["System"])
def model_status(role: str = Depends(verify_api_key)):
    """Check if all 5 ML models are trained and available."""
    saved = BASE_DIR / "models" / "saved"
    models = ["denial_model", "readmission_model", "fraud_model",
              "high_cost_model", "los_model"]
    status_map = {m: (saved / f"{m}.pkl").exists() for m in models}
    all_ready = all(status_map.values())
    return {
        "all_models_ready": all_ready,
        "models": status_map,
    }


@app.post("/predict-denial", response_model=PredictionResponse, tags=["Predictions"])
def predict_denial(req: DenialRequest, role: str = Depends(verify_api_key)):
    """Predict whether a claim will be denied."""
    try:
        from models.model_utils import predict_denial
        result = predict_denial(req.insurance_provider, req.payment_method, req.billed_amount)
        return {"status": "success", "result": result, "model": "XGBoost Classifier"}
    except FileNotFoundError:
        raise HTTPException(status_code=503, detail="Model not trained yet. Run train_models.py first.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict-readmission", response_model=PredictionResponse, tags=["Predictions"])
def predict_readmission(req: ReadmissionRequest, role: str = Depends(verify_api_key)):
    """Predict 30-day readmission risk."""
    try:
        from models.model_utils import predict_readmission
        result = predict_readmission(
            req.age, req.gender, req.insurance_type,
            req.visit_type, req.department, req.length_of_stay
        )
        return {"status": "success", "result": result, "model": "Random Forest Classifier"}
    except FileNotFoundError:
        raise HTTPException(status_code=503, detail="Model not trained yet.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict-fraud", response_model=PredictionResponse, tags=["Predictions"])
def predict_fraud(req: FraudRequest, role: str = Depends(verify_api_key)):
    """Detect fraudulent billing patterns."""
    try:
        from models.model_utils import predict_fraud
        result = predict_fraud(
            req.billed_amount, req.total_proc_cost,
            req.proc_count, req.insurance_provider, req.payment_method
        )
        return {"status": "success", "result": result, "model": "Isolation Forest"}
    except FileNotFoundError:
        raise HTTPException(status_code=503, detail="Model not trained yet.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict-high-cost", response_model=PredictionResponse, tags=["Predictions"])
def predict_high_cost(req: HighCostRequest, role: str = Depends(verify_api_key)):
    """Predict if a patient will be a high-cost patient."""
    try:
        from models.model_utils import predict_high_cost
        result = predict_high_cost(
            req.age, req.gender, req.insurance_type,
            req.marital_status, req.enc_count
        )
        return {"status": "success", "result": result, "model": "XGBoost Classifier"}
    except FileNotFoundError:
        raise HTTPException(status_code=503, detail="Model not trained yet.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict-los", response_model=PredictionResponse, tags=["Predictions"])
def predict_los(req: LOSRequest, role: str = Depends(verify_api_key)):
    """Predict length of hospital stay in days."""
    try:
        from models.model_utils import predict_los
        result = predict_los(
            req.age, req.gender, req.insurance_type,
            req.visit_type, req.department,
            req.admission_type, int(req.has_chronic)
        )
        return {"status": "success", "result": result, "model": "XGBoost Regressor"}
    except FileNotFoundError:
        raise HTTPException(status_code=503, detail="Model not trained yet.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ask", tags=["AI Agent"])
def ask_ai_agent(req: AskRequest, role: str = Depends(verify_api_key)):
    """Ask the RAG AI Agent a question about hospital data."""
    try:
        from rag.rag_engine import ask_ai
        api_key = req.groq_api_key or os.getenv("GROQ_API_KEY", "")
        if not api_key:
            raise HTTPException(status_code=400, detail="Groq API key required.")
        answer = ask_ai(req.question, api_key)
        return {
            "status":   "success",
            "question": req.question,
            "answer":   answer,
            "model":    "LLaMA 3.3 70B via Groq + TF-IDF RAG",
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── Run ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
