"""
FastAPI Application for Fraud Detection
Production-ready API with proper error handling
"""

import os
import joblib
import pandas as pd
import json
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# FastAPI app
app = FastAPI(
    title="Fraud Detection API",
    description="ML-powered fraud detection",
    version="1.0.0"
)

# Global state
model = None
feature_names = None
app_state = {"ready": False}

class TransactionFeatures(BaseModel):
    """Transaction features (V1-V28 + Amount)"""
    V1: float
    V2: float
    V3: float
    V4: float
    V5: float
    V6: float
    V7: float
    V8: float
    V9: float
    V10: float
    V11: float
    V12: float
    V13: float
    V14: float
    V15: float
    V16: float
    V17: float
    V18: float
    V19: float
    V20: float
    V21: float
    V22: float
    V23: float
    V24: float
    V25: float
    V26: float
    V27: float
    V28: float
    Amount: float = Field(..., description="Transaction amount")
    
    class Config:
        json_schema_extra = {
            "example": {
                "V1": -1.36, "V2": -0.07, "V3": 2.54, "V4": 1.38,
                "V5": -0.34, "V6": 0.46, "V7": 0.24, "V8": 0.10,
                "V9": 0.36, "V10": 0.09, "V11": -0.55, "V12": -0.62,
                "V13": -0.99, "V14": -0.31, "V15": 1.47, "V16": -0.47,
                "V17": 0.21, "V18": 0.03, "V19": 0.40, "V20": 0.25,
                "V21": -0.02, "V22": 0.28, "V23": -0.11, "V24": 0.07,
                "V25": 0.13, "V26": -0.19, "V27": 0.13, "V28": -0.02,
                "Amount": 149.62
            }
        }

class PredictionResponse(BaseModel):
    is_fraud: int
    fraud_probability: float

@app.on_event("startup")
async def startup():
    global model, feature_names
    logger.info("Starting up...")
    
    try:
        model_path = "models/model.pkl"
        model = joblib.load(model_path)
        feature_names = model.feature_names_in_.tolist()
        app_state["ready"] = True
        logger.info(f"Model loaded: {len(feature_names)} features")
    except Exception as e:
        logger.error(f"Startup failed: {e}")
        app_state["ready"] = False

@app.get("/health")
async def health():
    return {"status": "healthy"}

@app.get("/ready")
async def ready():
    if app_state["ready"]:
        return {"status": "ready", "model_loaded": True}
    raise HTTPException(status_code=503, detail="Model not loaded")

@app.post("/predict", response_model=PredictionResponse)
async def predict(features: TransactionFeatures):
    if not app_state["ready"]:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Convert to DataFrame
        features_dict = features.dict()
        df = pd.DataFrame([features_dict])
        df = df[feature_names]
        
        # Predict
        proba = model.predict_proba(df)
        fraud_prob = float(proba[0][1])
        is_fraud = 1 if fraud_prob > 0.5 else 0
        
        logger.info(f"Prediction: {is_fraud}, Prob: {fraud_prob:.4f}")
        
        return PredictionResponse(
            is_fraud=is_fraud,
            fraud_probability=fraud_prob
        )
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def root():
    return {
        "name": "Fraud Detection API",
        "version": "1.0.0",
        "endpoints": ["/health", "/ready", "/predict", "/docs"]
    }
