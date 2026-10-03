"""FastAPI serving the churn model."""

import os
from contextlib import asynccontextmanager

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException

from .schema import Customer, Health, Prediction

MODEL_PATH = os.getenv("MODEL_PATH", "outputs/model/model.pkl")
THRESHOLD = float(os.getenv("DECISION_THRESHOLD", "0.5"))

_model = None


def get_model():
    """Load the model once, on first use."""
    global _model
    if _model is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"No model at {MODEL_PATH}")
        _model = joblib.load(MODEL_PATH)
    return _model


@asynccontextmanager
async def lifespan(_: FastAPI):
    """Load at start-up, so a missing model fails immediately."""
    try:
        get_model()
        print(f"Model loaded from {MODEL_PATH}")
    except FileNotFoundError as error:
        print(f"WARNING: {error}")
    yield


app = FastAPI(
    title="Churn Prediction API",
    description="Predicts whether a customer is likely to leave.",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health", response_model=Health)
def health():
    """Liveness probe."""
    return Health(status="ok", model_loaded=_model is not None)


@app.post("/predict", response_model=Prediction)
def predict(customer: Customer):
    try:
        model = get_model()
    except FileNotFoundError as error:
        raise HTTPException(
            status_code=503, detail=str(error)
        ) from error

    frame = pd.DataFrame([customer.model_dump()])
    probability = float(model.predict_proba(frame)[0][1])

    return Prediction(
        churn=int(probability >= THRESHOLD),
        probability=round(probability, 4),
        threshold=THRESHOLD,
    )