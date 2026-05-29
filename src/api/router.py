import json
import os

from fastapi import APIRouter, HTTPException

from src.api.predictor import predict, get_loaded_models
from src.api.schemas import (
    PredictRequest,
    PredictResponse,
    BatchPredictRequest,
    ModelInfo,
)

router = APIRouter()

MODEL_DIR = os.getenv("MODEL_DIR", "models")


# ── health ────────────────────────────────────────────────────────────────────
@router.get("/health", tags=["Info"])
def health():
    models = get_loaded_models()
    return {
        "status": "ok" if models else "degraded",
        "models_loaded": models,
    }


# ── models list ───────────────────────────────────────────────────────────────
@router.get("/models", tags=["Info"], response_model=list[ModelInfo])
def list_models():
    """Returns training metrics for every model, from models/results.json."""
    results_path = os.path.join(MODEL_DIR, "results.json")
    if not os.path.exists(results_path):
        raise HTTPException(
            status_code=503,
            detail="results.json not found. Run train.py first.",
        )
    with open(results_path) as f:
        return json.load(f)


# ── single predict ────────────────────────────────────────────────────────────
@router.post("/predict", tags=["Prediction"], response_model=PredictResponse)
def predict_single(request: PredictRequest):
    # build features dict — handle aliased field names (v(g) etc.)
    features_dict = _features_to_dict(request.features)

    try:
        result = predict(features_dict, request.model)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    return PredictResponse(**result)


# ── batch predict ─────────────────────────────────────────────────────────────
@router.post("/predict/batch", tags=["Prediction"], response_model=list[PredictResponse])
def predict_batch(body: BatchPredictRequest):
    """Run up to 100 predictions in one call."""
    results = []
    for req in body.requests:
        features_dict = _features_to_dict(req.features)
        try:
            result = predict(features_dict, req.model)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        results.append(PredictResponse(**result))
    return results


# ── helper ────────────────────────────────────────────────────────────────────
def _features_to_dict(features) -> dict:
    """
    Convert Pydantic CodeFeatures → plain dict with the exact column names
    the pipeline produced (e.g. 'v(g)' not 'v_g').
    """
    alias_map = {
        "v_g":  "v(g)",
        "ev_g": "ev(g)",
        "iv_g": "iv(g)",
    }
    raw = features.model_dump(by_alias=True)
    # resolve any remaining snake_case → original column name
    result = {}
    for k, v in raw.items():
        result[alias_map.get(k, k)] = v
    return result