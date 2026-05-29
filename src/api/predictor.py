"""
src/api/predictor.py
Loads all model artifacts once at startup.
predict() is stateless — safe to call from any number of requests.
"""
import json
import os
from typing import Optional

import joblib
import numpy as np

MODEL_DIR = os.getenv("MODEL_DIR", "models")

# ── loaded once at import time ────────────────────────────────────────────────
_scaler           = None
_models: dict     = {}
_selected_features: list = []


def load_artifacts():
    """Call this once at app startup (lifespan)."""
    global _scaler, _models, _selected_features

    scaler_path   = os.path.join(MODEL_DIR, "scaler.pkl")
    features_path = os.path.join(MODEL_DIR, "selected_features.json")

    if not os.path.exists(scaler_path):
        raise FileNotFoundError(
            f"Scaler not found at {scaler_path}. Run: python src/train.py data/jm1_csv.csv"
        )

    _scaler = joblib.load(scaler_path)

    with open(features_path) as f:
        _selected_features = json.load(f)

    model_files = {
        "decision_tree": "decision_tree.pkl",
        "knn":           "knn.pkl",
        "random_forest": "random_forest.pkl",
        "svm":           "svm.pkl",
        "xgboost":       "xgboost.pkl",
    }

    for name, fname in model_files.items():
        path = os.path.join(MODEL_DIR, fname)
        if os.path.exists(path):
            _models[name] = joblib.load(path)
            print(f"  ✓ loaded {name}")
        else:
            print(f"  ✗ missing {path} — skipping")

    print(f"Predictor ready. Models: {list(_models.keys())}")


def get_loaded_models() -> list:
    return list(_models.keys())


def predict(features_dict: dict, model_name: str) -> dict:
    """
    features_dict: raw field values from the Pydantic schema
    model_name: one of the keys in MODEL_REGISTRY

    Returns dict with prediction, label, probability, confidence.
    """
    if model_name not in _models:
        raise ValueError(
            f"Model '{model_name}' not loaded. Available: {list(_models.keys())}"
        )

    # build numpy array in exact column order from training
    try:
        vec = np.array([[features_dict[col] for col in _selected_features]])
    except KeyError as e:
        raise ValueError(f"Missing feature: {e}. Expected: {_selected_features}")

    # scale using the fitted scaler
    vec_scaled = _scaler.transform(vec)

    model      = _models[model_name]
    prediction = int(model.predict(vec_scaled)[0])

    # probability
    probability: Optional[float] = None
    try:
        proba       = model.predict_proba(vec_scaled)[0]
        probability = round(float(proba[prediction]), 4)
    except AttributeError:
        pass

    # confidence band
    if probability is None:
        confidence = "Unknown"
    elif probability >= 0.75:
        confidence = "High"
    elif probability >= 0.55:
        confidence = "Medium"
    else:
        confidence = "Low"

    return {
        "prediction":  prediction,
        "label":       "Defective" if prediction == 1 else "No Defect",
        "probability": probability,
        "confidence":  confidence,
        "model_used":  model_name,
    }