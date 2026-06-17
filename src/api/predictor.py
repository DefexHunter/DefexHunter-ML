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

# Files in MODEL_DIR that are NOT a model (don't try to joblib.load these as a classifier).
_NON_MODEL_FILES = {"scaler.pkl"}

# ── loaded once at import time ────────────────────────────────────────────────
_scaler           = None
_models: dict     = {}
_selected_features: list = []


def load_artifacts():
    """
    Call this once at app startup (lifespan).

    Models are discovered by scanning MODEL_DIR for *.pkl files rather than
    a hardcoded name->filename map. train.py writes one <model_name>.pkl per
    entry in MODEL_REGISTRY, so this stays correct automatically whether
    MODEL_REGISTRY has 4 models or 6 (e.g. if lightgbm gets installed later).
    """
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

    if not os.path.isdir(MODEL_DIR):
        raise FileNotFoundError(f"Model directory not found: {MODEL_DIR}")

    for fname in sorted(os.listdir(MODEL_DIR)):
        if not fname.endswith(".pkl") or fname in _NON_MODEL_FILES:
            continue
        name = fname[: -len(".pkl")]
        path = os.path.join(MODEL_DIR, fname)
        try:
            _models[name] = joblib.load(path)
            print(f"  ✓ loaded {name}")
        except Exception as e:
            print(f"  ✗ failed to load {path}: {e}")

    if not _models:
        print("  ⚠ no model .pkl files found — did train.py run successfully?")

    print(f"Predictor ready. Models: {list(_models.keys())}")


def get_loaded_models() -> list:
    return list(_models.keys())


def predict(features_dict: dict, model_name: str) -> dict:
    """
    features_dict: raw field values from the Pydantic schema, keyed by the
                   exact uppercase training column names (e.g. 'CALL_PAIRS').
    model_name: one of the keys currently in _models (see get_loaded_models())

    Returns dict with prediction, label, probability, confidence.
    """
    if model_name not in _models:
        raise ValueError(
            f"Model '{model_name}' not loaded. Available: {list(_models.keys())}"
        )

    # build numpy array in exact column order from training
    vec = np.array([[features_dict.get(col, 0.0) for col in _selected_features]])

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