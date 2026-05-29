import sys
import json
import os

import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
    KFold,
    cross_val_score,
)
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)

# ── local imports ─────────────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.data.pipeline import build_pipeline
from src.models.registry import MODEL_REGISTRY

MODEL_DIR = "models"


# ── evaluation ───────────────────────────
def evaluate_model(name, model, X_tr, y_tr, X_te, y_te):
    y_pred  = model.predict(X_te)
    y_proba = model.predict_proba(X_te)[:, 1]

    acc    = accuracy_score(y_te, y_pred)
    kfold  = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_acc = cross_val_score(model, X_tr, y_tr, cv=kfold).mean()

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_te, y_pred, average="macro"
    )
    cm      = confusion_matrix(y_te, y_pred)
    roc_auc = roc_auc_score(y_te, y_proba)

    return {
        "model":       name,
        "accuracy":    round(float(acc),    4),
        "cv_accuracy": round(float(cv_acc), 4),
        "precision":   round(float(precision), 4),
        "recall":      round(float(recall),    4),
        "f1_score":    round(float(f1),        4),
        "roc_auc":     round(float(roc_auc),   4),
        "class_0_acc": round(float(cm[0, 0] / cm[0].sum()), 4),
        "class_1_acc": round(float(cm[1, 1] / cm[1].sum()), 4),
    }

