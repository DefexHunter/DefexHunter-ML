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

# ── main ──────────────────────────────────────────────────────────────────────
def train(data_path: str):
    os.makedirs(MODEL_DIR, exist_ok=True)

    # ── 1. build pipeline ────────────────────────────────
    print("=" * 55)
    print("STEP 1 — Running data pipeline")
    print("=" * 55)
    pipeline_result = build_pipeline(data_path)

    X_train          = pipeline_result["X_train"]
    X_test           = pipeline_result["X_test"]
    y_train          = pipeline_result["y_train"]
    y_test           = pipeline_result["y_test"]
    scaler           = pipeline_result["scaler"]
    selected_features = pipeline_result["selected_features"]

    # ── 2. save scaler + feature list ─────────────────────────────────────────
    joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler.pkl"))
    print(f"\nScaler saved → {MODEL_DIR}/scaler.pkl")

    with open(os.path.join(MODEL_DIR, "selected_features.json"), "w") as f:
        json.dump(selected_features, f)
    print(f"Features saved → {MODEL_DIR}/selected_features.json")
    print(f"Features ({len(selected_features)}): {selected_features}")

    # ── 3. train every model with GridSearchCV ────────────────────────────────
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    all_results = []

    for model_name, (estimator, param_grid) in MODEL_REGISTRY.items():
        print("\n" + "=" * 55)
        print(f"TRAINING — {model_name.upper()}")
        print("=" * 55)

        grid = GridSearchCV(
            estimator,
            param_grid,
            scoring="f1_macro",
            cv=cv,
            n_jobs=-1,
            verbose=1,
        )
        grid.fit(X_train, y_train)
        best_model = grid.best_estimator_

        print(f"Best params : {grid.best_params_}")
        print(f"Best CV F1  : {round(grid.best_score_, 4)}")

        # evaluate 
        metrics = evaluate_model(
            model_name, best_model, X_train, y_train, X_test, y_test
        )
        all_results.append(metrics)

        print(f"Test accuracy : {metrics['accuracy']}")
        print(f"Test F1 macro : {metrics['f1_score']}")
        print(f"ROC-AUC       : {metrics['roc_auc']}")

        # save model
        out_path = os.path.join(MODEL_DIR, f"{model_name}.pkl")
        joblib.dump(best_model, out_path)
        print(f"Saved → {out_path}")
    
    # ── 4. save results summary ───────────────────────────────────────────────
    results_path = os.path.join(MODEL_DIR, "results.json")
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved → {results_path}")

    # ── 5. print final leaderboard ────────────────────────────────────────────
    print("\n" + "=" * 55)
    print("FINAL LEADERBOARD (sorted by F1)")
    print("=" * 55)
    df = pd.DataFrame(all_results).sort_values("f1_score", ascending=False)
    print(df[["model", "accuracy", "f1_score", "roc_auc", "class_0_acc", "class_1_acc"]].to_string(index=False))

