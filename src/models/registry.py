import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

try:
    from lightgbm import LGBMClassifier
    _HAS_LGBM = True
except ImportError:
    _HAS_LGBM = False


class FeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Adds non-linear / interaction features on top of whatever pipeline.py
    hands it (already correlation-filtered and scaled). Fit on the
    training fold only, so it's safe inside GridSearchCV - no leakage.
    Input is already scaled (can be negative), so log/sqrt use a
    sign-safe version: sign(x) * f(|x|).
    """

    def __init__(self, top_k=6):
        self.top_k = top_k

    def fit(self, X, y=None):
        X = pd.DataFrame(X).reset_index(drop=True)
        y = pd.Series(np.asarray(y)).reset_index(drop=True)

        corr = X.apply(lambda col: col.corr(y)).abs().fillna(0)
        self.top_features_ = corr.sort_values(ascending=False).head(self.top_k).index.tolist()
        self.input_columns_ = X.columns.tolist()
        return self

    def transform(self, X):
        X = pd.DataFrame(np.asarray(X), columns=self.input_columns_)
        new_cols = {}

        for col in self.top_features_:
            new_cols[f"sq_{col}"] = X[col] ** 2
            new_cols[f"cube_{col}"] = X[col] ** 3
            new_cols[f"sqrt_{col}"] = np.sign(X[col]) * np.sqrt(np.abs(X[col]))
            new_cols[f"log_{col}"] = np.sign(X[col]) * np.log1p(np.abs(X[col]))

        for i, a in enumerate(self.top_features_):
            for b in self.top_features_[i + 1:]:
                new_cols[f"{a}_x_{b}"] = X[a] * X[b]

        if self.top_features_:
            new_cols["risk_score"] = X[self.top_features_].mean(axis=1)

        return pd.concat([X, pd.DataFrame(new_cols, index=X.index)], axis=1)


def _pipe(clf):
    return Pipeline([("feat_eng", FeatureEngineer()), ("clf", clf)])


MODEL_REGISTRY = {
    "decision_tree": (
        _pipe(DecisionTreeClassifier(random_state=42)),
        {
            "clf__max_depth": [None, 5, 10, 20],
            "clf__min_samples_split": [2, 5, 10],
            "clf__min_samples_leaf": [1, 2, 4],
            "clf__criterion": ["gini", "entropy"],
        },
    ),
    "knn": (
        _pipe(KNeighborsClassifier()),
        {
            "clf__n_neighbors": [3, 5, 7, 9, 11, 15],
            "clf__weights": ["uniform", "distance"],
            "clf__metric": ["euclidean", "manhattan"],
        },
    ),
    "random_forest": (
        _pipe(RandomForestClassifier(random_state=42)),
        {
            "clf__n_estimators": [200, 300, 500],
            "clf__max_depth": [None, 10, 20, 30],
            "clf__min_samples_split": [2, 5, 10],
            "clf__min_samples_leaf": [1, 2, 4],
            "clf__max_features": ["sqrt", "log2"],
        },
    ),
    "xgboost": (
        _pipe(XGBClassifier(eval_metric="logloss", random_state=42)),
        {
            "clf__n_estimators": [100, 200, 300],
            "clf__max_depth": [3, 5, 7],
            "clf__learning_rate": [0.01, 0.05, 0.1],
            "clf__subsample": [0.8, 1],
            "clf__colsample_bytree": [0.8, 1],
            "clf__min_child_weight": [1, 3, 5],
            "clf__scale_pos_weight": [1, 2, 3],
        },
    ),
}

# Optional 6th model - only registers if lightgbm is installed
if _HAS_LGBM:
    MODEL_REGISTRY["lightgbm"] = (
        _pipe(LGBMClassifier(random_state=42, verbose=-1)),
        {
            "clf__n_estimators": [200, 300, 500],
            "clf__num_leaves": [15, 31, 63],
            "clf__learning_rate": [0.01, 0.05, 0.1],
            "clf__subsample": [0.8, 1],
            "clf__colsample_bytree": [0.8, 1],
        },
    )